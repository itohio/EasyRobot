package kpu

import (
	"encoding/binary"
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
)

func le32(v ...uint32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}

const (
	w, h, inC, outC = 16, 16, 3, 4
	n               = w * h * outC
	argsOff         = 128
	wOff, bnOff     = 256, 384
	actOff          = 512
	srcAddr         = 100
	dstAddr         = 300
)

// model is conv 16×16×3 → 16×16×4 with main-memory output at 0, then
// Dequantize (scale 0.5, bias −1) to floats at n; output = the floats.
func model(t *testing.T) *kmodel.Model {
	t.Helper()
	var f []byte
	f = append(f, le32(3, 1, 0, 2, 0, n+4*n, 1)...)
	f = append(f, le32(n, 4*n)...)
	f = append(f, le32(uint32(kmodel.K210Conv), 24, uint32(kmodel.Dequantize), 24)...)
	f = append(f, le32(kmodel.FlagMainMemOut, 0, argsOff, wOff, bnOff, actOff)...)
	f = append(f, le32(0, 0, n, n, math.Float32bits(0.5), math.Float32bits(-1))...)
	f = append(f, make([]byte, actOff+256-len(f))...)
	// Parameters for the emulated engine: 1×1 weights, BN ×1, identity
	// activation (segment 10 at 0 with mul 1, segments 11–15 out of reach).
	for i := range inC * outC {
		f[wOff+i] = byte(i + 1)
	}
	for c := range outC {
		binary.LittleEndian.PutUint64(f[bnOff+8*c:], 1|uint64(c)<<24) // mul 1, add c, shift 0
	}
	for i := range 16 {
		start := uint64(0x800000000) // −2^35
		var mul uint64
		switch {
		case i == 10:
			start, mul = 0, 1
		case i > 10:
			start = 0x7_ffff_fff0 + uint64(i)
		}
		binary.LittleEndian.PutUint64(f[actOff+8*i:], start<<24|mul<<8)
	}
	var a kmodel.Args
	a[1] = srcAddr | dstAddr<<32
	a[2] = (inC - 1) | (outC-1)<<32
	a[3] = (w - 1) | (h-1)<<10 | (w-1)<<32 | (h-1)<<42
	a[7] = 16 | 1<<16 | 4<<28                  // input layout: 4 per line, 1 line per row, 16 per group
	a[8] = 16 | 1<<16 | 4<<20                  // output layout, same
	a[11] = 1 | (w*h-1)<<16 | (w*h*outC-1)<<32 // send_data_out (the runner must clear it) and byte counts
	for i, v := range a {
		binary.LittleEndian.PutUint64(f[argsOff+8*i:], v)
	}
	m, err := kmodel.Parse(AlignedCopy(f))
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// base stands in for the model's address: host heaps lie above 4 GiB, the
// K210's SRAM at 0x8000_0000.
const base = uintptr(0x8010_0000)

// fake is a KPU that records the layer arguments and writes a known output
// tensor at the destination address.
type fake struct {
	mem  []byte
	args []kmodel.Args
	err  error
}

func (f *fake) ram() []byte { return f.mem }

func (f *fake) now() uint64 { return 0 }

func (f *fake) conv(a *kmodel.Args) error {
	f.args = append(f.args, *a)
	if f.err != nil {
		return f.err
	}
	ow, oh, oc := a.OutShape()
	out := make([]byte, ow*oh*oc)
	for i := range out {
		out[i] = byte(i * 3)
	}
	return a.OutLayout().UploadTo(f.mem, a.DstAddr(), ow, oh, oc, out)
}

func TestRun(t *testing.T) {
	m := model(t)
	e := &fake{mem: make([]byte, kmodel.RAMSize)}
	r, err := newRunner(m, make([]byte, m.MainMemUsage), base, e)
	if err != nil {
		t.Fatal(err)
	}
	if r.InputSize() != w*h*inC {
		t.Fatalf("InputSize %d", r.InputSize())
	}
	in := make([]byte, w*h*inC)
	for i := range in {
		in[i] = byte(i ^ 0x5a)
	}
	if err := r.Run(in); err != nil {
		t.Fatal(err)
	}
	// Input uploaded at the first conv's source address in its input layout.
	got := make([]byte, len(in))
	if err := (kmodel.Layout{Group: 4, RowLines: 1, ChanLines: 16}).DownloadFrom(e.mem, srcAddr, w, h, inC, got); err != nil || string(got) != string(in) {
		t.Fatalf("input not uploaded at the source address (err %v)", err)
	}
	// Layer argument patched like kpu_conv, with int_en on and the FIFO
	// output off.
	a := e.args[0]
	if uint64(a[5]>>32) != uint64(base)+wOff || uint64(a[4]>>32) != uint64(base)+bnOff || uint64(a[7]>>32) != uint64(base)+actOff {
		t.Errorf("addresses: weights %#x bn %#x act %#x, base %#x", a[5]>>32, a[4]>>32, a[7]>>32, base)
	}
	if a[0]&1 == 0 || a[11]&1 != 0 {
		t.Errorf("int_en %d send_data_out %d, want 1 0", a[0]&1, a[11]&1)
	}
	if a[7]&0xffff_ffff != 16|1<<16|4<<28 {
		t.Errorf("input layout bits changed: %#x", a[7]&0xffff_ffff)
	}
	// Output read back from KPU memory as dense CHW and dequantised.
	o := r.Output(0)
	for i := range n {
		f := math.Float32frombits(binary.LittleEndian.Uint32(o[4*i:]))
		if want := float32(byte(i*3))*0.5 - 1; f != want {
			t.Fatalf("output %d: %v, want %v", i, f, want)
		}
	}
}

func TestRunErrors(t *testing.T) {
	m := model(t)
	main := make([]byte, m.MainMemUsage)
	if _, err := newRunner(m, main, base+8, &fake{}); !errors.Is(err, ErrAlign) {
		t.Errorf("misaligned base: %v", err)
	}
	if _, err := newRunner(m, main, 1<<32-128, &fake{}); !errors.Is(err, ErrAlign) {
		t.Errorf("model crossing 4 GiB: %v", err)
	}
	if _, err := newRunner(m, main[:len(main)-1], base, &fake{}); err == nil {
		t.Error("short main memory accepted")
	}
	e := &fake{mem: make([]byte, kmodel.RAMSize)}
	r, err := newRunner(m, main, base, e)
	if err != nil {
		t.Fatal(err)
	}
	if err := r.Run(make([]byte, 10)); !errors.Is(err, ErrInput) {
		t.Errorf("wrong input size: %v", err)
	}
	e.err = ErrTimeout
	if err := r.Run(make([]byte, w*h*inC)); !errors.Is(err, ErrTimeout) {
		t.Errorf("engine error not returned: %v", err)
	}
}

func TestNewOffK210(t *testing.T) {
	if _, err := New(model(t), nil); !errors.Is(err, ErrNoKPU) {
		t.Errorf("New on the host: %v", err)
	}
}

func TestNewEmulated(t *testing.T) {
	m := model(t)
	r, err := NewEmulated(m, make([]byte, m.MainMemUsage))
	if err != nil {
		t.Fatal(err)
	}
	in := make([]byte, w*h*inC)
	for i := range in {
		in[i] = byte(i % 7)
	}
	if err := r.Run(in); err != nil {
		t.Fatal(err)
	}
	// The same conv computed directly from the file's parameters.
	c, _ := m.Layers[0].Conv()
	p, err := m.ConvParams(c)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]byte, n)
	if err := p.Emulate(in, want); err != nil {
		t.Fatal(err)
	}
	o := r.Output(0)
	for i := range n {
		f := math.Float32frombits(binary.LittleEndian.Uint32(o[4*i:]))
		if f != float32(want[i])*0.5-1 {
			t.Fatalf("output %d: %v, want %v", i, f, float32(want[i])*0.5-1)
		}
	}
	if want[0] == want[1] && want[1] == want[2] {
		t.Fatal("test model gives a constant output; the check would be weak")
	}
}

// quiet is a fake KPU that allocates nothing: it writes a constant output
// tensor and ignores the arguments' contents.
type quiet struct {
	mem []byte
	out []byte
}

func (q *quiet) ram() []byte { return q.mem }
func (q *quiet) now() uint64 { return 0 }
func (q *quiet) conv(a *kmodel.Args) error {
	ow, oh, oc := a.OutShape()
	return a.OutLayout().UploadTo(q.mem, a.DstAddr(), ow, oh, oc, q.out)
}

// TestRunAllocatesNothing pins Run's zero-allocation promise on the host
// compiler too (the layer argument escaped through
// the engine interface once per conv).
func TestRunAllocatesNothing(t *testing.T) {
	m := model(t)
	e := &quiet{mem: make([]byte, kmodel.RAMSize), out: make([]byte, n)}
	r, err := newRunner(m, make([]byte, m.MainMemUsage), base, e)
	if err != nil {
		t.Fatal(err)
	}
	in := make([]byte, w*h*inC)
	if a := testing.AllocsPerRun(20, func() {
		if err := r.Run(in); err != nil {
			t.Fatal(err)
		}
	}); a != 0 {
		t.Errorf("Run allocates %v times per call", a)
	}
}

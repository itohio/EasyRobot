package kmodel

import (
	"bytes"
	"encoding/binary"
	"errors"
	"math"
	"testing"
)

// builder assembles a kmodel v3 file the way ncc lays it out: header,
// outputs, layer table, bodies, then (at 128-byte aligned offsets) conv
// layer arguments and parameter blobs.
type builder struct {
	flags, mainMem uint32
	outputs        []Output
	layers         []Layer
	tail           []byte // appended after the bodies; offsets fixed up by the caller
}

func le32(v ...uint32) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], x)
	}
	return b
}

func (b *builder) bytes(version, arch uint32) []byte {
	var out []byte
	out = append(out, le32(version, b.flags, arch, uint32(len(b.layers)), 0, b.mainMem, uint32(len(b.outputs)))...)
	for _, o := range b.outputs {
		out = append(out, le32(o.Address, o.Size)...)
	}
	for _, l := range b.layers {
		out = append(out, le32(uint32(l.Type), uint32(len(l.Body)))...)
	}
	for _, l := range b.layers {
		out = append(out, l.Body...)
	}
	return append(out, b.tail...)
}

// convArgs builds a layer argument for an in w×h×c → out w×h×oc conv with
// the given layouts and KPU addresses.
func convArgs(src, dst uint32, w, h, c, oc int, in, out Layout) Args {
	var a Args
	a[1] = uint64(src) | uint64(dst)<<32
	a[2] = uint64(c-1) | uint64(oc-1)<<32
	a[3] = uint64(w-1) | uint64(h-1)<<10 | uint64(w-1)<<32 | uint64(h-1)<<42
	a[7] = uint64(in.ChanLines) | uint64(in.RowLines)<<16 | uint64(in.Group)<<28
	a[8] = uint64(out.ChanLines) | uint64(out.RowLines)<<16 | uint64(out.Group)<<20
	a[11] = 1 // send_data_out set in the file: the runner must clear it
	return a
}

// tinyModel is conv (16×16×3 → 16×16×4, main-memory output at 0) then
// Dequantize (4·256 bytes at 0 → floats at 1024), output the floats.
func tinyModel(t testing.TB) []byte {
	t.Helper()
	const w, h, c, oc = 16, 16, 3, 4
	n := uint32(w * h * oc)
	b := builder{flags: 1, mainMem: n + 4*n, outputs: []Output{{n, 4 * n}}}
	// Bodies: conv (24 B) + dequantize (24 B); table and header first.
	head := uint32(28 + 8 + 2*8 + 24 + 24)
	argsOff := (head + 127) &^ 127
	wOff, bnOff, actOff := argsOff+128, argsOff+256, argsOff+384
	b.layers = []Layer{
		{K210Conv, le32(FlagMainMemOut, 0, argsOff, wOff, bnOff, actOff)},
		{Dequantize, le32(0, 0, n, n, math.Float32bits(0.5), math.Float32bits(-1))},
	}
	a := convArgs(100, 200, w, h, c, oc, UploadLayout(w, h), UploadLayout(w, h))
	tail := make([]byte, actOff+128-head)
	for i, v := range a {
		binary.LittleEndian.PutUint64(tail[argsOff-head+uint32(8*i):], v)
	}
	b.tail = tail
	return b.bytes(3, 0)
}

func TestParse(t *testing.T) {
	data := tinyModel(t)
	m, err := Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	if !m.EightBitMode() || m.MainMemUsage != 5*1024 || len(m.Layers) != 2 || len(m.Outputs) != 1 {
		t.Fatalf("header: %+v", m)
	}
	if m.Layers[0].Type != K210Conv || m.Layers[1].Type != Dequantize {
		t.Fatalf("types %v %v", m.Layers[0].Type, m.Layers[1].Type)
	}
	if err := m.Check(); err != nil {
		t.Fatal(err)
	}
	c, _ := m.Layers[0].Conv()
	a := m.Args(c)
	if a.SrcAddr() != 100 || a.DstAddr() != 200 {
		t.Fatalf("addrs %d %d", a.SrcAddr(), a.DstAddr())
	}
	if w, h, ch := a.OutShape(); w != 16 || h != 16 || ch != 4 {
		t.Fatalf("out shape %d %d %d", w, h, ch)
	}
	if a.InLayout() != (Layout{4, 1, 16}) || a.OutLayout() != (Layout{4, 1, 16}) {
		t.Fatalf("layouts %+v %+v", a.InLayout(), a.OutLayout())
	}
}

func TestParseErrors(t *testing.T) {
	good := tinyModel(t)
	set := func(i int, v uint32) []byte {
		b := bytes.Clone(good)
		binary.LittleEndian.PutUint32(b[4*i:], v)
		return b
	}
	cases := []struct {
		name string
		data []byte
		want error
	}{
		{"short", good[:20], ErrFormat},
		{"version 4", set(0, 4), ErrVersion},
		{"arch 1", set(2, 1), ErrVersion},
		{"too many layers", set(3, 1000), ErrFormat},
		{"output past main memory", set(8, 1<<20), ErrFormat},
		{"truncated body", good[:28+8+16+30], ErrFormat},
	}
	for _, c := range cases {
		if _, err := Parse(c.data); !errors.Is(err, c.want) {
			t.Errorf("%s: err %v, want %v", c.name, err, c.want)
		}
	}
}

func TestCheckErrors(t *testing.T) {
	good := tinyModel(t)
	body := 28 + 8 + 16 // conv body start
	mod := func(f func(b []byte)) *Model {
		b := bytes.Clone(good)
		f(b)
		m, err := Parse(b)
		if err != nil {
			t.Fatal(err)
		}
		return m
	}
	cases := []struct {
		name string
		m    *Model
		want error
	}{
		{"unsupported type", mod(func(b []byte) { binary.LittleEndian.PutUint32(b[28+8+8:], 15) }), ErrUnsupported},
		{"first layer not conv", mod(func(b []byte) { binary.LittleEndian.PutUint32(b[28+8:], uint32(Dequantize)) }), ErrFormat},
		{"args past the file", mod(func(b []byte) { binary.LittleEndian.PutUint32(b[body+8:], uint32(len(b))-8) }), ErrFormat},
		{"weights past the file", mod(func(b []byte) { binary.LittleEndian.PutUint32(b[body+12:], uint32(len(b))) }), ErrFormat},
		{"main-mem output past main memory", mod(func(b []byte) { binary.LittleEndian.PutUint32(b[body+4:], 4500) }), ErrFormat},
		{"dequantize past main memory", mod(func(b []byte) { binary.LittleEndian.PutUint32(b[body+24+12:], 2000) }), ErrFormat},
	}
	for _, c := range cases {
		if err := c.m.Check(); !errors.Is(err, c.want) {
			t.Errorf("%s: err %v, want %v", c.name, err, c.want)
		}
	}
}

// uploadCore is a literal port of kpu_upload_core (kpu.c, byte path).
func uploadCore(ram []byte, width, height, channels int, src []byte, addr int) {
	var rowPadding, rowGroup, rowLength int
	switch {
	case width <= 16:
		rowPadding, rowGroup, rowLength = 16, 4, 1
	case width <= 32:
		rowPadding, rowGroup, rowLength = 32, 2, 1
	default:
		rowPadding, rowGroup, rowLength = 64, 1, (width+63)/64
	}
	dest := addr * 64
	i := 0
	for oc := range channels {
		origin := dest + oc/rowGroup*rowLength*height*64 + oc%rowGroup*rowPadding
		for y := range height {
			yo := origin + y*rowLength*64
			for x := range width {
				ram[yo+x] = src[i]
				i++
			}
		}
	}
}

func TestUploadMatchesCRuntime(t *testing.T) {
	for _, s := range []struct{ w, h, c int }{{8, 8, 3}, {16, 16, 5}, {24, 7, 3}, {32, 32, 3}, {40, 5, 2}, {64, 4, 3}, {128, 3, 2}, {320, 2, 3}} {
		src := make([]byte, s.w*s.h*s.c)
		for i := range src {
			src[i] = byte(i*7 + i/s.w)
		}
		want := make([]byte, RAMSize)
		got := make([]byte, RAMSize)
		uploadCore(want, s.w, s.h, s.c, src, 37)
		if err := UploadLayout(s.w, s.h).UploadTo(got, 37, s.w, s.h, s.c, src); err != nil {
			t.Fatalf("%v: %v", s, err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%v: layout differs from kpu_upload_core", s)
		}
		back := make([]byte, len(src))
		if err := UploadLayout(s.w, s.h).DownloadFrom(got, 37, s.w, s.h, s.c, back); err != nil || !bytes.Equal(back, src) {
			t.Errorf("%v: download is not the inverse of upload (err %v)", s, err)
		}
	}
}

func TestLayoutBounds(t *testing.T) {
	ram := make([]byte, RAMSize)
	src := make([]byte, 64*64*3)
	l := UploadLayout(64, 64)
	if err := l.UploadTo(ram, RAMSize/64-64*3+1, 64, 64, 3, src); err == nil {
		t.Error("tensor ending past KPU memory accepted")
	}
	if err := l.UploadTo(ram, RAMSize/64-64*3, 64, 64, 3, src); err != nil {
		t.Errorf("tensor ending at the last line rejected: %v", err)
	}
	if err := (Layout{4, 1, 16}).UploadTo(ram, 0, 17, 16, 3, src); err == nil {
		t.Error("17-wide row in a 16-byte group slot accepted")
	}
	if err := (Layout{3, 1, 16}).UploadTo(ram, 0, 16, 16, 3, src); err == nil {
		t.Error("group of 3 accepted")
	}
}

func TestDequantize(t *testing.T) {
	main := make([]byte, 3+12)
	copy(main, []byte{0, 10, 255})
	Quant{In: 0, Out: 3, Count: 3, Scale: 0.5, Bias: -1}.Dequantize(main, new([256]uint32))
	for i, want := range []float32{-1, 4, 126.5} {
		if got := getF32(main[3:], uint32(i)); got != want {
			t.Errorf("q[%d]: %v, want %v", i, got, want)
		}
	}
}

func TestQuantize(t *testing.T) {
	in := []float32{-3, 0.25, 0.75, 1.25, 2, 200}
	main := make([]byte, 4*len(in)+len(in))
	for i, v := range in {
		putF32(main, uint32(i), v)
	}
	// (f + 1) / 0.5: −4 → 0 (clamped), 2.5 → 3, 3.5 → 4 (half away from
	// zero, like roundf), 4.5 → 5, 6, 402 → 255 (clamped).
	Quant{In: 0, Out: uint32(4 * len(in)), Count: uint32(len(in)), Scale: 0.5, Bias: -1}.Quantize(main)
	if got, want := main[4*len(in):], []byte{0, 3, 4, 5, 6, 255}; !bytes.Equal(got, want) {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestGAP(t *testing.T) {
	main := make([]byte, 4*6+4*2)
	for i, v := range []float32{1, 2, 3, 10, 20, 31} {
		putF32(main, uint32(i), v)
	}
	GAP{In: 0, Out: 24, KernelSize: 3, Channels: 2}.Run(main)
	if a, b := getF32(main[24:], 0), getF32(main[24:], 1); a != 2 || b != 61.0/3 {
		t.Errorf("got %v %v, want 2 %v", a, b, float32(61.0/3))
	}
}

func TestPadding(t *testing.T) {
	ram := make([]byte, RAMSize)
	main := []byte{1, 2, 3, 4, 5, 6}
	Padding{In: 0, Out: 10, Channels: 6}.AddPadding(main, ram)
	// Channel c at line 10 + c/4·4, byte c%4·16 (kpu_add_padding).
	for c, v := range main {
		if got := ram[640+c/4*256+c%4*16]; got != v {
			t.Errorf("channel %d at %d: %d, want %d", c, 640+c/4*256+c%4*16, got, v)
		}
	}
	m := make([]byte, 3*16+3)
	for c := range 3 {
		m[16*c] = byte(10 + c)
		m[16*c+1] = 99
	}
	Padding{In: 0, Out: 48, Channels: 3}.RemovePadding(m)
	if got := m[48:]; !bytes.Equal(got, []byte{10, 11, 12}) {
		t.Errorf("RemovePadding: %v", got)
	}
}

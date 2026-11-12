package kmodel

import (
	"bytes"
	"errors"
	"testing"
)

// buildFixture builds the check fixture's model (one layer of every type)
// with the Builder, from values, not bytes.
func buildFixture(t *testing.T) (*Builder, []BN, [16]ActSegment, Args, []byte) {
	t.Helper()
	b := NewBuilder(true, 32704)
	b.SetMainMem(fxMain)
	b.AddOutput(Output{Address: 1024, Size: 4 * fxN})
	a := NewArgs()
	must(t, a.SetSrcAddr(100))
	must(t, a.SetDstAddr(200))
	must(t, a.SetInShape(fxW, fxH, fxC))
	must(t, a.SetOutShape(fxW, fxH, fxOC))
	must(t, a.SetInLayout(UploadLayout(fxW, fxH)))
	must(t, a.SetOutLayout(UploadLayout(fxW, fxH)))
	must(t, a.SetKernel(1, false))
	a.SetPadValue(128)
	must(t, a.SetConvValue(-128, 0, -200, 0))
	must(t, a.SetArgAdd(25600))
	must(t, a.SetLoad(fxC*fxOC, 1, fxOC))
	weights := []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	bn := []BN{{Mul: 1 << 22, Add: -5, Shift: 15}, {Mul: -(1 << 23), Add: 1<<31 - 1, Shift: 0}, {Mul: 1, Add: -1 << 31, Shift: 7}, {Mul: 3, Add: 0, Shift: 1}}
	var act [16]ActSegment
	for i := range act {
		act[i] = ActSegment{StartX: int64(i)<<30 - 1<<35, Mul: int64(i*1000 - 8000), Shift: uint(i), Add: int64(i * 16)}
	}
	act[15] = ActSegment{StartX: 1<<35 - 1, Mul: 1<<15 - 1, Shift: 255, Add: 255}
	bnb, err := EncodeBN(bn)
	if err != nil {
		t.Fatal(err)
	}
	actb, err := EncodeAct(act)
	if err != nil {
		t.Fatal(err)
	}
	b.AddConvToMain(0, a, weights, bnb, actb)
	b.AddDequantize(Quant{In: 0, Out: 1024, Count: fxN, Scale: 0.5, Bias: -1})
	b.AddGAP(GAP{In: 1024, Out: 5120, KernelSize: fxW * fxH, Channels: fxOC})
	b.AddQuantize(Quant{In: 5120, Out: 5136, Count: fxOC, Scale: 0.1})
	b.AddAddPadding(Padding{In: 5136, Out: 1000, Channels: fxOC})
	b.AddUpload(Upload{In: 0, Out: 2000, Width: fxW, Height: fxH, Channels: fxOC})
	b.AddRemovePadding(Padding{In: 0, Out: 5140, Channels: fxOC})
	return b, bn, act, a, weights
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func TestBuilderRoundTrip(t *testing.T) {
	b, bn, act, a, weights := buildFixture(t)
	data, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	m, err := Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	if !m.EightBitMode() || m.MainMemUsage != fxMain || len(m.Outputs) != 1 || m.Outputs[0] != (Output{1024, 4 * fxN}) {
		t.Fatalf("header: flags %d main %d outputs %v", m.Flags, m.MainMemUsage, m.Outputs)
	}
	want := []LayerType{K210Conv, Dequantize, GlobalAveragePool2D, Quantize, K210AddPadding, K210Upload, K210RemovePadding}
	if len(m.Layers) != len(want) {
		t.Fatalf("%d layers", len(m.Layers))
	}
	end := headerSize + 8*len(m.Outputs) + 8*len(m.Layers)
	for i, l := range m.Layers {
		end += len(l.Body)
		if l.Type != want[i] || end%8 != 0 {
			t.Errorf("layer %d: type %v (want %v), body ends at %d (want a multiple of 8, as ncc pads)", i, l.Type, want[i], end)
		}
	}
	c, _ := m.Layers[0].Conv()
	if c.Flags != FlagMainMemOut || c.LayerOffset%8 != 0 || c.WeightsOffset%128 != 0 || c.BNOffset%128 != 0 || c.ActOffset%256 != 0 {
		t.Errorf("conv offsets not aligned as ncc's: %+v", c)
	}
	p, err := m.ConvParams(c)
	if err != nil {
		t.Fatal(err)
	}
	if p.Args != a || !bytes.Equal(p.Weights, weights) || p.Act != act {
		t.Errorf("conv: args/weights/act changed in the round trip")
	}
	for i := range bn {
		if p.BN[i] != bn[i] {
			t.Errorf("BN %d: %+v, want %+v", i, p.BN[i], bn[i])
		}
	}
	if p.Kernel != 1 || p.Depthwise || p.PadValue != 128 || p.ArgX != -128 || p.ArgW != -200 || p.ArgAdd != 25600 {
		t.Errorf("conv fields: %+v", p)
	}
	if q, _ := m.Layers[1].Quant(); q != (Quant{In: 0, Out: 1024, Count: fxN, Scale: 0.5, Bias: -1}) {
		t.Errorf("dequantize: %+v", q)
	}
	if g, _ := m.Layers[2].GAP(); g != (GAP{In: 1024, Out: 5120, KernelSize: fxW * fxH, Channels: fxOC}) {
		t.Errorf("GAP: %+v", g)
	}
	if u, _ := m.Layers[5].Upload(); u != (Upload{In: 0, Out: 2000, Width: fxW, Height: fxH, Channels: fxOC}) {
		t.Errorf("upload: %+v", u)
	}
	if p, _ := m.Layers[6].Padding(); p != (Padding{In: 0, Out: 5140, Channels: fxOC}) {
		t.Errorf("remove padding: %+v", p)
	}
}

func TestBuildChecks(t *testing.T) {
	b, _, _, _, _ := buildFixture(t)
	b.AddOutput(Output{Address: fxMain, Size: 1})
	if _, err := b.Build(); !errors.Is(err, ErrFormat) {
		t.Errorf("model with an output past main memory built: %v", err)
	}
}

func TestEncodeRanges(t *testing.T) {
	for _, e := range []BN{{Mul: 1 << 23}, {Mul: -(1 << 23) - 1}, {Add: 1 << 31}, {Add: -(1 << 31) - 1}, {Shift: 16}} {
		if _, err := EncodeBN([]BN{e}); !errors.Is(err, ErrFormat) {
			t.Errorf("BN %+v encoded", e)
		}
	}
	for _, s := range []ActSegment{{StartX: 1 << 35}, {StartX: -(1 << 35) - 1}, {Mul: 1 << 15}, {Mul: -(1 << 15) - 1}, {Shift: 256}, {Add: 256}, {Add: -1}} {
		var act [16]ActSegment
		act[3] = s
		if _, err := EncodeAct(act); !errors.Is(err, ErrFormat) {
			t.Errorf("activation segment %+v encoded", s)
		}
	}
}

// TestArgSettersKeepNeighbours sets every field to all ones and then to a
// value, on words that start all ones, so a mask that is too wide or too
// narrow shows up as a changed neighbour or a wrong read-back.
func TestArgSettersKeepNeighbours(t *testing.T) {
	var a Args
	for i := range a {
		a[i] = ^uint64(0)
	}
	must(t, a.SetSrcAddr(0x1234))
	must(t, a.SetDstAddr(0x0567))
	must(t, a.SetInShape(320, 240, 3))
	must(t, a.SetOutShape(160, 120, 16))
	must(t, a.SetInLayout(Layout{Group: 1, RowLines: 5, ChanLines: 1200}))
	must(t, a.SetOutLayout(Layout{Group: 2, RowLines: 1, ChanLines: 120}))
	if a.SrcAddr() != 0x1234 || a.DstAddr() != 0x0567 {
		t.Errorf("addresses %#x %#x", a.SrcAddr(), a.DstAddr())
	}
	if w, h, c := a.InShape(); w != 320 || h != 240 || c != 3 {
		t.Errorf("in shape %d %d %d", w, h, c)
	}
	if w, h, c := a.OutShape(); w != 160 || h != 120 || c != 16 {
		t.Errorf("out shape %d %d %d", w, h, c)
	}
	if a.InLayout() != (Layout{1, 5, 1200}) || a.OutLayout() != (Layout{2, 1, 120}) {
		t.Errorf("layouts %+v %+v", a.InLayout(), a.OutLayout())
	}
	// Bits outside the fields stay set: reserved bits of image_addr.
	if a[1]>>15&0x1ffff != 0x1ffff || a[1]>>47 != 0x1ffff {
		t.Errorf("image_addr reserved bits changed: %#x", a[1])
	}
	if a[11]&0xffff != 0xffff || a[11]>>16&0xffff != 160*120-1 || a[11]>>32 != 160*120*16-1 {
		t.Errorf("dma_parameter: %#x", a[11])
	}
	must(t, a.SetKernel(3, true))
	must(t, a.SetPool(5))
	a.SetPadValue(0x80)
	must(t, a.SetConvValue(-2, 3, 1<<23-1, 15))
	must(t, a.SetArgAdd(-(1 << 39)))
	must(t, a.SetLoad(30720, 3, 120))
	m := &Model{Data: make([]byte, 1)}
	p, err := m.ParamsAt(a, 0, 0, 0)
	if err == nil {
		t.Fatal("ParamsAt accepted tables past a 1-byte file")
	}
	m.Data = make([]byte, 1<<16)
	if p, err = m.ParamsAt(a, 0, 0, 0); err != nil {
		t.Fatal(err)
	}
	if p.Kernel != 3 || !p.Depthwise || p.Pool != 5 || p.PadValue != 0x80 || p.ArgX != -2 || p.ShrX != 3 || p.ArgW != 1<<23-1 || p.ShrW != 15 || p.ArgAdd != -(1<<39) {
		t.Errorf("decoded %+v", p)
	}
	if a[5]>>15&0x1ffff != 30720 || a[5]>>1&0x3f != 2 || a[2]>>48&0x3ff != 119 || a[5]&1 != 1 {
		t.Errorf("load fields: %#x %#x", a[5], a[2])
	}
	if a[4]>>8&0x3 != 3 || a[4]>>11&0x1f != 0x1f { // first_stride, bypass_conv, reserved untouched
		t.Errorf("kernel_pool_type_cfg neighbours changed: %#x", a[4])
	}
}

func TestNewArgsDefaults(t *testing.T) {
	a := NewArgs()
	// nncase rc5's constants (K210Conv2d.GenerateBinLayer).
	if a[4]>>10&1 != 1 || a[4]>>16&0xff != 15 || a[5]&1 != 1 || a[7]>>31 != 1 {
		t.Errorf("defaults: load_para %d burst %d load_coor %d load_act %d", a[4]>>10&1, a[4]>>16&0xff, a[5]&1, a[7]>>31)
	}
}

// TestSignedSettersStayInTheirFields: negative values have all high bits
// set; the setters must not spill them into the neighbouring fields.
func TestSignedSettersStayInTheirFields(t *testing.T) {
	var a Args
	must(t, a.SetConvValue(-1, 0, -1, 0))
	must(t, a.SetArgAdd(-1))
	if a[9] != 0x00ff_ffff_ffff_ff00 {
		t.Errorf("conv_value: %#x, want arg_x and arg_w all ones, shifts and reserved 0", a[9])
	}
	if a[10] != 1<<40-1 {
		t.Errorf("conv_value2: %#x, want 40 ones", a[10])
	}
}

// TestBuilderTableAlignment varies the BN table length so that 128- and
// 256-byte alignment differ: the activation table must land on 256.
func TestBuilderTableAlignment(t *testing.T) {
	for _, oc := range []int{4, 12, 20, 36} {
		b := NewBuilder(true, 0)
		b.SetMainMem(uint32(fxW * fxH * oc))
		a := NewArgs()
		must(t, a.SetInShape(fxW, fxH, fxC))
		must(t, a.SetOutShape(fxW, fxH, oc))
		must(t, a.SetInLayout(UploadLayout(fxW, fxH)))
		must(t, a.SetOutLayout(UploadLayout(fxW, fxH)))
		must(t, a.SetDstAddr(1000))
		bn, _ := EncodeBN(make([]BN, oc))
		act, _ := EncodeAct([16]ActSegment{})
		b.AddConvToMain(0, a, make([]byte, fxC*oc), bn, act)
		data, err := b.Build()
		if err != nil {
			t.Fatalf("oc %d: %v", oc, err)
		}
		m, _ := Parse(data)
		c, _ := m.Layers[0].Conv()
		if c.LayerOffset%8 != 0 || c.WeightsOffset%128 != 0 || c.BNOffset%128 != 0 || c.ActOffset%256 != 0 {
			t.Errorf("oc %d: offsets %+v", oc, c)
		}
	}
}

// TestSettersRejectOutOfRange: every setter refuses a value that its field
// cannot hold, returns ErrFormat and leaves the argument unchanged, and
// accepts the extreme values that fit (review R-K210-K8b B1: masking made
// width 1025 read back as 1).
func TestSettersRejectOutOfRange(t *testing.T) {
	bad := map[string]func(a *Args) error{
		"src addr 2^15":          func(a *Args) error { return a.SetSrcAddr(0x8000) },
		"dst addr 2^15":          func(a *Args) error { return a.SetDstAddr(0x8000) },
		"in width 1025":          func(a *Args) error { return a.SetInShape(1025, 1, 1) },
		"in width 0":             func(a *Args) error { return a.SetInShape(0, 1, 1) },
		"in height 513":          func(a *Args) error { return a.SetInShape(1, 513, 1) },
		"in channels 1025":       func(a *Args) error { return a.SetInShape(1, 1, 1025) },
		"out width 1025":         func(a *Args) error { return a.SetOutShape(1025, 1, 1) },
		"out height 0":           func(a *Args) error { return a.SetOutShape(1, 0, 1) },
		"out channels 0":         func(a *Args) error { return a.SetOutShape(1, 1, 0) },
		"in layout group 3":      func(a *Args) error { return a.SetInLayout(Layout{3, 1, 1}) },
		"in layout rows 16":      func(a *Args) error { return a.SetInLayout(Layout{1, 16, 1}) },
		"out layout chans 0":     func(a *Args) error { return a.SetOutLayout(Layout{1, 1, 0}) },
		"out layout chans 2^15":  func(a *Args) error { return a.SetOutLayout(Layout{1, 1, 0x8000}) },
		"kernel 2":               func(a *Args) error { return a.SetKernel(2, false) },
		"pool 7":                 func(a *Args) error { return a.SetPool(7) },
		"pool -1":                func(a *Args) error { return a.SetPool(-1) },
		"arg_x 2^23":             func(a *Args) error { return a.SetConvValue(1<<23, 0, 0, 0) },
		"arg_x -2^23-1":          func(a *Args) error { return a.SetConvValue(-(1<<23)-1, 0, 0, 0) },
		"arg_w 2^23":             func(a *Args) error { return a.SetConvValue(0, 0, 1<<23, 0) },
		"shr_x 16":               func(a *Args) error { return a.SetConvValue(0, 16, 0, 0) },
		"shr_w 16":               func(a *Args) error { return a.SetConvValue(0, 0, 0, 16) },
		"arg_add 2^39":           func(a *Args) error { return a.SetArgAdd(1 << 39) },
		"arg_add -2^39-1":        func(a *Args) error { return a.SetArgAdd(-(1 << 39) - 1) },
		"para_size 2^17":         func(a *Args) error { return a.SetLoad(1<<17, 1, 1) },
		"loads 0":                func(a *Args) error { return a.SetLoad(1, 0, 1) },
		"loads 65":               func(a *Args) error { return a.SetLoad(1, 65, 1) },
		"channels per load 1025": func(a *Args) error { return a.SetLoad(1, 1, 1025) },
	}
	for name, set := range bad {
		a := NewArgs()
		before := a
		if err := set(&a); !errors.Is(err, ErrFormat) {
			t.Errorf("%s: err %v, want ErrFormat", name, err)
		}
		if a != before {
			t.Errorf("%s: argument changed on error", name)
		}
	}
	good := map[string]func(a *Args) error{
		"src addr 2^15-1":        func(a *Args) error { return a.SetSrcAddr(0x7fff) },
		"in shape 1024×512×1024": func(a *Args) error { return a.SetInShape(1024, 512, 1024) },
		"out shape 1×1×1":        func(a *Args) error { return a.SetOutShape(1, 1, 1) },
		"layout 4/15/32767":      func(a *Args) error { return a.SetInLayout(Layout{4, 15, 0x7fff}) },
		"pool 6":                 func(a *Args) error { return a.SetPool(6) },
		"arg_x/arg_w extremes":   func(a *Args) error { return a.SetConvValue(-(1 << 23), 15, 1<<23-1, 15) },
		"arg_add extremes":       func(a *Args) error { return a.SetArgAdd(-(1 << 39)) },
		"arg_add max":            func(a *Args) error { return a.SetArgAdd(1<<39 - 1) },
		"load extremes":          func(a *Args) error { return a.SetLoad(1<<17-1, 64, 1024) },
	}
	for name, set := range good {
		a := NewArgs()
		if err := set(&a); err != nil {
			t.Errorf("%s: %v", name, err)
		}
	}
	a := NewArgs()
	must(t, a.SetInShape(1024, 512, 1024))
	if w, h, c := a.InShape(); w != 1024 || h != 512 || c != 1024 {
		t.Errorf("limits read back as %d×%d×%d", w, h, c)
	}
}

// TestChannelByteNum: dma_parameter.channel_byte_num is 16 bits (W·H − 1
// per channel) and only matters with send_data_out on. An output of more
// than 65,536 pixels per channel is refused while send_data_out is on, in
// either order of the two setters; with it off the shape is accepted.
func TestChannelByteNum(t *testing.T) {
	a := NewArgs()
	if err := a.SetOutShape(256, 256, 1); err != nil {
		t.Fatalf("65,536 pixels: %v", err)
	}
	if got := a[11] >> 16 & 0xffff; got != 65535 {
		t.Errorf("channel_byte_num %d, want 65535", got)
	}
	if err := a.SetSendDataOut(true); err != nil {
		t.Fatalf("send_data_out with 65,536 pixels: %v", err)
	}
	if err := a.SetOutShape(512, 256, 1); !errors.Is(err, ErrFormat) {
		t.Errorf("131,072 pixels with send_data_out on: err %v, want ErrFormat", err)
	}
	if w, h, _ := a.OutShape(); w != 256 || h != 256 {
		t.Errorf("rejected SetOutShape changed the shape to %d×%d", w, h)
	}
	b := NewArgs()
	if err := b.SetOutShape(512, 256, 1); err != nil {
		t.Fatalf("131,072 pixels with send_data_out off: %v", err)
	}
	if err := b.SetSendDataOut(true); !errors.Is(err, ErrFormat) {
		t.Errorf("send_data_out on a 131,072-pixel output: err %v, want ErrFormat", err)
	}
	if b[11]&1 != 0 {
		t.Error("rejected SetSendDataOut set the bit")
	}
}

// TestCheckLargeOutputChannelBytes (review R-K210-K9a K9a-2/K9a-3): with
// send_data_out off and more than 65,536 output pixels per channel the KPU
// ignores channel_byte_num, and ncc writes it truncated to 16 bits
// (K210Conv2d.cs) where SetOutShape saturates: Check accepts both, refuses
// any other value, and refuses such an output with send_data_out on.
// 512×200 has bit 15 set in W·H − 1 (0x18FFF), so the truncation must keep
// all 16 bits.
func TestCheckLargeOutputChannelBytes(t *testing.T) {
	build := func(w, h int, chb uint64, send bool) error {
		b := NewBuilder(true, 0)
		b.SetMainMem(uint32(w * h))
		b.AddOutput(Output{Address: 0, Size: uint32(w * h)})
		a := NewArgs()
		l := UploadLayout(w, h)
		must(t, a.SetSrcAddr(0))
		must(t, a.SetDstAddr(uint32(l.ChanLines)))
		must(t, a.SetInShape(w, h, 1))
		must(t, a.SetOutShape(w, h, 1))
		must(t, a.SetInLayout(l))
		must(t, a.SetOutLayout(l))
		must(t, a.SetKernel(1, false))
		must(t, a.SetLoad(1, 1, 1))
		a[11] = a[11]&^(0xffff<<16) | chb<<16
		if send {
			a[11] |= 1
		}
		bn, err := EncodeBN([]BN{{Mul: 1, Shift: 0}})
		must(t, err)
		var act [16]ActSegment
		for i := range act {
			act[i] = ActSegment{StartX: int64(i) - 1<<35}
		}
		actb, err := EncodeAct(act)
		must(t, err)
		b.AddConvToMain(0, a, []byte{1}, bn, actb)
		_, err = b.Build()
		return err
	}
	for _, c := range []struct {
		name string
		w, h int
		chb  uint64
		send bool
		ok   bool
	}{
		{"saturated (SetOutShape)", 300, 220, 65535, false, true},
		{"truncated (ncc)", 300, 220, (300*220 - 1) & 0xffff, false, true},
		{"truncated (ncc), bit 15 set", 512, 200, (512*200 - 1) & 0xffff, false, true},
		{"truncated to 15 bits", 512, 200, (512*200 - 1) & 0x7fff, false, false},
		{"other value", 300, 220, 1234, false, false},
		{"saturated with send_data_out", 300, 220, 65535, true, false},
	} {
		err := build(c.w, c.h, c.chb, c.send)
		if c.ok != (err == nil) || (err != nil && !errors.Is(err, ErrFormat)) {
			t.Errorf("%s %d×%d: err %v, want ok %v", c.name, c.w, c.h, err, c.ok)
		}
	}
}

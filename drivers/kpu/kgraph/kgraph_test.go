package kgraph_test

import (
	"encoding/binary"
	"errors"
	"hash/crc32"
	"testing"
	"unsafe"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/compile"
	"github.com/itohio/EasyRobot/drivers/kpu/kgraph"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// detector builds an SSD-shaped graph at 10×10 and runs it at 7×7 on its
// trained grid: a stride-1 conv and a stride-2 conv (the two feature maps,
// 7×7 and 4×4), on each a box head and a score head with their reshapes,
// the boxes concatenated and reshaped, the scores concatenated (the second
// input through a Rescale map) and passed through a Logistic. At 7×7 the
// stride-2 conv is a Pad(0,2,0,2) and a VALID conv, which stay on the CPU;
// the other convs lower. Outputs: boxes, scores.
func detector(t *testing.T) *qnet.Graph {
	t.Helper()
	const c, a, n = 4, 2, 10
	q := qnet.Quant{Scale: 0.05, ZP: -3}
	q2 := qnet.Quant{Scale: 0.08, ZP: 5}
	conv := func(from []int, in qi8.Shape, outC, k, stride int, oq qnet.Quant) qnet.Layer {
		l := qnet.Layer{Kind: qnet.Conv2D, From: from, In: in, Out: qi8.Shape{C: outC}, InQ: q, OutQ: oq,
			KH: k, KW: k, StrideH: stride, StrideW: stride, Padding: qi8.PaddingSame, ActMin: -128, ActMax: 127}
		out, err := l.OutShape()
		if err != nil {
			t.Fatal(err)
		}
		l.Out = out
		return l
	}
	reshape := func(in, out qi8.Shape, oq qnet.Quant) qnet.Layer {
		return qnet.Layer{Kind: qnet.Reshape, In: in, Out: out, InQ: oq, OutQ: oq}
	}
	f1 := conv(nil, qi8.Shape{H: n, W: n, C: c}, c, 3, 1, q) // 0
	f2 := conv(nil, f1.Out, c, 3, 2, q)                      // 1
	n1, n2 := f1.Out.H*f1.Out.W*a, f2.Out.H*f2.Out.W*a
	b1 := conv([]int{0}, f1.Out, a*4, 1, 1, q)  // 2
	b2 := conv([]int{1}, f2.Out, a*4, 1, 1, q)  // 4
	c1 := conv([]int{0}, f1.Out, a*3, 1, 1, q)  // 6
	c2 := conv([]int{1}, f2.Out, a*3, 1, 1, q2) // 8
	m, err := qi8.QuantizeConcatRescale(q2.Scale, q2.ZP, q.Scale, q.ZP)
	if err != nil {
		t.Fatal(err)
	}
	lp, err := qi8.QuantizeLogistic(q.Scale, q.ZP)
	if err != nil {
		t.Fatal(err)
	}
	boxes, scores := qi8.Shape{H: n1 + n2, W: 1, C: 4}, qi8.Shape{H: 1, W: n1 + n2, C: 3}
	g := &qnet.Graph{Layers: []qnet.Layer{
		f1, f2,
		b1, reshape(b1.Out, qi8.Shape{H: n1, W: 1, C: 4}, q), // 2, 3
		b2, reshape(b2.Out, qi8.Shape{H: n2, W: 1, C: 4}, q), // 4, 5
		c1, reshape(c1.Out, qi8.Shape{H: 1, W: n1, C: 3}, q), // 6, 7
		c2, reshape(c2.Out, qi8.Shape{H: 1, W: n2, C: 3}, q2), // 8, 9
		{Kind: qnet.Concat, From: []int{3, 5}, Axis: 0, In: qi8.Shape{H: n1, W: 1, C: 4}, Out: boxes, InQ: q, OutQ: q}, // 10
		reshape(boxes, qi8.Shape{H: 1, W: n1 + n2, C: 4}, q),                                                           // 11
		{Kind: qnet.Concat, From: []int{7, 9}, Axis: 1, In: qi8.Shape{H: 1, W: n1, C: 3}, Out: scores, InQ: q, OutQ: q,
			Rescale: []*[256]int8{nil, m}}, // 12
		{Kind: qnet.Logistic, In: scores, Out: scores, InQ: q, OutQ: qnet.LogisticOutQuant, LogisticParams: &lp}, // 13
	}, Outputs: []int{11, 13}}
	for i := range g.Layers {
		l := &g.Layers[i]
		nw, nb, nr := l.ParamCounts()
		l.Weights, l.Bias, l.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
	}
	if err := qnet.SyntheticParams(g, 7); err != nil {
		t.Fatal(err)
	}
	if err := g.Validate(); err != nil {
		t.Fatal(err)
	}
	r, err := compile.ResizeAligned(g, 7, 7)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func input(n int) []int8 {
	in := make([]int8, n)
	for i := range in {
		in[i] = int8(i*37 + i/11)
	}
	return in
}

func runner(t *testing.T, g *qnet.Graph, b qnet.Backend) *qnet.Runner {
	t.Helper()
	p, err := qnet.PlanMemory(g, b)
	if err != nil {
		t.Fatal(err)
	}
	r, err := qnet.NewRunner(g, make([]int8, p.Size()), b)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// encoded compiles the detector and returns it, its backend and its file,
// held at a 128-byte-aligned address.
func encoded(t *testing.T) (*qnet.Graph, *compile.Backend, []byte) {
	t.Helper()
	g := detector(t)
	be, err := compile.Compile(g, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	if n := len(be.Segments()); n < 3 {
		t.Fatalf("%d KPU segments, the fixture needs several", n)
	}
	data, err := be.File(g, kgraph.Extra{Tag: "anchors", Data: kgraph.Float32s([]float32{0.5, 0.25, -1, 3e9})}, kgraph.Extra{Tag: "note", Data: []byte("x")})
	if err != nil {
		t.Fatal(err)
	}
	return g, be, kpu.AlignedCopy(data)
}

func inside(data []byte, p unsafe.Pointer, n int) bool {
	lo := uintptr(unsafe.Pointer(unsafe.SliceData(data)))
	return uintptr(p) >= lo && uintptr(p)+uintptr(n) <= lo+uintptr(len(data))
}

// TestRoundTrip: the parsed graph has the compiled graph's layers; the
// layers of KPU segments carry no parameters, the CPU layers' parameters
// equal the originals and lie inside the file, as do the kmodels (each at a
// multiple of 128 bytes) and the extras.
func TestRoundTrip(t *testing.T) {
	g, be, data := encoded(t)
	f, err := kgraph.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	onKPU := map[int]bool{}
	if len(f.Segments) != len(be.Segments()) {
		t.Fatalf("%d segments, want %d", len(f.Segments), len(be.Segments()))
	}
	for k, s := range be.Segments() {
		fs := f.Segments[k]
		if fs.First != s.First || fs.Count != s.Count || string(fs.Kmodel) != string(s.Model.Data) {
			t.Errorf("segment %d: layers %d+%d, %d kmodel bytes; want %d+%d, %d", k, fs.First, fs.Count, len(fs.Kmodel), s.First, s.Count, len(s.Model.Data))
		}
		if !inside(data, unsafe.Pointer(unsafe.SliceData(fs.Kmodel)), len(fs.Kmodel)) || uintptr(unsafe.Pointer(unsafe.SliceData(fs.Kmodel)))%128 != 0 {
			t.Errorf("segment %d: the kmodel is not inside the file at a 128-byte-aligned address", k)
		}
		for i := s.First; i < s.First+s.Count; i++ {
			onKPU[i] = true
		}
	}
	if len(f.Graph.Layers) != len(g.Layers) || len(f.Graph.Outputs) != 2 || f.Graph.Outputs[0] != g.Outputs[0] || f.Graph.Outputs[1] != g.Outputs[1] {
		t.Fatalf("%d layers, outputs %v; want %d, %v", len(f.Graph.Layers), f.Graph.Outputs, len(g.Layers), g.Outputs)
	}
	cpuConvs := 0
	for i := range g.Layers {
		want, got := g.Layers[i], f.Graph.Layers[i]
		if onKPU[i] {
			if len(got.Weights)+len(got.Bias)+len(got.Requant) != 0 {
				t.Errorf("layer %d runs on the KPU and still has parameters in the file", i)
			}
			want.Weights, want.Bias, want.Requant = nil, nil, nil
		}
		if len(got.From) != len(want.From) || (got.From == nil) != (want.From == nil) {
			t.Errorf("layer %d: From %v, want %v", i, got.From, want.From)
		}
		for k := range want.From {
			if got.From[k] != want.From[k] {
				t.Errorf("layer %d: From %v, want %v", i, got.From, want.From)
			}
		}
		if string(i8b(got.Weights)) != string(i8b(want.Weights)) || len(got.Bias) != len(want.Bias) || len(got.Requant) != len(want.Requant) {
			t.Errorf("layer %d: parameters differ", i)
		}
		for k := range want.Bias {
			if got.Bias[k] != want.Bias[k] {
				t.Errorf("layer %d: bias %d", i, k)
			}
		}
		for k := range want.Requant {
			if got.Requant[k] != want.Requant[k] {
				t.Errorf("layer %d: requant %d", i, k)
			}
		}
		if len(got.Weights) > 0 {
			cpuConvs++
			if !inside(data, unsafe.Pointer(unsafe.SliceData(got.Weights)), len(got.Weights)) ||
				!inside(data, unsafe.Pointer(unsafe.SliceData(got.Bias)), 4*len(got.Bias)) ||
				!inside(data, unsafe.Pointer(unsafe.SliceData(got.Requant)), 8*len(got.Requant)) {
				t.Errorf("layer %d: parameters were copied out of the file", i)
			}
		}
		if (got.LogisticParams == nil) != (want.LogisticParams == nil) || (want.LogisticParams != nil && *got.LogisticParams != *want.LogisticParams) {
			t.Errorf("layer %d: logistic table", i)
		}
		if len(got.Rescale) != len(want.Rescale) {
			t.Fatalf("layer %d: %d rescale maps, want %d", i, len(got.Rescale), len(want.Rescale))
		}
		for k := range want.Rescale {
			if (got.Rescale[k] == nil) != (want.Rescale[k] == nil) || (want.Rescale[k] != nil && *got.Rescale[k] != *want.Rescale[k]) {
				t.Errorf("layer %d: rescale map %d", i, k)
			}
		}
		// Everything else: compare with the pointers and slices cleared.
		for _, l := range []*qnet.Layer{&want, &got} {
			l.From, l.Weights, l.Bias, l.Requant, l.LogisticParams, l.Rescale = nil, nil, nil, nil, nil, nil
		}
		if !equalScalars(&got, &want) {
			t.Errorf("layer %d: %+v, want %+v", i, got, want)
		}
	}
	if cpuConvs == 0 {
		t.Fatal("the fixture has no convolution on the CPU")
	}
	if a, ok := kgraph.ReadFloat32s(f.Extra("anchors")); !ok || len(a) != 4 || a[0] != 0.5 || a[1] != 0.25 || a[2] != -1 || a[3] != 3e9 {
		t.Errorf("anchors extra %v", a)
	}
	if string(f.Extra("note")) != "x" || f.Extra("none") != nil {
		t.Errorf("extras: note %q, none %v", f.Extra("note"), f.Extra("none"))
	}
}

func equalScalars(a, b *qnet.Layer) bool {
	return a.Kind == b.Kind && a.In == b.In && a.Out == b.Out && a.InQ == b.InQ && a.OutQ == b.OutQ &&
		a.KH == b.KH && a.KW == b.KW && a.StrideH == b.StrideH && a.StrideW == b.StrideW && a.Padding == b.Padding &&
		a.TiesUp == b.TiesUp && a.ActMin == b.ActMin && a.ActMax == b.ActMax && a.Round == b.Round && a.WZP == b.WZP &&
		a.Axis == b.Axis && a.Pads == b.Pads && a.PadValue == b.PadValue
}

func i8b(s []int8) []byte {
	b := make([]byte, len(s))
	for i, v := range s {
		b[i] = byte(v)
	}
	return b
}

// TestRunFromFile: the graph loaded from the file, run on the emulated KPU,
// gives both outputs byte for byte as the compiled graph does (the same
// kmodels and the same CPU kernels), with one upload buffer and one output
// buffer for all segments, and no allocation per run.
func TestRunFromFile(t *testing.T) {
	g, be, data := encoded(t)
	f, err := kgraph.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	fb, err := kgraph.NewBackend(f, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	in := input(g.Layers[0].In.Len())
	ro, rf := runner(t, g, be), runner(t, &f.Graph, fb)
	if _, err := ro.Run(in); err != nil {
		t.Fatal(err)
	}
	if _, err := rf.Run(in); err != nil {
		t.Fatal(err)
	}
	for k := range 2 {
		want, got := ro.Output(in, k), rf.Output(in, k)
		if string(i8b(got)) != string(i8b(want)) {
			t.Errorf("output %d differs from the compiled graph's", k)
		}
	}
	var kpuSegments, sumIn, maxIn, sumOut, maxOut int
	for _, s := range rf.Plan().Segments {
		if s.Backend == nil {
			continue
		}
		kpuSegments++
		in, out := f.Graph.Layers[s.First].In.Len(), f.Graph.Layers[s.First+s.Count-1].Out.Len()
		sumIn, maxIn, sumOut, maxOut = sumIn+in, max(maxIn, in), sumOut+out, max(maxOut, out)
	}
	if kpuSegments != len(be.Segments()) {
		t.Errorf("%d segments on the KPU, want %d", kpuSegments, len(be.Segments()))
	}
	if up, out := fb.BufferBytes(); up != maxIn || out != maxOut || up >= sumIn || out >= sumOut {
		t.Errorf("buffers: upload %d B, output %d B; want the largest input %d (sum %d) and output %d (sum %d)", up, out, maxIn, sumIn, maxOut, sumOut)
	}
	if n := testing.AllocsPerRun(5, func() {
		if _, err := rf.Run(in); err != nil {
			t.Fatal(err)
		}
	}); n != 0 {
		t.Errorf("Run allocates %v times", n)
	}
}

// fixCRC recomputes the file's checksum after a deliberate change.
func fixCRC(data []byte) {
	binary.LittleEndian.PutUint32(data[12:], crc32.ChecksumIEEE(data[16:]))
}

// TestParseRefuses: a file at an unaligned address, a damaged one, and one
// whose tables point outside it (with a valid checksum) are refused, without
// a panic.
func TestParseRefuses(t *testing.T) {
	_, _, data := encoded(t)
	if _, err := kgraph.Parse(data); err != nil {
		t.Fatal(err)
	}
	shifted := kpu.AlignedCopy(append([]byte{0}, data...))[1:]
	if _, err := kgraph.Parse(shifted); !errors.Is(err, kgraph.ErrAlign) {
		t.Errorf("file at an odd address: %v, want ErrAlign", err)
	}
	mutate := func(name string, want error, change func(d []byte) []byte) {
		t.Helper()
		d := change(kpu.AlignedCopy(data))
		if _, err := kgraph.Parse(d); !errors.Is(err, want) {
			t.Errorf("%s: %v, want %v", name, err, want)
		}
	}
	mutate("too short", kgraph.ErrFormat, func(d []byte) []byte { return d[:40] })
	mutate("truncated", kgraph.ErrFormat, func(d []byte) []byte { return d[:len(d)-1] })
	mutate("magic", kgraph.ErrFormat, func(d []byte) []byte { d[0] ^= 1; return d })
	mutate("version", kgraph.ErrVersion, func(d []byte) []byte { d[4]++; fixCRC(d); return d })
	mutate("a flipped weight", kgraph.ErrChecksum, func(d []byte) []byte { d[len(d)-200] ^= 0x10; return d })
	u32 := binary.LittleEndian
	layers := int(u32.Uint32(data[20:]))
	segs := int(u32.Uint32(data[28:]))
	mutate("layer count", kgraph.ErrFormat, func(d []byte) []byte { u32.PutUint32(d[16:], 1<<30); fixCRC(d); return d })
	mutate("layer table offset", kgraph.ErrFormat, func(d []byte) []byte { u32.PutUint32(d[20:], uint32(len(d)-8)); fixCRC(d); return d })
	mutate("kmodel offset not a multiple of 128", kgraph.ErrFormat, func(d []byte) []byte {
		u32.PutUint32(d[segs+8:], u32.Uint32(d[segs+8:])+16)
		fixCRC(d)
		return d
	})
	mutate("kmodel past the end", kgraph.ErrFormat, func(d []byte) []byte { u32.PutUint32(d[segs+12:], uint32(len(d))); fixCRC(d); return d })
	mutate("segment past the layers", kgraph.ErrFormat, func(d []byte) []byte { u32.PutUint32(d[segs+4:], 1000); fixCRC(d); return d })
	// Every layer record's offsets and counts set to huge values in turn.
	const record = 96
	for i := range int(u32.Uint32(data[16:])) {
		for off := 60; off < 92; off += 4 {
			d := kpu.AlignedCopy(data)
			u32.PutUint32(d[layers+i*record+off:], 0xFFFFFFF0)
			fixCRC(d)
			if f, err := kgraph.Parse(d); err == nil {
				// A field this layer's kind does not use: the file still parses.
				_ = f
			} else if !errors.Is(err, kgraph.ErrFormat) {
				t.Errorf("layer %d field at %d: %v, want ErrFormat", i, off, err)
			}
		}
	}
}

// TestEncodeRefuses: a kind the format does not hold and segments that are
// not ordered runs of layers.
func TestEncodeRefuses(t *testing.T) {
	g := detector(t)
	be, err := compile.Compile(g, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	var segs []kgraph.Segment
	for _, s := range be.Segments() {
		segs = append(segs, kgraph.Segment{First: s.First, Count: s.Count, Kmodel: s.Model.Data})
	}
	if _, err := kgraph.Encode(g, segs); err != nil {
		t.Fatal(err)
	}
	swapped := append([]kgraph.Segment(nil), segs...)
	swapped[0], swapped[1] = swapped[1], swapped[0]
	if _, err := kgraph.Encode(g, swapped); !errors.Is(err, kgraph.ErrFormat) {
		t.Errorf("segments out of order: %v, want ErrFormat", err)
	}
	long := append([]kgraph.Segment(nil), segs...)
	long[len(long)-1].Count = len(g.Layers)
	if _, err := kgraph.Encode(g, long); !errors.Is(err, kgraph.ErrFormat) {
		t.Errorf("segment past the last layer: %v, want ErrFormat", err)
	}
	if _, err := kgraph.Encode(g, segs, kgraph.Extra{Tag: "a tag too long"}); !errors.Is(err, kgraph.ErrFormat) {
		t.Errorf("long tag: %v, want ErrFormat", err)
	}
	s := qi8.Shape{H: 4, W: 4, C: 2}
	q := qnet.Quant{Scale: 0.1}
	sm := &qnet.Graph{Layers: []qnet.Layer{
		{Kind: qnet.ReLU, In: s, Out: s, InQ: q, OutQ: q},
		{Kind: qnet.ResizeNearest, In: s, Out: qi8.Shape{H: 8, W: 8, C: 2}, InQ: q, OutQ: q},
	}}
	if err := sm.Validate(); err != nil {
		t.Fatal(err)
	}
	if _, err := kgraph.Encode(sm, nil); !errors.Is(err, kgraph.ErrUnsupported) {
		t.Errorf("a resize layer: %v, want ErrUnsupported", err)
	}
}

// TestBackendRefusesMismatch: a segment whose kmodel is another segment's
// (other input and output sizes) is refused when the backend is built.
func TestBackendRefusesMismatch(t *testing.T) {
	g, be, _ := encoded(t)
	var segs []kgraph.Segment
	for _, s := range be.Segments() {
		segs = append(segs, kgraph.Segment{First: s.First, Count: s.Count, Kmodel: s.Model.Data})
	}
	segs[0].Kmodel, segs[1].Kmodel = segs[1].Kmodel, segs[0].Kmodel
	data, err := kgraph.Encode(g, segs)
	if err != nil {
		t.Fatal(err)
	}
	f, err := kgraph.Parse(kpu.AlignedCopy(data))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := kgraph.NewBackend(f, kpu.NewEmulated); !errors.Is(err, kgraph.ErrFormat) {
		t.Errorf("swapped kmodels: %v, want ErrFormat", err)
	}
}

// TestCHW: NHWC int8 to the KPU's CHW uint8 (value + 128) and back.
func TestCHW(t *testing.T) {
	src := []int8{-128, 0, 1, 2, 3, 127} // 3 pixels × 2 channels
	chw := make([]byte, 6)
	kgraph.ToCHW(chw, src, 3, 2)
	if want := []byte{0, 129, 131, 128, 130, 255}; string(chw) != string(want) {
		t.Fatalf("CHW %v, want %v", chw, want)
	}
	back := make([]int8, 6)
	kgraph.FromCHW(back, chw, 3, 2)
	for i := range src {
		if back[i] != src[i] {
			t.Fatalf("back %v, want %v", back, src)
		}
	}
}

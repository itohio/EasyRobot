package compile

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// mobileNetPad builds MobileNetV1 at width alpha and input size×size×3 the
// nncase way: every stride-2 conv is Pad(1,1) + 3×3 VALID (windows on even
// pixels; mobileNetSAME is the TF SAME variant). It is cut at the last
// pointwise conv (conv_pw_13), with qnet's synthetic quantisation and
// parameters.
func mobileNetPad(t *testing.T, alpha float32, size int) *qnet.Graph {
	t.Helper()
	blocks := [13]struct{ c, s int }{
		{64, 1}, {128, 2}, {128, 1}, {256, 2}, {256, 1}, {512, 2},
		{512, 1}, {512, 1}, {512, 1}, {512, 1}, {512, 1}, {1024, 2}, {1024, 1},
	}
	act := qnet.MobileNetActQuant
	lo, hi := qi8.ActReLU6(act.Scale, act.ZP)
	var ls []qnet.Layer
	s := qi8.Shape{H: size, W: size, C: 3}
	conv := func(kind qnet.Kind, inQ qnet.Quant, outC, k, stride int) {
		in := s
		pad := qi8.PaddingSame
		if stride == 2 {
			ls = append(ls, qnet.Layer{Kind: qnet.Pad, In: s, Out: qi8.Shape{H: s.H + 2, W: s.W + 2, C: s.C},
				InQ: inQ, OutQ: inQ, Pads: [6]int{1, 1, 1, 1, 0, 0}, PadValue: inQ.ZP})
			in = ls[len(ls)-1].Out
			pad = qi8.PaddingValid
		}
		l := qnet.Layer{Kind: kind, In: in, Out: qi8.Shape{C: outC}, InQ: inQ, OutQ: act,
			KH: k, KW: k, StrideH: stride, StrideW: stride, Padding: pad, ActMin: lo, ActMax: hi}
		out, err := l.OutShape()
		if err != nil {
			t.Fatal(err)
		}
		l.Out = out
		nw, nb, nr := l.ParamCounts()
		l.Weights, l.Bias, l.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
		ls = append(ls, l)
		s = out
	}
	conv(qnet.Conv2D, qnet.MobileNetInputQuant, int(32*alpha), 3, 2)
	for _, b := range blocks {
		conv(qnet.DepthwiseConv2D, act, s.C, 3, b.s)
		conv(qnet.Conv2D, act, int(float32(b.c)*alpha), 1, 1)
	}
	g := &qnet.Graph{Layers: ls}
	if err := qnet.SyntheticParams(g, 210); err != nil {
		t.Fatal(err)
	}
	if err := g.Validate(); err != nil {
		t.Fatal(err)
	}
	return g
}

func newRunner(t *testing.T, g *qnet.Graph, backends ...qnet.Backend) *qnet.Runner {
	t.Helper()
	p, err := qnet.PlanMemory(g, backends...)
	if err != nil {
		t.Fatal(err)
	}
	r, err := qnet.NewRunner(g, make([]int8, p.Size()), backends...)
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func randomInput(n int, seed uint64) []int8 {
	r := rand.New(rand.NewPCG(seed, 7))
	in := make([]int8, n)
	for i := range in {
		in[i] = int8(r.IntN(256) - 128)
	}
	return in
}

// sampleEven takes the even pixels of a full-resolution NHWC tensor whose
// strided shape is half.
func sampleEven(full []int8, half qi8.Shape) []int8 {
	out := make([]int8, half.Len())
	w := 2 * half.W
	for y := range half.H {
		for x := range half.W {
			copy(out[(y*half.W+x)*half.C:(y*half.W+x+1)*half.C], full[(2*y*w+2*x)*half.C:])
		}
	}
	return out
}

// chw2nhwc converts a uint8 CHW tensor from KPU memory to int8 NHWC.
func chw2nhwc(b []byte, s qi8.Shape) []int8 {
	out := make([]int8, len(b))
	for y := range s.H {
		for x := range s.W {
			for c := range s.C {
				out[(y*s.W+x)*s.C+c] = int8(b[(c*s.H+y)*s.W+x] ^ 0x80)
			}
		}
	}
	return out
}

// TestMobileNetOnKPU: explicit-pad MobileNetV1-0.25 at 256×256 cut at
// conv_pw_13 runs as one KPU segment; each KPU layer, from the KPU's own
// input, is within 1 LSB of qnet's CPU kernels; Run allocates nothing.
func TestMobileNetOnKPU(t *testing.T) {
	size := 256
	if testing.Short() {
		size = 64
	}
	g := mobileNetPad(t, 0.25, size)
	b, err := Compile(g, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	r := newRunner(t, g, b)
	segs := r.Plan().Segments
	if len(segs) != 1 || segs[0].Backend != b || segs[0].Count != len(g.Layers) {
		t.Fatalf("plan: %d segments, first %+v; want one KPU segment of %d layers", len(segs), segs[0], len(g.Layers))
	}
	ks := b.Segments()[0]
	if len(ks.Groups) != 27 {
		t.Fatalf("%d KPU layers, want 27", len(ks.Groups))
	}

	// Capture every KPU layer's output during the run (a Full depthwise's
	// at full resolution).
	outs := make([][]int8, len(ks.Groups))
	ks.Runner.OnLayer = func(i int) {
		gr := ks.Groups[i]
		s := g.Layers[gr.First+gr.Count-1].Out
		if gr.Mode.Full {
			s.H, s.W = 2*s.H, 2*s.W
		}
		buf := make([]byte, s.Len())
		if err := ks.Runner.LayerOutput(i, buf); err != nil {
			t.Fatal(err)
		}
		outs[i] = chw2nhwc(buf, s)
	}
	in := randomInput(g.Layers[0].In.Len(), 1)
	got, err := r.Run(in)
	if err != nil {
		t.Fatal(err)
	}
	ks.Runner.OnLayer = nil

	// Per layer: qnet's kernels on the KPU's own input to that layer.
	var hist [2]int
	x := in
	for i, gr := range ks.Groups {
		y := x
		for k := gr.First; k < gr.First+gr.Count; k++ {
			l := &g.Layers[k]
			dst := make([]int8, l.Out.Len())
			if err := qnet.RunLayer(l, dst, y); err != nil {
				t.Fatal(err)
			}
			y = dst
		}
		got := outs[i]
		if gr.Mode.Full { // full-resolution depthwise: compare at even pixels
			got = sampleEven(got, g.Layers[gr.First+gr.Count-1].Out)
		}
		for j := range y {
			d := int(got[j]) - int(y[j])
			d = max(d, -d)
			if d > 1 {
				t.Fatalf("KPU layer %d (graph layers %d..%d, %v): output %d: KPU %d, qnet %d", i, gr.First, gr.First+gr.Count-1, g.Layers[gr.First].Kind, j, got[j], y[j])
			}
			hist[d]++
		}
		x = got
	}
	t.Logf("%d×%d: 27 KPU layers, %d outputs: %d equal, %d at 1 LSB", size, size, hist[0]+hist[1], hist[0], hist[1])

	// The final output against the CPU-only runner (informational).
	cpu, err := newRunner(t, g).Run(in)
	if err != nil {
		t.Fatal(err)
	}
	var fh [4]int
	for j := range cpu {
		d := int(got[j]) - int(cpu[j])
		fh[min(max(d, -d), 3)]++
	}
	t.Logf("final output %v vs CPU runner: %d equal, %d at 1, %d at 2, %d at ≥3", g.Layers[len(g.Layers)-1].Out, fh[0], fh[1], fh[2], fh[3])

	if a := testing.AllocsPerRun(3, func() {
		if _, err := r.Run(in); err != nil {
			t.Fatal(err)
		}
	}); a != 0 {
		t.Errorf("Run with the KPU backend allocates %v times", a)
	}
}

// TestMixedGraph: a layer the KPU cannot run (a 3×3 stride-1 max pool)
// splits the graph into KPU, CPU and KPU segments; the KPU-backed result
// follows the CPU runner.
func TestMixedGraph(t *testing.T) {
	base := mobileNetPad(t, 0.25, 64)
	k := -1 // a pointwise conv in the middle, followed by a stride-1 depthwise
	for i := len(base.Layers) / 3; i < len(base.Layers)-1; i++ {
		if l, n := base.Layers[i], base.Layers[i+1]; l.Kind == qnet.Conv2D && l.KH == 1 && n.Kind == qnet.DepthwiseConv2D && n.StrideH == 1 {
			k = i
			break
		}
	}
	prev := base.Layers[k]
	pool := qnet.Layer{Kind: qnet.MaxPool2D, In: prev.Out, Out: prev.Out, InQ: prev.OutQ, OutQ: prev.OutQ,
		KH: 3, KW: 3, StrideH: 1, StrideW: 1, Padding: qi8.PaddingSame}
	ls := append(append(append([]qnet.Layer(nil), base.Layers[:k+1]...), pool), base.Layers[k+1:]...)
	g := &qnet.Graph{Layers: ls}
	if err := g.Validate(); err != nil {
		t.Fatal(err)
	}
	b, err := Compile(g, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	r := newRunner(t, g, b)
	var kinds []string
	for _, s := range r.Plan().Segments {
		if s.Backend != nil {
			kinds = append(kinds, "kpu")
		} else {
			kinds = append(kinds, g.Layers[s.First].Kind.String())
		}
	}
	if len(b.Segments()) != 2 || len(r.Plan().Segments) != 3 || r.Plan().Segments[1].Backend != nil || g.Layers[r.Plan().Segments[1].First].Kind != qnet.MaxPool2D {
		t.Fatalf("plan %v; want KPU, the 3×3 max pool on the CPU, KPU", kinds)
	}
	in := randomInput(g.Layers[0].In.Len(), 2)
	got, err := r.Run(in)
	if err != nil {
		t.Fatal(err)
	}
	cpu, err := newRunner(t, g).Run(in)
	if err != nil {
		t.Fatal(err)
	}
	var fh [4]int
	for j := range cpu {
		d := int(got[j]) - int(cpu[j])
		fh[min(max(d, -d), 3)]++
	}
	if fh[2]+fh[3] != 0 {
		t.Errorf("mixed graph output more than 1 LSB from the CPU runner: %v", fh)
	}
	t.Logf("plan %v; final output vs CPU runner: %d equal, %d at 1, %d at 2, %d at ≥3", kinds, fh[0], fh[1], fh[2], fh[3])
}

// TestSegmentBoundaries: a layer read by two layers, or a graph output,
// ends a segment (its tensor must reach main memory), and qnet's planner
// accepts the claims.
func TestSegmentBoundaries(t *testing.T) {
	base := mobileNetPad(t, 0.25, 32)
	// Cut to the first 8 layers: Pad, conv0, dw1, pw1, Pad, dw2, pw2, dw3.
	ls := append([]qnet.Layer(nil), base.Layers[:8]...)
	cases := map[string]struct {
		g       qnet.Graph
		wantKPU [][2]int // segments (first, count)
	}{
		"plain chain": {qnet.Graph{Layers: ls}, [][2]int{{0, 8}}},
		// pw1 (layer 3) is also a graph output: the first segment ends there.
		"intermediate output": {qnet.Graph{Layers: ls, Outputs: []int{3, 7}}, [][2]int{{0, 4}, {4, 4}}},
		// The stride-2 depthwise (layer 5) is a graph output: the next 1×1
		// cannot sample it, so a Sampler does and the segment ends there.
		"stride-2 depthwise output": {qnet.Graph{Layers: ls, Outputs: []int{5, 7}}, [][2]int{{0, 6}, {6, 2}}},
	}
	// dw1's output (layer 2) is read by a second depthwise (layer 3) and by
	// an Add of both: the first segment must end at layer 2, and the second
	// depthwise is a segment of its own.
	dw := ls[2]
	dw.In = ls[2].Out
	add := qnet.Layer{Kind: qnet.Add, From: []int{2, 3}, In: ls[2].Out, Out: ls[2].Out, InQ: ls[2].OutQ, In2Q: ls[2].OutQ, OutQ: ls[2].OutQ}
	lmin, lmax := qi8.ActNone()
	add.ActMin, add.ActMax = lmin, lmax
	rq, err := qi8.QuantizeAddMultipliers(0.5, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	add.AddRQ = rq
	branch := append(append([]qnet.Layer(nil), ls[:3]...), dw, add)
	cases["branch"] = struct {
		g       qnet.Graph
		wantKPU [][2]int
	}{qnet.Graph{Layers: branch}, [][2]int{{0, 3}, {3, 1}}}

	for name, c := range cases {
		g := c.g
		if err := g.Validate(); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		b, err := Compile(&g, kpu.NewEmulated)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		var got [][2]int
		for _, s := range b.Segments() {
			got = append(got, [2]int{s.First, s.Count})
		}
		if len(got) != len(c.wantKPU) {
			t.Fatalf("%s: segments %v, want %v", name, got, c.wantKPU)
		}
		for i := range got {
			if got[i] != c.wantKPU[i] {
				t.Errorf("%s: segments %v, want %v", name, got, c.wantKPU)
			}
		}
		if _, err := qnet.PlanMemory(&g, b); err != nil {
			t.Errorf("%s: planner rejects the claims: %v", name, err)
		}
	}
}

// TestSamplerEven: a stride-2 depthwise with its Pad(1,1) and nothing
// after it runs at full resolution, and a Sampler takes the even pixels;
// the output is within 1 LSB of qnet (the depthwise's rounding), as for any
// single KPU layer.
func TestSamplerEven(t *testing.T) {
	base := mobileNetPad(t, 0.25, 32)
	g := &qnet.Graph{Layers: append([]qnet.Layer(nil), base.Layers[4:6]...)} // Pad, dw2
	b, err := Compile(g, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	if ss := b.Segments(); len(ss) != 1 || len(ss[0].Groups) != 2 || !ss[0].Groups[1].Mode.Sampler || ss[0].Groups[1].Mode.OddSample {
		t.Fatalf("segments %+v; want the Full depthwise and an even Sampler", ss)
	}
	in := randomInput(g.Layers[0].In.Len(), 5)
	got, err := newRunner(t, g, b).Run(in)
	if err != nil {
		t.Fatal(err)
	}
	want, err := newRunner(t, g).Run(in)
	if err != nil {
		t.Fatal(err)
	}
	for j := range want {
		if d := int(got[j]) - int(want[j]); d > 1 || d < -1 {
			t.Fatalf("output %d: KPU %d, qnet %d", j, got[j], want[j])
		}
	}
}

// TestSegmentAddresses: in every compiled layer of MobileNets at several
// sizes and widths, the input region of KPU memory (read from SrcAddr, one
// row in for an OddSample reader) and the output region, as the layouts in
// the layer argument address them, do not overlap and stay inside KPU
// memory (the KPU reads its input while it writes; the emulator would not
// notice either).
func TestSegmentAddresses(t *testing.T) {
	graphs := map[string]*qnet.Graph{}
	for _, size := range []int{32, 96, 160, 224, 256} {
		for _, alpha := range []float32{0.25, 0.5} {
			same := mobileNetSAME(t, alpha, size)
			graphs[fmt.Sprintf("Pad + VALID α %v %d²", alpha, size)] = mobileNetPad(t, alpha, size)
			graphs[fmt.Sprintf("TF SAME α %v %d²", alpha, size)] = same
			// From pw1: the Full depthwise layers write at the top of
			// memory, so their readers' extra row must be reserved there.
			graphs[fmt.Sprintf("TF SAME from pw1 α %v %d²", alpha, size)] = &qnet.Graph{Layers: same.Layers[2:]}
		}
	}
	// The footprint of a tensor as the KPU addresses it: from its first
	// line, ChanLines per channel group (the layout written into the
	// argument, not this package's lines()).
	region := func(addr uint32, l kmodel.Layout, c int) (int, int) {
		return int(addr), int(addr) + l.ChanLines*((c+l.Group-1)/l.Group)
	}
	for name, g := range graphs {
		b, err := Compile(g, kpu.NewEmulated)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		for _, seg := range b.Segments() {
			m := seg.Model
			for i, l := range m.Layers {
				c, err := l.Conv()
				if err != nil {
					t.Fatal(err)
				}
				a := m.Args(c)
				_, _, ic := a.InShape()
				_, _, oc := a.OutShape()
				in0, in1 := region(a.SrcAddr(), a.InLayout(), ic)
				out0, out1 := region(a.DstAddr(), a.OutLayout(), oc)
				if in0 < out1 && out0 < in1 {
					t.Errorf("%s layer %d: input lines [%d,%d) overlap output [%d,%d)", name, i, in0, in1, out0, out1)
				}
				if in1 > ramLines || out1 > ramLines {
					t.Errorf("%s layer %d: input [%d,%d) or output [%d,%d) beyond KPU memory", name, i, in0, in1, out0, out1)
				}
			}
		}
	}
}

// mobileNetSAME is qnet's own MobileNetV1 (TF SAME stride 2, like the
// official TFLite model) cut at conv_pw_13, with synthetic parameters.
func mobileNetSAME(t *testing.T, alpha float32, size int) *qnet.Graph {
	t.Helper()
	ls := make([]qnet.Layer, qnet.MobileNetV1Layers(0))
	full, err := qnet.MobileNetV1(alpha, size, size, 0, ls)
	if err != nil {
		t.Fatal(err)
	}
	g := &qnet.Graph{Layers: full.Layers[:len(full.Layers)-1]} // drop the global average pool
	for i := range g.Layers {
		l := &g.Layers[i]
		nw, nb, nr := l.ParamCounts()
		l.Weights, l.Bias, l.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
	}
	if err := qnet.SyntheticParams(g, 210); err != nil {
		t.Fatal(err)
	}
	if err := g.Validate(); err != nil {
		t.Fatal(err)
	}
	return g
}

// TestOfficialMobileNetSAME: TF SAME stride 2 runs exactly on the KPU
// model. Each stride-2 layer runs at full resolution; the 1×1 conv after a
// depthwise samples its odd pixels, and an identity Sampler those of the
// stem (a 3×3 depthwise follows it). One segment of 28 KPU layers; per
// layer within 1 LSB of qnet, the Sampler exact.
func TestOfficialMobileNetSAME(t *testing.T) {
	size := 256
	if testing.Short() {
		size = 64
	}
	g := mobileNetSAME(t, 0.25, size)
	b, err := Compile(g, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	r := newRunner(t, g, b)
	if segs := r.Plan().Segments; len(segs) != 1 || segs[0].Backend != b {
		var got []int
		for _, s := range b.Segments() {
			got = append(got, s.First, s.Count)
		}
		t.Fatalf("plan: %d segments (KPU %v); want one KPU segment of %d layers", len(segs), got, len(g.Layers))
	}
	ks := b.Segments()[0]
	if len(ks.Groups) != 28 || !ks.Groups[1].Mode.Sampler || !ks.Groups[0].Mode.Full {
		t.Fatalf("groups %+v; want the Full stem, its Sampler, then 26 more", ks.Groups)
	}
	outs := make([][]int8, len(ks.Groups))
	ks.Runner.OnLayer = func(i int) {
		gr := ks.Groups[i]
		s := g.Layers[gr.First+gr.Count-1].Out
		if gr.Mode.Full {
			s.H, s.W = 2*s.H, 2*s.W
		}
		buf := make([]byte, s.Len())
		if err := ks.Runner.LayerOutput(i, buf); err != nil {
			t.Fatal(err)
		}
		outs[i] = chw2nhwc(buf, s)
	}
	in := randomInput(g.Layers[0].In.Len(), 1)
	got, err := r.Run(in)
	if err != nil {
		t.Fatal(err)
	}
	ks.Runner.OnLayer = nil
	var hist [2]int
	x := in
	for i, gr := range ks.Groups {
		y := x
		for k := gr.First; k < gr.First+gr.Count; k++ {
			l := &g.Layers[k]
			dst := make([]int8, l.Out.Len())
			if err := qnet.RunLayer(l, dst, y); err != nil {
				t.Fatal(err)
			}
			y = dst
		}
		out := g.Layers[gr.First+gr.Count-1].Out
		kpuOut := outs[i]
		if gr.Mode.Full { // odd pixels of the full-resolution layer
			kpuOut = sampleAt(kpuOut, out, 1)
		}
		for j := range y {
			d := int(kpuOut[j]) - int(y[j])
			d = max(d, -d)
			if d > 1 || (d > 0 && gr.Mode.Sampler) { // the Sampler copies
				t.Fatalf("KPU layer %d (graph %d..%d %v, mode %+v): output %d: KPU %d, qnet %d", i, gr.First, gr.First+gr.Count-1, g.Layers[gr.First].Kind, gr.Mode, j, kpuOut[j], y[j])
			}
			hist[d]++
		}
		x = kpuOut
	}
	cpu, err := newRunner(t, g).Run(in)
	if err != nil {
		t.Fatal(err)
	}
	same := 0
	for j := range cpu {
		if cpu[j] == got[j] {
			same++
		}
	}
	t.Logf("%d×%d SAME: %d KPU layers, per layer %d equal, %d at 1 LSB; final %d of %d equal to the CPU runner", size, size, len(ks.Groups), hist[0], hist[1], same, len(cpu))
	if same != len(cpu) && hist[1] == 0 {
		t.Errorf("final output differs from the CPU runner although every layer is exact")
	}
}

// sampleAt takes the pixels (2y+off, 2x+off) of a full-resolution NHWC
// tensor whose strided shape is half.
func sampleAt(full []int8, half qi8.Shape, off int) []int8 {
	out := make([]int8, half.Len())
	w := 2 * half.W
	for y := range half.H {
		for x := range half.W {
			copy(out[(y*half.W+x)*half.C:(y*half.W+x+1)*half.C], full[((2*y+off)*w+2*x+off)*half.C:])
		}
	}
	return out
}

package compile

import (
	"errors"
	"testing"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// The segment planner's rules (review R-K210-K9de D3), each checked on a
// small graph through Compile and the KPU-backed runner against qnet's CPU
// runner.

var ruleQ = qnet.Quant{Scale: 0.05, ZP: -3}

// ruleConv is a 3×3 SAME (or 1×1) stride-1 Conv2D s → c channels.
func ruleConv(s qi8.Shape, c, k int) qnet.Layer {
	l := qnet.Layer{Kind: qnet.Conv2D, In: s, Out: qi8.Shape{H: s.H, W: s.W, C: c}, InQ: ruleQ, OutQ: ruleQ,
		KH: k, KW: k, StrideH: 1, StrideW: 1, Padding: qi8.PaddingSame, ActMin: -128, ActMax: 127}
	nw, nb, nr := l.ParamCounts()
	l.Weights, l.Bias, l.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
	return l
}

func ruleAct(kind qnet.Kind, s qi8.Shape) qnet.Layer {
	return qnet.Layer{Kind: kind, In: s, Out: s, InQ: ruleQ, OutQ: ruleQ}
}

func ruleGraph(t *testing.T, g *qnet.Graph) *qnet.Graph {
	t.Helper()
	if err := qnet.SyntheticParams(g, 11); err != nil {
		t.Fatal(err)
	}
	if err := g.Validate(); err != nil {
		t.Fatal(err)
	}
	return g
}

// compileRun compiles g, runs it with the KPU backend and on the CPU, and
// returns the segments; outputs must agree within 1 LSB.
func compileRun(t *testing.T, g *qnet.Graph) []*Segment {
	t.Helper()
	b, err := Compile(g, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	in := randomInput(g.Layers[0].In.Len(), 3)
	got, err := newRunner(t, g, b).Run(in)
	if err != nil {
		t.Fatal(err)
	}
	want, err := newRunner(t, g).Run(in)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if d := int(got[i]) - int(want[i]); d > 1 || d < -1 {
			t.Fatalf("output %d: KPU path %d, CPU %d", i, got[i], want[i])
		}
	}
	return b.Segments()
}

// TestFusesActAndPool: a standalone ReLU/ReLU6 and a 2×2 pool after a conv
// become part of its KPU layer (one group covering all of them).
func TestFusesActAndPool(t *testing.T) {
	s := qi8.Shape{H: 8, W: 8, C: 4}
	for _, act := range []qnet.Kind{qnet.ReLU, qnet.ReLU6} {
		for _, pool := range []qnet.Kind{qnet.MaxPool2D, qnet.AvgPool2D} {
			c := ruleConv(s, 4, 3)
			p := qnet.Layer{Kind: pool, In: c.Out, Out: qi8.Shape{H: 4, W: 4, C: 4}, InQ: ruleQ, OutQ: ruleQ,
				KH: 2, KW: 2, StrideH: 2, StrideW: 2, Padding: qi8.PaddingValid}
			g := ruleGraph(t, &qnet.Graph{Layers: []qnet.Layer{c, ruleAct(act, c.Out), p}})
			segs := compileRun(t, g)
			if len(segs) != 1 || len(segs[0].Groups) != 1 || segs[0].Groups[0].Count != 3 {
				t.Errorf("%v + %v: segments %+v; want one KPU layer of 3 graph layers", act, pool, segs)
			}
		}
	}
}

// TestFusedLayerOutputsEndGroups: a layer whose output is a graph output,
// or has a second reader, is not fused into the following ReLU (its tensor
// must reach main memory).
func TestFusedLayerOutputsEndGroups(t *testing.T) {
	s := qi8.Shape{H: 8, W: 8, C: 4}
	c := ruleConv(s, 4, 3)
	g := ruleGraph(t, &qnet.Graph{Layers: []qnet.Layer{c, ruleAct(qnet.ReLU, c.Out)}, Outputs: []int{0, 1}})
	for _, seg := range compileRun(t, g) {
		for _, gr := range seg.Groups {
			if gr.First == 0 && gr.Count > 1 {
				t.Errorf("conv with a graph output fused with its ReLU: %+v", seg.Groups)
			}
		}
	}
	// conv0's output read by the ReLU and by an Add of both.
	add := qnet.Layer{Kind: qnet.Add, From: []int{0, 1}, In: s, Out: s, InQ: ruleQ, In2Q: ruleQ, OutQ: ruleQ}
	add.ActMin, add.ActMax = qi8.ActNone()
	rq, err := qi8.QuantizeAddMultipliers(0.5, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	add.AddRQ = rq
	g = ruleGraph(t, &qnet.Graph{Layers: []qnet.Layer{c, ruleAct(qnet.ReLU, c.Out), add}})
	for _, seg := range compileRun(t, g) {
		for _, gr := range seg.Groups {
			if gr.First == 0 && gr.Count > 1 {
				t.Errorf("conv with a second reader fused with its ReLU: %+v", seg.Groups)
			}
		}
	}
}

// TestParallelBranch: a conv that reads an earlier layer, not its
// predecessor, starts a new segment rather than chaining onto the layer
// before it, even when that layer's output has a single reader elsewhere
// (c1 → c3 here, c2 reads c0).
func TestParallelBranch(t *testing.T) {
	s := qi8.Shape{H: 8, W: 8, C: 4}
	c0, c1, c2, c3 := ruleConv(s, 4, 3), ruleConv(s, 4, 1), ruleConv(s, 4, 1), ruleConv(s, 4, 1)
	c2.From = []int{0} // parallel to c1
	c3.From = []int{1}
	g := ruleGraph(t, &qnet.Graph{Layers: []qnet.Layer{c0, c1, c2, c3}, Outputs: []int{2, 3}})
	for _, seg := range compileRun(t, g) {
		if seg.First <= 1 && seg.First+seg.Count > 2 {
			t.Errorf("segment %d..%d chains the parallel branch c2 onto c1", seg.First, seg.First+seg.Count-1)
		}
	}
}

// TestBackendRunChecks: Run refuses layers that are not exactly a compiled
// segment or buffers of the wrong size; Claim refuses a slice shorter than
// the segment.
func TestBackendRunChecks(t *testing.T) {
	s := qi8.Shape{H: 8, W: 8, C: 4}
	g := ruleGraph(t, &qnet.Graph{Layers: []qnet.Layer{ruleConv(s, 4, 3), ruleConv(s, 4, 1)}})
	b, err := Compile(g, kpu.NewEmulated)
	if err != nil {
		t.Fatal(err)
	}
	if n := b.Claim(g.Layers); n != 2 {
		t.Fatalf("Claim of the whole graph: %d, want 2", n)
	}
	if n := b.Claim(g.Layers[:1]); n != 0 {
		t.Errorf("Claim of a slice shorter than the segment: %d, want 0", n)
	}
	src, dst := make([]int8, s.Len()), make([]int8, s.Len())
	for name, err := range map[string]error{
		"one layer of two":  b.Run(g.Layers[:1], dst, src),
		"short source":      b.Run(g.Layers, dst, src[:1]),
		"short destination": b.Run(g.Layers, dst[:1], src),
		"no layers":         b.Run(nil, dst, src),
	} {
		if !errors.Is(err, ErrSegment) {
			t.Errorf("%s: err %v, want ErrSegment", name, err)
		}
	}
	if err := b.Run(g.Layers, dst, src); err != nil {
		t.Errorf("the segment itself: %v", err)
	}
}

// TestFullLayerTooBigStaysOnCPU: a stride-2 depthwise conv whose
// full-resolution output would not fit KPU memory next to its input is not
// lowered (the pair stays on the CPU) and the graph still compiles and runs.
func TestFullLayerTooBigStaysOnCPU(t *testing.T) {
	s := qi8.Shape{H: 256, W: 256, C: 24} // full resolution: 24 + 24 channels × 1,024 lines > 32,768
	dw := qnet.Layer{Kind: qnet.DepthwiseConv2D, In: s, Out: qi8.Shape{H: 128, W: 128, C: 24}, InQ: ruleQ, OutQ: ruleQ,
		KH: 3, KW: 3, StrideH: 2, StrideW: 2, Padding: qi8.PaddingSame, ActMin: -128, ActMax: 127}
	nw, nb, nr := dw.ParamCounts()
	dw.Weights, dw.Bias, dw.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
	pw := ruleConv(dw.Out, 8, 1)
	g := ruleGraph(t, &qnet.Graph{Layers: []qnet.Layer{dw, pw}})
	for _, seg := range compileRun(t, g) {
		if seg.First == 0 {
			t.Errorf("the oversized stride-2 depthwise was lowered: %+v", seg.Groups)
		}
	}
}

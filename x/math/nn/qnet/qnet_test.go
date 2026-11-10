package qnet_test

import (
	"bytes"
	"math/rand/v2"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

var (
	actQ = qnet.Quant{Scale: 6.0 / 255, ZP: -128}
	inQ  = qnet.Quant{Scale: 1.0 / 128, ZP: 3}
	outQ = qnet.Quant{Scale: 1.0 / 16, ZP: -5}
)

// smallLayers covers every kind:
//
//	0 conv 3×3/1 SAME 7×6×3→7×6×4   (ReLU6 fused)
//	1 dw   3×3/2 SAME 7×6×4→4×3×4
//	2 conv 1×1/1      4×3×4→4×3×8
//	3 relu6           4×3×8
//	4 maxpool 2×2/2 SAME →2×2×8
//	5 relu            2×2×8
//	6 avgpool 2×2 VALID →1×1×8
//	7 fc              8→5 (no activation)
func smallLayers() []qnet.Layer {
	lo, hi := qi8.ActReLU6(actQ.Scale, actQ.ZP)
	nlo, nhi := qi8.ActNone()
	conv := func(k qnet.Kind, in qi8.Shape, q qnet.Quant, outC, kk, s int) qnet.Layer {
		l := qnet.Layer{Kind: k, In: in, Out: qi8.Shape{C: outC}, InQ: q, OutQ: actQ,
			KH: kk, KW: kk, StrideH: s, StrideW: s, Padding: qi8.PaddingSame, ActMin: lo, ActMax: hi}
		return l
	}
	ls := []qnet.Layer{
		conv(qnet.Conv2D, qi8.Shape{H: 7, W: 6, C: 3}, inQ, 4, 3, 1),
		conv(qnet.DepthwiseConv2D, qi8.Shape{H: 7, W: 6, C: 4}, actQ, 4, 3, 2),
		conv(qnet.Conv2D, qi8.Shape{H: 4, W: 3, C: 4}, actQ, 8, 1, 1),
		{Kind: qnet.ReLU6, In: qi8.Shape{H: 4, W: 3, C: 8}, InQ: actQ, OutQ: actQ},
		{Kind: qnet.MaxPool2D, In: qi8.Shape{H: 4, W: 3, C: 8}, InQ: actQ, OutQ: actQ, KH: 2, KW: 2, StrideH: 2, StrideW: 2},
		{Kind: qnet.ReLU, In: qi8.Shape{H: 2, W: 2, C: 8}, InQ: actQ, OutQ: actQ},
		{Kind: qnet.AvgPool2D, In: qi8.Shape{H: 2, W: 2, C: 8}, InQ: actQ, OutQ: actQ, KH: 2, KW: 2, StrideH: 1, StrideW: 1, Padding: qi8.PaddingValid},
		{Kind: qnet.FullyConnected, In: qi8.Shape{H: 1, W: 1, C: 8}, Out: qi8.Shape{C: 5}, InQ: actQ, OutQ: outQ, ActMin: nlo, ActMax: nhi},
	}
	for i := range ls {
		out, err := ls[i].OutShape()
		if err != nil {
			panic(err)
		}
		ls[i].Out = out
	}
	return ls
}

// bound returns a graph over layers with freshly allocated synthetic params.
func bound(t testing.TB, layers []qnet.Layer, seed uint64) qnet.Graph {
	t.Helper()
	g := qnet.Graph{Layers: layers}
	nw, nb, nr := qnet.ParamCounts(&g)
	if err := qnet.BindParams(&g, make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)); err != nil {
		t.Fatal(err)
	}
	if err := qnet.SyntheticParams(&g, seed); err != nil {
		t.Fatal(err)
	}
	if err := g.Validate(); err != nil {
		t.Fatal(err)
	}
	return g
}

func randInput(n int, seed uint64) []int8 {
	r := rand.New(rand.NewPCG(seed, 1))
	s := make([]int8, n)
	for i := range s {
		s[i] = int8(r.IntN(256) - 128)
	}
	return s
}

// runByHand calls the qi8 kernels layer by layer, building the qi8 params
// from the layer fields directly.
func runByHand(t testing.TB, g *qnet.Graph, input []int8) []int8 {
	t.Helper()
	src := input
	for i := range g.Layers {
		l := &g.Layers[i]
		dst := make([]int8, l.Out.Len())
		var err error
		switch l.Kind {
		case qnet.Conv2D:
			err = qi8.Conv2D(dst, src, l.Weights, l.Bias, l.Requant, qi8.Conv2DParams{
				In: l.In, OutC: l.Out.C, KH: l.KH, KW: l.KW, StrideH: l.StrideH, StrideW: l.StrideW,
				Padding: l.Padding, InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, ActMin: l.ActMin, ActMax: l.ActMax})
		case qnet.DepthwiseConv2D:
			err = qi8.DepthwiseConv2D(dst, src, l.Weights, l.Bias, l.Requant, qi8.DepthwiseConv2DParams{
				In: l.In, KH: l.KH, KW: l.KW, StrideH: l.StrideH, StrideW: l.StrideW,
				Padding: l.Padding, InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, ActMin: l.ActMin, ActMax: l.ActMax})
		case qnet.FullyConnected:
			err = qi8.FullyConnected(dst, src, l.Weights, l.Bias, l.Requant, qi8.FCParams{
				In: l.In.Len(), Out: l.Out.C, InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, ActMin: l.ActMin, ActMax: l.ActMax})
		case qnet.MaxPool2D:
			err = qi8.MaxPool2D(dst, src, qi8.Pool2DParams{In: l.In, KH: l.KH, KW: l.KW, StrideH: l.StrideH, StrideW: l.StrideW, Padding: l.Padding})
		case qnet.AvgPool2D:
			err = qi8.AvgPool2D(dst, src, qi8.Pool2DParams{In: l.In, KH: l.KH, KW: l.KW, StrideH: l.StrideH, StrideW: l.StrideW, Padding: l.Padding})
		case qnet.ReLU:
			err = qi8.ReLU(dst, src, l.InQ.ZP)
		case qnet.ReLU6:
			err = qi8.ReLU6(dst, src, l.InQ.ZP, qi8.Quantize6(l.InQ.Scale, l.InQ.ZP))
		default:
			t.Fatalf("layer %d: kind %v", i, l.Kind)
		}
		if err != nil {
			t.Fatalf("layer %d (%v): %v", i, l.Kind, err)
		}
		src = dst
	}
	return src
}

func TestValidate(t *testing.T) {
	cases := []struct {
		name  string
		mod   func(ls []qnet.Layer) []qnet.Layer
		idx   int
		error error
	}{
		{"ok", func(ls []qnet.Layer) []qnet.Layer { return ls }, -1, nil},
		{"nil bias ok", func(ls []qnet.Layer) []qnet.Layer { ls[2].Bias = nil; return ls }, -1, nil},
		{"empty", func([]qnet.Layer) []qnet.Layer { return nil }, -1, qnet.ErrEmpty},
		{"unknown kind", func(ls []qnet.Layer) []qnet.Layer { ls[3].Kind = 0; return ls }, 3, qnet.ErrKind},
		{"wrong out", func(ls []qnet.Layer) []qnet.Layer { ls[1].Out.H = 3; return ls }, 1, qnet.ErrShape},
		{"zero kernel", func(ls []qnet.Layer) []qnet.Layer { ls[4].KH = 0; return ls }, 4, qnet.ErrShape},
		{"valid kernel > input", func(ls []qnet.Layer) []qnet.Layer { ls[6].KH = 3; return ls }, 6, qnet.ErrShape},
		{"chain shape", func(ls []qnet.Layer) []qnet.Layer {
			ls[5].In.C, ls[5].Out.C = 4, 4 // self-consistent, but != layer 4 out
			return ls
		}, 5, qnet.ErrShape},
		{"chain quant", func(ls []qnet.Layer) []qnet.Layer { ls[1].InQ.ZP = 0; return ls }, 1, qnet.ErrQuant},
		{"zero scale", func(ls []qnet.Layer) []qnet.Layer { ls[0].InQ.Scale = 0; return ls }, 0, qnet.ErrQuant},
		{"pool requant", func(ls []qnet.Layer) []qnet.Layer { ls[4].OutQ.ZP = 0; return ls }, 4, qnet.ErrQuant},
		{"act range", func(ls []qnet.Layer) []qnet.Layer { ls[2].ActMin, ls[2].ActMax = 5, 4; return ls }, 2, qnet.ErrActivation},
		{"missing weights", func(ls []qnet.Layer) []qnet.Layer { ls[1].Weights = nil; return ls }, 1, qnet.ErrParams},
		{"short weights", func(ls []qnet.Layer) []qnet.Layer { ls[7].Weights = ls[7].Weights[:39]; return ls }, 7, qnet.ErrParams},
		{"short bias", func(ls []qnet.Layer) []qnet.Layer { ls[0].Bias = ls[0].Bias[:3]; return ls }, 0, qnet.ErrParams},
		{"missing requant", func(ls []qnet.Layer) []qnet.Layer { ls[2].Requant = nil; return ls }, 2, qnet.ErrParams},
		{"invalid requant", func(ls []qnet.Layer) []qnet.Layer { ls[2].Requant[3].Mult = 5; return ls }, 2, qnet.ErrParams},
		{"params on pool", func(ls []qnet.Layer) []qnet.Layer { ls[6].Weights = ls[0].Weights; return ls }, 6, qnet.ErrParams},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g := bound(t, smallLayers(), 1)
			g.Layers = c.mod(g.Layers)
			idx, err := g.Check()
			if err != c.error || idx != c.idx {
				t.Fatalf("Check = %d, %v; want %d, %v", idx, err, c.idx, c.error)
			}
			if err := g.Validate(); err != c.error {
				t.Fatalf("Validate = %v", err)
			}
		})
	}
}

func TestParamCountsBind(t *testing.T) {
	g := qnet.Graph{Layers: smallLayers()}
	nw, nb, nr := qnet.ParamCounts(&g)
	// conv 4·9·3 + dw 9·4 + pw 8·4 + fc 5·8; bias/requant 4+4+8+5.
	if nw != 108+36+32+40 || nb != 21 || nr != 21 {
		t.Fatalf("ParamCounts = %d %d %d", nw, nb, nr)
	}
	w, b, rq := make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
	if err := qnet.BindParams(&g, w[:nw-1], b, rq); err != qnet.ErrParams {
		t.Fatalf("short weights: %v", err)
	}
	if err := qnet.BindParams(&g, w, b[:1], rq); err != qnet.ErrParams {
		t.Fatalf("short bias: %v", err)
	}
	if err := qnet.BindParams(&g, w, b, rq); err != nil {
		t.Fatal(err)
	}
	if &g.Layers[1].Weights[0] != &w[108] || &g.Layers[2].Bias[0] != &b[8] || &g.Layers[7].Requant[0] != &rq[16] {
		t.Fatal("params not sliced in layer order")
	}
	if cap(g.Layers[0].Weights) != 108 || g.Layers[3].Weights != nil || g.Layers[3].Requant != nil {
		t.Fatal("bad slicing of capacity / param-free layers")
	}
	if err := qnet.BindParams(&g, w, nil, rq); err != nil || g.Layers[0].Bias != nil {
		t.Fatalf("nil bias: %v", err)
	}
}

func TestBudget(t *testing.T) {
	g := qnet.Graph{Layers: smallLayers()}
	want := []qnet.LayerBudget{
		{In: 126, Out: 168, Weights: 108, Params: 48, MACs: 7 * 6 * 108},
		{In: 168, Out: 48, Weights: 36, Params: 48, MACs: 4 * 3 * 36},
		{In: 48, Out: 96, Weights: 32, Params: 96, MACs: 4 * 3 * 32},
		{In: 96, Out: 96},
		{In: 96, Out: 32},
		{In: 32, Out: 32},
		{In: 32, Out: 8},
		{In: 8, Out: 5, Weights: 40, Params: 60, MACs: 40},
	}
	for i, w := range want {
		if got := g.Budget(i); got != w {
			t.Errorf("Budget(%d) = %+v, want %+v", i, got, w)
		}
	}
	if g.WeightBytes() != 216 || g.ParamBytes() != 252 || g.MACs() != 4536+432+384+40 {
		t.Fatalf("totals %d %d %d", g.WeightBytes(), g.ParamBytes(), g.MACs())
	}
}

// planSize returns the arena bytes the plan for g needs.
func planSize(t testing.TB, g *qnet.Graph, backends ...qnet.Backend) int {
	t.Helper()
	p, err := qnet.PlanMemory(g, backends...)
	if err != nil {
		t.Fatal(err)
	}
	return p.Size()
}

// branchQ quantises residual branches differently from actQ, so every Add
// rescales both inputs; logitQ is the FC output feeding Softmax.
var (
	branchQ = qnet.Quant{Scale: 0.05, ZP: 3}
	logitQ  = qnet.Quant{Scale: 0.125, ZP: 2}
)

// resLayers is a small residual net covering From, Add (incl. a skip from the
// graph input), Reshape and Softmax:
//
//	0 conv 1×1  7×6×3 → 7×6×3  linear, branchQ
//	1 add(0, input)            → actQ, ReLU6
//	2 conv 3×3  → 7×6×8        ReLU6, actQ
//	3 conv 3×3  → 7×6×8        linear, branchQ
//	4 add(3, 2)                → actQ, ReLU6
//	5 dw 3×3/2  → 4×3×8        actQ
//	6 avgpool 4×3 VALID → 1×1×8
//	7 reshape   → 2×2×2
//	8 fc 8 → 5                 logitQ
//	9 softmax                  SoftmaxOutQuant
func resLayers(round qi8.Rounding) []qnet.Layer {
	lo, hi := qi8.ActReLU6(actQ.Scale, actQ.ZP)
	nlo, nhi := qi8.ActNone()
	conv := func(k qnet.Kind, in qi8.Shape, q, oq qnet.Quant, outC, kk, s int, lo, hi int8) qnet.Layer {
		return qnet.Layer{Kind: k, In: in, Out: qi8.Shape{C: outC}, InQ: q, OutQ: oq,
			KH: kk, KW: kk, StrideH: s, StrideW: s, Padding: qi8.PaddingSame, ActMin: lo, ActMax: hi, Round: round}
	}
	add := func(from []int, in qi8.Shape, q, q2 qnet.Quant) qnet.Layer {
		return qnet.Layer{Kind: qnet.Add, From: from, In: in, InQ: q, In2Q: q2, OutQ: actQ, ActMin: lo, ActMax: hi, Round: round}
	}
	s73, s78 := qi8.Shape{H: 7, W: 6, C: 3}, qi8.Shape{H: 7, W: 6, C: 8}
	ls := []qnet.Layer{
		conv(qnet.Conv2D, s73, inQ, branchQ, 3, 1, 1, nlo, nhi),
		add([]int{0, -1}, s73, branchQ, inQ),
		conv(qnet.Conv2D, s73, actQ, actQ, 8, 3, 1, lo, hi),
		conv(qnet.Conv2D, s78, actQ, branchQ, 8, 3, 1, nlo, nhi),
		add([]int{3, 2}, s78, branchQ, actQ),
		conv(qnet.DepthwiseConv2D, s78, actQ, actQ, 8, 3, 2, lo, hi),
		{Kind: qnet.AvgPool2D, In: qi8.Shape{H: 4, W: 3, C: 8}, InQ: actQ, OutQ: actQ, KH: 4, KW: 3, StrideH: 1, StrideW: 1, Padding: qi8.PaddingValid},
		{Kind: qnet.Reshape, In: qi8.Shape{H: 1, W: 1, C: 8}, Out: qi8.Shape{H: 2, W: 2, C: 2}, InQ: actQ, OutQ: actQ},
		{Kind: qnet.FullyConnected, In: qi8.Shape{H: 2, W: 2, C: 2}, Out: qi8.Shape{C: 5}, InQ: actQ, OutQ: logitQ, ActMin: nlo, ActMax: nhi, Round: round},
		{Kind: qnet.Softmax, In: qi8.Shape{H: 1, W: 1, C: 5}, InQ: logitQ, OutQ: qnet.SoftmaxOutQuant},
	}
	for i := range ls {
		out, err := ls[i].OutShape()
		if err != nil {
			panic(err)
		}
		ls[i].Out = out
	}
	return ls
}

// boundRes binds synthetic params and derives the Add and Softmax params.
func boundRes(t testing.TB, round qi8.Rounding, seed uint64) qnet.Graph {
	t.Helper()
	ls := resLayers(round)
	g := qnet.Graph{Layers: ls}
	nw, nb, nr := qnet.ParamCounts(&g)
	if err := qnet.BindParams(&g, make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)); err != nil {
		t.Fatal(err)
	}
	if err := qnet.SyntheticParams(&g, seed); err != nil {
		t.Fatal(err)
	}
	for i := range g.Layers {
		l := &g.Layers[i]
		var err error
		switch l.Kind {
		case qnet.Add:
			if round == qi8.RoundTFLite {
				l.AddTFLite, err = qi8.QuantizeAddTFLite(l.InQ.Scale, l.In2Q.Scale, l.OutQ.Scale)
			} else {
				so := float64(l.OutQ.Scale)
				l.AddRQ, err = qi8.QuantizeAddMultipliers(float64(l.InQ.Scale)/so, float64(l.In2Q.Scale)/so)
			}
		case qnet.Softmax:
			l.SoftmaxParams, err = qi8.QuantizeSoftmax(1, l.InQ.Scale)
		}
		if err != nil {
			t.Fatal(err)
		}
	}
	if layer, err := g.Check(); err != nil {
		t.Fatalf("layer %d: %v", layer, err)
	}
	return g
}

// reference runs g layer by layer with qnet.RunLayer (qi8 free functions),
// following From.
func reference(t testing.TB, g *qnet.Graph, input []int8) []int8 {
	t.Helper()
	outs := make([][]int8, len(g.Layers))
	get := func(i, k int) []int8 {
		p := i - 1
		if f := g.Layers[i].From; f != nil {
			p = f[k]
		}
		if p < 0 {
			return input
		}
		return outs[p]
	}
	for i := range g.Layers {
		l := &g.Layers[i]
		outs[i] = make([]int8, l.Out.Len())
		var srcs [][]int8
		switch {
		case l.Kind == qnet.Const:
		case l.From == nil:
			srcs = [][]int8{get(i, 0)}
		default:
			for k := range l.From {
				srcs = append(srcs, get(i, k))
			}
		}
		if err := qnet.RunLayer(l, outs[i], srcs...); err != nil {
			t.Fatalf("layer %d (%v): %v", i, l.Kind, err)
		}
	}
	return outs[len(outs)-1]
}

func TestValidateDAG(t *testing.T) {
	cases := []struct {
		name  string
		mod   func(ls []qnet.Layer)
		idx   int
		error error
	}{
		{"ok", func([]qnet.Layer) {}, -1, nil},
		{"from out of range", func(ls []qnet.Layer) { ls[4].From = []int{5, 2} }, 4, qnet.ErrShape},
		{"from below -1", func(ls []qnet.Layer) { ls[4].From = []int{3, -2} }, 4, qnet.ErrShape},
		{"add needs two", func(ls []qnet.Layer) { ls[4].From = []int{3} }, 4, qnet.ErrShape},
		{"add nil from", func(ls []qnet.Layer) { ls[4].From = nil }, 4, qnet.ErrShape},
		{"conv two inputs", func(ls []qnet.Layer) { ls[2].From = []int{1, 0} }, 2, qnet.ErrShape},
		{"layer 0 reads a layer", func(ls []qnet.Layer) { ls[0].From = []int{0} }, 0, qnet.ErrShape},
		{"explicit from ok", func(ls []qnet.Layer) { ls[2].From = []int{1} }, -1, nil},
		{"producer shape", func(ls []qnet.Layer) { ls[2].From = []int{-1} }, 2, qnet.ErrQuant}, // input is 7×6×3 like layer 1, but inQ != actQ
		{"add second shape", func(ls []qnet.Layer) { ls[4].From = []int{3, 1} }, 4, qnet.ErrShape},
		{"add second quant", func(ls []qnet.Layer) { ls[4].In2Q = branchQ }, 4, qnet.ErrQuant},
		{"add multipliers", func(ls []qnet.Layer) { ls[4].AddRQ = qi8.AddRequant{} }, 4, qnet.ErrParams},
		{"add act range", func(ls []qnet.Layer) { ls[4].ActMin, ls[4].ActMax = 5, 4 }, 4, qnet.ErrActivation},
		{"params on add", func(ls []qnet.Layer) { ls[4].Weights = ls[2].Weights }, 4, qnet.ErrParams},
		{"reshape length", func(ls []qnet.Layer) { ls[7].Out = qi8.Shape{H: 3, W: 1, C: 2} }, 7, qnet.ErrShape},
		{"reshape requant", func(ls []qnet.Layer) { ls[7].OutQ = branchQ }, 7, qnet.ErrQuant},
		{"softmax out quant", func(ls []qnet.Layer) { ls[9].OutQ = actQ }, 9, qnet.ErrQuant},
		{"softmax params", func(ls []qnet.Layer) { ls[9].SoftmaxParams = qi8.SoftmaxParams{} }, 9, qnet.ErrParams},
		{"bad rounding", func(ls []qnet.Layer) { ls[2].Round = 7 }, 2, qnet.ErrParams},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g := boundRes(t, qi8.RoundExact, 1)
			c.mod(g.Layers)
			idx, err := g.Check()
			if err != c.error || idx != c.idx {
				t.Fatalf("Check = %d, %v; want %d, %v", idx, err, c.idx, c.error)
			}
		})
	}
}

// pairBackend fuses a depthwise conv followed by a 1×1 conv, running both
// through qi8 with its own scratch buffer for the intermediate tensor.
type pairBackend struct {
	scratch []int8
	runs    int
}

func (*pairBackend) Name() string { return "pair" }

func (*pairBackend) Claim(ls []qnet.Layer) int {
	if len(ls) >= 2 && ls[0].Kind == qnet.DepthwiseConv2D && ls[1].Kind == qnet.Conv2D && ls[1].KH == 1 && ls[1].KW == 1 {
		return 2
	}
	return 0
}

func (b *pairBackend) Run(ls []qnet.Layer, dst, src []int8) error {
	if len(ls) != 2 {
		return qnet.ErrSegment
	}
	b.runs++
	mid := b.scratch[:ls[0].Out.Len()]
	if err := qnet.RunLayer(&ls[0], mid, src); err != nil {
		return err
	}
	return qnet.RunLayer(&ls[1], dst, mid)
}

// claimer claims a fixed count at the layers listed in at (every layer when
// at is nil).
type claimer struct {
	n  int
	at []int
	g  *qnet.Graph
}

func (claimer) Name() string { return "claimer" }

func (c claimer) Claim(ls []qnet.Layer) int {
	if c.at == nil {
		return c.n
	}
	first := len(c.g.Layers) - len(ls)
	for _, a := range c.at {
		if a == first {
			return c.n
		}
	}
	return 0
}

func (claimer) Run([]qnet.Layer, []int8, []int8) error { return nil }

func planString(p []qnet.Segment) string {
	var b bytes.Buffer
	for _, s := range p {
		name := "cpu"
		if s.Backend != nil {
			name = s.Backend.Name()
		}
		b.WriteString(name)
		b.WriteByte(byte('0' + s.First))
		b.WriteByte(byte('0' + s.Count))
		b.WriteByte(' ')
	}
	return b.String()
}

// checkPlan verifies the plan's invariants independently: every arena
// tensor lies inside the activation region, tensors whose lifetimes overlap
// do not overlap in memory, Reshape outputs alias their input and Const
// outputs are their Weights (Off −2).
func checkPlan(t *testing.T, g *qnet.Graph, p *qnet.MemPlan) {
	t.Helper()
	n := len(g.Layers)
	seg := make([]int, n)
	lastOf := make([]bool, n)
	for k, s := range p.Segments {
		for i := s.First; i < s.First+s.Count; i++ {
			seg[i] = k
		}
		lastOf[s.First+s.Count-1] = true
	}
	prod := func(i, k int) int {
		if f := g.Layers[i].From; f != nil {
			return f[k]
		}
		return i - 1
	}
	// live interval of each layer's output in segment steps (alias-aware)
	born, end := make([]int, n), make([]int, n)
	for i := range g.Layers {
		born[i], end[i] = seg[i], seg[i]
	}
	end[n-1] = len(p.Segments)
	for i := n - 1; i >= 0; i-- {
		nin := 1
		switch k := g.Layers[i].Kind; {
		case k == qnet.Add || k == qnet.Mul:
			nin = 2
		case k == qnet.Const:
			nin = 0
		case k == qnet.Concat && g.Layers[i].From != nil:
			nin = len(g.Layers[i].From)
		}
		for k := range nin {
			if pr := prod(i, k); pr >= 0 {
				end[pr] = max(end[pr], seg[i], end[i]*boolInt(g.Layers[i].Kind == qnet.Reshape))
			}
		}
	}
	for i := range g.Layers {
		o := p.Out[i]
		if !lastOf[i] {
			if o.Len != 0 {
				t.Fatalf("fused intermediate %d has region %+v", i, o)
			}
			continue
		}
		if o.Len != g.Layers[i].Out.Len() {
			t.Fatalf("layer %d region %+v, want len %d", i, o, g.Layers[i].Out.Len())
		}
		if g.Layers[i].Kind == qnet.Const {
			if o.Off != -2 {
				t.Fatalf("const %d region %+v is not its Weights", i, o)
			}
			continue
		}
		if g.Layers[i].Kind == qnet.Reshape {
			pr := prod(i, 0)
			if (pr < 0 && o.Off != -1) || (pr >= 0 && o.Off != p.Out[pr].Off) {
				t.Fatalf("reshape %d region %+v does not alias its input", i, o)
			}
			continue
		}
		if o.Off < p.Persistent || o.Off%16 != 0 || o.Off+o.Len > p.Persistent+p.Activations {
			t.Fatalf("layer %d region %+v outside activations [%d,%d)", i, o, p.Persistent, p.Persistent+p.Activations)
		}
		for j := range i {
			q := p.Out[j]
			if !lastOf[j] || g.Layers[j].Kind == qnet.Reshape || g.Layers[j].Kind == qnet.Const || q.Len == 0 {
				continue
			}
			lifeOverlap := born[i] <= end[j] && born[j] <= end[i]
			memOverlap := o.Off < q.Off+q.Len && q.Off < o.Off+o.Len
			if lifeOverlap && memOverlap {
				t.Fatalf("layers %d %+v [%d,%d] and %d %+v [%d,%d] overlap", i, o, born[i], end[i], j, q, born[j], end[j])
			}
		}
	}
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func TestRunnerSequential(t *testing.T) {
	g := bound(t, smallLayers(), 7)
	in := randInput(g.Layers[0].In.Len(), 3)
	want := runByHand(t, &g, in)

	p, err := qnet.PlanMemory(&g)
	if err != nil {
		t.Fatal(err)
	}
	checkPlan(t, &g, p)
	// Layer outputs (16-aligned): 176, 48, 96, 96, 32, 32, 16, 16. The
	// greedy plan reaches the ping-pong bound max(in+out) = 176+48.
	if p.Activations != 224 {
		t.Fatalf("activations %d", p.Activations)
	}
	if qi8.Backend() == "generic" && p.Persistent != 0 {
		t.Fatalf("generic build needs no scratch, got %d", p.Persistent)
	}
	if _, err := qnet.NewRunner(&g, make([]int8, p.Size()-1)); err != qnet.ErrArena {
		t.Fatalf("short arena: %v", err)
	}
	r, err := qnet.NewRunner(&g, make([]int8, p.Size()))
	if err != nil {
		t.Fatal(err)
	}
	if got := planString(r.Plan().Segments); got != "cpu01 cpu11 cpu21 cpu31 cpu41 cpu51 cpu61 cpu71 " {
		t.Fatalf("plan %q", got)
	}
	for range 2 { // repeatable
		out, err := r.Run(in)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(i8b(out), i8b(want)) {
			t.Fatalf("out %v, want %v", out, want)
		}
	}
	if _, err := r.Run(in[1:]); err != qnet.ErrInput {
		t.Fatalf("short input: %v", err)
	}
	var zero qnet.Runner
	if _, err := zero.Run(in); err != qnet.ErrNotReady {
		t.Fatalf("zero runner: %v", err)
	}
}

func TestRunnerDAG(t *testing.T) {
	for _, round := range []qi8.Rounding{qi8.RoundExact, qi8.RoundTFLite} {
		g := boundRes(t, round, 5)
		p, err := qnet.PlanMemory(&g)
		if err != nil {
			t.Fatal(err)
		}
		checkPlan(t, &g, p)
		r, err := qnet.NewRunner(&g, make([]int8, p.Size()))
		if err != nil {
			t.Fatal(err)
		}
		for seed := range uint64(5) {
			in := randInput(g.Layers[0].In.Len(), seed)
			want := reference(t, &g, in)
			got, err := r.Run(in)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(i8b(got), i8b(want)) {
				t.Fatalf("round %d seed %d: out %v, want %v", round, seed, got, want)
			}
		}
		in := randInput(126, 1)
		if n := testing.AllocsPerRun(10, func() { _, err = r.Run(in) }); n != 0 || err != nil {
			t.Fatalf("Run allocs %v, %v", n, err)
		}
	}
	// The rounding mode reaches the kernels: the two modes differ somewhere.
	ge, gt := boundRes(t, qi8.RoundExact, 5), boundRes(t, qi8.RoundTFLite, 5)
	differ := false
	for seed := range uint64(40) {
		in := randInput(126, seed)
		if !bytes.Equal(i8b(reference(t, &ge, in)), i8b(reference(t, &gt, in))) {
			differ = true
			break
		}
	}
	if !differ {
		t.Fatal("RoundExact and RoundTFLite graphs agree on 40 inputs")
	}
}

func TestRunnerReshapeOfInput(t *testing.T) {
	q := qnet.Quant{Scale: 0.5, ZP: 1}
	g := qnet.Graph{Layers: []qnet.Layer{
		{Kind: qnet.Reshape, In: qi8.Shape{H: 2, W: 3, C: 1}, Out: qi8.Shape{H: 1, W: 1, C: 6}, InQ: q, OutQ: q},
	}}
	r, err := qnet.NewRunner(&g, make([]int8, 64))
	if err != nil {
		t.Fatal(err)
	}
	if a := r.Plan().Activations; a != 0 {
		t.Fatalf("activations %d", a)
	}
	in := []int8{1, 2, 3, 4, 5, 6}
	out, err := r.Run(in)
	if err != nil || &out[0] != &in[0] || len(out) != 6 {
		t.Fatalf("reshape of the input must alias it: %v %v", out, err)
	}
}

func TestRunnerPlanErrors(t *testing.T) {
	g := bound(t, smallLayers(), 7)
	arena := make([]int8, 4096)
	var r qnet.Runner
	if err := r.Init(&g, arena, claimer{n: 9}); err != qnet.ErrClaim {
		t.Fatalf("over-claim: %v", err)
	}
	if err := r.Init(&g, arena, claimer{n: -1}); err != qnet.ErrClaim {
		t.Fatalf("negative claim: %v", err)
	}
	if _, err := r.Run(make([]int8, 126)); err != qnet.ErrNotReady {
		t.Fatalf("after failed Init: %v", err)
	}
	g.Layers[2].Requant = nil
	if err := r.Init(&g, arena); err != qnet.ErrParams {
		t.Fatalf("unbound: %v", err)
	}

	gr := boundRes(t, qi8.RoundExact, 3)
	for _, c := range []struct {
		name string
		cl   claimer
	}{
		{"claim over an Add", claimer{n: 2, at: []int{0}, g: &gr}},
		{"intermediate read outside", claimer{n: 2, at: []int{2}, g: &gr}}, // layer 2 also feeds Add 4
	} {
		if _, err := qnet.PlanMemory(&gr, c.cl); err != qnet.ErrClaim {
			t.Fatalf("%s: %v", c.name, err)
		}
	}
	if _, err := qnet.PlanMemory(&gr, claimer{n: 2, at: []int{5}, g: &gr}); err != nil {
		t.Fatalf("dw+pool chain: %v", err)
	}
}

func TestRunnerFusedSegments(t *testing.T) {
	g := bound(t, smallLayers(), 11)
	in := randInput(g.Layers[0].In.Len(), 5)
	want := runByHand(t, &g, in)
	pair := &pairBackend{scratch: make([]int8, 256)}

	p, err := qnet.PlanMemory(&g, pair)
	if err != nil {
		t.Fatal(err)
	}
	checkPlan(t, &g, p)
	// Segment 1-2 reads layer 0's 176 B and writes 96 B while both live.
	if p.Activations != 272 {
		t.Fatalf("activations %d", p.Activations)
	}
	var r qnet.Runner
	if err := r.Init(&g, make([]int8, p.Size()), pair); err != nil {
		t.Fatal(err)
	}
	if got := planString(r.Plan().Segments); got != "cpu01 pair12 cpu31 cpu41 cpu51 cpu61 cpu71 " {
		t.Fatalf("plan %q", got)
	}
	out, err := r.Run(in)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(i8b(out), i8b(want)) || pair.runs != 1 {
		t.Fatalf("out %v, want %v (pair runs %d)", out, want, pair.runs)
	}
}

func i8b(s []int8) []byte {
	b := make([]byte, len(s))
	for i, v := range s {
		b[i] = byte(v)
	}
	return b
}

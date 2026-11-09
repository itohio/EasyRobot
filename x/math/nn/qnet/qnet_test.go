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
	// max(168, 168+48, 48+96, 96+96, 96+32, 32+32, 32+8, 8+5): layer 0 reads the input.
	if g.ArenaSize() != 216 {
		t.Fatalf("ArenaSize = %d", g.ArenaSize())
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

// claimer claims a fixed count for every layer.
type claimer int

func (claimer) Name() string                           { return "claimer" }
func (c claimer) Claim([]qnet.Layer) int               { return int(c) }
func (claimer) Run([]qnet.Layer, []int8, []int8) error { return nil }

// convOnly claims only Conv2D layers (and runs them generically).
type convOnly struct{ qnet.Generic }

func (convOnly) Claim(ls []qnet.Layer) int {
	if len(ls) > 0 && ls[0].Kind == qnet.Conv2D {
		return 1
	}
	return 0
}

func planString(p []qnet.Segment) string {
	var b bytes.Buffer
	for _, s := range p {
		b.WriteString(s.Backend.Name())
		b.WriteByte(byte('0' + s.First))
		b.WriteByte(byte('0' + s.Count))
		b.WriteByte(' ')
	}
	return b.String()
}

func TestRunnerGeneric(t *testing.T) {
	g := bound(t, smallLayers(), 7)
	in := randInput(g.Layers[0].In.Len(), 3)
	want := runByHand(t, &g, in)

	if _, err := qnet.NewRunner(&g, make([]int8, 215), qnet.Generic{}); err != qnet.ErrArena {
		t.Fatalf("arena 215: %v", err)
	}
	r, err := qnet.NewRunner(&g, make([]int8, 216), qnet.Generic{})
	if err != nil {
		t.Fatal(err)
	}
	if got := planString(r.Plan()); got != "generic01 generic11 generic21 generic31 generic41 generic51 generic61 generic71 " {
		t.Fatalf("plan %q", got)
	}
	if r.ArenaSize() != 216 {
		t.Fatalf("ArenaSize %d", r.ArenaSize())
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

func TestRunnerPlanErrors(t *testing.T) {
	g := bound(t, smallLayers(), 7)
	arena := make([]int8, 1024)
	var r qnet.Runner
	if err := r.Init(&g, arena); err != qnet.ErrUnclaimed {
		t.Fatalf("no backends: %v", err)
	}
	if err := r.Init(&g, arena, convOnly{}); err != qnet.ErrUnclaimed {
		t.Fatalf("conv only: %v", err)
	}
	if err := r.Init(&g, arena, claimer(9)); err != qnet.ErrClaim {
		t.Fatalf("over-claim: %v", err)
	}
	if err := r.Init(&g, arena, claimer(-1)); err != qnet.ErrClaim {
		t.Fatalf("negative claim: %v", err)
	}
	if _, err := r.Run(make([]int8, 126)); err != qnet.ErrNotReady {
		t.Fatalf("after failed Init: %v", err)
	}
	g.Layers[2].Requant = nil
	if err := r.Init(&g, arena, qnet.Generic{}); err != qnet.ErrParams {
		t.Fatalf("unbound: %v", err)
	}

	// MaxSegments+1 ReLU layers cannot be planned one per segment.
	many := make([]qnet.Layer, qnet.MaxSegments+1)
	for i := range many {
		many[i] = qnet.Layer{Kind: qnet.ReLU, In: qi8.Shape{H: 1, W: 1, C: 2}, Out: qi8.Shape{H: 1, W: 1, C: 2}, InQ: actQ, OutQ: actQ}
	}
	gm := qnet.Graph{Layers: many}
	if err := r.Init(&gm, arena, qnet.Generic{}); err != qnet.ErrCapacity {
		t.Fatalf("too many segments: %v", err)
	}
	if err := r.Init(&gm, arena, claimer(qnet.MaxSegments+1)); err != nil {
		t.Fatalf("one segment: %v", err)
	}
}

func TestRunnerFusedSegments(t *testing.T) {
	g := bound(t, smallLayers(), 11)
	in := randInput(g.Layers[0].In.Len(), 5)
	want := runByHand(t, &g, in)
	pair := &pairBackend{scratch: make([]int8, 256)}

	// The fused segment 1-2 reads 168 B and writes 96 B: more than the
	// per-layer maximum of 216.
	var r qnet.Runner
	if err := r.Init(&g, make([]int8, 216), pair, qnet.Generic{}); err != qnet.ErrArena {
		t.Fatalf("arena 216: %v", err)
	}
	if err := r.Init(&g, make([]int8, 264), pair, qnet.Generic{}); err != nil {
		t.Fatal(err)
	}
	if got := planString(r.Plan()); got != "generic01 pair12 generic31 generic41 generic51 generic61 generic71 " {
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

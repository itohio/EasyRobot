package qnet_test

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

var (
	mulQ  = qnet.Quant{Scale: 0.011, ZP: -9}
	c5Q   = qnet.Quant{Scale: 0.02, ZP: 3}
	meanQ = qnet.Quant{Scale: 0.007, ZP: 5}
	sQ    = qnet.Quant{Scale: 0.03, ZP: -2}
	prodQ = qnet.Quant{Scale: 0.004, ZP: 1}
)

// opsLayers covers the kinds added for TFLite face detection and YOLO:
//
//	0 pad (1,0 | 0,2 | 0,1)   6×5×4 → 7×7×5   inQ
//	1 resize nearest, half pixel → 4×9×5
//	2 logistic                → LogisticOutQuant
//	3 const 1×1×5             c5Q
//	4 mul(2, 3) per channel   → mulQ, clamp [−9, 127]
//	5 requantize(1)           → mulQ
//	6 mul(4, 5) element-wise  → mulQ
//	7 concat(6, 5) on C       → 4×9×10
//	8 concat(7, 7) on W       → 4×18×10
//	9 mean                    → 1×1×10 meanQ
//	10 const 1×1×1            sQ
//	11 mul(9, 10) scalar      → prodQ
//	12 reshape                → 1×2×5
func opsLayers(t testing.TB, round qi8.Rounding) qnet.Graph {
	t.Helper()
	s45 := qi8.Shape{H: 4, W: 9, C: 5}
	c5 := randInput(5, 11)
	sc := randInput(1, 12)
	mul := func(from []int, in qi8.Shape, q, q2, oq qnet.Quant, lo int8) qnet.Layer {
		rq, err := qi8.QuantizeMul(q.Scale, q2.Scale, oq.Scale)
		if err != nil {
			t.Fatal(err)
		}
		return qnet.Layer{Kind: qnet.Mul, From: from, In: in, InQ: q, In2Q: q2, OutQ: oq, ActMin: lo, ActMax: 127, Round: round, RQ: rq}
	}
	lp, err := qi8.QuantizeLogistic(inQ.Scale, inQ.ZP)
	if err != nil {
		t.Fatal(err)
	}
	rrq, err := qi8.QuantizeRequantize(inQ.Scale, mulQ.Scale)
	if err != nil {
		t.Fatal(err)
	}
	mrq, err := qi8.QuantizeMean(mulQ.Scale, meanQ.Scale)
	if err != nil {
		t.Fatal(err)
	}
	ls := []qnet.Layer{
		{Kind: qnet.Pad, In: qi8.Shape{H: 6, W: 5, C: 4}, InQ: inQ, OutQ: inQ, Pads: [6]int{1, 0, 0, 2, 0, 1}, PadValue: inQ.ZP},
		{Kind: qnet.ResizeNearest, In: qi8.Shape{H: 7, W: 7, C: 5}, Out: s45, InQ: inQ, OutQ: inQ, HalfPixel: true},
		{Kind: qnet.Logistic, In: s45, InQ: inQ, OutQ: qnet.LogisticOutQuant, LogisticParams: &lp},
		{Kind: qnet.Const, In: qi8.Shape{H: 1, W: 1, C: 5}, Out: qi8.Shape{H: 1, W: 1, C: 5}, InQ: c5Q, OutQ: c5Q, Weights: c5},
		mul([]int{2, 3}, s45, qnet.LogisticOutQuant, c5Q, mulQ, -9),
		{Kind: qnet.Requantize, From: []int{1}, In: s45, InQ: inQ, OutQ: mulQ, Round: round, RQ: rrq},
		mul([]int{4, 5}, s45, mulQ, mulQ, mulQ, -128),
		{Kind: qnet.Concat, From: []int{6, 5}, In: s45, Out: qi8.Shape{H: 4, W: 9, C: 10}, InQ: mulQ, OutQ: mulQ, Axis: 2},
		{Kind: qnet.Concat, From: []int{7, 7}, In: qi8.Shape{H: 4, W: 9, C: 10}, Out: qi8.Shape{H: 4, W: 18, C: 10}, InQ: mulQ, OutQ: mulQ, Axis: 1},
		{Kind: qnet.Mean, In: qi8.Shape{H: 4, W: 18, C: 10}, InQ: mulQ, OutQ: meanQ, Round: round, RQ: mrq},
		{Kind: qnet.Const, In: qi8.Shape{H: 1, W: 1, C: 1}, Out: qi8.Shape{H: 1, W: 1, C: 1}, InQ: sQ, OutQ: sQ, Weights: sc},
		mul([]int{9, 10}, qi8.Shape{H: 1, W: 1, C: 10}, meanQ, sQ, prodQ, -128),
		{Kind: qnet.Reshape, In: qi8.Shape{H: 1, W: 1, C: 10}, Out: qi8.Shape{H: 1, W: 2, C: 5}, InQ: prodQ, OutQ: prodQ},
	}
	for i := range ls {
		out, err := ls[i].OutShape()
		if err != nil {
			t.Fatalf("layer %d: %v", i, err)
		}
		ls[i].Out = out
	}
	g := qnet.Graph{Layers: ls}
	if layer, err := g.Check(); err != nil {
		t.Fatalf("layer %d: %v", layer, err)
	}
	return g
}

func TestRunnerOpsKinds(t *testing.T) {
	for _, round := range []qi8.Rounding{qi8.RoundExact, qi8.RoundTFLite} {
		g := opsLayers(t, round)
		p, err := qnet.PlanMemory(&g)
		if err != nil {
			t.Fatal(err)
		}
		checkPlan(t, &g, p)
		for _, i := range []int{3, 10} {
			if p.Out[i].Off != -2 {
				t.Fatalf("const %d region %+v, want the Weights (Off -2)", i, p.Out[i])
			}
		}
		r, err := qnet.NewRunner(&g, make([]int8, p.Size()))
		if err != nil {
			t.Fatal(err)
		}
		for seed := uint64(1); seed <= 20; seed++ {
			in := randInput(g.Layers[0].In.Len(), seed)
			want := reference(t, &g, in)
			got, err := r.Run(in)
			if err != nil {
				t.Fatal(err)
			}
			if string(u8s(got)) != string(u8s(want)) {
				t.Fatalf("round %d seed %d: runner %v, reference %v", round, seed, got, want)
			}
		}
		in := randInput(g.Layers[0].In.Len(), 99)
		if n := testing.AllocsPerRun(20, func() { _, _ = r.Run(in) }); n != 0 {
			t.Fatalf("Run allocates %v", n)
		}
	}
}

func TestValidateOpsKinds(t *testing.T) {
	cases := []struct {
		name  string
		mod   func(ls []qnet.Layer)
		idx   int
		error error
	}{
		{"ok", func([]qnet.Layer) {}, -1, nil},
		{"logistic out quant", func(ls []qnet.Layer) { ls[2].OutQ = mulQ }, 2, qnet.ErrQuant},
		{"logistic table missing", func(ls []qnet.Layer) { ls[2].LogisticParams = nil }, 2, qnet.ErrParams},
		{"const with from", func(ls []qnet.Layer) { ls[3].From = []int{2} }, 3, qnet.ErrShape},
		{"const data length", func(ls []qnet.Layer) { ls[3].Weights = ls[3].Weights[:4] }, 3, qnet.ErrParams},
		{"const requantises", func(ls []qnet.Layer) { ls[3].OutQ = mulQ }, 3, qnet.ErrQuant},
		{"mul bad broadcast", func(ls []qnet.Layer) { ls[4].From = []int{2, 0} }, 4, qnet.ErrShape},
		{"mul needs two", func(ls []qnet.Layer) { ls[4].From = []int{2} }, 4, qnet.ErrShape},
		{"mul second quant", func(ls []qnet.Layer) { ls[4].In2Q = mulQ }, 4, qnet.ErrQuant},
		{"mul act range", func(ls []qnet.Layer) { ls[4].ActMin, ls[4].ActMax = 3, 2 }, 4, qnet.ErrActivation},
		{"mul multiplier", func(ls []qnet.Layer) { ls[4].RQ = qi8.Requant{Mult: 5, Shift: 0} }, 4, qnet.ErrParams},
		{"mul zero multiplier ok", func(ls []qnet.Layer) { ls[4].RQ = qi8.Requant{} }, -1, nil},
		{"requantize rounding", func(ls []qnet.Layer) { ls[5].Round = 9 }, 5, qnet.ErrParams},
		{"concat input quant", func(ls []qnet.Layer) { ls[7].From = []int{6, 2} }, 7, qnet.ErrQuant},
		{"concat wrong axis", func(ls []qnet.Layer) { ls[8].Axis = 0 }, 8, qnet.ErrShape},
		{"concat sum", func(ls []qnet.Layer) { ls[8].Out.W = 17 }, 8, qnet.ErrShape},
		{"concat bad axis", func(ls []qnet.Layer) { ls[8].Axis = 3 }, 8, qnet.ErrShape},
		{"concat empty from", func(ls []qnet.Layer) { ls[8].From = []int{} }, 8, qnet.ErrShape},
		{"pad relabels quant", func(ls []qnet.Layer) { ls[0].OutQ, ls[1].InQ = mulQ, mulQ }, -1, nil},
		{"pad negative", func(ls []qnet.Layer) { ls[0].Pads[2] = -1 }, 0, qnet.ErrShape},
		{"resize relabels quant", func(ls []qnet.Layer) { ls[1].OutQ, ls[2].InQ, ls[5].InQ = mulQ, mulQ, mulQ }, -1, nil},
		{"mean shape", func(ls []qnet.Layer) { ls[9].Out = qi8.Shape{H: 1, W: 2, C: 5} }, 9, qnet.ErrShape},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			g := opsLayers(t, qi8.RoundTFLite)
			c.mod(g.Layers)
			idx, err := g.Check()
			if err != c.error || idx != c.idx {
				t.Fatalf("Check = %d, %v; want %d, %v", idx, err, c.idx, c.error)
			}
		})
	}
	// A Const cannot be layer 0 (layer 0 defines the graph input).
	g := qnet.Graph{Layers: []qnet.Layer{{Kind: qnet.Const, In: qi8.Shape{H: 1, W: 1, C: 1}, Out: qi8.Shape{H: 1, W: 1, C: 1}, InQ: sQ, OutQ: sQ, Weights: []int8{1}}}}
	if _, err := g.Check(); err != qnet.ErrShape {
		t.Fatalf("const at layer 0: %v", err)
	}
}

func u8s(v []int8) []byte {
	b := make([]byte, len(v))
	for i, x := range v {
		b[i] = byte(x)
	}
	return b
}

// yoloLayers: ResizeBilinear (half pixel) of the input, LeakyReLU, then
// ResizeBilinear (align corners) — the kinds YOLOX and tiny YOLOv2 add.
func yoloLayers(t testing.TB, round qi8.Rounding) qnet.Graph {
	t.Helper()
	lq := qnet.Quant{Scale: 0.009, ZP: -70}
	id, alpha, err := qi8.QuantizeLeakyReLU(inQ.Scale, lq.Scale, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	g := qnet.Graph{Layers: []qnet.Layer{
		{Kind: qnet.ResizeBilinear, In: qi8.Shape{H: 5, W: 7, C: 6}, Out: qi8.Shape{H: 3, W: 4, C: 6}, InQ: inQ, OutQ: inQ, HalfPixel: true},
		{Kind: qnet.LeakyReLU, In: qi8.Shape{H: 3, W: 4, C: 6}, Out: qi8.Shape{H: 3, W: 4, C: 6}, InQ: inQ, OutQ: lq, Round: round, RQ: id, RQAlpha: alpha},
		{Kind: qnet.ResizeBilinear, In: qi8.Shape{H: 3, W: 4, C: 6}, Out: qi8.Shape{H: 8, W: 9, C: 6}, InQ: lq, OutQ: lq, AlignCorners: true},
	}}
	if layer, err := g.Check(); err != nil {
		t.Fatalf("layer %d: %v", layer, err)
	}
	return g
}

func TestRunnerYOLOKinds(t *testing.T) {
	for _, round := range []qi8.Rounding{qi8.RoundExact, qi8.RoundTFLite} {
		g := yoloLayers(t, round)
		r, err := qnet.NewRunner(&g, make([]int8, 4096))
		if err != nil {
			t.Fatal(err)
		}
		for seed := uint64(1); seed <= 10; seed++ {
			in := randInput(g.Layers[0].In.Len(), seed)
			want := reference(t, &g, in)
			got, err := r.Run(in)
			if err != nil {
				t.Fatal(err)
			}
			if string(u8s(got)) != string(u8s(want)) {
				t.Fatalf("round %d seed %d: runner differs from RunLayer", round, seed)
			}
		}
	}
	cases := []struct {
		name string
		mod  func(ls []qnet.Layer)
		want error
	}{
		{"leaky alpha multiplier", func(ls []qnet.Layer) { ls[1].RQAlpha = qi8.Requant{Mult: 3} }, qnet.ErrParams},
		{"leaky identity multiplier", func(ls []qnet.Layer) { ls[1].RQ = qi8.Requant{Mult: 3} }, qnet.ErrParams},
		{"leaky rounding", func(ls []qnet.Layer) { ls[1].Round = 5 }, qnet.ErrParams},
		{"bilinear both flags", func(ls []qnet.Layer) { ls[0].AlignCorners = true }, qnet.ErrShape},
		{"bilinear size", func(ls []qnet.Layer) { ls[2].Out.C = 7 }, qnet.ErrShape},
	}
	for _, c := range cases {
		g := yoloLayers(t, qi8.RoundTFLite)
		c.mod(g.Layers)
		if _, err := g.Check(); err != c.want {
			t.Errorf("%s: %v, want %v", c.name, err, c.want)
		}
	}
}

// Transpose (the YOLOX head tail): runner equals RunLayer; the output shape
// follows Perm, quantisation must pass through, and Perm must be a
// permutation.
func TestRunnerTranspose(t *testing.T) {
	mk := func() qnet.Graph {
		return qnet.Graph{Layers: []qnet.Layer{
			{Kind: qnet.Transpose, In: qi8.Shape{H: 4, W: 5, C: 6}, Out: qi8.Shape{H: 6, W: 4, C: 5}, InQ: inQ, OutQ: inQ, Perm: [3]int{2, 0, 1}},
			{Kind: qnet.Reshape, In: qi8.Shape{H: 6, W: 4, C: 5}, Out: qi8.Shape{H: 1, W: 6, C: 20}, InQ: inQ, OutQ: inQ},
			{Kind: qnet.Transpose, In: qi8.Shape{H: 1, W: 6, C: 20}, Out: qi8.Shape{H: 1, W: 20, C: 6}, InQ: inQ, OutQ: inQ, Perm: [3]int{0, 2, 1}},
		}}
	}
	g := mk()
	if layer, err := g.Check(); err != nil {
		t.Fatalf("layer %d: %v", layer, err)
	}
	r, err := qnet.NewRunner(&g, make([]int8, 4096))
	if err != nil {
		t.Fatal(err)
	}
	for seed := uint64(1); seed <= 5; seed++ {
		in := randInput(g.Layers[0].In.Len(), seed)
		got, err := r.Run(in)
		if err != nil {
			t.Fatal(err)
		}
		if string(u8s(got)) != string(u8s(reference(t, &g, in))) {
			t.Fatalf("seed %d: runner differs from RunLayer", seed)
		}
		// Perm [2,0,1] then flatten and swap back = the input's H×W×C
		// order with C outermost: out[w][c] = in at row-major (h, w', c) …
		// checked directly for one value: in[h=1][w=2][c=3] lands at
		// transpose-1 index ((3·4+1)·5+2) = 67, i.e. reshape (0, 3, 7),
		// then transpose-2 index (7·6+3) = 45.
		if got[45] != in[(1*5+2)*6+3] {
			t.Fatalf("seed %d: value moved to the wrong place", seed)
		}
	}
	cases := []struct {
		name string
		mod  func(ls []qnet.Layer)
		want error
	}{
		{"perm not a permutation", func(ls []qnet.Layer) { ls[0].Perm = [3]int{2, 2, 1} }, qnet.ErrShape},
		{"out shape not permuted", func(ls []qnet.Layer) { ls[0].Out = qi8.Shape{H: 4, W: 6, C: 5} }, qnet.ErrShape},
		{"quantisation changed", func(ls []qnet.Layer) { ls[2].OutQ = qnet.Quant{Scale: 0.1, ZP: 0} }, qnet.ErrQuant},
	}
	for _, c := range cases {
		g := mk()
		c.mod(g.Layers)
		if _, err := g.Check(); err != c.want {
			t.Errorf("%s: %v, want %v", c.name, err, c.want)
		}
	}
}

// Mul with a per-pixel (H×W×1) second input: runner equals RunLayer, which
// equals Mul against the operand expanded over C; the flag and the
// producer's shape must agree.
func TestRunnerMulPerPixel(t *testing.T) {
	oq := qnet.Quant{Scale: 0.03, ZP: 5}
	rq, err := qi8.QuantizeMul(inQ.Scale, 1.0/256, oq.Scale)
	if err != nil {
		t.Fatal(err)
	}
	mk := func() qnet.Graph {
		return qnet.Graph{Layers: []qnet.Layer{
			{Kind: qnet.Logistic, In: qi8.Shape{H: 3, W: 4, C: 1}, Out: qi8.Shape{H: 3, W: 4, C: 1}, InQ: inQ, OutQ: qnet.LogisticOutQuant,
				LogisticParams: mustLogistic(t, inQ)},
			{Kind: qnet.Const, From: nil, In: qi8.Shape{H: 3, W: 4, C: 5}, Out: qi8.Shape{H: 3, W: 4, C: 5}, InQ: inQ, OutQ: inQ, Weights: randInput(60, 99)},
			{Kind: qnet.Mul, From: []int{1, 0}, In: qi8.Shape{H: 3, W: 4, C: 5}, Out: qi8.Shape{H: 3, W: 4, C: 5}, InQ: inQ, In2Q: qnet.LogisticOutQuant,
				OutQ: oq, ActMin: -128, ActMax: 127, Round: qi8.RoundTFLite, RQ: rq, PerPixel: true},
		}}
	}
	g := mk()
	if layer, err := g.Check(); err != nil {
		t.Fatalf("layer %d: %v", layer, err)
	}
	r, err := qnet.NewRunner(&g, make([]int8, 4096))
	if err != nil {
		t.Fatal(err)
	}
	in := randInput(12, 3)
	got, err := r.Run(in)
	if err != nil {
		t.Fatal(err)
	}
	if string(u8s(got)) != string(u8s(reference(t, &g, in))) {
		t.Fatal("runner differs from RunLayer")
	}
	sig := make([]int8, 12)
	if err := qnet.RunLayer(&g.Layers[0], sig, in); err != nil {
		t.Fatal(err)
	}
	exp := make([]int8, 60)
	for i := range exp {
		exp[i] = sig[i/5]
	}
	want := make([]int8, 60)
	p := g.Layers[2].MulParams()
	p.Repeat = 0
	if err := qi8.Mul(want, g.Layers[1].Weights, exp, p); err != nil {
		t.Fatal(err)
	}
	if string(u8s(got)) != string(u8s(want)) {
		t.Fatal("per-pixel Mul differs from Mul against the expanded operand")
	}
	g = mk()
	g.Layers[2].PerPixel = false
	if _, err := g.Check(); err != qnet.ErrShape {
		t.Errorf("H×W×1 operand without PerPixel: %v, want ErrShape", err)
	}
}

func mustLogistic(t testing.TB, q qnet.Quant) *qi8.LogisticParams {
	t.Helper()
	lp, err := qi8.QuantizeLogistic(q.Scale, q.ZP)
	if err != nil {
		t.Fatal(err)
	}
	return &lp
}

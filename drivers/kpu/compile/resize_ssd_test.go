package compile

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// ssdHeads builds an SSD-shaped graph at h×w: a stride-1 conv and a
// stride-2 conv (the two feature maps), on each a box head (A·4 channels,
// reshaped to [cells·A, 1, 4]) and a score head (A·3 channels, reshaped to
// [1, cells·A, 3]); the boxes concatenated along H and reshaped to
// [1, boxes, 4], the scores concatenated along W (the second input through a
// Rescale map) and passed through a Logistic. Outputs: boxes, scores.
func ssdHeads(t *testing.T, h, w int) *qnet.Graph {
	t.Helper()
	const c, a = 4, 2
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
	s0 := qi8.Shape{H: h, W: w, C: c}
	f1 := conv(nil, s0, c, 3, 1, q)     // 0
	f2 := conv(nil, f1.Out, c, 3, 2, q) // 1
	n1, n2 := f1.Out.H*f1.Out.W*a, f2.Out.H*f2.Out.W*a
	b1 := conv([]int{0}, f1.Out, a*4, 1, 1, q)  // 2
	b2 := conv([]int{1}, f2.Out, a*4, 1, 1, q)  // 4
	c1 := conv([]int{0}, f1.Out, a*3, 1, 1, q)  // 6
	c2 := conv([]int{1}, f2.Out, a*3, 1, 1, q2) // 8
	m, err := qi8.QuantizeConcatRescale(q2.Scale, q2.ZP, q.Scale, q.ZP)
	if err != nil {
		t.Fatal(err)
	}
	lq := qnet.LogisticOutQuant
	lp, err := qi8.QuantizeLogistic(q.Scale, q.ZP)
	if err != nil {
		t.Fatal(err)
	}
	g := &qnet.Graph{Layers: []qnet.Layer{
		f1, f2,
		b1, reshape(b1.Out, qi8.Shape{H: n1, W: 1, C: 4}, q), // 2, 3
		b2, reshape(b2.Out, qi8.Shape{H: n2, W: 1, C: 4}, q), // 4, 5
		c1, reshape(c1.Out, qi8.Shape{H: 1, W: n1, C: 3}, q), // 6, 7
		c2, reshape(c2.Out, qi8.Shape{H: 1, W: n2, C: 3}, q2), // 8, 9
		{Kind: qnet.Concat, From: []int{3, 5}, Axis: 0, In: qi8.Shape{H: n1, W: 1, C: 4}, Out: qi8.Shape{H: n1 + n2, W: 1, C: 4}, InQ: q, OutQ: q}, // 10
		reshape(qi8.Shape{H: n1 + n2, W: 1, C: 4}, qi8.Shape{H: 1, W: n1 + n2, C: 4}, q),                                                           // 11
		{Kind: qnet.Concat, From: []int{7, 9}, Axis: 1, In: qi8.Shape{H: 1, W: n1, C: 3}, Out: qi8.Shape{H: 1, W: n1 + n2, C: 3}, InQ: q, OutQ: q,
			Rescale: []*[256]int8{nil, m}}, // 12
		{Kind: qnet.Logistic, In: qi8.Shape{H: 1, W: n1 + n2, C: 3}, Out: qi8.Shape{H: 1, W: n1 + n2, C: 3}, InQ: q, OutQ: lq, LogisticParams: &lp}, // 13
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
		t.Fatalf("ssdHeads %d×%d: %v", h, w, err)
	}
	return g
}

// TestResizeSSDHeads: Resize re-derives the head reshapes and the
// concatenations of an SSD-shaped graph. The graph built at 8×8 and resized
// to 16×24 has the shapes of the one built at 16×24, shares its Rescale
// maps, and gives the same outputs on the same input.
func TestResizeSSDHeads(t *testing.T) {
	small, big := ssdHeads(t, 8, 8), ssdHeads(t, 16, 24)
	r, err := Resize(small, 16, 24)
	if err != nil {
		t.Fatal(err)
	}
	for i := range big.Layers {
		if r.Layers[i].In != big.Layers[i].In || r.Layers[i].Out != big.Layers[i].Out {
			t.Errorf("layer %d (%v): %v → %v, want %v → %v", i, r.Layers[i].Kind, r.Layers[i].In, r.Layers[i].Out, big.Layers[i].In, big.Layers[i].Out)
		}
	}
	if r.Layers[12].Rescale[1] != small.Layers[12].Rescale[1] || r.Layers[12].Rescale[0] != nil {
		t.Error("the Concat's Rescale maps did not carry over")
	}
	if small.Layers[3].Out.H != 8*8*2 {
		t.Error("Resize changed its input graph")
	}
	in := randomInput(16*24*4, 5)
	ra, rb := newRunner(t, r), newRunner(t, big)
	if _, err := ra.Run(in); err != nil {
		t.Fatal(err)
	}
	if _, err := rb.Run(in); err != nil {
		t.Fatal(err)
	}
	for k := range 2 {
		a, b := ra.Output(in, k), rb.Output(in, k)
		if len(a) != len(b) {
			t.Fatalf("output %d: %d values, want %d", k, len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("output %d value %d: %d, want %d", k, i, a[i], b[i])
			}
		}
	}
}

// TestResizeReshapeRefused: a Reshape that does not keep its channels with
// one of H, W at 1 has no size-independent meaning.
func TestResizeReshapeRefused(t *testing.T) {
	q := qnet.Quant{Scale: 0.05, ZP: -3}
	for _, out := range []qi8.Shape{{H: 4, W: 16, C: 4}, {H: 1, W: 1, C: 256}} {
		g := &qnet.Graph{Layers: []qnet.Layer{
			{Kind: qnet.ReLU, In: qi8.Shape{H: 8, W: 8, C: 4}, Out: qi8.Shape{H: 8, W: 8, C: 4}, InQ: q, OutQ: q},
			{Kind: qnet.Reshape, In: qi8.Shape{H: 8, W: 8, C: 4}, Out: out, InQ: q, OutQ: q},
		}}
		if err := g.Validate(); err != nil {
			t.Fatal(err)
		}
		if _, err := Resize(g, 16, 16); !errors.Is(err, ErrUnsupported) {
			t.Errorf("reshape to %v: %v, want ErrUnsupported", out, err)
		}
		// At the same size nothing needs deriving.
		if _, err := Resize(g, 8, 8); err != nil {
			t.Errorf("reshape to %v at the original size: %v", out, err)
		}
	}
}

// TestSmallMapsStayOnCPU: the planner gives the KPU no layer whose input or
// output map is smaller than 4×4 (not checked on the K210; the maker's
// compiler never emits one). On ssdHeads at 8×8 the stride-2 map is 4×4, so
// everything lowers; at 4×4 it is 2×2, and the layers reading it stay on
// the CPU, as does the stride-2 conv producing it. The graph's outputs stay
// within 1 LSB of qnet's CPU run before the logistic.
func TestSmallMapsStayOnCPU(t *testing.T) {
	for _, c := range []struct {
		size int
		cpu  []int // conv layers left to the CPU
	}{{8, nil}, {4, []int{1, 4, 8}}} {
		g := ssdHeads(t, c.size, c.size)
		be, err := Compile(g, kpu.NewEmulated)
		if err != nil {
			t.Fatal(err)
		}
		onKPU := map[int]bool{}
		for _, s := range be.Segments() {
			for i := s.First; i < s.First+s.Count; i++ {
				onKPU[i] = true
				l := &g.Layers[i]
				if min(l.In.H, l.In.W, l.Out.H, l.Out.W) < MinSide {
					t.Errorf("%d×%d: layer %d (%v → %v) is on the KPU", c.size, c.size, i, l.In, l.Out)
				}
			}
		}
		var cpu []int
		for i, l := range g.Layers {
			if l.Kind == qnet.Conv2D && !onKPU[i] {
				cpu = append(cpu, i)
			}
		}
		if !slices.Equal(cpu, c.cpu) {
			t.Errorf("%d×%d: conv layers on the CPU %v, want %v", c.size, c.size, cpu, c.cpu)
		}
		in := randomInput(g.Layers[0].In.Len(), 11)
		rk, rc := newRunner(t, g, be), newRunner(t, g)
		if _, err := rk.Run(in); err != nil {
			t.Fatal(err)
		}
		if _, err := rc.Run(in); err != nil {
			t.Fatal(err)
		}
		a, b := rk.Output(in, 0), rc.Output(in, 0) // the boxes: no logistic after them
		for i := range a {
			if d := int(a[i]) - int(b[i]); d > 2 || d < -2 {
				t.Fatalf("%d×%d: box value %d: KPU %d, CPU %d", c.size, c.size, i, a[i], b[i])
			}
		}
	}
}

// TestKmodelRefusesSmallMap: Kmodel names the limit.
func TestKmodelRefusesSmallMap(t *testing.T) {
	g, err := Cut(ssdHeads(t, 4, 4), 1) // 4×4 → 2×2
	if err != nil {
		t.Fatal(err)
	}
	g.Outputs = nil
	if _, _, err := Kmodel(g); !errors.Is(err, ErrUnsupported) || !strings.Contains(err.Error(), "smaller than 4×4") {
		t.Errorf("Kmodel on a graph ending at 2×2: %v", err)
	}
}

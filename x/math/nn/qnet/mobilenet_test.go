package qnet_test

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// mobileNet is a bound MobileNetV1-0.25 with synthetic params and its arena.
type mobileNet struct {
	g     qnet.Graph
	arena []int8
	in    []int8
}

func newMobileNet(t testing.TB, h, w, classes int) *mobileNet {
	t.Helper()
	g, err := qnet.MobileNetV1(0.25, h, w, classes, make([]qnet.Layer, qnet.MobileNetV1Layers(classes)))
	if err != nil {
		t.Fatal(err)
	}
	nw, nb, nr := qnet.ParamCounts(&g)
	if err := qnet.BindParams(&g, make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)); err != nil {
		t.Fatal(err)
	}
	if err := qnet.SyntheticParams(&g, 42); err != nil {
		t.Fatal(err)
	}
	return &mobileNet{g: g, arena: make([]int8, g.ArenaSize()), in: randInput(h*w*3, uint64(h))}
}

func TestMobileNetV1Shapes(t *testing.T) {
	type sh = [3]int // H, W, C
	for _, c := range []struct {
		h, w int
		outs []sh // layer outputs, conv0 then (dw, pw) pairs, pool
	}{
		{96, 96, []sh{{48, 48, 8}, {48, 48, 8}, {48, 48, 16}, {24, 24, 16}, {24, 24, 32}, {24, 24, 32}, {24, 24, 32},
			{12, 12, 32}, {12, 12, 64}, {12, 12, 64}, {12, 12, 64}, {6, 6, 64}, {6, 6, 128},
			{6, 6, 128}, {6, 6, 128}, {6, 6, 128}, {6, 6, 128}, {6, 6, 128}, {6, 6, 128}, {6, 6, 128}, {6, 6, 128}, {6, 6, 128}, {6, 6, 128},
			{3, 3, 128}, {3, 3, 256}, {3, 3, 256}, {3, 3, 256}, {1, 1, 256}}},
		{128, 96, []sh{{64, 48, 8}, {64, 48, 8}, {64, 48, 16}, {32, 24, 16}, {32, 24, 32}, {32, 24, 32}, {32, 24, 32},
			{16, 12, 32}, {16, 12, 64}, {16, 12, 64}, {16, 12, 64}, {8, 6, 64}, {8, 6, 128},
			{8, 6, 128}, {8, 6, 128}, {8, 6, 128}, {8, 6, 128}, {8, 6, 128}, {8, 6, 128}, {8, 6, 128}, {8, 6, 128}, {8, 6, 128}, {8, 6, 128},
			{4, 3, 128}, {4, 3, 256}, {4, 3, 256}, {4, 3, 256}, {1, 1, 256}}},
	} {
		g, err := qnet.MobileNetV1(0.25, c.h, c.w, 0, make([]qnet.Layer, 40))
		if err != nil {
			t.Fatal(err)
		}
		if len(g.Layers) != 28 || len(c.outs) != 28 {
			t.Fatalf("layers %d", len(g.Layers))
		}
		for i, l := range g.Layers {
			if want := (qi8.Shape{H: c.outs[i][0], W: c.outs[i][1], C: c.outs[i][2]}); l.Out != want {
				t.Errorf("%dx%d layer %d (%v) out %v, want %v", c.h, c.w, i, l.Kind, l.Out, want)
			}
		}
		pool := g.Layers[27]
		if pool.Kind != qnet.AvgPool2D || pool.KH != c.h/32 || pool.KW != c.w/32 || pool.Padding != qi8.PaddingValid {
			t.Errorf("pool %+v", pool)
		}
		if l := g.Layers[1]; l.Kind != qnet.DepthwiseConv2D || l.KH != 3 || l.StrideH != 1 || l.ActMin != -128 || l.ActMax != 127 {
			t.Errorf("dw %+v", l)
		}
		if l := g.Layers[0]; l.StrideH != 2 || l.InQ != qnet.MobileNetInputQuant || l.OutQ != qnet.MobileNetActQuant {
			t.Errorf("stem %+v", l)
		}
	}

	g, err := qnet.MobileNetV1(0.25, 96, 96, 1000, make([]qnet.Layer, 29))
	if err != nil {
		t.Fatal(err)
	}
	fc := g.Layers[28]
	if fc.Kind != qnet.FullyConnected || fc.Out != (qi8.Shape{H: 1, W: 1, C: 1000}) || fc.OutQ != qnet.MobileNetLogitQuant || fc.ActMin != -128 || fc.ActMax != 127 {
		t.Fatalf("fc %+v", fc)
	}
	nw, nb, nr := qnet.ParamCounts(&g)
	// stem 216, depthwise 9·1240, pointwise 196224, fc 256000.
	if nw != 216+11160+196224+256000 || nb != 2736+1000 || nr != nb {
		t.Fatalf("ParamCounts %d %d %d", nw, nb, nr)
	}
	if err := g.Validate(); err != qnet.ErrParams {
		t.Fatalf("unbound graph validates: %v", err)
	}

	if _, err := qnet.MobileNetV1(0.25, 96, 96, 1000, make([]qnet.Layer, 28)); err != qnet.ErrCapacity {
		t.Fatalf("capacity: %v", err)
	}
	if _, err := qnet.MobileNetV1(0, 96, 96, 0, make([]qnet.Layer, 28)); err != qnet.ErrShape {
		t.Fatalf("alpha 0: %v", err)
	}
	if _, err := qnet.MobileNetV1(0.25, 0, 96, 0, make([]qnet.Layer, 28)); err != qnet.ErrShape {
		t.Fatalf("height 0: %v", err)
	}
}

func TestMobileNetV1MatchesKernels(t *testing.T) {
	for _, size := range []int{96, 128} {
		for _, classes := range []int{0, 1000} {
			t.Run(fmt.Sprintf("%d_c%d", size, classes), func(t *testing.T) {
				m := newMobileNet(t, size, size, classes)
				want := runByHand(t, &m.g, m.in)
				var r qnet.Runner
				if err := r.Init(&m.g, m.arena, qnet.Generic{}); err != nil {
					t.Fatal(err)
				}
				got, err := r.Run(m.in)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(i8b(got), i8b(want)) {
					t.Fatal("runner output differs from layer-by-layer qi8")
				}
				// Fusing every dw+pw pair gives the same bytes.
				pair := &pairBackend{scratch: make([]int8, m.g.ArenaSize())}
				var rf qnet.Runner
				if err := rf.Init(&m.g, make([]int8, 2*m.g.ArenaSize()), pair, qnet.Generic{}); err != nil {
					t.Fatal(err)
				}
				if n := len(rf.Plan()); n != len(m.g.Layers)-13 {
					t.Fatalf("fused plan has %d segments", n)
				}
				got, err = rf.Run(m.in)
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(i8b(got), i8b(want)) || pair.runs != 13 {
					t.Fatalf("fused output differs (pair runs %d)", pair.runs)
				}
			})
		}
	}
}

// TestSyntheticActivations checks that the synthetic parameters keep every
// layer's activations spread out: not stuck at the ReLU floor, rarely at the
// ReLU6 ceiling, and logits mostly unsaturated.
func TestSyntheticActivations(t *testing.T) {
	m := newMobileNet(t, 128, 128, 1000)
	src := m.in
	for i := range m.g.Layers {
		l := &m.g.Layers[i]
		dst := make([]int8, l.Out.Len())
		if err := qnet.RunLayer(l, dst, src); err != nil {
			t.Fatal(err)
		}
		var lo, hi int
		seen := map[int8]bool{}
		for _, v := range dst {
			seen[v] = true
			if v == l.ActMin {
				lo++
			}
			if v == l.ActMax {
				hi++
			}
		}
		n := float64(len(dst))
		fl, fh := float64(lo)/n, float64(hi)/n
		t.Logf("layer %2d %-9v floor %.2f ceil %.3f distinct %d", i, l.Kind, fl, fh, len(seen))
		switch l.Kind {
		case qnet.Conv2D, qnet.DepthwiseConv2D:
			if fl < 0.3 || fl > 0.7 || fh > 0.01 || len(seen) < 48 {
				t.Errorf("layer %d degenerate: floor %.2f ceil %.3f distinct %d", i, fl, fh, len(seen))
			}
		case qnet.FullyConnected:
			if fl+fh > 0.01 || len(seen) < 64 {
				t.Errorf("logits degenerate: sat %.3f distinct %d", fl+fh, len(seen))
			}
		}
		src = dst
	}
}

func TestNoAllocs(t *testing.T) {
	layers := make([]qnet.Layer, qnet.MobileNetV1Layers(1000))
	var g qnet.Graph
	var err error
	if n := testing.AllocsPerRun(10, func() { g, err = qnet.MobileNetV1(0.25, 96, 96, 1000, layers) }); n != 0 || err != nil {
		t.Fatalf("MobileNetV1: %v allocs, %v", n, err)
	}
	nw, nb, nr := qnet.ParamCounts(&g)
	w, b, rq := make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
	if n := testing.AllocsPerRun(10, func() { err = qnet.BindParams(&g, w, b, rq) }); n != 0 || err != nil {
		t.Fatalf("BindParams: %v allocs, %v", n, err)
	}
	if n := testing.AllocsPerRun(2, func() { err = qnet.SyntheticParams(&g, 1) }); n != 0 || err != nil {
		t.Fatalf("SyntheticParams: %v allocs, %v", n, err)
	}
	arena := make([]int8, g.ArenaSize())
	var r qnet.Runner
	if n := testing.AllocsPerRun(10, func() { err = r.Init(&g, arena, qnet.Generic{}) }); n != 0 || err != nil {
		t.Fatalf("Init: %v allocs, %v", n, err)
	}
	pair := &pairBackend{scratch: make([]int8, g.ArenaSize())}
	var rf qnet.Runner
	if err := rf.Init(&g, make([]int8, 2*g.ArenaSize()), pair, qnet.Generic{}); err != nil {
		t.Fatal(err)
	}
	in := randInput(96*96*3, 1)
	for _, rr := range []*qnet.Runner{&r, &rf} {
		if n := testing.AllocsPerRun(3, func() { _, err = rr.Run(in) }); n != 0 || err != nil {
			t.Fatalf("Run: %v allocs, %v", n, err)
		}
	}
}

// TestBudgetTable logs the per-layer budget (run with -v) and pins totals.
// 224×224 with 1000 classes reproduces the paper's 41 M mult-adds and 0.47 M
// parameters for MobileNetV1-0.25.
func TestBudgetTable(t *testing.T) {
	for _, c := range []struct {
		size, classes   int
		weights, params int
		macs            int64
		arena           int
	}{
		{96, 0, 207600, 32832, 7489152, 55296},
		{96, 1000, 463600, 44832, 7745152, 55296},
		{128, 0, 207600, 32832, 13314048, 98304},
		{128, 1000, 463600, 44832, 13570048, 98304},
		{224, 1000, 463600, 44832, 41030272, 301056},
	} {
		g, err := qnet.MobileNetV1(0.25, c.size, c.size, c.classes, make([]qnet.Layer, 29))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("MobileNetV1-0.25 %dx%d classes=%d", c.size, c.size, c.classes)
		t.Logf("%3s %-9s %-12s %-12s %7s %7s %7s %7s %10s", "#", "kind", "in", "out", "inB", "outB", "wB", "bias+rqB", "MACs")
		for i := range g.Layers {
			l, b := &g.Layers[i], g.Budget(i)
			t.Logf("%3d %-9v %-12s %-12s %7d %7d %7d %7d %10d", i, l.Kind,
				fmt.Sprintf("%dx%dx%d", l.In.H, l.In.W, l.In.C), fmt.Sprintf("%dx%dx%d", l.Out.H, l.Out.W, l.Out.C),
				b.In, b.Out, b.Weights, b.Params, b.MACs)
		}
		t.Logf("total: weights %d B, bias+requant %d B, MACs %d, arena %d B", g.WeightBytes(), g.ParamBytes(), g.MACs(), g.ArenaSize())
		if g.WeightBytes() != c.weights || g.ParamBytes() != c.params || g.MACs() != c.macs || g.ArenaSize() != c.arena {
			t.Errorf("%d/%d totals: want weights %d, params %d, MACs %d, arena %d", c.size, c.classes, c.weights, c.params, c.macs, c.arena)
		}
	}
}

func benchMobileNet(b *testing.B, size, classes int) {
	m := newMobileNet(b, size, size, classes)
	var r qnet.Runner
	if err := r.Init(&m.g, m.arena, qnet.Generic{}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if _, err := r.Run(m.in); err != nil {
			b.Fatal(err)
		}
	}
	b.ReportMetric(b.Elapsed().Seconds()*1e3/float64(b.N), "ms/frame")
	b.ReportMetric(float64(m.g.MACs())*float64(b.N)/b.Elapsed().Seconds()/1e6, "MMAC/s")
}

func BenchmarkMobileNetV1(b *testing.B) {
	for _, size := range []int{96, 128} {
		for _, classes := range []int{0, 1000} {
			b.Run(fmt.Sprintf("%dx%d_c%d", size, size, classes), func(b *testing.B) { benchMobileNet(b, size, classes) })
		}
	}
}

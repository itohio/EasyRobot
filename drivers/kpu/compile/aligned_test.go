package compile

import (
	"math/rand/v2"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// strideChain builds, at n×n: a 3×3 stride-2 conv, a 3×3 stride-2 depthwise
// conv, a 1×1 conv, a 3×3 stride-2 conv and a 3×3 stride-1 conv (all TF
// SAME), the last reading the layer before it by an explicit From. Outputs:
// the 1×1 conv and the last layer.
func strideChain(t *testing.T, n int) *qnet.Graph {
	t.Helper()
	q := qnet.Quant{Scale: 0.05, ZP: -7}
	in := qi8.Shape{H: n, W: n, C: 3}
	var ls []qnet.Layer
	add := func(kind qnet.Kind, outC, k, stride int) {
		l := qnet.Layer{Kind: kind, In: in, Out: qi8.Shape{C: outC}, InQ: q, OutQ: q,
			KH: k, KW: k, StrideH: stride, StrideW: stride, Padding: qi8.PaddingSame, ActMin: -128, ActMax: 127}
		out, err := l.OutShape()
		if err != nil {
			t.Fatal(err)
		}
		l.Out = out
		nw, nb, nr := l.ParamCounts()
		l.Weights, l.Bias, l.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
		ls = append(ls, l)
		in = out
	}
	add(qnet.Conv2D, 4, 3, 2)
	add(qnet.DepthwiseConv2D, 4, 3, 2)
	add(qnet.Conv2D, 6, 1, 1)
	add(qnet.Conv2D, 6, 3, 2)
	add(qnet.Conv2D, 5, 3, 1)
	ls[4].From = []int{3}
	g := &qnet.Graph{Layers: ls, Outputs: []int{2, 4}}
	if err := qnet.SyntheticParams(g, 23); err != nil {
		t.Fatal(err)
	}
	if err := g.Validate(); err != nil {
		t.Fatal(err)
	}
	return g
}

// lastPixel returns the last input row (or column) that cell j of layer
// last's output reads in g, a chain of SAME convolutions.
func lastPixel(g *qnet.Graph, last, j int) int {
	for i := last; i >= 0; i-- {
		l := &g.Layers[i]
		before := max((l.Out.H-1)*l.StrideH+l.KH-l.In.H, 0) / 2
		j = j*l.StrideH - before + l.KH - 1
	}
	return j
}

// TestResizeAligned: the aligned graph run on the top-left h×w pixels of an
// image gives, in every cell that reads nothing beyond those pixels, the
// value the original graph gives on the whole image. Resize alone moves the
// sampling grid when a stride-2 input changes parity, so its cells differ.
func TestResizeAligned(t *testing.T) {
	for _, c := range []struct {
		n, h, w int
		kinds   string // the aligned graph's layers: c conv, d depthwise, p Pad
		pads    [][6]int
		outputs []int
	}{
		// 42 → 21 → 11 → 6; 32 → 16 → 8 → 4: both odd inputs became even.
		{42, 32, 32, "cpdcpcc", [][6]int{{1, 1, 1, 1, 0, 0}, {1, 1, 1, 1, 0, 0}}, []int{3, 6}},
		// 44 → 22 → 11 → 6; 36 → 18 → 9 → 5: the same parities, nothing to do.
		{44, 36, 36, "cdccc", nil, []int{2, 4}},
		// 26 → 13: an even input became odd (two rows after, none before).
		{44, 26, 26, "cpdccc", [][6]int{{0, 2, 0, 2, 0, 0}}, []int{3, 5}},
		// 26 × 36: only the rows change parity.
		{44, 26, 36, "cpdccc", [][6]int{{0, 2, 0, 1, 0, 0}}, []int{3, 5}},
	} {
		g := strideChain(t, c.n)
		a, err := ResizeAligned(g, c.h, c.w)
		if err != nil {
			t.Fatalf("%d → %d×%d: %v", c.n, c.h, c.w, err)
		}
		var kinds string
		var pads [][6]int
		for _, l := range a.Layers {
			switch l.Kind {
			case qnet.Pad:
				kinds += "p"
				pads = append(pads, l.Pads)
			case qnet.DepthwiseConv2D:
				kinds += "d"
			default:
				kinds += "c"
			}
		}
		if kinds != c.kinds || len(pads) != len(c.pads) || len(a.Outputs) != 2 || a.Outputs[0] != c.outputs[0] || a.Outputs[1] != c.outputs[1] {
			t.Fatalf("%d → %d×%d: layers %s pads %v outputs %v, want %s %v %v", c.n, c.h, c.w, kinds, pads, a.Outputs, c.kinds, c.pads, c.outputs)
		}
		for i := range pads {
			if pads[i] != c.pads[i] {
				t.Errorf("%d → %d×%d: Pad %d is %v, want %v", c.n, c.h, c.w, i, pads[i], c.pads[i])
			}
		}
		if len(g.Layers) != 5 || g.Layers[4].From[0] != 3 {
			t.Fatal("ResizeAligned changed its input graph")
		}

		r := rand.New(rand.NewPCG(uint64(c.n*c.h+c.w), 3))
		full := make([]int8, c.n*c.n*3)
		for i := range full {
			full[i] = int8(r.IntN(256) - 128)
		}
		crop := make([]int8, 0, c.h*c.w*3)
		for y := range c.h {
			crop = append(crop, full[y*c.n*3:][:c.w*3]...)
		}
		plain, err := Resize(g, c.h, c.w)
		if err != nil {
			t.Fatal(err)
		}
		ro, ra, rp := newRunner(t, g), newRunner(t, a), newRunner(t, plain)
		for _, run := range []struct {
			r  *qnet.Runner
			in []int8
		}{{ro, full}, {ra, crop}, {rp, crop}} {
			if _, err := run.r.Run(run.in); err != nil {
				t.Fatal(err)
			}
		}
		for k, last := range g.Outputs {
			so, sa := g.Layers[last].Out, a.Layers[a.Outputs[k]].Out
			if sp := plain.Layers[last].Out; sa != sp {
				t.Fatalf("output %d: aligned shape %v, resized %v", k, sa, sp)
			}
			want, got, moved := ro.Output(full, k), ra.Output(crop, k), rp.Output(crop, k)
			cells, differ := 0, 0
			for y := 0; y < sa.H && lastPixel(g, last, y) < c.h; y++ {
				for x := 0; x < sa.W && lastPixel(g, last, x) < c.w; x++ {
					cells++
					for ch := range sa.C {
						w := want[(y*so.W+x)*so.C+ch]
						if v := got[(y*sa.W+x)*sa.C+ch]; v != w {
							t.Fatalf("%d → %d×%d output %d cell (%d,%d,%d): %d, the original graph gives %d", c.n, c.h, c.w, k, y, x, ch, v, w)
						}
						if moved[(y*sa.W+x)*sa.C+ch] != w {
							differ++
						}
					}
				}
			}
			if cells == 0 {
				t.Fatalf("%d → %d×%d output %d: no cell inside the crop", c.n, c.h, c.w, k)
			}
			if len(c.pads) > 0 && k == 1 && differ == 0 {
				t.Errorf("%d → %d×%d: Resize alone already matches the original grid; the case tests nothing", c.n, c.h, c.w)
			}
		}
	}
}

// TestResizeAlignedSSDHeads: the heads' reshapes and concatenations follow
// an inserted Pad layer. The stride-2 conv of ssdHeads reads an even map at
// 10×10 and at 6×6 (no Pad), an odd one at 7×7 (two rows after).
func TestResizeAlignedSSDHeads(t *testing.T) {
	g := ssdHeads(t, 10, 10)
	a, err := ResizeAligned(g, 6, 6)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Layers) != len(g.Layers) {
		t.Fatalf("6×6: %d layers, want %d", len(a.Layers), len(g.Layers))
	}
	a, err = ResizeAligned(g, 7, 7)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Layers) != len(g.Layers)+1 || a.Layers[1].Kind != qnet.Pad || a.Layers[1].Pads != [6]int{0, 2, 0, 2, 0, 0} {
		t.Fatalf("7×7: layers %d, layer 1 %v %v", len(a.Layers), a.Layers[1].Kind, a.Layers[1].Pads)
	}
	// Boxes: 7·7·2 from the first map, 4·4·2 from the second.
	if out := a.Layers[a.Outputs[0]].Out; out != (qi8.Shape{H: 1, W: (49 + 16) * 2, C: 4}) {
		t.Errorf("7×7: boxes %v", out)
	}
	in := randomInput(7*7*4, 9)
	if _, err := newRunner(t, a).Run(in); err != nil {
		t.Fatal(err)
	}
}

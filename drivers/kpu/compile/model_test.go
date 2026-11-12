package compile

import (
	"errors"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// TestKmodelRuns: Kmodel compiles the SAME MobileNet into one kmodel; run
// by drivers/kpu (emulated) on the CHW input, its output equals qnet's CPU
// runner.
func TestKmodelRuns(t *testing.T) {
	g := mobileNetSAME(t, 0.25, 64)
	data, info, err := Kmodel(g)
	if err != nil {
		t.Fatal(err)
	}
	if info.KPULayers != 28 || info.Output != g.Layers[len(g.Layers)-1].Out {
		t.Fatalf("info %+v", info)
	}
	m, err := kmodel.Parse(kpu.AlignedCopy(data))
	if err != nil {
		t.Fatal(err)
	}
	r, err := kpu.NewEmulated(m, make([]byte, m.MainMemUsage))
	if err != nil {
		t.Fatal(err)
	}
	in := randomInput(info.Input.Len(), 3)
	chw := nhwc2chw(in, info.Input)
	if err := r.Run(chw); err != nil {
		t.Fatal(err)
	}
	got := chw2nhwc(r.Output(0), info.Output)
	cpu, err := newRunner(t, g).Run(in)
	if err != nil {
		t.Fatal(err)
	}
	for j := range cpu {
		if got[j] != cpu[j] {
			t.Fatalf("output %d: kmodel %d, CPU %d", j, got[j], cpu[j])
		}
	}
}

func TestKmodelRefuses(t *testing.T) {
	g := mobileNetSAME(t, 0.25, 64)
	last := g.Layers[len(g.Layers)-1]
	pool := qnet.Layer{Kind: qnet.MaxPool2D, In: last.Out, Out: last.Out, InQ: last.OutQ, OutQ: last.OutQ,
		KH: 3, KW: 3, StrideH: 1, StrideW: 1, Padding: qi8.PaddingSame}
	full := &qnet.Graph{Layers: append(append([]qnet.Layer(nil), g.Layers...), pool)}
	if _, _, err := Kmodel(full); !errors.Is(err, ErrUnsupported) {
		t.Errorf("graph ending in a 3×3 max pool: err %v, want ErrUnsupported", err)
	}
	two := &qnet.Graph{Layers: g.Layers, Outputs: []int{3, len(g.Layers) - 1}}
	if _, _, err := Kmodel(two); !errors.Is(err, ErrUnsupported) {
		t.Errorf("two outputs: err %v, want ErrUnsupported", err)
	}
}

// TestResize: qnet's MobileNet built at 64×64 and resized to 96×96 has the
// shapes of the one built at 96×96; Cut keeps a prefix.
func TestResize(t *testing.T) {
	a := mobileNetSAME(t, 0.25, 64)
	b := mobileNetSAME(t, 0.25, 96)
	r, err := Resize(a, 96, 96)
	if err != nil {
		t.Fatal(err)
	}
	for i := range b.Layers {
		if r.Layers[i].In != b.Layers[i].In || r.Layers[i].Out != b.Layers[i].Out {
			t.Fatalf("layer %d: %v→%v, want %v→%v", i, r.Layers[i].In, r.Layers[i].Out, b.Layers[i].In, b.Layers[i].Out)
		}
	}
	if a.Layers[0].In.H != 64 {
		t.Error("Resize changed its input graph")
	}
	c, err := Cut(r, 5)
	if err != nil || len(c.Layers) != 6 || c.Layers[5].Out != r.Layers[5].Out {
		t.Errorf("Cut: %v, %d layers", err, len(c.Layers))
	}
	if _, err := Cut(r, len(r.Layers)); err == nil {
		t.Error("cut past the end accepted")
	}
	fc := &qnet.Graph{Layers: []qnet.Layer{{Kind: qnet.FullyConnected}}}
	if _, err := Resize(fc, 8, 8); !errors.Is(err, ErrUnsupported) {
		t.Errorf("resize of a fully connected layer: %v", err)
	}
}

// nhwc2chw converts int8 NHWC to the KPU's uint8 CHW (q + 128).
func nhwc2chw(in []int8, s qi8.Shape) []byte {
	chw := make([]byte, s.Len())
	for y := range s.H {
		for x := range s.W {
			for c := range s.C {
				chw[(c*s.H+y)*s.W+x] = byte(in[(y*s.W+x)*s.C+c]) ^ 0x80
			}
		}
	}
	return chw
}

// TestResizeResidualOnInput (review R-K210-K9same S1): Resize re-derives an
// Add or Mul from its first input (qnet's In; only the second may
// broadcast), including the graph input (From −1), which used to index
// layer −1 and panic, and a layer whose output differs from the input size.
func TestResizeResidualOnInput(t *testing.T) {
	s := qi8.Shape{H: 8, W: 8, C: 4}
	q := qnet.Quant{Scale: 0.05, ZP: -3}
	conv := func(in qi8.Shape, stride int) qnet.Layer {
		out := qi8.Shape{H: in.H / stride, W: in.W / stride, C: in.C}
		return qnet.Layer{Kind: qnet.Conv2D, In: in, Out: out, InQ: q, OutQ: q,
			KH: 3, KW: 3, StrideH: stride, StrideW: stride, Padding: qi8.PaddingSame, ActMin: -128, ActMax: 127}
	}
	elem := func(kind qnet.Kind, in qi8.Shape, from ...int) qnet.Layer {
		l := qnet.Layer{Kind: kind, From: from, In: in, Out: in, InQ: q, In2Q: q, OutQ: q}
		l.ActMin, l.ActMax = qi8.ActNone()
		return l
	}
	mean := qnet.Layer{Kind: qnet.AvgPool2D, In: s, Out: qi8.Shape{H: 1, W: 1, C: 4}, InQ: q, OutQ: q,
		KH: 8, KW: 8, StrideH: 8, StrideW: 8, Padding: qi8.PaddingValid}
	half := qi8.Shape{H: 4, W: 4, C: 4}
	for _, c := range []struct {
		name   string
		layers []qnet.Layer
		want   qi8.Shape // the last layer's In and Out after resizing to 16×24
	}{
		{"add[-1, 0]", []qnet.Layer{conv(s, 1), elem(qnet.Add, s, -1, 0)}, qi8.Shape{H: 16, W: 24, C: 4}},
		// The input times a per-channel mean (1×1×C, the broadcast operand).
		{"mul[-1, mean]", []qnet.Layer{conv(s, 1), mean, elem(qnet.Mul, s, -1, 1)}, qi8.Shape{H: 16, W: 24, C: 4}},
		{"mul[0, mean]", []qnet.Layer{conv(s, 1), mean, elem(qnet.Mul, s, 0, 1)}, qi8.Shape{H: 16, W: 24, C: 4}},
		// After a stride-2 conv: the input size would be wrong.
		{"stride 2, add[1, 0]", []qnet.Layer{conv(s, 2), conv(half, 1), elem(qnet.Add, half, 1, 0)}, qi8.Shape{H: 8, W: 12, C: 4}},
		{"stride 2, add[0, 1]", []qnet.Layer{conv(s, 2), conv(half, 1), elem(qnet.Add, half, 0, 1)}, qi8.Shape{H: 8, W: 12, C: 4}},
	} {
		g := &qnet.Graph{Layers: c.layers}
		for i := range g.Layers {
			l := &g.Layers[i]
			nw, nb, nr := l.ParamCounts()
			l.Weights, l.Bias, l.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
			if l.Kind == qnet.Add {
				rq, err := qi8.QuantizeAddMultipliers(0.5, 0.5)
				if err != nil {
					t.Fatal(err)
				}
				l.AddRQ = rq
			}
		}
		if err := qnet.SyntheticParams(g, 1); err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if err := g.Validate(); err != nil {
			t.Fatalf("%s: the graph itself: %v", c.name, err)
		}
		r, err := Resize(g, 16, 24)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if l := r.Layers[len(r.Layers)-1]; l.In != c.want || l.Out != c.want {
			t.Errorf("%s: last layer %v → %v after resize, want %v", c.name, l.In, l.Out, c.want)
		}
	}
}

// TestKmodelRefusalReasons (review R-K210-K9same S2): refusals name the
// actual limit. A Pad(1,1)+VALID net too tall for the KPU is refused for
// its 256-row input limit (the Pad's conv), not because a Pad is not a
// convolution; a global pool is refused as a non-convolution, a TiesUp
// average pool for its rounding.
func TestKmodelRefusalReasons(t *testing.T) {
	for _, c := range []struct {
		name string
		g    *qnet.Graph
		want string
	}{
		{"512² Pad + VALID", mobileNetPad(t, 0.25, 512), "at most 256"},
		{"SAME with a global pool", func() *qnet.Graph {
			g := mobileNetSAME(t, 0.25, 64)
			last := g.Layers[len(g.Layers)-1]
			pool := qnet.Layer{Kind: qnet.AvgPool2D, In: last.Out, Out: qi8.Shape{H: 1, W: 1, C: last.Out.C}, InQ: last.OutQ, OutQ: last.OutQ,
				KH: last.Out.H, KW: last.Out.W, StrideH: 1, StrideW: 1, Padding: qi8.PaddingValid}
			return &qnet.Graph{Layers: append(append([]qnet.Layer(nil), g.Layers...), pool)}
		}(), "runs convolutions"},
		// Review R-K210-QU8 Q6: a TFLite uint8 average pool after the last
		// conv is not fused (the KPU floors the mean); the refusal says so.
		{"SAME with a TiesUp average pool", func() *qnet.Graph {
			g := mobileNetSAME(t, 0.25, 64)
			last := g.Layers[len(g.Layers)-1]
			pool := qnet.Layer{Kind: qnet.AvgPool2D, In: last.Out, Out: qi8.Shape{H: last.Out.H / 2, W: last.Out.W / 2, C: last.Out.C}, InQ: last.OutQ, OutQ: last.OutQ,
				KH: 2, KW: 2, StrideH: 2, StrideW: 2, Padding: qi8.PaddingValid, TiesUp: true}
			return &qnet.Graph{Layers: append(append([]qnet.Layer(nil), g.Layers...), pool)}
		}(), "rounds ties up"},
		// A Pad whose next layer does not read it: the Pad's own reason,
		// not that layer's (review recheck of 881d789).
		{"Pad not read by the next layer", func() *qnet.Graph {
			s := qi8.Shape{H: 8, W: 8, C: 4}
			pad := qnet.Layer{Kind: qnet.Pad, In: s, Out: qi8.Shape{H: 10, W: 10, C: 4}, InQ: ruleQ, OutQ: ruleQ, Pads: [6]int{1, 1, 1, 1, 0, 0}}
			c2 := ruleConv(s, 4, 3)
			c2.From = []int{0} // the conv after the Pad reads layer 0
			c3 := ruleConv(pad.Out, 4, 3)
			c3.Padding, c3.Out, c3.From = qi8.PaddingValid, s, []int{1} // the Pad's reader
			add := qnet.Layer{Kind: qnet.Add, From: []int{2, 3}, In: s, Out: s, InQ: ruleQ, In2Q: ruleQ, OutQ: ruleQ}
			add.ActMin, add.ActMax = qi8.ActNone()
			rq, err := qi8.QuantizeAddMultipliers(0.5, 0.5)
			if err != nil {
				t.Fatal(err)
			}
			add.AddRQ = rq
			return ruleGraph(t, &qnet.Graph{Layers: []qnet.Layer{ruleConv(s, 4, 3), pad, c2, c3, add}})
		}(), "Pad lowers only"},
		// A trailing Pad has no layer after it to borrow a reason from.
		{"trailing Pad", func() *qnet.Graph {
			s := qi8.Shape{H: 8, W: 8, C: 4}
			pad := qnet.Layer{Kind: qnet.Pad, In: s, Out: qi8.Shape{H: 10, W: 10, C: 4}, InQ: ruleQ, OutQ: ruleQ, Pads: [6]int{1, 1, 1, 1, 0, 0}}
			return ruleGraph(t, &qnet.Graph{Layers: []qnet.Layer{ruleConv(s, 4, 3), pad}})
		}(), "Pad lowers only"},
	} {
		_, _, err := Kmodel(c.g)
		if !errors.Is(err, ErrUnsupported) || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want ErrUnsupported naming %q", c.name, err, c.want)
		}
	}
}

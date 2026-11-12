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

// TestResizeResidualOnInput (review R-K210-K9same S1): an Add whose first
// input is the graph input (From [-1, 0]) takes the resized input shape;
// Resize used to index layer −1 and panic.
func TestResizeResidualOnInput(t *testing.T) {
	s := qi8.Shape{H: 8, W: 8, C: 4}
	q := qnet.Quant{Scale: 0.05, ZP: -3}
	c := qnet.Layer{Kind: qnet.Conv2D, In: s, Out: s, InQ: q, OutQ: q,
		KH: 3, KW: 3, StrideH: 1, StrideW: 1, Padding: qi8.PaddingSame, ActMin: -128, ActMax: 127}
	nw, nb, nr := c.ParamCounts()
	c.Weights, c.Bias, c.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
	add := qnet.Layer{Kind: qnet.Add, From: []int{-1, 0}, In: s, Out: s, InQ: q, In2Q: q, OutQ: q}
	add.ActMin, add.ActMax = qi8.ActNone()
	rq, err := qi8.QuantizeAddMultipliers(0.5, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	add.AddRQ = rq
	g := &qnet.Graph{Layers: []qnet.Layer{c, add}}
	if err := qnet.SyntheticParams(g, 1); err != nil {
		t.Fatal(err)
	}
	r, err := Resize(g, 16, 24)
	if err != nil {
		t.Fatal(err)
	}
	want := qi8.Shape{H: 16, W: 24, C: 4}
	if r.Layers[1].In != want || r.Layers[1].Out != want {
		t.Errorf("Add after resize: %v → %v, want %v", r.Layers[1].In, r.Layers[1].Out, want)
	}
}

// TestKmodelRefusalReasons (review R-K210-K9same S2): refusals name the
// actual limit. A Pad(1,1)+VALID net too tall for the KPU is refused for
// its 256-row input limit (the Pad's conv), not because a Pad is not a
// convolution; a global pool is refused as a non-convolution.
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
	} {
		_, _, err := Kmodel(c.g)
		if !errors.Is(err, ErrUnsupported) || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: err %v, want ErrUnsupported naming %q", c.name, err, c.want)
		}
	}
}

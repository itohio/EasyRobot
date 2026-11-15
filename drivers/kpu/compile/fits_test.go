package compile

import (
	"testing"

	"github.com/itohio/EasyRobot/drivers/kpu"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// TestFitsKPUMemory: a 1×1 conv on 256×256 maps whose input and output
// together fill the 2 MiB of KPU memory exactly (16 + 16 channels, 1,024
// lines each) runs on the KPU; one more output channel does not fit, so the
// layer stays on the CPU and the graph still compiles and runs.
func TestFitsKPUMemory(t *testing.T) {
	for _, tc := range []struct {
		oc    int
		onKPU bool
	}{{16, true}, {17, false}} {
		s := qi8.Shape{H: 256, W: 256, C: 16}
		l := qnet.Layer{Kind: qnet.Conv2D, In: s, Out: qi8.Shape{H: 256, W: 256, C: tc.oc},
			InQ: qnet.Quant{Scale: 0.05, ZP: 3}, OutQ: qnet.Quant{Scale: 0.05, ZP: -5},
			KH: 1, KW: 1, StrideH: 1, StrideW: 1, Padding: qi8.PaddingValid, ActMin: -128, ActMax: 127}
		nw, nb, nr := l.ParamCounts()
		l.Weights, l.Bias, l.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
		g := &qnet.Graph{Layers: []qnet.Layer{l}}
		if err := qnet.SyntheticParams(g, 7); err != nil {
			t.Fatal(err)
		}
		b, err := Compile(g, kpu.NewEmulated)
		if err != nil {
			t.Fatalf("%d output channels: %v", tc.oc, err)
		}
		if got := len(b.Segments()) == 1; got != tc.onKPU {
			t.Errorf("%d output channels: on the KPU %v, want %v", tc.oc, got, tc.onKPU)
		}
		if _, err := newRunner(t, g, b).Run(randomInput(s.Len(), 1)); err != nil {
			t.Errorf("%d output channels: run: %v", tc.oc, err)
		}
	}
}

// TestReaderExtraRowFits: the 1×1 conv that
// samples a TF SAME stride-2 depthwise conv's odd pixels reads one row past
// its input, so that row counts in its memory fit. 256×256×15 depthwise
// (15,360 lines in and out, plus the 4-line row) and 128×128 outputs of 256
// lines per channel: 67 channels fit, 68 would overlap the reader's input
// by 4 lines, so the pair stays on the CPU. At 62×66×9 the reader's input
// row (2 lines) and its output row (1 line) differ, so the row must be the
// input's: 1,020 channels fit, 1,021 do not.
func TestReaderExtraRowFits(t *testing.T) {
	for _, tc := range []struct {
		in    qi8.Shape
		oc    int
		onKPU bool
	}{
		{qi8.Shape{H: 256, W: 256, C: 15}, 67, true},
		{qi8.Shape{H: 256, W: 256, C: 15}, 68, false},
		{qi8.Shape{H: 62, W: 66, C: 9}, 1020, true},
		{qi8.Shape{H: 62, W: 66, C: 9}, 1021, false},
	} {
		s := tc.in
		dw := qnet.Layer{Kind: qnet.DepthwiseConv2D, In: s, Out: qi8.Shape{H: s.H / 2, W: s.W / 2, C: s.C}, InQ: ruleQ, OutQ: ruleQ,
			KH: 3, KW: 3, StrideH: 2, StrideW: 2, Padding: qi8.PaddingSame, ActMin: -128, ActMax: 127}
		nw, nb, nr := dw.ParamCounts()
		dw.Weights, dw.Bias, dw.Requant = make([]int8, nw), make([]int32, nb), make([]qi8.Requant, nr)
		g := ruleGraph(t, &qnet.Graph{Layers: []qnet.Layer{dw, ruleConv(dw.Out, tc.oc, 1)}})
		onKPU := false
		for _, seg := range compileRun(t, g) {
			onKPU = onKPU || seg.First == 0
		}
		if onKPU != tc.onKPU {
			t.Errorf("%v, %d reader channels: pair on the KPU %v, want %v", s, tc.oc, onKPU, tc.onKPU)
		}
	}
}

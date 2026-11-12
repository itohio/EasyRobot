package tfl

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/json"
	"io"
	"os"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
)

// TestU8NetMatchesInterpreter loads testdata/u8net.tflite (gen_u8net.py: a
// uint8 model in the form of TF's official "quant" MobileNetV1 exports, with
// asymmetric weights and every operator the loader accepts as uint8) and
// requires every layer output of the qnet Runner to equal the LiteRT
// reference interpreter's operator output (u8net.json.gz, 64 inputs, half of
// them near the input zero point so rounding ties are common). qnet holds
// uint8 tensors as int8 (value − 128); the fixture is in that form.
func TestU8NetMatchesInterpreter(t *testing.T) {
	model, err := os.ReadFile("testdata/u8net.tflite")
	if err != nil {
		t.Fatal(err)
	}
	var fx struct {
		Ops   []string `json:"ops"`
		Cases []struct {
			Input string   `json:"input"`
			Outs  []string `json:"outs"`
		} `json:"cases"`
	}
	f, err := os.Open("testdata/u8net.json.gz")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(zr)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatal(err)
	}
	g, err := Load(model)
	if err != nil {
		t.Fatal(err)
	}
	want := []qnet.Kind{qnet.Conv2D, qnet.Conv2D, qnet.DepthwiseConv2D, qnet.MaxPool2D, qnet.AvgPool2D,
		qnet.Reshape, qnet.FullyConnected, qnet.Softmax}
	if len(g.Layers) != len(want) || len(fx.Ops) != len(want) {
		t.Fatalf("%d layers, %d operators, want %d", len(g.Layers), len(fx.Ops), len(want))
	}
	for i, k := range want {
		if g.Layers[i].Kind != k {
			t.Fatalf("layer %d is %v (operator %s), want %v", i, g.Layers[i].Kind, fx.Ops[i], k)
		}
	}
	// uint8 zero points move by −128; weights keep theirs as WZP (generator: 140, 77, 200, 99).
	if q := g.Layers[0].InQ; q != (qnet.Quant{Scale: 1.0 / 16, ZP: 120 - 128}) {
		t.Fatalf("graph input quantisation %+v", q)
	}
	for i, wzp := range map[int]int8{0: 140 - 128, 1: 77 - 128, 2: 200 - 128, 6: 99 - 128} {
		if g.Layers[i].WZP != wzp {
			t.Fatalf("layer %d WZP %d, want %d", i, g.Layers[i].WZP, wzp)
		}
	}
	if !g.Layers[4].TiesUp {
		t.Fatal("uint8 average pool must round ties up")
	}
	p, err := qnet.PlanMemory(&g)
	if err != nil {
		t.Fatal(err)
	}
	r, err := qnet.NewRunner(&g, make([]int8, p.Size()))
	if err != nil {
		t.Fatal(err)
	}
	dec := func(s string) []int8 {
		b, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			t.Fatal(err)
		}
		return i8(b)
	}
	for ci, c := range fx.Cases {
		in := dec(c.Input)
		// Compare each tensor right after its segment: the arena reuses it.
		for s, seg := range r.Plan().Segments {
			if err := r.RunSegment(in, s); err != nil {
				t.Fatal(err)
			}
			i := seg.First + seg.Count - 1
			if got, exp := r.SegmentOutput(in, s), dec(c.Outs[i]); !bytes.Equal(u8(got), u8(exp)) {
				t.Fatalf("case %d layer %d (%s): %d bytes differ from the interpreter", ci, i, fx.Ops[i], diffCount(got, exp))
			}
		}
	}
	in := dec(fx.Cases[0].Input)
	if a := testing.AllocsPerRun(20, func() { _, _ = r.Run(in) }); a != 0 {
		t.Fatalf("Run allocates %v times", a)
	}
}

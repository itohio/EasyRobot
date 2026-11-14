package tfl

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"os"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// TestYOLOXTailMatchesInterpreter loads testdata/yoloxtail.tflite
// (gen_yoloxtail.py: a YOLOX-like head — objectness CONV_2D + LOGISTIC times
// the features as a per-pixel MUL, TRANSPOSE [0,3,1,2], RESHAPE,
// CONCATENATION, TRANSPOSE [0,2,1] — plus a chain of rank-4 permutations,
// with int8 → uint8 QUANTIZEd graph outputs) and requires every layer
// output of the qnet Runner to equal the LiteRT reference interpreter's
// operator output, and each graph output + 128 to equal the interpreter's
// uint8 output (yoloxtail.json.gz, 32 inputs).
func TestYOLOXTailMatchesInterpreter(t *testing.T) {
	model, err := os.ReadFile("testdata/yoloxtail.tflite")
	if err != nil {
		t.Fatal(err)
	}
	var fx struct {
		Ops   []string `json:"ops"`
		Cases []struct {
			Input string   `json:"input"`
			Outs  []string `json:"outs"`
			Graph []string `json:"graph"`
		} `json:"cases"`
	}
	f, err := os.Open("testdata/yoloxtail.json.gz")
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
	want := []qnet.Kind{qnet.Conv2D, qnet.Logistic, qnet.Mul, qnet.ResizeNearest, qnet.Transpose, qnet.Reshape, qnet.Transpose, qnet.Reshape, qnet.Concat,
		qnet.Transpose, qnet.Transpose, qnet.Transpose, qnet.Transpose, qnet.Transpose}
	if len(g.Layers) != len(want) || len(fx.Ops) != len(want) {
		t.Fatalf("%d layers, %d operators, want %d", len(g.Layers), len(fx.Ops), len(want))
	}
	for i := range want {
		if g.Layers[i].Kind != want[i] {
			t.Fatalf("layer %d is %v (operator %s), want %v", i, g.Layers[i].Kind, fx.Ops[i], want[i])
		}
	}
	// The uint8 QUANTIZEs are cut: the outputs are the head-tail TRANSPOSE
	// and the last layer, int8 with the uint8 zero point − 128.
	if !g.Layers[2].PerPixel || g.Layers[2].In != (qi8.Shape{H: 5, W: 6, C: 7}) {
		t.Fatalf("MUL layer %+v, want the 5×6×7 features times the per-pixel objectness", g.Layers[2])
	}
	if len(g.Outputs) != 2 || g.Outputs[0] != 9 || g.Outputs[1] != 13 {
		t.Fatalf("outputs %v, want [9 13]", g.Outputs)
	}
	if q := g.Layers[9].OutQ; q != (qnet.Quant{Scale: 0.05, ZP: 3}) {
		t.Fatalf("output 0 quantisation %+v, want {0.05 3} (uint8 zero point 131 − 128)", q)
	}
	r, err := qnet.NewRunner(&g, make([]int8, 1<<14))
	if err != nil {
		t.Fatal(err)
	}
	dec := func(s string) []byte {
		b, err := base64.StdEncoding.DecodeString(s)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	for ci, c := range fx.Cases {
		in := i8(dec(c.Input))
		for s, seg := range r.Plan().Segments {
			if err := r.RunSegment(in, s); err != nil {
				t.Fatal(err)
			}
			i := seg.First + seg.Count - 1
			if got, exp := r.SegmentOutput(in, s), i8(dec(c.Outs[i])); !bytes.Equal(u8(got), u8(exp)) {
				t.Fatalf("case %d layer %d (%s): %d bytes differ from the interpreter", ci, i, fx.Ops[i], diffCount(got, exp))
			}
		}
		if _, err := r.Run(in); err != nil {
			t.Fatal(err)
		}
		for k := range 2 {
			got, exp := r.Output(in, k), dec(c.Graph[k])
			if len(got) != len(exp) {
				t.Fatalf("case %d output %d: %d values, want %d", ci, k, len(got), len(exp))
			}
			for j := range got {
				if byte(int(got[j])+128) != exp[j] {
					t.Fatalf("case %d output %d [%d]: int8 %d + 128 != uint8 %d", ci, k, j, got[j], exp[j])
				}
			}
		}
	}
}

// The loader refuses TRANSPOSE that moves the batch axis or changes the
// quantisation, and int8 → uint8 QUANTIZEs that are not a plain uint8 graph
// output (not an output, other scale, zero point not shifted by 128), each
// with an *OpError naming the operator.
func TestTransposeAndUint8OutputRejects(t *testing.T) {
	perm := func(b *mb, p ...int32) int32 {
		return int32(b.tensor("p", []int32{int32(len(p))}, typeInt32, b.buffer(le32(p...)), nil, nil, 0))
	}
	cases := []struct {
		name  string
		build func(b *mb) ([]int32, []int32)
	}{
		{"transpose moves the batch axis", func(b *mb) ([]int32, []int32) {
			x, y := b.act([]int32{1, 2, 3, 4}, 0.1, 0), b.act([]int32{2, 1, 3, 4}, 0.1, 0)
			b.op(opTranspose, []int32{x, perm(b, 1, 0, 2, 3)}, []int32{y}, 0, nil)
			return []int32{x}, []int32{y}
		}},
		{"transpose changes the quantisation", func(b *mb) ([]int32, []int32) {
			x, y := b.act([]int32{1, 2, 3, 4}, 0.1, 0), b.act([]int32{1, 4, 2, 3}, 0.2, 0)
			b.op(opTranspose, []int32{x, perm(b, 0, 3, 1, 2)}, []int32{y}, 0, nil)
			return []int32{x}, []int32{y}
		}},
		{"transpose output shape wrong", func(b *mb) ([]int32, []int32) {
			x, y := b.act([]int32{1, 2, 3, 4}, 0.1, 0), b.act([]int32{1, 2, 4, 3}, 0.1, 0)
			b.op(opTranspose, []int32{x, perm(b, 0, 3, 1, 2)}, []int32{y}, 0, nil)
			return []int32{x}, []int32{y}
		}},
		{"uint8 quantize not a graph output", func(b *mb) ([]int32, []int32) {
			x, y := b.act([]int32{1, 4}, 0.1, 0), b.act([]int32{1, 4}, 1.0/256, -128)
			u := int32(b.tensor("u", []int32{1, 4}, typeUInt8, 0, []float32{0.1}, []int64{128}, 0))
			b.op(opQuantize, []int32{x}, []int32{u}, 0, nil)
			b.op(opSoftmax, []int32{x}, []int32{y}, optSoftmax, optsSoftmax(1))
			return []int32{x}, []int32{y}
		}},
		{"uint8 output with another scale", func(b *mb) ([]int32, []int32) {
			x := b.act([]int32{1, 4}, 0.1, 0)
			u := int32(b.tensor("u", []int32{1, 4}, typeUInt8, 0, []float32{0.2}, []int64{128}, 0))
			b.op(opQuantize, []int32{x}, []int32{u}, 0, nil)
			return []int32{x}, []int32{u}
		}},
		{"uint8 output zero point not +128", func(b *mb) ([]int32, []int32) {
			x := b.act([]int32{1, 4}, 0.1, 0)
			u := int32(b.tensor("u", []int32{1, 4}, typeUInt8, 0, []float32{0.1}, []int64{127}, 0))
			b.op(opQuantize, []int32{x}, []int32{u}, 0, nil)
			return []int32{x}, []int32{u}
		}},
	}
	for _, c := range cases {
		b := newMB()
		in, out := c.build(b)
		b.io(in, out)
		_, err := Load(b.bytes())
		var oe *OpError
		if !errors.As(err, &oe) {
			t.Errorf("%s: %v, want an *OpError", c.name, err)
		}
	}
}

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
)

// TestOpsNetMatchesInterpreter loads testdata/opsnet.tflite (gen_opsnet.py:
// float I/O, QUANTIZE, PAD, PADV2, RESIZE_NEAREST_NEIGHBOR, LOGISTIC, MUL,
// CONCATENATION, MEAN, ADD with constants, two DEQUANTIZEd outputs) and
// requires every layer output of the qnet Runner to equal the LiteRT
// reference interpreter's operator output (opsnet.json.gz, 64 inputs).
func TestOpsNetMatchesInterpreter(t *testing.T) {
	model, err := os.ReadFile("testdata/opsnet.tflite")
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
	f, err := os.Open("testdata/opsnet.json.gz")
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
	// QUANTIZE of the float input and the two output DEQUANTIZEs are cut.
	if q := g.Layers[0].InQ; q != (qnet.Quant{Scale: 0.05, ZP: 3}) {
		t.Fatalf("graph input quantisation %+v", q)
	}
	var layers []int // non-Const layers, one per interpreter operator
	for i := range g.Layers {
		if g.Layers[i].Kind != qnet.Const {
			layers = append(layers, i)
		}
	}
	want := []qnet.Kind{qnet.Pad, qnet.ResizeNearest, qnet.ResizeNearest, qnet.Logistic, qnet.Mul, qnet.Requantize, qnet.Mul,
		qnet.Requantize, qnet.Concat, qnet.Concat, qnet.Mean, qnet.Mul, qnet.Add, qnet.Pad,
		qnet.LeakyReLU, qnet.ResizeBilinear, qnet.ResizeBilinear}
	const meanOp, padV2Op = 10, 13
	if len(layers) != len(want) || len(fx.Ops) != len(want) {
		t.Fatalf("%d layers, %d operators, want %d", len(layers), len(fx.Ops), len(want))
	}
	for k, i := range layers {
		if g.Layers[i].Kind != want[k] {
			t.Fatalf("layer %d is %v (operator %s), want %v", i, g.Layers[i].Kind, fx.Ops[k], want[k])
		}
	}
	meanAt, padV2At, last := layers[meanOp], layers[padV2Op], len(g.Layers)-1
	if len(g.Outputs) != 3 || g.Outputs[0] != padV2At || g.Outputs[1] != meanAt || g.Outputs[2] != last {
		t.Fatalf("outputs %v, want [%d %d %d]", g.Outputs, padV2At, meanAt, last)
	}
	r, err := qnet.NewRunner(&g, make([]int8, 1<<16))
	if err != nil {
		t.Fatal(err)
	}
	opOf := make(map[int]int) // layer → interpreter operator
	for k, i := range layers {
		opOf[i] = k
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
			k, ok := opOf[i]
			if !ok {
				continue
			}
			if got, exp := r.SegmentOutput(in, s), dec(c.Outs[k]); !bytes.Equal(u8(got), u8(exp)) {
				t.Fatalf("case %d layer %d (%s): %d bytes differ from the interpreter", ci, i, fx.Ops[k], diffCount(got, exp))
			}
		}
		out, err := r.Run(in)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(u8(out), u8(dec(c.Outs[padV2Op]))) {
			t.Fatalf("case %d: graph output 0 (PADV2) differs", ci)
		}
		if !bytes.Equal(u8(r.Output(in, 1)), u8(dec(c.Outs[meanOp]))) {
			t.Fatalf("case %d: graph output 1 (MEAN) differs", ci)
		}
		if !bytes.Equal(u8(r.Output(in, 2)), u8(dec(c.Outs[len(c.Outs)-1]))) {
			t.Fatalf("case %d: graph output 2 (RESIZE_BILINEAR) differs", ci)
		}
	}
}

// TestLoadIOCuts: a uint8 graph input quantised by the first operator is cut
// (the graph takes the QUANTIZE output); a DEQUANTIZE that is not a graph
// output is rejected by name.
func TestLoadIOCuts(t *testing.T) {
	b := newMB()
	u := int32(b.tensor("u", []int32{1, 4}, typeUInt8, 0, []float32{0.5}, []int64{128}, 0))
	x := b.act([]int32{1, 4}, 0.5, 0)
	z := b.act([]int32{1, 4}, 1.0/256, -128)
	b.op(opQuantize, []int32{u}, []int32{x}, 0, nil)
	b.op(opSoftmax, []int32{x}, []int32{z}, optSoftmax, optsSoftmax(1))
	b.io([]int32{u}, []int32{z})
	g, err := Load(b.bytes())
	if err != nil {
		t.Fatal(err)
	}
	if len(g.Layers) != 1 || g.Layers[0].Kind != qnet.Softmax || g.Layers[0].InQ != (qnet.Quant{Scale: 0.5, ZP: 0}) || g.Outputs != nil {
		t.Fatalf("uint8 input cut: %+v", g)
	}

	b = newMB()
	x = b.act([]int32{1, 4}, 0.5, 0)
	z = b.act([]int32{1, 4}, 1.0/256, -128)
	f := int32(b.tensor("f", []int32{1, 4}, typeFloat32, 0, nil, nil, 0))
	b.op(opSoftmax, []int32{x}, []int32{z}, optSoftmax, optsSoftmax(1))
	b.op(opDequantize, []int32{z}, []int32{f}, 0, nil)
	b.io([]int32{x}, []int32{z})
	var oe *OpError
	if _, err := Load(b.bytes()); !errors.As(err, &oe) || oe.Op != "DEQUANTIZE" || oe.Index != 1 {
		t.Fatalf("dequantize of a non-output: %v", err)
	}
}

func u8(v []int8) []byte {
	b := make([]byte, len(v))
	for i, x := range v {
		b[i] = byte(x)
	}
	return b
}

func diffCount(a, b []int8) int {
	if len(a) != len(b) {
		return max(len(a), len(b))
	}
	n := 0
	for i := range a {
		if a[i] != b[i] {
			n++
		}
	}
	return n
}

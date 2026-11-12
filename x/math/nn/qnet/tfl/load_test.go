package tfl

import (
	"bytes"
	"errors"
	"math/rand/v2"
	"strings"
	"testing"
	"unsafe"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// mb wraps tflBuilder with helpers for activation and constant tensors.
type mb struct {
	*tflBuilder
	r *rand.Rand
}

func newMB() *mb { return &mb{newTFLBuilder(), rand.New(rand.NewPCG(7, 9))} }

func (b *mb) act(shape []int32, scale float32, zp int64) int32 {
	return int32(b.tensor("a", shape, typeInt8, 0, []float32{scale}, []int64{zp}, 0))
}

func (b *mb) randBytes(n int) []byte {
	d := make([]byte, n)
	for i := range d {
		d[i] = byte(b.r.IntN(256))
	}
	return d
}

// weights adds a constant int8 tensor with per-channel scales along qdim.
func (b *mb) weights(shape []int32, scales []float32, qdim int32) (int32, []byte) {
	n := 1
	for _, s := range shape {
		n *= int(s)
	}
	d := b.randBytes(n)
	zp := make([]int64, len(scales))
	return int32(b.tensor("w", shape, typeInt8, b.buffer(d), scales, zp, qdim)), d
}

func (b *mb) bias(n int) (int32, []int32) {
	v := make([]int32, n)
	for i := range v {
		v[i] = int32(b.r.IntN(4001) - 2000)
	}
	return int32(b.tensor("b", []int32{int32(n)}, typeInt32, b.buffer(le32(v...)), nil, nil, 0)), v
}

func i8(d []byte) []int8 {
	out := make([]int8, len(d))
	for i, v := range d {
		out[i] = int8(v)
	}
	return out
}

func randIn(n int, seed uint64) []int8 {
	r := rand.New(rand.NewPCG(seed, 3))
	s := make([]int8, n)
	for i := range s {
		s[i] = int8(r.IntN(256) - 128)
	}
	return s
}

// runGraph runs g through the qnet Runner.
func runGraph(t *testing.T, g *qnet.Graph, in []int8) []int8 {
	t.Helper()
	p, err := qnet.PlanMemory(g)
	if err != nil {
		t.Fatal(err)
	}
	r, err := qnet.NewRunner(g, make([]int8, p.Size()))
	if err != nil {
		t.Fatal(err)
	}
	out, err := r.Run(in)
	if err != nil {
		t.Fatal(err)
	}
	return append([]int8(nil), out...)
}

func tfliteRQ(t *testing.T, sIn, sW, sOut float32) qi8.Requant {
	t.Helper()
	r, err := qi8.QuantizeMultiplier(float64(sIn) * float64(sW) / float64(sOut))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

// TestLoadConvChain: conv (per-channel, RELU6) → depthwise (multiplier 1) →
// avgpool → reshape → fc → softmax, as in MobileNet-style classifiers.
func TestLoadConvChain(t *testing.T) {
	b := newMB()
	in := b.act([]int32{1, 6, 5, 3}, 1.0/128, -1)
	wScales := []float32{0.01, 0.02, 0.004, 0.03}
	w0, w0d := b.weights([]int32{4, 3, 3, 3}, wScales, 0)
	b0, b0v := b.bias(4)
	t0 := b.act([]int32{1, 3, 3, 4}, 0.05, -128)
	b.op(opConv2D, []int32{in, w0, b0}, []int32{t0}, optConv2D, optsConv2D(padSame, 2, 2, actRelu6, 1, 1))

	w1, _ := b.weights([]int32{1, 3, 3, 4}, []float32{0.02, 0.01, 0.03, 0.02}, 3)
	b1, _ := b.bias(4)
	t1 := b.act([]int32{1, 3, 3, 4}, 0.04, -128)
	b.op(opDepthwiseConv2D, []int32{t0, w1, b1}, []int32{t1}, optDepthwiseConv2D, optsDepthwiseConv2D(padSame, 1, 1, 1, actRelu, 1, 1))

	t2 := b.act([]int32{1, 1, 1, 4}, 0.04, -128)
	b.op(opAveragePool2D, []int32{t1}, []int32{t2}, optPool2D, optsPool2D(padValid, 1, 1, 3, 3, actNone))
	shape := int32(b.tensor("s", []int32{2}, typeInt32, b.buffer(le32(1, 4)), nil, nil, 0))
	t3 := b.act([]int32{1, 4}, 0.04, -128)
	b.op(opReshape, []int32{t2, shape}, []int32{t3}, optReshape, optsReshape([]int32{1, 4}))
	w4, _ := b.weights([]int32{3, 4}, []float32{0.015}, 0) // per-tensor
	t4 := b.act([]int32{1, 3}, 0.1, 5)
	b.op(opFullyConnected, []int32{t3, w4, -1}, []int32{t4}, optFullyConnected, optsFullyConnected(actNone))
	t5 := b.act([]int32{1, 3}, 1.0/256, -128)
	b.op(opSoftmax, []int32{t4}, []int32{t5}, optSoftmax, optsSoftmax(1))
	b.io([]int32{in}, []int32{t5})
	model := b.bytes()

	g, err := Load(model)
	if err != nil {
		t.Fatal(err)
	}
	kinds := []qnet.Kind{qnet.Conv2D, qnet.DepthwiseConv2D, qnet.AvgPool2D, qnet.Reshape, qnet.FullyConnected, qnet.Softmax}
	if len(g.Layers) != len(kinds) {
		t.Fatalf("%d layers", len(g.Layers))
	}
	for i, k := range kinds {
		l := &g.Layers[i]
		if l.Kind != k || l.From != nil {
			t.Fatalf("layer %d: %v from %v", i, l.Kind, l.From)
		}
		want := qi8.RoundTFLite
		if k == qnet.FullyConnected {
			want = qi8.RoundExact // TFLite's FC rounds once, half away from zero
		}
		if (k == qnet.Conv2D || k == qnet.DepthwiseConv2D || k == qnet.FullyConnected) && l.Round != want {
			t.Fatalf("layer %d rounding %v", i, l.Round)
		}
	}
	c := &g.Layers[0]
	if c.In != (qi8.Shape{H: 6, W: 5, C: 3}) || c.Out != (qi8.Shape{H: 3, W: 3, C: 4}) || c.StrideH != 2 || c.Padding != qi8.PaddingSame {
		t.Fatalf("conv geometry %+v", c)
	}
	if &c.Weights[0] != (*int8)(unsafe.Pointer(&model[bytes.Index(model, w0d)])) {
		t.Fatal("weights must alias the model bytes")
	}
	for o, s := range wScales {
		if c.Requant[o] != tfliteRQ(t, 1.0/128, s, 0.05) || c.Bias[o] != b0v[o] {
			t.Fatalf("channel %d: requant %+v bias %d", o, c.Requant[o], c.Bias[o])
		}
	}
	lo, hi := qi8.ActReLU6TFLite(0.05, -128)
	if c.ActMin != lo || c.ActMax != hi {
		t.Fatalf("relu6 range [%d,%d]", c.ActMin, c.ActMax)
	}
	if d := &g.Layers[1]; d.ActMin != -128 || d.ActMax != 127 {
		t.Fatalf("relu with zp -128 is [-128,127], got [%d,%d]", d.ActMin, d.ActMax)
	}
	if fc := &g.Layers[4]; fc.Bias != nil || fc.Requant[0] != fc.Requant[2] || fc.In != (qi8.Shape{H: 1, W: 1, C: 4}) {
		t.Fatalf("fc %+v", fc)
	}
	sp, _ := qi8.QuantizeSoftmax(1, 0.1)
	if g.Layers[5].SoftmaxParams != sp {
		t.Fatal("softmax params")
	}
	// The runner (qi8 prepared layers) equals the free-function reference.
	for seed := range uint64(4) {
		x := randIn(90, seed)
		want := x
		for i := range g.Layers {
			dst := make([]int8, g.Layers[i].Out.Len())
			if err := qnet.RunLayer(&g.Layers[i], dst, want); err != nil {
				t.Fatal(err)
			}
			want = dst
		}
		if got := runGraph(t, &g, x); !bytes.Equal(i8b(got), i8b(want)) {
			t.Fatalf("seed %d: %v want %v", seed, got, want)
		}
	}
}

func i8b(s []int8) []byte {
	b := make([]byte, len(s))
	for i, v := range s {
		b[i] = byte(v)
	}
	return b
}

// TestLoadDepthwiseMultiplier: a one-channel depthwise layer with multiplier
// M becomes a Conv2D whose output equals TFLite's depthwise definition
// (output channel m uses weight [0][ky][kx][m]).
func TestLoadDepthwiseMultiplier(t *testing.T) {
	b := newMB()
	in := b.act([]int32{1, 5, 4, 1}, 0.02, 3)
	scales := []float32{0.01, 0.02, 0.015, 0.03, 0.005, 0.02}
	w, wd := b.weights([]int32{1, 3, 3, 6}, scales, 3)
	bi, bv := b.bias(6)
	out := b.act([]int32{1, 3, 2, 6}, 0.03, -7)
	b.op(opDepthwiseConv2D, []int32{in, w, bi}, []int32{out}, optDepthwiseConv2D, optsDepthwiseConv2D(padSame, 2, 2, 6, actNone, 1, 1))
	b.io([]int32{in}, []int32{out})
	g, err := Load(b.bytes())
	if err != nil {
		t.Fatal(err)
	}
	if g.Layers[0].Kind != qnet.Conv2D || g.Layers[0].Out.C != 6 {
		t.Fatalf("layer %+v", g.Layers[0])
	}
	x := randIn(20, 1)
	got := runGraph(t, &g, x)
	// direct depthwise-with-multiplier computation (SAME, stride 2)
	wv := i8(wd)
	for oy := range 3 {
		for ox := range 2 {
			for m := range 6 {
				acc := bv[m]
				for ky := range 3 {
					for kx := range 3 {
						iy, ix := oy*2-1+ky, ox*2+kx // SAME: pad top 1 (total 2), left 0 (total 1)
						if iy < 0 || iy >= 5 || ix < 0 || ix >= 4 {
							continue
						}
						acc += (int32(x[iy*4+ix]) - 3) * int32(wv[(ky*3+kx)*6+m])
					}
				}
				r := tfliteRQ(t, 0.02, scales[m], 0.03)
				want := int8(min(max(r.ApplyTFLite(acc)-7, -128), 127))
				if g := got[(oy*2+ox)*6+m]; g != want {
					t.Fatalf("(%d,%d,%d) = %d want %d", oy, ox, m, g, want)
				}
			}
		}
	}
}

// TestLoadResidualAdd: an ADD of a conv output and the graph input becomes a
// two-producer qnet Add with TFLite multipliers.
func TestLoadResidualAdd(t *testing.T) {
	b := newMB()
	in := b.act([]int32{1, 4, 4, 2}, 0.05, 2)
	w, _ := b.weights([]int32{2, 1, 1, 2}, []float32{0.01, 0.02}, 0)
	t0 := b.act([]int32{1, 4, 4, 2}, 0.03, -4)
	b.op(opConv2D, []int32{in, w, -1}, []int32{t0}, optConv2D, optsConv2D(padValid, 1, 1, actNone, 1, 1))
	t1 := b.act([]int32{1, 4, 4, 2}, 0.06, -128)
	b.op(opAdd, []int32{t0, in}, []int32{t1}, optAdd, optsAdd(actRelu))
	b.io([]int32{in}, []int32{t1})
	g, err := Load(b.bytes())
	if err != nil {
		t.Fatal(err)
	}
	a := &g.Layers[1]
	rq, _ := qi8.QuantizeAddTFLite(0.03, 0.05, 0.06)
	if a.Kind != qnet.Add || len(a.From) != 2 || a.From[0] != 0 || a.From[1] != -1 || a.AddTFLite != rq || a.In2Q != (qnet.Quant{Scale: 0.05, ZP: 2}) {
		t.Fatalf("add %+v", a)
	}
	x := randIn(32, 4)
	c := make([]int8, 32)
	if err := qnet.RunLayer(&g.Layers[0], c, x); err != nil {
		t.Fatal(err)
	}
	want := make([]int8, 32)
	if err := qi8.Add(want, c, x, qi8.AddParams{ZPA: -4, ZPB: 2, ZPOut: -128, Min: -128, Max: 127, Round: qi8.RoundTFLite, TFLite: rq}); err != nil {
		t.Fatal(err)
	}
	if got := runGraph(t, &g, x); !bytes.Equal(i8b(got), i8b(want)) {
		t.Fatalf("%v want %v", got, want)
	}
}

// A multiplier below 2^-32 (a dead channel) loads as the zero multiplier.
func TestLoadZeroMultiplier(t *testing.T) {
	b := newMB()
	in := b.act([]int32{1, 1, 1, 4}, 0.03, 0)
	w, _ := b.weights([]int32{2, 4}, []float32{1e-11, 0.01}, 0)
	out := b.act([]int32{1, 2}, 0.02, -9)
	b.op(opFullyConnected, []int32{in, w}, []int32{out}, optFullyConnected, optsFullyConnected(actNone))
	b.io([]int32{in}, []int32{out})
	g, err := Load(b.bytes())
	if err != nil {
		t.Fatal(err)
	}
	if g.Layers[0].Requant[0] != (qi8.Requant{}) {
		t.Fatalf("requant %+v", g.Layers[0].Requant)
	}
	if got := runGraph(t, &g, randIn(4, 1)); got[0] != -9 {
		t.Fatalf("dead channel output %d, want the zero point", got[0])
	}
}

func TestLoadErrors(t *testing.T) {
	conv := func(mod func(b *mb, in, w, out int32) (code int32, opt uint8, opts *tableSpec)) []byte {
		b := newMB()
		in := b.act([]int32{1, 4, 4, 2}, 0.05, 2)
		w, _ := b.weights([]int32{2, 3, 3, 2}, []float32{0.01, 0.02}, 0)
		out := b.act([]int32{1, 4, 4, 2}, 0.03, -4)
		code, opt, opts := int32(opConv2D), uint8(optConv2D), optsConv2D(padSame, 1, 1, actNone, 1, 1)
		if mod != nil {
			code, opt, opts = mod(b, in, w, out)
		}
		b.op(code, []int32{in, w, -1}, []int32{out}, opt, opts)
		b.io([]int32{in}, []int32{out})
		return b.bytes()
	}
	// asU8 makes tensor i uint8 with zero point zp (all its channels).
	asU8 := func(b *mb, i int32, zp ...int64) {
		b.tensorTable(int(i)).u8(tensorType, typeUInt8).sub(tensorQuantization).vecI64(quantZeroPoint, zp)
	}
	unsupported := []struct {
		name, op string
		model    []byte
		tensor   string // named in the reason, when set
	}{
		{"unknown op", "SUB", conv(func(*mb, int32, int32, int32) (int32, uint8, *tableSpec) { return opSub, 0, nil }), ""},
		{"unnamed op", "BUILTIN_145", conv(func(*mb, int32, int32, int32) (int32, uint8, *tableSpec) { return 145, 0, nil }), ""},
		{"dilation", "CONV_2D", conv(func(*mb, int32, int32, int32) (int32, uint8, *tableSpec) {
			return opConv2D, optConv2D, optsConv2D(padSame, 1, 1, actNone, 2, 1)
		}), ""},
		{"uint8 input, int8 weights", "CONV_2D", conv(func(b *mb, in, _, _ int32) (int32, uint8, *tableSpec) {
			asU8(b, in, 130)
			return opConv2D, optConv2D, optsConv2D(padSame, 1, 1, actNone, 1, 1)
		}), "w"},
		{"per-channel uint8 weights", "CONV_2D", conv(func(b *mb, in, w, out int32) (int32, uint8, *tableSpec) {
			asU8(b, in, 130)
			asU8(b, w, 120, 140)
			asU8(b, out, 124)
			return opConv2D, optConv2D, optsConv2D(padSame, 1, 1, actNone, 1, 1)
		}), "w"},
		{"asymmetric int8 weights", "CONV_2D", conv(func(b *mb, _, w, _ int32) (int32, uint8, *tableSpec) {
			b.tensorTable(int(w)).sub(tensorQuantization).vecI64(quantZeroPoint, []int64{0, 3})
			return opConv2D, optConv2D, optsConv2D(padSame, 1, 1, actNone, 1, 1)
		}), "w"},
		{"tanh activation", "CONV_2D", conv(func(*mb, int32, int32, int32) (int32, uint8, *tableSpec) {
			return opConv2D, optConv2D, optsConv2D(padSame, 1, 1, 4, 1, 1)
		}), ""},
		{"sparse weights", "CONV_2D", conv(func(b *mb, _, w, _ int32) (int32, uint8, *tableSpec) {
			b.sparse(int(w))
			return opConv2D, optConv2D, optsConv2D(padSame, 1, 1, actNone, 1, 1)
		}), ""},
	}
	for _, c := range unsupported {
		_, err := Load(c.model)
		var oe *OpError
		if !errors.As(err, &oe) || !errors.Is(err, ErrUnsupported) || oe.Op != c.op || oe.Index != 0 {
			t.Fatalf("%s: %v", c.name, err)
		}
		if c.tensor != "" && !strings.Contains(oe.Reason, `tensor "`+c.tensor+`"`) {
			t.Fatalf("%s: reason %q does not name tensor %q", c.name, oe.Reason, c.tensor)
		}
		t.Logf("%s: %v", c.name, err)
	}

	// uint8 on an operator not verified for it (ADD) is refused, naming the tensor
	b := newMB()
	x := b.act([]int32{1, 2, 2, 2}, 0.1, 0)
	y := b.act([]int32{1, 2, 2, 2}, 0.1, 0)
	z := b.act([]int32{1, 2, 2, 2}, 0.1, 0)
	for _, i := range []int32{x, y, z} {
		asU8(b, i, 128)
	}
	b.op(opMaxPool2D, []int32{x}, []int32{y}, optPool2D, optsPool2D(padSame, 1, 1, 1, 1, actNone))
	b.op(opAdd, []int32{x, y}, []int32{z}, optAdd, optsAdd(actNone))
	b.io([]int32{x}, []int32{z})
	var oe *OpError
	if _, err := Load(b.bytes()); !errors.As(err, &oe) || oe.Op != "ADD" || oe.Index != 1 || !strings.Contains(oe.Reason, `tensor "a"`) {
		t.Fatalf("uint8 add: %v", err)
	}

	// broadcasting Add, pool with fused activation, softmax output quant,
	// and an output that is not the last operator
	b = newMB()
	x = b.act([]int32{1, 2, 2, 2}, 0.1, 0)
	y = b.act([]int32{1, 1, 1, 2}, 0.1, 0)
	z = b.act([]int32{1, 2, 2, 2}, 0.1, 0)
	b.op(opAveragePool2D, []int32{x}, []int32{y}, optPool2D, optsPool2D(padValid, 1, 1, 2, 2, actNone))
	b.op(opAdd, []int32{x, y}, []int32{z}, optAdd, optsAdd(actNone))
	b.io([]int32{x}, []int32{z})
	if _, err := Load(b.bytes()); !errors.As(err, &oe) || oe.Op != "ADD" || oe.Index != 1 {
		t.Fatalf("broadcast add: %v", err)
	}
	b = newMB()
	x = b.act([]int32{1, 2, 2, 2}, 0.1, 0)
	z = b.act([]int32{1, 1, 1, 2}, 0.1, 0)
	b.op(opAveragePool2D, []int32{x}, []int32{z}, optPool2D, optsPool2D(padValid, 1, 1, 2, 2, actRelu))
	b.io([]int32{x}, []int32{z})
	if _, err := Load(b.bytes()); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("pool + relu: %v", err)
	}
	b = newMB()
	x = b.act([]int32{1, 4}, 0.1, 0)
	z = b.act([]int32{1, 4}, 0.1, 0)
	b.op(opSoftmax, []int32{x}, []int32{z}, optSoftmax, optsSoftmax(1))
	b.io([]int32{x}, []int32{z})
	if _, err := Load(b.bytes()); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("softmax output quant: %v", err)
	}
	b = newMB()
	x = b.act([]int32{1, 4}, 0.1, 0)
	z = b.act([]int32{1, 2, 2, 1}, 0.1, 0)
	z2 := b.act([]int32{1, 4}, 1.0/256, -128)
	shape := int32(b.tensor("s", []int32{4}, typeInt32, b.buffer(le32(1, 2, 2, 1)), nil, nil, 0))
	b.op(opReshape, []int32{x, shape}, []int32{z}, optReshape, optsReshape([]int32{1, 2, 2, 1}))
	b.op(opSoftmax, []int32{x}, []int32{z2}, optSoftmax, optsSoftmax(1))
	b.io([]int32{x}, []int32{z})
	if _, err := Load(b.bytes()); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("output not last: %v", err)
	}

	// two subgraphs; malformed bytes never panic
	b = newMB()
	b.newSubgraph()
	if _, err := Load(b.bytes()); err != ErrUnsupported {
		t.Fatalf("two subgraphs: %v", err)
	}
	good := conv(nil)
	if _, err := Load(good); err != nil {
		t.Fatal(err)
	}
	for n := range len(good) {
		if _, err := Load(good[:n]); err == nil {
			t.Fatalf("truncated to %d loaded", n)
		}
	}
	bad := append([]byte(nil), good...)
	copy(bad[4:8], "XXXX")
	if _, err := Load(bad); err != ErrFormat {
		t.Fatalf("bad identifier: %v", err)
	}
}

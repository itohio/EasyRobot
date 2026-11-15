package tfl

import (
	"encoding/binary"
	"math"
	"unsafe"

	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// Load converts the first subgraph of an int8 or uint8 TFLite model into a
// qnet graph that reproduces TFLite's reference kernels: Conv2D, depthwise
// and Add use qi8.RoundTFLite; FullyConnected uses qi8.RoundExact, because
// TFLite's FULLY_CONNECTED kernel rounds once, half away from zero
// (established against the LiteRT interpreter, see qi8/testdata/README.md);
// multipliers are computed from the float32 scales the way TFLite does, and
// fused activation ranges use TFLite's float32 rounding.
//
// uint8 models (TF's "quant" exports: uint8 per-tensor activations and
// asymmetric per-tensor uint8 weights) load as int8 by subtracting 128 from
// every value and zero point, which is exact: the weights' zero point
// becomes Layer.WZP, the uint8 average pool's rounding Layer.TiesUp, and
// multipliers use TFLite's uint8 form (sIn·sW in float32). The caller feeds
// the graph input as x − 128 and reads uint8 outputs as y + 128. uint8 is
// accepted on CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, AVERAGE_POOL_2D,
// MAX_POOL_2D, RESHAPE and SOFTMAX (each checked against the interpreter);
// an operator mixing int8 and uint8 tensors, uint8 on any other operator
// and per-channel uint8 weights are refused with the tensor named.
//
// int8 weights alias b (no copy), so b must stay alive and unchanged while
// the graph is used; it can live in flash. uint8 weights are converted into
// a copy. Bias aliases b when 4-byte aligned on a little-endian machine
// (copied otherwise); requantisation multipliers and the transposed weights
// of a channel-multiplier depthwise layer are allocated.
//
// Supported operators: CONV_2D, DEPTHWISE_CONV_2D (multiplier 1, or an
// input with one channel and any multiplier, which becomes a Conv2D),
// FULLY_CONNECTED, AVERAGE_POOL_2D and MAX_POOL_2D (no fused activation),
// RESHAPE, SOFTMAX, ADD (same shapes), MUL (same shapes, or the smaller
// operand per channel, per pixel (H×W×1) or scalar), LOGISTIC, MEAN (over H and W of a rank-4
// tensor), CONCATENATION (any non-batch axis, inputs quantised as the
// output), PAD and PADV2 (constant paddings, no batch padding),
// RESIZE_NEAREST_NEIGHBOR and RESIZE_BILINEAR (rank 4), LEAKY_RELU,
// TRANSPOSE (rank 2–4, constant permutation keeping the batch axis first),
// QUANTIZE (int8 → int8 becomes a Requantize layer) and DEQUANTIZE. A constant int8 operand of ADD, MUL or
// CONCATENATION becomes a Const layer (not for the first operator). A float
// or uint8 graph input quantised by the first operator (QUANTIZE), and
// DEQUANTIZEs producing graph outputs, and int8 → uint8 QUANTIZEs producing
// graph outputs with the same scale and the zero point + 128, are cut: the
// graph then takes the int8 input quantised as Layers[0].InQ and returns
// int8 outputs quantised as their layers' OutQ; the caller converts (for a
// cut uint8 output: add 128). Tensors must be int8 or uint8
// as above (bias int32) with per-tensor activation quantisation, symmetric
// int8 weights (per-channel or per-tensor) and batch 1; one input and one or more
// outputs (Graph.Outputs in the model's order, nil when the only output is
// the last layer), the last operator's among them. Anything else returns an
// *OpError naming the operator (it wraps ErrUnsupported), or ErrFormat for a
// malformed file.
func Load(b []byte) (qnet.Graph, error) {
	m, err := Open(b)
	if err != nil {
		return qnet.Graph{}, err
	}
	c := converter{m: m, prod: make(map[int]int)}
	return c.run()
}

type converter struct {
	m      *Model
	layers []qnet.Layer
	prod   map[int]int // tensor index → producing layer (−1: graph input)
	op     int
	opName string
}

func (c *converter) fail(reason string) error {
	return &OpError{Index: c.op, Op: c.opName, Reason: reason}
}

func (c *converter) run() (qnet.Graph, error) {
	m := c.m
	f := &m.f
	if m.inputs.n != 1 || m.outputs.n < 1 {
		c.op, c.opName = -1, "SUBGRAPH"
		return qnet.Graph{}, c.fail("need exactly one input and at least one output")
	}
	in := int(f.vecI32(m.inputs, 0))
	c.prod[in] = -1
	for i := range m.ops.n {
		o, err := m.operator(i)
		if err != nil {
			return qnet.Graph{}, err
		}
		c.op, c.opName = i, o.name()
		if err := c.convert(&o); err != nil {
			return qnet.Graph{}, err
		}
	}
	outs := make([]int, m.outputs.n)
	lastIsOut := false
	for k := range outs {
		p, ok := c.prod[int(f.vecI32(m.outputs, k))]
		if !ok || p < 0 {
			c.op, c.opName = -1, "SUBGRAPH"
			return qnet.Graph{}, c.fail("every graph output must be produced by an operator")
		}
		outs[k] = p
		lastIsOut = lastIsOut || p == len(c.layers)-1
	}
	if !lastIsOut {
		c.op, c.opName = -1, "SUBGRAPH"
		return qnet.Graph{}, c.fail("the last operator's output must be a graph output")
	}
	g := qnet.Graph{Layers: c.layers}
	if len(outs) > 1 || outs[0] != len(c.layers)-1 {
		g.Outputs = outs
	}
	if layer, err := g.Check(); err != nil {
		c.op, c.opName = layer, "SUBGRAPH"
		if layer >= 0 {
			c.opName = g.Layers[layer].Kind.String()
		}
		return qnet.Graph{}, c.fail("converted layer is invalid: " + err.Error())
	}
	return g, f.err
}

// isOutput reports whether tensor idx is a graph output.
func (c *converter) isOutput(idx int) bool {
	for k := range c.m.outputs.n {
		if int(c.m.f.vecI32(c.m.outputs, k)) == idx {
			return true
		}
	}
	return false
}

// act is an activation tensor input of an operator.
type act struct {
	shape qi8.Shape
	q     qnet.Quant
	from  int // producing layer (−1: graph input)
}

func (c *converter) tensorAt(o *operator, k int, outputs bool) (tensor, int, error) {
	v := o.inputs
	if outputs {
		v = o.outputs
	}
	if k >= v.n {
		return tensor{}, -1, c.fail("missing operand")
	}
	idx := int(c.m.f.vecI32(v, k))
	if idx < 0 {
		return tensor{}, -1, nil
	}
	t, err := c.m.tensor(idx)
	if err != nil {
		if err == ErrUnsupported {
			return tensor{}, idx, c.fail("tensor with custom quantisation or rank > 8")
		}
		return tensor{}, idx, err
	}
	if t.sparse {
		return tensor{}, idx, c.fail("sparse tensors are not supported")
	}
	return t, idx, nil
}

// shape maps a TFLite shape with batch 1 to NHWC.
func (c *converter) shape(t *tensor) (qi8.Shape, error) {
	s := t.shape
	for _, d := range s {
		if d <= 0 {
			return qi8.Shape{}, c.fail("non-positive dimension")
		}
	}
	switch len(s) {
	case 0:
		return qi8.Shape{H: 1, W: 1, C: 1}, nil
	case 1:
		return qi8.Shape{H: 1, W: 1, C: int(s[0])}, nil
	case 2:
		if s[0] == 1 {
			return qi8.Shape{H: 1, W: 1, C: int(s[1])}, nil
		}
	case 3:
		if s[0] == 1 {
			return qi8.Shape{H: 1, W: int(s[1]), C: int(s[2])}, nil
		}
	case 4:
		if s[0] == 1 {
			return qi8.Shape{H: int(s[1]), W: int(s[2]), C: int(s[3])}, nil
		}
	}
	return qi8.Shape{}, c.fail("batch must be 1 (rank ≤ 4)")
}

// quant reads a per-tensor activation quantisation. uint8 tensors are
// loaded as int8 (value − 128, exact), so their zero point moves by −128.
func (c *converter) quant(t *tensor) (qnet.Quant, error) {
	var off int64
	switch t.typ {
	case typeInt8:
	case typeUInt8:
		off = -128
	default:
		return qnet.Quant{}, c.failT(t, "activations must be int8 or uint8 (float models are not supported)")
	}
	if t.scale.n != 1 || t.zp.n != 1 {
		return qnet.Quant{}, c.failT(t, "activations must have one scale and zero point")
	}
	s := c.m.scaleAt(t, 0)
	zp := c.m.zpAt(t, 0) + off
	if !(s > 0) || math.IsInf(float64(s), 0) || zp < -128 || zp > 127 {
		return qnet.Quant{}, c.failT(t, "invalid activation scale or zero point")
	}
	return qnet.Quant{Scale: s, ZP: int8(zp)}, nil
}

// failT is fail naming tensor t.
func (c *converter) failT(t *tensor, reason string) error {
	return c.fail("tensor \"" + string(t.name) + "\": " + reason)
}

// uint8Ops are the operators whose uint8 TFLite kernels are verified equal
// to qnet on the value − 128 int8 form (official uint8 MobileNetV1
// models and one-operator models against the LiteRT interpreter).
var uint8Ops = map[int32]bool{
	opConv2D: true, opDepthwiseConv2D: true, opFullyConnected: true,
	opAveragePool2D: true, opMaxPool2D: true, opReshape: true, opSoftmax: true,
}

// dtype checks that o's int8/uint8 tensors (activations, weights, constant
// operands) share one element type, and that uint8 appears only on
// operators in uint8Ops. QUANTIZE and DEQUANTIZE change the type; they are
// checked where they are converted.
func (c *converter) dtype(o *operator) error {
	if o.code == opQuantize || o.code == opDequantize {
		return nil
	}
	var first tensor
	have := false
	for _, outputs := range []bool{false, true} {
		n := o.inputs.n
		if outputs {
			n = o.outputs.n
		}
		for k := range n {
			t, idx, err := c.tensorAt(o, k, outputs)
			if err != nil {
				return err
			}
			if idx < 0 || (t.typ != typeInt8 && t.typ != typeUInt8) {
				continue
			}
			if !have {
				first, have = t, true
				continue
			}
			if t.typ != first.typ {
				return c.failT(&t, "mixed int8 and uint8 tensors in one operator")
			}
		}
	}
	if have && first.typ == typeUInt8 && !uint8Ops[o.code] {
		return c.failT(&first, "uint8 is supported only for CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, AVERAGE_POOL_2D, MAX_POOL_2D, RESHAPE and SOFTMAX")
	}
	return nil
}

// input resolves activation input k of o.
func (c *converter) input(o *operator, k int) (act, error) {
	t, idx, err := c.tensorAt(o, k, false)
	if err != nil {
		return act{}, err
	}
	if idx < 0 {
		return act{}, c.fail("missing activation input")
	}
	p, ok := c.prod[idx]
	if !ok {
		return act{}, c.fail("activation input is a constant or not produced yet")
	}
	s, err := c.shape(&t)
	if err != nil {
		return act{}, err
	}
	q, err := c.quant(&t)
	if err != nil {
		return act{}, err
	}
	return act{shape: s, q: q, from: p}, nil
}

// output resolves o's single output.
func (c *converter) output(o *operator) (int, qi8.Shape, qnet.Quant, error) {
	if o.outputs.n != 1 {
		return 0, qi8.Shape{}, qnet.Quant{}, c.fail("need exactly one output")
	}
	t, idx, err := c.tensorAt(o, 0, true)
	if err != nil {
		return 0, qi8.Shape{}, qnet.Quant{}, err
	}
	if idx < 0 {
		return 0, qi8.Shape{}, qnet.Quant{}, ErrFormat
	}
	s, err := c.shape(&t)
	if err != nil {
		return 0, qi8.Shape{}, qnet.Quant{}, err
	}
	q, err := c.quant(&t)
	return idx, s, q, err
}

// emit appends l (reading the producers in from) and records its output.
func (c *converter) emit(l qnet.Layer, outIdx int, from ...int) error {
	n := len(c.layers)
	if len(from) == 1 && from[0] == n-1 && n > 0 {
		l.From = nil // sequential default
	} else if n == 0 && len(from) == 1 && from[0] == -1 {
		l.From = nil
	} else {
		l.From = append([]int(nil), from...)
	}
	if out, err := l.OutShape(); err != nil || out != l.Out {
		return c.fail("output shape does not match the operator's geometry")
	}
	c.layers = append(c.layers, l)
	c.prod[outIdx] = n
	return nil
}

func (c *converter) convert(o *operator) error {
	if len(o.custom) > 0 {
		return c.fail("custom operators are not supported")
	}
	if err := c.dtype(o); err != nil {
		return err
	}
	switch o.code {
	case opConv2D:
		return c.conv(o, false)
	case opDepthwiseConv2D:
		return c.conv(o, true)
	case opFullyConnected:
		return c.fc(o)
	case opAveragePool2D, opMaxPool2D:
		return c.pool(o)
	case opReshape:
		return c.reshape(o)
	case opSoftmax:
		return c.softmax(o)
	case opAdd:
		return c.add(o)
	case opLogistic:
		return c.logistic(o)
	case opMul:
		return c.mul(o)
	case opMean:
		return c.mean(o)
	case opQuantize:
		return c.quantize(o)
	case opDequantize:
		return c.dequantize(o)
	case opConcatenation:
		return c.concat(o)
	case opPad, opPadV2:
		return c.pad(o)
	case opResizeNearest, opResizeBilinear:
		return c.resize(o)
	case opLeakyRelu:
		return c.leakyRelu(o)
	case opTranspose:
		return c.transpose(o)
	}
	return c.fail("operator not supported")
}

// actRange is TFLite's CalculateActivationRangeQuantized for int8.
func (c *converter) actRange(fused uint8, q qnet.Quant) (int8, int8, error) {
	quantize := func(v float32) int32 {
		return int32(q.ZP) + int32(math.Round(float64(v/q.Scale)))
	}
	lo, hi := int32(-128), int32(127)
	switch fused {
	case actNone:
	case actRelu:
		lo = max(lo, quantize(0))
	case actRelu6:
		lo, hi = max(lo, quantize(0)), min(hi, quantize(6))
	case actReluN1To1:
		lo, hi = max(lo, quantize(-1)), min(hi, quantize(1))
	default:
		return 0, 0, c.fail("fused activation not supported")
	}
	if lo > hi {
		return 0, 0, c.fail("empty activation range")
	}
	return int8(lo), int8(hi), nil
}

func padding(p uint8) (qi8.Padding, bool) {
	switch p {
	case padSame:
		return qi8.PaddingSame, true
	case padValid:
		return qi8.PaddingValid, true
	}
	return 0, false
}

// constant returns the int8 data of constant input k and its tensor.
func (c *converter) constant(o *operator, k int) (tensor, []byte, error) {
	t, idx, err := c.tensorAt(o, k, false)
	if err != nil {
		return tensor{}, nil, err
	}
	if idx < 0 {
		return tensor{}, nil, nil
	}
	if _, produced := c.prod[idx]; produced {
		return tensor{}, nil, c.fail("weights or bias must be constant")
	}
	d, err := c.m.bufferData(t.buffer)
	if err != nil {
		if err == ErrUnsupported {
			return tensor{}, nil, c.fail("buffers stored outside the flatbuffer are not supported")
		}
		return tensor{}, nil, err
	}
	return t, d, nil
}

func int8s(b []byte) []int8 {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Slice((*int8)(unsafe.Pointer(&b[0])), len(b))
}

var littleEndian = binary.NativeEndian.Uint16([]byte{1, 0}) == 1

// int32s views (or decodes) little-endian int32 data.
func int32s(b []byte) []int32 {
	n := len(b) / 4
	if n == 0 {
		return nil
	}
	if littleEndian && uintptr(unsafe.Pointer(&b[0]))%4 == 0 {
		return unsafe.Slice((*int32)(unsafe.Pointer(&b[0])), n)
	}
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out
}

// weights reads a weight tensor with n channels along quantised dimension
// dim and returns its data, per-channel scales and zero point. int8 weights
// must be symmetric (per-channel or per-tensor); uint8 weights must be
// per-tensor (asymmetric allowed) and are returned as int8 (value − 128, a
// copy) with their zero point moved by −128.
func (c *converter) weights(o *operator, k int, wantShape []int32, n int, dim int32) ([]int8, []float32, int8, error) {
	t, d, err := c.constant(o, k)
	if err != nil {
		return nil, nil, 0, err
	}
	if d == nil {
		return nil, nil, 0, c.fail("missing weights")
	}
	if t.typ != typeInt8 && t.typ != typeUInt8 {
		return nil, nil, 0, c.failT(&t, "weights must be int8 or uint8")
	}
	if len(t.shape) != len(wantShape) {
		return nil, nil, 0, c.failT(&t, "unexpected weight shape")
	}
	size := 1
	for i, s := range t.shape {
		if s != wantShape[i] {
			return nil, nil, 0, c.failT(&t, "unexpected weight shape")
		}
		size *= int(s)
	}
	if len(d) != size {
		return nil, nil, 0, ErrFormat
	}
	if t.scale.n != 1 && (t.scale.n != n || t.qdim != dim) {
		return nil, nil, 0, c.failT(&t, "weight quantisation must be per-tensor or per output channel")
	}
	if t.zp.n != 0 && t.zp.n != t.scale.n {
		return nil, nil, 0, ErrFormat
	}
	scales := make([]float32, n)
	for ch := range n {
		k := 0
		if t.scale.n > 1 {
			k = ch
		}
		scales[ch] = c.m.scaleAt(&t, k)
	}
	if t.typ == typeInt8 {
		for k := range t.zp.n {
			if c.m.zpAt(&t, k) != 0 {
				return nil, nil, 0, c.failT(&t, "int8 weights must be symmetric (zero point 0)")
			}
		}
		return int8s(d), scales, 0, nil
	}
	if t.scale.n != 1 {
		return nil, nil, 0, c.failT(&t, "per-channel uint8 (asymmetric) weights are not supported")
	}
	var zp int64
	if t.zp.n == 1 {
		zp = c.m.zpAt(&t, 0)
	}
	if zp < 0 || zp > 255 {
		return nil, nil, 0, c.failT(&t, "invalid weight zero point")
	}
	w := make([]int8, len(d))
	for i, v := range d {
		w[i] = int8(v ^ 0x80) // v − 128
	}
	return w, scales, int8(zp - 128), nil
}

// bias reads optional int32 bias input k with n entries.
func (c *converter) bias(o *operator, k, n int) ([]int32, error) {
	if k >= o.inputs.n {
		return nil, nil
	}
	t, d, err := c.constant(o, k)
	if err != nil || d == nil {
		return nil, err
	}
	if t.typ != typeInt32 {
		return nil, c.fail("bias must be int32")
	}
	if len(d) != 4*n {
		return nil, c.fail("bias length does not match the output channels")
	}
	return int32s(d), nil
}

// requants computes TFLite's per-channel multipliers sIn·sW/sOut. Like
// TFLite's QuantizeMultiplier, a multiplier below 2^-32 (or a zero weight
// scale) becomes the zero multiplier: the channel outputs its zero point.
// For uint8 models TFLite forms sIn·sW in float32 before widening
// (GetQuantizedConvolutionMultipler); int8 models widen first.
func (c *converter) requants(sIn float32, sW []float32, sOut float32, uint8Model bool) ([]qi8.Requant, error) {
	rq := make([]qi8.Requant, len(sW))
	for i, w := range sW {
		real := float64(sIn) * float64(w) / float64(sOut)
		if uint8Model {
			real = float64(sIn*w) / float64(sOut)
		}
		if real >= 0 && real < 0x1p-32 {
			continue // zero multiplier
		}
		r, err := qi8.QuantizeMultiplier(real)
		if err != nil {
			return nil, c.fail("requantisation multiplier out of range")
		}
		rq[i] = r
	}
	return rq, nil
}

func (c *converter) conv(o *operator, depthwise bool) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	f := &c.m.f
	var pad, act uint8
	var sw, sh, dw, dh, mult int32
	if depthwise {
		if o.optType != optDepthwiseConv2D {
			return ErrFormat
		}
		pad, sw, sh = f.fieldU8(o.opts, dwPadding, 0), f.fieldI32(o.opts, dwStrideW, 1), f.fieldI32(o.opts, dwStrideH, 1)
		mult, act = f.fieldI32(o.opts, dwMultiplier, 1), f.fieldU8(o.opts, dwAct, 0)
		dw, dh = f.fieldI32(o.opts, dwDilW, 1), f.fieldI32(o.opts, dwDilH, 1)
	} else {
		if o.optType != optConv2D {
			return ErrFormat
		}
		pad, sw, sh = f.fieldU8(o.opts, convPadding, 0), f.fieldI32(o.opts, convStrideW, 1), f.fieldI32(o.opts, convStrideH, 1)
		act = f.fieldU8(o.opts, convAct, 0)
		dw, dh = f.fieldI32(o.opts, convDilW, 1), f.fieldI32(o.opts, convDilH, 1)
	}
	if dw != 1 || dh != 1 {
		return c.fail("dilation is not supported")
	}
	p, ok := padding(pad)
	if !ok {
		return ErrFormat
	}
	wt, _, err := c.tensorAt(o, 1, false)
	if err != nil {
		return err
	}
	if len(wt.shape) != 4 {
		return c.fail("weights must be rank 4")
	}
	kh, kw := int(wt.shape[1]), int(wt.shape[2])
	lo, hi, err := c.actRange(act, oq)
	if err != nil {
		return err
	}
	l := qnet.Layer{
		In: x.shape, InQ: x.q, Out: os, OutQ: oq,
		KH: kh, KW: kw, StrideH: int(sh), StrideW: int(sw), Padding: p,
		ActMin: lo, ActMax: hi, Round: qi8.RoundTFLite,
	}
	var scales []float32
	switch {
	case !depthwise:
		l.Kind = qnet.Conv2D
		l.Weights, scales, l.WZP, err = c.weights(o, 1, []int32{int32(os.C), int32(kh), int32(kw), int32(x.shape.C)}, os.C, 0)
	case os.C == x.shape.C:
		l.Kind = qnet.DepthwiseConv2D
		l.Weights, scales, l.WZP, err = c.weights(o, 1, []int32{1, int32(kh), int32(kw), int32(os.C)}, os.C, 3)
	case x.shape.C == 1 && int(mult) == os.C:
		// Depthwise over one channel with multiplier M is a Conv2D with M
		// outputs: weights [1][kh][kw][M] → OHWI [M][kh][kw][1].
		var w []int8
		w, scales, l.WZP, err = c.weights(o, 1, []int32{1, int32(kh), int32(kw), int32(os.C)}, os.C, 3)
		if err == nil {
			l.Kind = qnet.Conv2D
			l.Weights = make([]int8, len(w))
			for m := range os.C {
				for t := range kh * kw {
					l.Weights[m*kh*kw+t] = w[t*os.C+m]
				}
			}
		}
	default:
		return c.fail("depthwise channel multiplier > 1 on a multi-channel input is not supported")
	}
	if err != nil {
		return err
	}
	if l.Bias, err = c.bias(o, 2, os.C); err != nil {
		return err
	}
	if l.Requant, err = c.requants(x.q.Scale, scales, oq.Scale, wt.typ == typeUInt8); err != nil {
		return err
	}
	return c.emit(l, outIdx, x.from)
}

func (c *converter) fc(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if o.optType != optFullyConnected && o.optType != 0 {
		return ErrFormat
	}
	f := &c.m.f
	if f.fieldU8(o.opts, fcWeightsFmt, 0) != 0 {
		return c.fail("shuffled weight formats are not supported")
	}
	n := os.Len()
	lo, hi, err := c.actRange(f.fieldU8(o.opts, fcAct, 0), oq)
	if err != nil {
		return err
	}
	l := qnet.Layer{
		Kind: qnet.FullyConnected, In: x.shape, InQ: x.q, Out: qi8.Shape{H: 1, W: 1, C: n}, OutQ: oq,
		ActMin: lo, ActMax: hi, Round: qi8.RoundExact, // TFLite's FC (int8 and uint8) rounds once, half away from zero
	}
	var scales []float32
	if l.Weights, scales, l.WZP, err = c.weights(o, 1, []int32{int32(n), int32(x.shape.Len())}, n, 0); err != nil {
		return err
	}
	if l.Bias, err = c.bias(o, 2, n); err != nil {
		return err
	}
	wt, _, err := c.tensorAt(o, 1, false)
	if err != nil {
		return err
	}
	if l.Requant, err = c.requants(x.q.Scale, scales, oq.Scale, wt.typ == typeUInt8); err != nil {
		return err
	}
	if err := c.emit(l, outIdx, x.from); err != nil {
		return err
	}
	if os != l.Out {
		// keep_num_dims or a rank-4 output: reshape to the declared shape
		return c.emit(qnet.Layer{Kind: qnet.Reshape, In: l.Out, InQ: oq, Out: os, OutQ: oq}, outIdx, len(c.layers)-1)
	}
	return nil
}

func (c *converter) pool(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if o.optType != optPool2D {
		return ErrFormat
	}
	f := &c.m.f
	if f.fieldU8(o.opts, poolAct, 0) != actNone {
		return c.fail("fused activation on pooling is not supported")
	}
	if oq != x.q {
		return c.fail("pooling must keep the quantisation")
	}
	p, ok := padding(f.fieldU8(o.opts, poolPadding, 0))
	if !ok {
		return ErrFormat
	}
	kind := qnet.AvgPool2D
	if o.code == opMaxPool2D {
		kind = qnet.MaxPool2D
	}
	xt, _, err := c.tensorAt(o, 0, false)
	if err != nil {
		return err
	}
	l := qnet.Layer{
		Kind: kind, In: x.shape, InQ: x.q, Out: os, OutQ: oq,
		KH: int(f.fieldI32(o.opts, poolFilterH, 1)), KW: int(f.fieldI32(o.opts, poolFilterW, 1)),
		StrideH: int(f.fieldI32(o.opts, poolStrideH, 1)), StrideW: int(f.fieldI32(o.opts, poolStrideW, 1)),
		Padding: p,
		// TFLite's uint8 average pool rounds ties up ((Σu + n/2)/n), its int8 one away from zero.
		TiesUp: kind == qnet.AvgPool2D && xt.typ == typeUInt8,
	}
	return c.emit(l, outIdx, x.from)
}

func (c *converter) reshape(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if oq != x.q {
		return c.fail("reshape must keep the quantisation")
	}
	return c.emit(qnet.Layer{Kind: qnet.Reshape, In: x.shape, InQ: x.q, Out: os, OutQ: oq}, outIdx, x.from)
}

func (c *converter) softmax(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if o.optType != optSoftmax {
		return ErrFormat
	}
	if oq != qnet.SoftmaxOutQuant {
		return c.fail("int8 softmax output must have scale 1/256 and zero point -128")
	}
	sp, err := qi8.QuantizeSoftmax(c.m.f.fieldF32(o.opts, softmaxBeta, 1), x.q.Scale)
	if err != nil {
		return c.fail("softmax scaling out of range")
	}
	if os != x.shape {
		return c.fail("softmax output shape differs from its input")
	}
	return c.emit(qnet.Layer{Kind: qnet.Softmax, In: x.shape, InQ: x.q, Out: os, OutQ: oq, SoftmaxParams: sp}, outIdx, x.from)
}

func (c *converter) add(o *operator) error {
	a, err := c.operand(o, 0)
	if err != nil {
		return err
	}
	b, err := c.operand(o, 1)
	if err != nil {
		return err
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if a.shape != b.shape || os != a.shape {
		return c.fail("broadcasting is not supported")
	}
	if o.optType != optAdd && o.optType != 0 {
		return ErrFormat
	}
	lo, hi, err := c.actRange(c.m.f.fieldU8(o.opts, addAct, 0), oq)
	if err != nil {
		return err
	}
	rq, err := qi8.QuantizeAddTFLite(a.q.Scale, b.q.Scale, oq.Scale)
	if err != nil {
		return c.fail("add multipliers out of range")
	}
	l := qnet.Layer{
		Kind: qnet.Add, In: a.shape, InQ: a.q, In2Q: b.q, Out: os, OutQ: oq,
		ActMin: lo, ActMax: hi, Round: qi8.RoundTFLite, AddTFLite: rq,
	}
	return c.emit(l, outIdx, a.from, b.from)
}

// Rounding of the requantising operators added for face detection and YOLO,
// as TFLite's reference kernels round them (see qi8 SPEC.md).
const (
	roundMul        = qi8.RoundTFLite
	roundMean       = qi8.RoundTFLite
	roundRequantize = qi8.RoundTFLite
	roundLeakyRelu  = qi8.RoundTFLite
)

// operand resolves input k of o like input, except that a constant int8
// tensor becomes a Const layer (one per tensor; its data aliases the model).
func (c *converter) operand(o *operator, k int) (act, error) {
	t, idx, err := c.tensorAt(o, k, false)
	if err != nil {
		return act{}, err
	}
	if idx < 0 {
		return act{}, c.fail("missing activation input")
	}
	if _, ok := c.prod[idx]; ok {
		return c.input(o, k)
	}
	s, err := c.shape(&t)
	if err != nil {
		return act{}, err
	}
	q, err := c.quant(&t)
	if err != nil {
		return act{}, err
	}
	_, d, err := c.constant(o, k)
	if err != nil {
		return act{}, err
	}
	if len(d) != s.Len() {
		return act{}, c.fail("constant operand without data of its shape")
	}
	if len(c.layers) == 0 {
		return act{}, c.fail("a constant operand of the first operator is not supported")
	}
	n := len(c.layers)
	c.layers = append(c.layers, qnet.Layer{Kind: qnet.Const, In: s, Out: s, InQ: q, OutQ: q, Weights: int8s(d)})
	c.prod[idx] = n
	return act{shape: s, q: q, from: n}, nil
}

// ints reads constant integer input k (int32 or int64) of o and its rank.
func (c *converter) ints(o *operator, k int) ([]int64, []int32, error) {
	t, d, err := c.constant(o, k)
	if err != nil {
		return nil, nil, err
	}
	if d == nil && len(t.shape) == 0 {
		return nil, nil, c.fail("missing constant operand")
	}
	var v []int64
	switch t.typ {
	case typeInt32:
		for i := 0; i+4 <= len(d); i += 4 {
			v = append(v, int64(int32(binary.LittleEndian.Uint32(d[i:]))))
		}
	case typeInt64:
		for i := 0; i+8 <= len(d); i += 8 {
			v = append(v, int64(binary.LittleEndian.Uint64(d[i:])))
		}
	default:
		return nil, nil, c.fail("integer operand must be int32 or int64")
	}
	n := 1
	for _, s := range t.shape {
		n *= int(s)
	}
	if len(v) != n {
		return nil, nil, c.fail("integer operand without data of its shape")
	}
	return v, t.shape, nil
}

// rank returns the TFLite rank of activation input k of o.
func (c *converter) rank(o *operator, k int) (int, error) {
	t, _, err := c.tensorAt(o, k, false)
	return len(t.shape), err
}

// nhwcAxis maps axis a of a rank-r batch-1 tensor to the qnet axis (0 = H,
// 1 = W, 2 = C), following shape's mapping of ranks 2–4 to NHWC.
func nhwcAxis(a int64, r int) (int, bool) {
	if a < 0 {
		a += int64(r)
	}
	if r < 2 || r > 4 || a < 1 || a >= int64(r) {
		return 0, false
	}
	return int(a) + 3 - r, true
}

func (c *converter) logistic(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if oq != qnet.LogisticOutQuant {
		return c.fail("int8 logistic output must have scale 1/256 and zero point -128")
	}
	if os != x.shape {
		return c.fail("logistic output shape differs from its input")
	}
	lp, err := qi8.QuantizeLogistic(x.q.Scale, x.q.ZP)
	if err != nil {
		return c.fail("logistic input scaling out of range")
	}
	return c.emit(qnet.Layer{Kind: qnet.Logistic, In: x.shape, InQ: x.q, Out: os, OutQ: oq, LogisticParams: &lp}, outIdx, x.from)
}

func (c *converter) mul(o *operator) error {
	a, err := c.operand(o, 0)
	if err != nil {
		return err
	}
	b, err := c.operand(o, 1)
	if err != nil {
		return err
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if o.optType != optMul && o.optType != 0 {
		return ErrFormat
	}
	if b.shape.Len() > a.shape.Len() {
		a, b = b, a // the product is symmetric; broadcast the second
	}
	bc := b.shape == qi8.Shape{H: 1, W: 1, C: a.shape.C} || b.shape == qi8.Shape{H: 1, W: 1, C: 1}
	perPixel := !bc && b.shape == qi8.Shape{H: a.shape.H, W: a.shape.W, C: 1} && a.shape.C > 1
	if (b.shape != a.shape && !bc && !perPixel) || os != a.shape {
		return c.fail("broadcasting other than per channel, per pixel or scalar is not supported")
	}
	lo, hi, err := c.actRange(c.m.f.fieldU8(o.opts, mulAct, 0), oq)
	if err != nil {
		return err
	}
	rq, err := qi8.QuantizeMul(a.q.Scale, b.q.Scale, oq.Scale)
	if err != nil {
		return c.fail("mul multiplier out of range")
	}
	l := qnet.Layer{
		Kind: qnet.Mul, In: a.shape, InQ: a.q, In2Q: b.q, Out: os, OutQ: oq,
		ActMin: lo, ActMax: hi, Round: roundMul, RQ: rq, PerPixel: perPixel,
	}
	return c.emit(l, outIdx, a.from, b.from)
}

func (c *converter) mean(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	r, err := c.rank(o, 0)
	if err != nil {
		return err
	}
	axes, _, err := c.ints(o, 1)
	if err != nil {
		return err
	}
	var hw [2]bool
	for _, a := range axes {
		ax, ok := nhwcAxis(a, r)
		if !ok || r != 4 || ax == 2 || hw[ax] {
			return c.fail("only the mean over H and W of a rank-4 tensor is supported")
		}
		hw[ax] = true
	}
	if !hw[0] || !hw[1] {
		return c.fail("only the mean over H and W of a rank-4 tensor is supported")
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if o.optType != optReducer && o.optType != 0 {
		return ErrFormat
	}
	rq, err := qi8.QuantizeMean(x.q.Scale, oq.Scale)
	if err != nil {
		return c.fail("mean multiplier out of range")
	}
	l := qnet.Layer{Kind: qnet.Mean, In: x.shape, InQ: x.q, Out: os, OutQ: oq, Round: roundMean, RQ: rq}
	return c.emit(l, outIdx, x.from)
}

// quantize converts QUANTIZE: int8 → int8 is a Requantize layer; a float or
// uint8 graph input quantised by the first operator is cut (the graph then
// takes the int8 tensor as its input).
func (c *converter) quantize(o *operator) error {
	t, idx, err := c.tensorAt(o, 0, false)
	if err != nil {
		return err
	}
	if ot, oi, err := c.tensorAt(o, 0, true); err == nil {
		if oi < 0 {
			return ErrFormat // no output tensor: malformed, as in output()
		}
		if ot.typ == typeUInt8 && t.typ == typeInt8 {
			return c.quantizeOutput(o, &ot, oi)
		}
		if ot.typ != typeInt8 {
			return c.failT(&ot, "QUANTIZE to a non-int8 type is not supported")
		}
	}
	if t.typ == typeFloat32 || t.typ == typeUInt8 {
		if p, ok := c.prod[idx]; !ok || p != -1 || len(c.layers) != 0 {
			return c.fail("only a float or uint8 graph input may be quantised")
		}
		outIdx, _, _, err := c.output(o)
		if err != nil {
			return err
		}
		c.prod[outIdx] = -1
		delete(c.prod, idx)
		return nil
	}
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if os != x.shape {
		return c.fail("quantize output shape differs from its input")
	}
	rq, err := qi8.QuantizeRequantize(x.q.Scale, oq.Scale)
	if err != nil {
		return c.fail("requantisation multiplier out of range")
	}
	l := qnet.Layer{Kind: qnet.Requantize, In: x.shape, InQ: x.q, Out: os, OutQ: oq, Round: roundRequantize, RQ: rq}
	return c.emit(l, outIdx, x.from)
}

// dequantize cuts a DEQUANTIZE of the graph output: the graph returns the
// int8 tensor instead.
// quantizeOutput cuts an int8 → uint8 QUANTIZE that produces a graph output
// with the same scale and the zero point shifted by 128 (a uint8 model
// output): the graph returns the int8 tensor and the caller adds 128.
func (c *converter) quantizeOutput(o *operator, ot *tensor, outIdx int) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	if !c.isOutput(outIdx) {
		return c.failT(ot, "QUANTIZE to uint8 is supported only for a graph output")
	}
	if s, err := c.shape(ot); err != nil || s != x.shape {
		return c.failT(ot, "quantize output shape differs from its input")
	}
	if ot.scale.n != 1 || ot.zp.n != 1 || c.m.scaleAt(ot, 0) != x.q.Scale || c.m.zpAt(ot, 0) != int64(x.q.ZP)+128 {
		return c.failT(ot, "uint8 graph output must keep the scale and shift the zero point by 128")
	}
	c.prod[outIdx] = x.from
	return nil
}

func (c *converter) dequantize(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	if o.outputs.n != 1 {
		return c.fail("need exactly one output")
	}
	t, outIdx, err := c.tensorAt(o, 0, true)
	if err != nil {
		return err
	}
	if outIdx < 0 || t.typ != typeFloat32 || !c.isOutput(outIdx) {
		return c.fail("only a dequantised graph output is supported")
	}
	c.prod[outIdx] = x.from
	return nil
}

func (c *converter) concat(o *operator) error {
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if o.optType != optConcatenation {
		return ErrFormat
	}
	if c.m.f.fieldU8(o.opts, concatAct, 0) != actNone {
		return c.fail("fused activation on concatenation is not supported")
	}
	t, _, err := c.tensorAt(o, 0, true)
	if err != nil {
		return err
	}
	axis, ok := nhwcAxis(int64(c.m.f.fieldI32(o.opts, concatAxis, 0)), len(t.shape))
	if !ok {
		return c.fail("concatenation along the batch axis or of rank > 4 is not supported")
	}
	if o.inputs.n < 1 {
		return c.fail("missing activation input")
	}
	from := make([]int, o.inputs.n)
	var first act
	for k := range o.inputs.n {
		x, err := c.operand(o, k)
		if err != nil {
			return err
		}
		if x.q != oq {
			return c.fail("concatenation inputs must share the output's quantisation")
		}
		if k == 0 {
			first = x
		}
		from[k] = x.from
	}
	l := qnet.Layer{Kind: qnet.Concat, In: first.shape, InQ: first.q, Out: os, OutQ: oq, Axis: axis}
	return c.emit(l, outIdx, from...)
}

func (c *converter) pad(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	r, err := c.rank(o, 0)
	if err != nil {
		return err
	}
	pads, ps, err := c.ints(o, 1)
	if err != nil {
		return err
	}
	if len(ps) != 2 || int(ps[0]) != r || ps[1] != 2 || r < 1 || r > 4 {
		return c.fail("paddings must be [rank, 2] for a rank ≤ 4 input")
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	l := qnet.Layer{Kind: qnet.Pad, In: x.shape, InQ: x.q, Out: os, OutQ: oq, PadValue: oq.ZP}
	for d := range r {
		before, after := pads[2*d], pads[2*d+1]
		if before < 0 || after < 0 || before > qi8.MaxElements || after > qi8.MaxElements {
			return c.fail("negative or oversized padding")
		}
		ax, ok := nhwcAxis(int64(d), r)
		if !ok {
			if before != 0 || after != 0 {
				return c.fail("padding the batch dimension is not supported")
			}
			continue
		}
		l.Pads[2*ax], l.Pads[2*ax+1] = int(before), int(after)
	}
	if o.code == opPadV2 && o.inputs.n > 2 {
		vt, v, err := c.constant(o, 2)
		if err != nil {
			return err
		}
		if vt.typ != typeInt8 || len(v) != 1 {
			return c.fail("PADV2 constant must be one int8 value")
		}
		if vq, err := c.quant(&vt); err != nil || vq != oq {
			return c.fail("PADV2 constant must share the output's quantisation")
		}
		l.PadValue = int8(v[0])
	}
	return c.emit(l, outIdx, x.from)
}

func (c *converter) resize(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	if r, err := c.rank(o, 0); err != nil || r != 4 {
		return c.fail("resize input must be rank 4")
	}
	size, _, err := c.ints(o, 1)
	if err != nil {
		return err
	}
	if len(size) != 2 {
		return c.fail("resize size must have two entries")
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	kind, optT, alignSlot, halfSlot := qnet.ResizeNearest, uint8(optResizeNearest), resizeAlignCorners, resizeHalfPixel
	if o.code == opResizeBilinear {
		kind, optT, alignSlot, halfSlot = qnet.ResizeBilinear, optResizeBilinear, resizeBilinearAlignCorners, resizeBilinearHalfPixel
	}
	if o.optType != optT && o.optType != 0 {
		return ErrFormat
	}
	if int64(os.H) != size[0] || int64(os.W) != size[1] {
		return c.fail("resize output shape differs from its size operand")
	}
	f := &c.m.f
	l := qnet.Layer{
		Kind: kind, In: x.shape, InQ: x.q, Out: os, OutQ: oq,
		AlignCorners: f.fieldU8(o.opts, alignSlot, 0) != 0,
		HalfPixel:    f.fieldU8(o.opts, halfSlot, 0) != 0,
	}
	return c.emit(l, outIdx, x.from)
}

// transpose converts TRANSPOSE of a rank 2–4 tensor with a constant
// permutation that keeps the batch axis first. Tensor axis a of a rank-r
// tensor is qnet axis a+3−r (the leading qnet axes of a lower rank have
// size 1 and stay in place).
func (c *converter) transpose(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	r, err := c.rank(o, 0)
	if err != nil || r < 2 || r > 4 {
		return c.fail("transpose input must be rank 2 to 4")
	}
	perm, _, err := c.ints(o, 1)
	if err != nil {
		return err
	}
	if len(perm) != r || perm[0] != 0 {
		return c.fail("transpose permutation must keep the batch axis first")
	}
	l := qnet.Layer{Kind: qnet.Transpose, In: x.shape, InQ: x.q, Perm: [3]int{0, 1, 2}}
	for a := 1; a < r; a++ {
		if perm[a] < 1 || perm[a] >= int64(r) {
			return c.fail("transpose permutation out of range")
		}
		l.Perm[a+3-r] = int(perm[a]) + 3 - r
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if oq != x.q {
		return c.fail("transpose must keep the quantisation")
	}
	l.Out, l.OutQ = os, oq
	return c.emit(l, outIdx, x.from) // emit checks Perm (a permutation) against Out
}

func (c *converter) leakyRelu(o *operator) error {
	x, err := c.input(o, 0)
	if err != nil {
		return err
	}
	outIdx, os, oq, err := c.output(o)
	if err != nil {
		return err
	}
	if o.optType != optLeakyRelu && o.optType != 0 {
		return ErrFormat
	}
	if os != x.shape {
		return c.fail("leaky relu output shape differs from its input")
	}
	id, alpha, err := qi8.QuantizeLeakyReLU(x.q.Scale, oq.Scale, c.m.f.fieldF32(o.opts, leakyReluAlpha, 0))
	if err != nil {
		return c.fail("leaky relu alpha or multipliers out of range")
	}
	l := qnet.Layer{Kind: qnet.LeakyReLU, In: x.shape, InQ: x.q, Out: os, OutQ: oq, Round: roundLeakyRelu, RQ: id, RQAlpha: alpha}
	return c.emit(l, outIdx, x.from)
}

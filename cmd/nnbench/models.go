package main

import (
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// spec is one benchmarked network.
type spec struct {
	name, note    string
	build         func() (qnet.Graph, error)
	wSeed, inSeed uint64
}

var specs = []spec{
	{
		name:  "TinyCNN",
		note:  "32x32x3: 3x(conv3x3/1 SAME + ReLU6, maxpool2) 16/32/64ch, global avgpool, FC->10",
		build: tinyCNN, wSeed: 1, inSeed: 101,
	},
	{
		name: "MobileNetV1-0.25",
		note: "96x96x3 feature extractor (qnet.MobileNetV1, classes 0), output 1x1x256",
		build: func() (qnet.Graph, error) {
			return qnet.MobileNetV1(0.25, 96, 96, 0, make([]qnet.Layer, qnet.MobileNetV1Layers(0)))
		},
		wSeed: 2, inSeed: 102,
	},
	{
		name:  "DS-CNN-style",
		note:  "64x64x3: conv3x3/2->32, 4x(dw3x3/1 + pw1x1->64), global avgpool, FC->12 (depthwise-separable)",
		build: dsCNN, wSeed: 3, inSeed: 103,
	},
	{
		name:  "ResNet-8",
		note:  "32x32x3 (MLPerf Tiny image classification shape): conv3x3->16, 3 residual stacks 16/32/64 (2x conv3x3, stride 2 + 1x1 projection shortcut in stacks 2-3, requantised Add + ReLU6), global avgpool, FC->10",
		build: resNet8, wSeed: 4, inSeed: 104,
	},
}

// Quantisation of residual-branch outputs (no activation) and projection
// shortcuts; deliberately different from qnet.MobileNetActQuant so that every
// Add rescales both inputs.
var (
	branchQuant   = qnet.Quant{Scale: 0.05, ZP: 3}
	shortcutQuant = qnet.Quant{Scale: 0.04, ZP: -5}
)

// graphBuilder appends layers, each reading the current layer (by default
// the previous one; −1 is the graph input). The first error sticks.
type graphBuilder struct {
	layers []qnet.Layer
	cur    int
	in     qi8.Shape
	err    error
}

func newBuilder(in qi8.Shape) *graphBuilder {
	return &graphBuilder{cur: -1, in: in}
}

// out returns the shape and quantisation produced by layer i (−1 = input).
func (b *graphBuilder) out(i int) (qi8.Shape, qnet.Quant) {
	if i < 0 {
		return b.in, qnet.MobileNetInputQuant
	}
	return b.layers[i].Out, b.layers[i].OutQ
}

// add appends l reading b.cur (From is set only when that is not the
// previous layer).
func (b *graphBuilder) add(l qnet.Layer) {
	if b.err != nil {
		return
	}
	l.In, l.InQ = b.out(b.cur)
	if b.cur != len(b.layers)-1 {
		l.From = []int{b.cur}
	}
	b.push(l)
}

func (b *graphBuilder) push(l qnet.Layer) {
	out, err := l.OutShape()
	if err != nil {
		b.err = err
		return
	}
	l.Out = out
	b.layers = append(b.layers, l)
	b.cur = len(b.layers) - 1
}

// conv adds a SAME-padded square Conv2D (kind qnet.Conv2D) or depthwise
// layer with fused ReLU6, output quantised as qnet.MobileNetActQuant.
func (b *graphBuilder) conv(kind qnet.Kind, outC, k, stride int) {
	act := qnet.MobileNetActQuant
	lo, hi := qi8.ActReLU6(act.Scale, act.ZP)
	b.convQ(kind, outC, k, stride, act, lo, hi)
}

// convLinear adds a SAME-padded Conv2D without activation, output quantised
// as q.
func (b *graphBuilder) convLinear(outC, k, stride int, q qnet.Quant) {
	lo, hi := qi8.ActNone()
	b.convQ(qnet.Conv2D, outC, k, stride, q, lo, hi)
}

func (b *graphBuilder) convQ(kind qnet.Kind, outC, k, stride int, q qnet.Quant, lo, hi int8) {
	b.add(qnet.Layer{
		Kind: kind, Out: qi8.Shape{C: outC}, OutQ: q,
		KH: k, KW: k, StrideH: stride, StrideW: stride, Padding: qi8.PaddingSame,
		ActMin: lo, ActMax: hi,
	})
}

// addReLU6 adds the requantised sum of layers x and y (equal shapes) with a
// fused ReLU6, output quantised as qnet.MobileNetActQuant.
func (b *graphBuilder) addReLU6(x, y int) {
	if b.err != nil {
		return
	}
	sx, qx := b.out(x)
	_, qy := b.out(y)
	q := qnet.MobileNetActQuant
	rq, err := qi8.QuantizeAddMultipliers(float64(qx.Scale)/float64(q.Scale), float64(qy.Scale)/float64(q.Scale))
	if err != nil {
		b.err = err
		return
	}
	lo, hi := qi8.ActReLU6(q.Scale, q.ZP)
	b.push(qnet.Layer{
		Kind: qnet.Add, From: []int{x, y}, In: sx, InQ: qx, In2Q: qy, OutQ: q,
		ActMin: lo, ActMax: hi, AddRQ: rq,
	})
}

func (b *graphBuilder) maxPool(k int) {
	_, q := b.out(b.cur)
	b.add(qnet.Layer{Kind: qnet.MaxPool2D, OutQ: q, KH: k, KW: k, StrideH: k, StrideW: k, Padding: qi8.PaddingValid})
}

func (b *graphBuilder) globalAvgPool() {
	s, q := b.out(b.cur)
	b.add(qnet.Layer{Kind: qnet.AvgPool2D, OutQ: q, KH: s.H, KW: s.W, StrideH: 1, StrideW: 1, Padding: qi8.PaddingValid})
}

func (b *graphBuilder) fc(out int) {
	lo, hi := qi8.ActNone()
	b.add(qnet.Layer{Kind: qnet.FullyConnected, Out: qi8.Shape{C: out}, OutQ: qnet.MobileNetLogitQuant, ActMin: lo, ActMax: hi})
}

func (b *graphBuilder) graph() (qnet.Graph, error) {
	return qnet.Graph{Layers: b.layers}, b.err
}

func tinyCNN() (qnet.Graph, error) {
	b := newBuilder(qi8.Shape{H: 32, W: 32, C: 3})
	for _, c := range []int{16, 32, 64} {
		b.conv(qnet.Conv2D, c, 3, 1)
		b.maxPool(2)
	}
	b.globalAvgPool()
	b.fc(10)
	return b.graph()
}

func dsCNN() (qnet.Graph, error) {
	b := newBuilder(qi8.Shape{H: 64, W: 64, C: 3})
	b.conv(qnet.Conv2D, 32, 3, 2)
	for range 4 {
		b.conv(qnet.DepthwiseConv2D, 0, 3, 1)
		b.conv(qnet.Conv2D, 64, 1, 1)
	}
	b.globalAvgPool()
	b.fc(12)
	return b.graph()
}

// resNet8 is the MLPerf Tiny ResNet-8 topology (batch norm folded into the
// convolutions; ReLU6 instead of ReLU, as the activation quantisation is
// qnet.MobileNetActQuant).
func resNet8() (qnet.Graph, error) {
	b := newBuilder(qi8.Shape{H: 32, W: 32, C: 3})
	b.conv(qnet.Conv2D, 16, 3, 1)
	for i, c := range []int{16, 32, 64} {
		x := b.cur
		stride := 1
		if i > 0 {
			stride = 2
		}
		b.conv(qnet.Conv2D, c, 3, stride)
		b.convLinear(c, 3, 1, branchQuant)
		y := b.cur
		if i > 0 {
			b.cur = x
			b.convLinear(c, 1, 2, shortcutQuant)
			x = b.cur
		}
		b.addReLU6(y, x)
	}
	b.globalAvgPool()
	b.fc(10)
	return b.graph()
}

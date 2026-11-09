package qnet

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// Kind is the operation of a Layer.
type Kind uint8

// Layer kinds. The zero value is invalid.
const (
	// Conv2D is a 2-D convolution (weights OHWI, per-output-channel requant).
	Conv2D Kind = iota + 1
	// DepthwiseConv2D is a depthwise convolution with channel multiplier 1.
	DepthwiseConv2D
	// FullyConnected is a GEMV over the flattened input (Out is 1×1×N).
	FullyConnected
	// MaxPool2D is spatial max pooling (In/Out quantisation equal).
	MaxPool2D
	// AvgPool2D is spatial average pooling (In/Out quantisation equal).
	// Global average pooling is AvgPool2D with KH×KW = In.H×In.W, VALID.
	AvgPool2D
	// ReLU is a standalone max(x, zp) (In/Out quantisation equal).
	ReLU
	// ReLU6 is a standalone clamp(x, zp, Quantize6(scale, zp)).
	ReLU6
)

// String returns the kind name.
func (k Kind) String() string {
	switch k {
	case Conv2D:
		return "conv2d"
	case DepthwiseConv2D:
		return "dwconv2d"
	case FullyConnected:
		return "fc"
	case MaxPool2D:
		return "maxpool2d"
	case AvgPool2D:
		return "avgpool2d"
	case ReLU:
		return "relu"
	case ReLU6:
		return "relu6"
	}
	return "invalid"
}

// Quant is the asymmetric quantisation of an activation tensor:
// real = Scale·(q − ZP).
type Quant struct {
	Scale float32
	ZP    int8
}

func (q Quant) valid() bool {
	s := float64(q.Scale)
	return s > 0 && !math.IsInf(s, 0)
}

// Layer is one operation of a Graph. It is plain data so that an accelerator
// backend can inspect and compile it; the qi8 parameter structs are derived
// from it (Conv2DParams, DepthwiseParams, FCParams, PoolParams).
//
// Field use per Kind:
//
//	Conv2D, DepthwiseConv2D: geometry, ActMin/ActMax, Weights, Bias, Requant
//	FullyConnected:          ActMin/ActMax, Weights, Bias, Requant
//	MaxPool2D, AvgPool2D:    geometry
//	ReLU, ReLU6:             none (zero point and scale from InQ)
//
// Out.C is the output channel count of Conv2D and FullyConnected; every other
// part of Out is implied by In and the geometry (see OutShape).
type Layer struct {
	Kind Kind

	In, Out   qi8.Shape // NHWC, batch 1
	InQ, OutQ Quant

	// Spatial geometry (Conv2D, DepthwiseConv2D, MaxPool2D, AvgPool2D).
	KH, KW           int
	StrideH, StrideW int
	Padding          qi8.Padding

	// Fused activation clamp (Conv2D, DepthwiseConv2D, FullyConnected).
	// The zero value [0, 0] clamps everything to 0.
	ActMin, ActMax int8

	// Weights: Conv2D OHWI [Out.C][KH][KW][In.C]; DepthwiseConv2D [KH][KW][C];
	// FullyConnected [Out.C][In.Len()]. Symmetric int8.
	Weights []int8
	// Bias is nil or one int32 per output channel (accumulator scale).
	Bias []int32
	// Requant has one entry per output channel.
	Requant []qi8.Requant
}

// Conv2DParams returns the qi8 parameters of a Conv2D layer.
func (l *Layer) Conv2DParams() qi8.Conv2DParams {
	return qi8.Conv2DParams{
		In: l.In, OutC: l.Out.C,
		KH: l.KH, KW: l.KW, StrideH: l.StrideH, StrideW: l.StrideW, Padding: l.Padding,
		InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, ActMin: l.ActMin, ActMax: l.ActMax,
	}
}

// DepthwiseParams returns the qi8 parameters of a DepthwiseConv2D layer.
func (l *Layer) DepthwiseParams() qi8.DepthwiseConv2DParams {
	return qi8.DepthwiseConv2DParams{
		In: l.In,
		KH: l.KH, KW: l.KW, StrideH: l.StrideH, StrideW: l.StrideW, Padding: l.Padding,
		InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, ActMin: l.ActMin, ActMax: l.ActMax,
	}
}

// FCParams returns the qi8 parameters of a FullyConnected layer.
func (l *Layer) FCParams() qi8.FCParams {
	return qi8.FCParams{
		In: l.In.Len(), Out: l.Out.C,
		InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, ActMin: l.ActMin, ActMax: l.ActMax,
	}
}

// PoolParams returns the qi8 parameters of a MaxPool2D or AvgPool2D layer.
func (l *Layer) PoolParams() qi8.Pool2DParams {
	return qi8.Pool2DParams{
		In: l.In,
		KH: l.KH, KW: l.KW, StrideH: l.StrideH, StrideW: l.StrideW, Padding: l.Padding,
	}
}

// OutShape returns the output shape implied by Kind, In, the geometry and
// (for Conv2D and FullyConnected) Out.C. Validate requires Out to equal it.
func (l *Layer) OutShape() (qi8.Shape, error) {
	var (
		s   qi8.Shape
		err error
	)
	switch l.Kind {
	case Conv2D:
		s, err = l.Conv2DParams().OutShape()
	case DepthwiseConv2D:
		s, err = l.DepthwiseParams().OutShape()
	case FullyConnected:
		if l.In.H <= 0 || l.In.W <= 0 || l.In.C <= 0 || l.Out.C <= 0 {
			return qi8.Shape{}, ErrShape
		}
		return qi8.Shape{H: 1, W: 1, C: l.Out.C}, nil
	case MaxPool2D, AvgPool2D:
		s, err = l.PoolParams().OutShape()
	case ReLU, ReLU6:
		if l.In.H <= 0 || l.In.W <= 0 || l.In.C <= 0 {
			return qi8.Shape{}, ErrShape
		}
		return l.In, nil
	default:
		return qi8.Shape{}, ErrKind
	}
	if err != nil {
		return qi8.Shape{}, ErrShape
	}
	return s, nil
}

// ParamCounts returns the number of weights, bias and requant entries the
// layer needs (all zero for parameter-free kinds). Bias may also be nil.
func (l *Layer) ParamCounts() (weights, bias, requant int) {
	switch l.Kind {
	case Conv2D:
		return l.Out.C * l.KH * l.KW * l.In.C, l.Out.C, l.Out.C
	case DepthwiseConv2D:
		return l.KH * l.KW * l.In.C, l.In.C, l.In.C
	case FullyConnected:
		return l.Out.C * l.In.Len(), l.Out.C, l.Out.C
	}
	return 0, 0, 0
}

// hasParams reports whether the kind carries weights/bias/requant.
func (k Kind) hasParams() bool {
	return k == Conv2D || k == DepthwiseConv2D || k == FullyConnected
}

// RunLayer executes one layer with the qi8 kernels: dst has Out.Len()
// elements, src In.Len(). Errors are qnet or qi8 sentinels. It does not
// allocate. Backends may use it as their fallback.
func RunLayer(l *Layer, dst, src []int8) error {
	switch l.Kind {
	case Conv2D:
		return qi8.Conv2D(dst, src, l.Weights, l.Bias, l.Requant, l.Conv2DParams())
	case DepthwiseConv2D:
		return qi8.DepthwiseConv2D(dst, src, l.Weights, l.Bias, l.Requant, l.DepthwiseParams())
	case FullyConnected:
		return qi8.FullyConnected(dst, src, l.Weights, l.Bias, l.Requant, l.FCParams())
	case MaxPool2D:
		return qi8.MaxPool2D(dst, src, l.PoolParams())
	case AvgPool2D:
		return qi8.AvgPool2D(dst, src, l.PoolParams())
	case ReLU:
		return qi8.ReLU(dst, src, l.InQ.ZP)
	case ReLU6:
		return qi8.ReLU6(dst, src, l.InQ.ZP, qi8.Quantize6(l.InQ.Scale, l.InQ.ZP))
	}
	return ErrKind
}

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
	// Add is the requantised element-wise sum of two producers (From has two
	// entries; In2Q is the second input's quantisation).
	Add
	// Reshape changes the shape only (Out given, same length and
	// quantisation); its output aliases its input.
	Reshape
	// Softmax is TFLite's int8 softmax over C (OutQ must be {1/256, −128}).
	Softmax
	// Logistic is TFLite's int8 sigmoid (OutQ must be LogisticOutQuant).
	Logistic
	// Mul is the requantised element-wise product of two producers; the
	// second may be broadcast (its shape equal to In, 1×1×In.C or 1×1×1).
	Mul
	// Mean averages over H and W (Out is 1×1×C), requantising to OutQ.
	Mean
	// Requantize changes the quantisation (TFLite QUANTIZE int8 → int8).
	Requantize
	// Concat joins its producers (From has one or more entries) along Axis;
	// all inputs and the output share one quantisation.
	Concat
	// Pad pads H, W and C with PadValue; values are copied unchanged, as
	// TFLite does even when OutQ differs from InQ.
	Pad
	// ResizeNearest is nearest-neighbour resizing of H and W to Out.H×Out.W;
	// values are copied unchanged (any OutQ, as TFLite).
	ResizeNearest
	// Const is a constant tensor: Weights holds its Out.Len() values, In and
	// InQ equal Out and OutQ; it reads nothing and must not be layer 0.
	Const
	// LeakyReLU is TFLite's int8 LEAKY_RELU: x for x ≥ zp, alpha·x below,
	// requantised to OutQ (RQ for the identity branch, RQAlpha for alpha).
	LeakyReLU
	// ResizeBilinear is TFLite's integer bilinear resize of H and W to
	// Out.H×Out.W on raw values (10-bit fixed-point weights).
	ResizeBilinear
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
	case Add:
		return "add"
	case Reshape:
		return "reshape"
	case Softmax:
		return "softmax"
	case Logistic:
		return "logistic"
	case Mul:
		return "mul"
	case Mean:
		return "mean"
	case Requantize:
		return "requantize"
	case Concat:
		return "concat"
	case Pad:
		return "pad"
	case ResizeNearest:
		return "resize_nearest"
	case Const:
		return "const"
	case LeakyReLU:
		return "leaky_relu"
	case ResizeBilinear:
		return "resize_bilinear"
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

// SoftmaxOutQuant is the output quantisation TFLite requires of int8 softmax.
var SoftmaxOutQuant = Quant{Scale: 1.0 / 256, ZP: -128}

// LogisticOutQuant is the output quantisation TFLite requires of int8
// logistic.
var LogisticOutQuant = Quant{Scale: 1.0 / 256, ZP: -128}

// Layer is one operation of a Graph. It is plain data so that an accelerator
// backend can inspect and compile it; the qi8 parameter structs are derived
// from it (Conv2DParams, DepthwiseParams, FCParams, PoolParams, AddParams,
// MulParams, MeanParams, RequantizeParams, PadParams, ResizeParams).
//
// Field use per Kind:
//
//	Conv2D, DepthwiseConv2D: geometry, ActMin/ActMax, Weights, Bias, Requant, Round
//	FullyConnected:          ActMin/ActMax, Weights, Bias, Requant, Round
//	MaxPool2D, AvgPool2D:    geometry
//	ReLU, ReLU6:             none (zero point and scale from InQ)
//	Add:                     From (two producers), In2Q, ActMin/ActMax, Round, AddRQ or AddTFLite
//	Mul:                     From (two producers), In2Q, ActMin/ActMax, Round, RQ
//	Mean, Requantize:        Round, RQ
//	Reshape:                 Out (any shape of the same length)
//	Softmax:                 SoftmaxParams
//	Logistic:                LogisticParams
//	Concat:                  From (one or more producers), Axis, Out
//	Pad:                     Pads, PadValue
//	ResizeNearest:           Out.H, Out.W, AlignCorners, HalfPixel
//	ResizeBilinear:          Out.H, Out.W, AlignCorners, HalfPixel
//	Const:                   Out, Weights (the data)
//	LeakyReLU:               Round, RQ (identity), RQAlpha
//
// Out.C is the output channel count of Conv2D and FullyConnected; Out of
// Reshape, Concat, Const and (H, W) of the resizes are given; every other
// part of Out is implied by In and the geometry (see OutShape).
type Layer struct {
	Kind Kind

	// From lists the producing layers (indices below this one; −1 is the
	// graph input). nil means the previous layer (the graph input for layer
	// 0). Single-input kinds take at most one entry, Add and Mul exactly two,
	// Concat one or more, Const none (nil).
	From []int

	In, Out   qi8.Shape // NHWC, batch 1
	InQ, OutQ Quant
	// In2Q is the quantisation of Add's or Mul's second input.
	In2Q Quant

	// Spatial geometry (Conv2D, DepthwiseConv2D, MaxPool2D, AvgPool2D).
	KH, KW           int
	StrideH, StrideW int
	Padding          qi8.Padding

	// Fused activation clamp (Conv2D, DepthwiseConv2D, FullyConnected, Add,
	// Mul). The zero value [0, 0] clamps everything to 0.
	ActMin, ActMax int8

	// Round is the requantisation rounding (Conv2D, DepthwiseConv2D,
	// FullyConnected, Add, Mul, Mean, Requantize): qi8.RoundExact (zero
	// value) or qi8.RoundTFLite.
	Round qi8.Rounding

	// Weights: Conv2D OHWI [Out.C][KH][KW][In.C]; DepthwiseConv2D [KH][KW][C];
	// FullyConnected [Out.C][In.Len()] (symmetric int8); Const its data.
	Weights []int8
	// Bias is nil or one int32 per output channel (accumulator scale).
	Bias []int32
	// Requant has one entry per output channel.
	Requant []qi8.Requant

	// AddRQ (Round == RoundExact) or AddTFLite (RoundTFLite) are Add's
	// multipliers.
	AddRQ     qi8.AddRequant
	AddTFLite qi8.AddRequantTFLite

	// SoftmaxParams is Softmax's prepared input scaling.
	SoftmaxParams qi8.SoftmaxParams
	// LogisticParams is Logistic's table (by pointer: it is 256 bytes and
	// every Layer would carry it otherwise).
	LogisticParams *qi8.LogisticParams

	// RQ is the multiplier of Mul (sA·sB/sOut), Mean (from qi8.QuantizeMean),
	// Requantize (sIn/sOut) and LeakyReLU's identity branch; RQAlpha is
	// LeakyReLU's alpha branch (both from qi8.QuantizeLeakyReLU).
	RQ, RQAlpha qi8.Requant

	// Axis is Concat's axis: 0 = H, 1 = W, 2 = C.
	Axis int
	// Pads is Pad's top, bottom, left, right, front (C) and back (C)
	// padding; PadValue fills it.
	Pads     [6]int
	PadValue int8
	// AlignCorners and HalfPixel select the resizes' index mapping
	// (TFLite's align_corners and half_pixel_centers).
	AlignCorners, HalfPixel bool
}

// Conv2DParams returns the qi8 parameters of a Conv2D layer.
func (l *Layer) Conv2DParams() qi8.Conv2DParams {
	return qi8.Conv2DParams{
		In: l.In, OutC: l.Out.C,
		KH: l.KH, KW: l.KW, StrideH: l.StrideH, StrideW: l.StrideW, Padding: l.Padding,
		InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, ActMin: l.ActMin, ActMax: l.ActMax, Round: l.Round,
	}
}

// DepthwiseParams returns the qi8 parameters of a DepthwiseConv2D layer.
func (l *Layer) DepthwiseParams() qi8.DepthwiseConv2DParams {
	return qi8.DepthwiseConv2DParams{
		In: l.In,
		KH: l.KH, KW: l.KW, StrideH: l.StrideH, StrideW: l.StrideW, Padding: l.Padding,
		InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, ActMin: l.ActMin, ActMax: l.ActMax, Round: l.Round,
	}
}

// FCParams returns the qi8 parameters of a FullyConnected layer.
func (l *Layer) FCParams() qi8.FCParams {
	return qi8.FCParams{
		In: l.In.Len(), Out: l.Out.C,
		InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, ActMin: l.ActMin, ActMax: l.ActMax, Round: l.Round,
	}
}

// PoolParams returns the qi8 parameters of a MaxPool2D or AvgPool2D layer.
func (l *Layer) PoolParams() qi8.Pool2DParams {
	return qi8.Pool2DParams{
		In: l.In,
		KH: l.KH, KW: l.KW, StrideH: l.StrideH, StrideW: l.StrideW, Padding: l.Padding,
	}
}

// AddParams returns the qi8 parameters of an Add layer.
func (l *Layer) AddParams() qi8.AddParams {
	return qi8.AddParams{
		ZPA: l.InQ.ZP, ZPB: l.In2Q.ZP, ZPOut: l.OutQ.ZP,
		RQ: l.AddRQ, Min: l.ActMin, Max: l.ActMax, Round: l.Round, TFLite: l.AddTFLite,
	}
}

// MulParams returns the qi8 parameters of a Mul layer.
func (l *Layer) MulParams() qi8.MulParams {
	return qi8.MulParams{
		ZPA: l.InQ.ZP, ZPB: l.In2Q.ZP, ZPOut: l.OutQ.ZP,
		RQ: l.RQ, Min: l.ActMin, Max: l.ActMax, Round: l.Round,
	}
}

// MeanParams returns the qi8 parameters of a Mean layer.
func (l *Layer) MeanParams() qi8.MeanParams {
	return qi8.MeanParams{InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, RQ: l.RQ, Round: l.Round}
}

// RequantizeParams returns the qi8 parameters of a Requantize layer.
func (l *Layer) RequantizeParams() qi8.RequantizeParams {
	return qi8.RequantizeParams{InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, RQ: l.RQ, Round: l.Round}
}

// PadParams returns the qi8 parameters of a Pad layer.
func (l *Layer) PadParams() qi8.PadParams {
	p := l.Pads
	return qi8.PadParams{In: l.In, Top: p[0], Bottom: p[1], Left: p[2], Right: p[3], Front: p[4], Back: p[5], Value: l.PadValue}
}

// ResizeParams returns the qi8 parameters of a ResizeNearest or
// ResizeBilinear layer.
func (l *Layer) ResizeParams() qi8.ResizeParams {
	return qi8.ResizeParams{In: l.In, OutH: l.Out.H, OutW: l.Out.W, AlignCorners: l.AlignCorners, HalfPixel: l.HalfPixel}
}

// LeakyReLUParams returns the qi8 parameters of a LeakyReLU layer.
func (l *Layer) LeakyReLUParams() qi8.LeakyReLUParams {
	return qi8.LeakyReLUParams{InZP: l.InQ.ZP, OutZP: l.OutQ.ZP, Identity: l.RQ, Alpha: l.RQAlpha, Round: l.Round}
}

// concatOuter is the number of rows Concat interleaves (the product of the
// dimensions before Axis).
func (l *Layer) concatOuter() int {
	switch l.Axis {
	case 0:
		return 1
	case 1:
		return l.In.H
	}
	return l.In.H * l.In.W
}

func positive(s qi8.Shape) bool { return s.H > 0 && s.W > 0 && s.C > 0 }

// OutShape returns the output shape implied by Kind, In, the geometry and
// (for Conv2D and FullyConnected) Out.C; for Reshape it is Out when that has
// In's length; for Concat and Const it is Out when positive (Check verifies
// Concat against its producers); for ResizeNearest it is Out.H×Out.W×In.C.
// Validate requires Out to equal it.
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
		if !positive(l.In) || l.Out.C <= 0 {
			return qi8.Shape{}, ErrShape
		}
		return qi8.Shape{H: 1, W: 1, C: l.Out.C}, nil
	case MaxPool2D, AvgPool2D:
		s, err = l.PoolParams().OutShape()
	case ReLU, ReLU6, Add, Softmax, Logistic, Mul, Requantize, LeakyReLU:
		if !positive(l.In) {
			return qi8.Shape{}, ErrShape
		}
		return l.In, nil
	case Mean:
		if !positive(l.In) {
			return qi8.Shape{}, ErrShape
		}
		return qi8.Shape{H: 1, W: 1, C: l.In.C}, nil
	case Reshape:
		if !positive(l.In) || !positive(l.Out) || l.Out.Len() != l.In.Len() {
			return qi8.Shape{}, ErrShape
		}
		return l.Out, nil
	case Concat:
		if !positive(l.In) || !positive(l.Out) || l.Axis < 0 || l.Axis > 2 {
			return qi8.Shape{}, ErrShape
		}
		return l.Out, nil
	case Const:
		if !positive(l.Out) {
			return qi8.Shape{}, ErrShape
		}
		return l.Out, nil
	case Pad:
		s, err = l.PadParams().OutShape()
	case ResizeNearest, ResizeBilinear:
		if !positive(l.In) || l.Out.H <= 0 || l.Out.W <= 0 {
			return qi8.Shape{}, ErrShape
		}
		// ResizeBilinear also rejects align_corners with half_pixel_centers
		// and upscales TFLite would read past the input for; qi8 checks the
		// geometry before the (nil) buffer lengths.
		if l.Kind == ResizeBilinear && qi8.ResizeBilinear(nil, nil, l.ResizeParams()) != qi8.ErrLength {
			return qi8.Shape{}, ErrShape
		}
		return qi8.Shape{H: l.Out.H, W: l.Out.W, C: l.In.C}, nil
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

// inputs returns how many producers the kind reads (−1: any number ≥ 1).
func (k Kind) inputs() int {
	switch k {
	case Add, Mul:
		return 2
	case Concat:
		return -1
	case Const:
		return 0
	}
	return 1
}

// RunLayer executes one layer with the qi8 reference (free-function)
// kernels: dst has Out.Len() elements; srcs are the producers' outputs in
// From order (none for Const, which copies its data; Reshape copies). Errors
// are qnet or qi8 sentinels. It does not allocate. It is the reference the
// Runner is tested against.
func RunLayer(l *Layer, dst []int8, srcs ...[]int8) error {
	if n := l.Kind.inputs(); (n >= 0 && len(srcs) != n) || (n < 0 && len(srcs) == 0) {
		return ErrShape
	}
	switch l.Kind {
	case Conv2D:
		return qi8.Conv2D(dst, srcs[0], l.Weights, l.Bias, l.Requant, l.Conv2DParams())
	case DepthwiseConv2D:
		return qi8.DepthwiseConv2D(dst, srcs[0], l.Weights, l.Bias, l.Requant, l.DepthwiseParams())
	case FullyConnected:
		return qi8.FullyConnected(dst, srcs[0], l.Weights, l.Bias, l.Requant, l.FCParams())
	case MaxPool2D:
		return qi8.MaxPool2D(dst, srcs[0], l.PoolParams())
	case AvgPool2D:
		return qi8.AvgPool2D(dst, srcs[0], l.PoolParams())
	case ReLU:
		return qi8.ReLU(dst, srcs[0], l.InQ.ZP)
	case ReLU6:
		return qi8.ReLU6(dst, srcs[0], l.InQ.ZP, qi8.Quantize6(l.InQ.Scale, l.InQ.ZP))
	case Add:
		return qi8.Add(dst, srcs[0], srcs[1], l.AddParams())
	case Reshape, Const:
		src := l.Weights
		if l.Kind == Reshape {
			src = srcs[0]
		}
		if len(dst) != len(src) {
			return qi8.ErrLength
		}
		copy(dst, src)
		return nil
	}
	return runStateless(l, dst, srcs)
}

// runStateless runs the kinds that need no prepared form.
func runStateless(l *Layer, dst []int8, srcs [][]int8) error {
	switch l.Kind {
	case Softmax:
		return qi8.Softmax(dst, srcs[0], l.In.C, l.SoftmaxParams)
	case Logistic:
		return qi8.Logistic(dst, srcs[0], *l.LogisticParams)
	case Mul:
		return qi8.Mul(dst, srcs[0], srcs[1], l.MulParams())
	case Mean:
		return qi8.MeanHW(dst, srcs[0], l.In, l.MeanParams())
	case Requantize:
		return qi8.Requantize(dst, srcs[0], l.RequantizeParams())
	case Concat:
		return qi8.Concat(dst, srcs, l.concatOuter())
	case Pad:
		return qi8.Pad(dst, srcs[0], l.PadParams())
	case ResizeNearest:
		return qi8.ResizeNearest(dst, srcs[0], l.ResizeParams())
	case ResizeBilinear:
		return qi8.ResizeBilinear(dst, srcs[0], l.ResizeParams())
	case LeakyReLU:
		return qi8.LeakyReLU(dst, srcs[0], l.LeakyReLUParams())
	}
	return ErrKind
}

package qnet

import (
	"math"

	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// Synthetic MobileNetV1 activation quantisation.
var (
	// MobileNetInputQuant: int8 image in [-1, 1).
	MobileNetInputQuant = Quant{Scale: 1.0 / 128, ZP: 0}
	// MobileNetActQuant: ReLU6 output, [0, 6] over the full int8 range.
	MobileNetActQuant = Quant{Scale: 6.0 / 255, ZP: -128}
	// MobileNetLogitQuant: classifier output (no activation).
	MobileNetLogitQuant = Quant{Scale: 1.0 / 16, ZP: 0}
)

// mobileNetBlocks are the depthwise-separable blocks of MobileNetV1 at
// alpha 1: pointwise output channels and depthwise stride.
var mobileNetBlocks = [13]struct{ c, s int }{
	{64, 1}, {128, 2}, {128, 1}, {256, 2}, {256, 1}, {512, 2},
	{512, 1}, {512, 1}, {512, 1}, {512, 1}, {512, 1},
	{1024, 2}, {1024, 1},
}

// MobileNetV1Layers returns the number of layers MobileNetV1 builds:
// stem conv + 13×(depthwise + pointwise) + global average pool (+ FC).
func MobileNetV1Layers(classes int) int {
	n := 1 + 2*len(mobileNetBlocks) + 1
	if classes > 0 {
		n++
	}
	return n
}

// MobileNetV1 fills layers[:MobileNetV1Layers(classes)] with a
// MobileNetV1-shaped graph and returns it (no allocation):
//
//	conv 3×3/2 → int(32·alpha), then 13 × (depthwise 3×3 + pointwise 1×1)
//	with pointwise channels int(alpha·{64, 128/2, 128, 256/2, 256, 512/2,
//	512×5, 1024/2, 1024}) (/2 = depthwise stride 2), global average pool,
//	and, when classes > 0, a fully connected layer to classes logits.
//
// The input is inputH×inputW×3 quantised as MobileNetInputQuant. Every conv
// fuses ReLU6 with output MobileNetActQuant; the FC has no activation and
// outputs MobileNetLogitQuant. Padding is SAME. Weights, Bias and Requant are
// left nil: size backing arrays with ParamCounts, then BindParams.
// Errors: ErrShape (alpha <= 0, a channel count < 1, non-positive input,
// classes < 0), ErrCapacity (len(layers) too small).
func MobileNetV1(alpha float32, inputH, inputW, classes int, layers []Layer) (Graph, error) {
	if !(alpha > 0) || classes < 0 || inputH <= 0 || inputW <= 0 {
		return Graph{}, ErrShape
	}
	n := MobileNetV1Layers(classes)
	if len(layers) < n {
		return Graph{}, ErrCapacity
	}
	layers = layers[:n]
	act := MobileNetActQuant

	c := int(32 * alpha)
	if c < 1 {
		return Graph{}, ErrShape
	}
	layers[0] = mobileNetConv(Conv2D, qi8.Shape{H: inputH, W: inputW, C: 3}, MobileNetInputQuant, c, 3, 2)
	s, err := setOut(&layers[0])
	if err != nil {
		return Graph{}, err
	}
	i := 1
	for _, b := range mobileNetBlocks {
		layers[i] = mobileNetConv(DepthwiseConv2D, s, act, s.C, 3, b.s)
		if s, err = setOut(&layers[i]); err != nil {
			return Graph{}, err
		}
		if c = int(float32(b.c) * alpha); c < 1 {
			return Graph{}, ErrShape
		}
		layers[i+1] = mobileNetConv(Conv2D, s, act, c, 1, 1)
		if s, err = setOut(&layers[i+1]); err != nil {
			return Graph{}, err
		}
		i += 2
	}
	layers[i] = Layer{
		Kind: AvgPool2D, In: s, InQ: act, OutQ: act,
		KH: s.H, KW: s.W, StrideH: 1, StrideW: 1, Padding: qi8.PaddingValid,
	}
	if s, err = setOut(&layers[i]); err != nil {
		return Graph{}, err
	}
	if classes > 0 {
		lmin, lmax := qi8.ActNone()
		layers[i+1] = Layer{
			Kind: FullyConnected, In: s, Out: qi8.Shape{C: classes},
			InQ: act, OutQ: MobileNetLogitQuant, ActMin: lmin, ActMax: lmax,
		}
		if _, err = setOut(&layers[i+1]); err != nil {
			return Graph{}, err
		}
	}
	g := Graph{Layers: layers}
	if err := g.checkShapes(); err != nil {
		return Graph{}, err
	}
	return g, nil
}

// mobileNetConv returns a SAME-padded square conv/depthwise layer with fused
// ReLU6 and MobileNetActQuant output; Out.C = outC, Out.H/W unset.
func mobileNetConv(kind Kind, in qi8.Shape, inQ Quant, outC, k, stride int) Layer {
	act := MobileNetActQuant
	lo, hi := qi8.ActReLU6(act.Scale, act.ZP)
	return Layer{
		Kind: kind, In: in, Out: qi8.Shape{C: outC}, InQ: inQ, OutQ: act,
		KH: k, KW: k, StrideH: stride, StrideW: stride, Padding: qi8.PaddingSame,
		ActMin: lo, ActMax: hi,
	}
}

// setOut sets l.Out to l.OutShape() and returns it.
func setOut(l *Layer) (qi8.Shape, error) {
	out, err := l.OutShape()
	if err != nil {
		return out, err
	}
	l.Out = out
	return out, nil
}

// Synthetic parameter statistics: weights are uniform in [-127, 127]
// (rms ≈ 73.3); the graph input is assumed uniform over int8 (|x − zp| rms ≈
// 73.9) and every later layer's input is the output of a synthetic layer
// (pre-activation std synthTarget LSB, about half floored by ReLU: rms ≈
// synthTarget/√2).
const (
	synthWeightRMS = 73.3
	synthInputRMS  = 73.9
	synthTarget    = 48
)

// SyntheticParams fills the bound Weights, Bias and Requant of every layer of
// g deterministically from seed, with magnitudes chosen so activations stay
// spread out without saturating: weights uniform in [-127, 127]; per channel,
// the requant multiplier is synthTarget/σ, σ = sqrt(fanIn)·73.3·xRMS (fan-in
// KH·KW·InC, KH·KW or In; xRMS 73.9 for the first parameter layer, then
// synthTarget/√2), jittered by ±25 %; the bias is uniform in ±σ/4. Returns
// ErrParams if a layer's slices are not bound with ParamCounts lengths (Bias
// may be nil). Does not allocate.
func SyntheticParams(g *Graph, seed uint64) error {
	rng := splitMix{seed}
	xRMS := synthInputRMS
	for i := range g.Layers {
		l := &g.Layers[i]
		nw, nb, nr := l.ParamCounts()
		if len(l.Weights) != nw || len(l.Requant) != nr || (len(l.Bias) != 0 && len(l.Bias) != nb) {
			return ErrParams
		}
		if nw == 0 {
			continue
		}
		fanIn := nw / nr
		for j := range l.Weights {
			l.Weights[j] = int8(int(rng.next()%255) - 127)
		}
		sigma := math.Sqrt(float64(fanIn)) * synthWeightRMS * xRMS
		for j := range l.Requant {
			jitter := 0.75 + 0.5*rng.unit()
			r, err := qi8.QuantizeMultiplier(synthTarget * jitter / sigma)
			if err != nil {
				return err
			}
			l.Requant[j] = r
		}
		for j := range l.Bias {
			l.Bias[j] = int32((rng.unit() - 0.5) * sigma / 2)
		}
		xRMS = synthTarget / math.Sqrt2
	}
	return nil
}

// splitMix is SplitMix64.
type splitMix struct{ s uint64 }

func (r *splitMix) next() uint64 {
	r.s += 0x9e3779b97f4a7c15
	z := r.s
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// unit returns a uniform float64 in [0, 1).
func (r *splitMix) unit() float64 { return float64(r.next()>>11) / (1 << 53) }

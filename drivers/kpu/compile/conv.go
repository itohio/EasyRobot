// Package compile lowers qnet graphs to K210 KPU layers (kmodel layer
// arguments and tables) for drivers/kpu. Design and milestones:
// docs/architecture/kpu-compiler.md; hardware format:
// docs/architecture/k210-kpu.md.
package compile

import (
	"errors"
	"fmt"
	"math"

	"github.com/itohio/EasyRobot/drivers/kpu/kmodel"
	"github.com/itohio/EasyRobot/x/math/nn/qnet"
	"github.com/itohio/EasyRobot/x/math/primitive/qi8"
)

// ErrUnsupported is returned for a layer the KPU cannot run as lowered here.
var ErrUnsupported = errors.New("compile: layer not supported on the KPU")

// Conv is one lowered KPU conv: its layer argument (weight, BN and
// activation addresses left zero; drivers/kpu patches them) and its tables.
type Conv struct {
	Args             kmodel.Args
	Weights, BN, Act []byte
}

// Limits of the scaling search (see LowerConv).
const (
	maxPostShift = 24      // P: output scale 2^P carried from BN to the activation
	yBound       = 1 << 34 // |BN output| stays inside the 36-bit activation compare
	mulBound     = 1 << 23 // BN norm_mul is signed 24-bit
	maxRows      = 256     // KPU input rows (258 never finish on silicon)
)

// Fuse names the qnet layers lowered together with a conv into one KPU
// layer. Each must directly chain with the conv (shapes and quantisation).
type Fuse struct {
	// Pad is a Pad(1,1,1,1) directly before a stride-2 VALID 3×3 conv: the
	// pair becomes a stride-1 SAME conv on the unpadded input (pad_value =
	// PadValue) followed by the KPU's left-top 2×2 sampler, which centres
	// the windows on even pixels exactly as the explicit pad does.
	Pad *qnet.Layer
	// Act is a standalone ReLU or ReLU6 directly after the conv; it
	// tightens the conv's clamp.
	Act *qnet.Layer
	// Pool is a MaxPool2D or AvgPool2D with kernel = stride ∈ {2, 4},
	// VALID, directly after a stride-1 regular conv (or after Act). The
	// KPU does not pool in depthwise mode (ER-K210-K9 evidence, K9e), so a
	// depthwise conv takes no pool.
	Pool *qnet.Layer
	// Full lowers a stride-2 depthwise conv (with its Pad), or a TF SAME
	// stride-2 3×3 regular conv, at full resolution: stride 1, no pool,
	// output the size of its unpadded input. The KPU cannot sample a
	// depthwise layer, nor the odd pixels of a regular one, so the next KPU
	// layer samples: a 1×1 conv lowered with Subsample, or the identity
	// Sampler. This is nncase's stride-2 depthwise
	// (K210SeparableConv2dTransform).
	Full bool
	// Subsample lowers a 1×1 stride-1 conv whose input arrives at twice its
	// height and width, from a Full layer, and applies the left-top
	// sampler. A 1×1 conv commutes with the subsampling, so the pair is
	// exact.
	Subsample bool
	// OddSample, with Subsample, samples the odd rows and columns instead:
	// the right-top sampler (pool_type 6, odd columns) on the input read
	// from one row down (odd rows). That is TF SAME stride 2 on an even
	// size, whose windows centre on odd pixels; the Full layer before it
	// computed them at full resolution. The reader reads one row past its
	// writer's tensor (never sampled): the plan reserves it.
	OddSample bool
}

// Sampler returns the identity 1×1 conv on l's output (same shape and
// quantisation): lowered with Subsample (and OddSample), it samples a Full
// layer's output when the graph's next layer is not a 1×1 conv. Exact:
// weights 1 on the diagonal and multiplier 1 make every rounding vanish.
func Sampler(l *qnet.Layer) qnet.Layer {
	c := l.Out.C
	w := make([]int8, c*c)
	for i := range c {
		w[i*c+i] = 1
	}
	one, _ := qi8.QuantizeMultiplier(1)
	rq := make([]qi8.Requant, c)
	for i := range rq {
		rq[i] = one
	}
	return qnet.Layer{
		Kind: qnet.Conv2D, In: l.Out, Out: l.Out, InQ: l.OutQ, OutQ: l.OutQ,
		KH: 1, KW: 1, StrideH: 1, StrideW: 1, Padding: qi8.PaddingValid,
		ActMin: -128, ActMax: 127, Weights: w, Requant: rq,
	}
}

// LowerConv lowers a single qnet conv layer (see Lower) without fusion.
func LowerConv(l *qnet.Layer, src, dst uint32) (Conv, error) { return Lower(l, Fuse{}, src, dst) }

// Lower lowers a qnet Conv2D (1×1 or 3×3) or DepthwiseConv2D (3×3,
// multiplier 1), together with the layers in f, with its input at KPU line
// address src and its output at dst, both in the standard layout for their
// width (kmodel.UploadLayout).
//
// Supported: stride 1 SAME (1×1 also VALID); stride 2 as Pad(1,1) + 3×3
// VALID or as 1×1 VALID, on even H and W (the left-top sampler); TF SAME
// stride 2 on even H and W, whose windows centre on odd pixels, as Full
// followed by an OddSample reader; a fused 2×2 or 4×4 max or average pool
// after a stride-1 conv; a fused ReLU or ReLU6. The KPU layer's input and
// output are at most 256 rows (a 258-row input never finishes on silicon,
// ER-K210-K9 evidence).
//
// The integers are exact up to requantisation (design note, "Integer
// mapping"): KPU tensors are uint8 x′ = q + 128 and weights w′ = q_w + 128;
// with a weight zero point z_w (0 for symmetric weights), arg_x = −(z_w +
// 128), arg_w = −(zp_in + 128), arg_add = (z_w + 128)·(zp_in + 128)·k² (the
// hardware multiplies by IC for a conv) and pad_value = zp_in + 128
// (PadValue + 128 with a fused Pad), so the KPU accumulator equals TFLite's
// Σ (q_w − z_w)·(q_x − zp_in). qnet layers carry symmetric weights until the
// ER-QU8 loader adds their zero point, so Lower uses z_w = 0.
//
// Requantisation is split: per channel, BN computes y = (acc·mul >> s) +
// add ≈ (acc + bias)·M_c·2^P; the layer's activation table then rounds
// y/2^P (y_mul 1, shift P), adds zp_out + 128 and clamps to
// [ActMin, ActMax] + 128 with two flat segments. P is the largest value
// ≤ 24 that keeps every channel's y inside 2^34, its BN add inside int32 and
// its mul inside 24 bits. The two roundings (BN floors, the activation
// rounds half up) make the result differ from qnet's single rounding by at
// most 1 LSB; a fused average pool floors where qnet rounds, one LSB more.
func Lower(l *qnet.Layer, f Fuse, src, dst uint32) (Conv, error) {
	return lowerZW(l, f, src, dst, 0)
}

// lowerZW is Lower for weights with zero point zw (int8, as the weights).
func lowerZW(l *qnet.Layer, f Fuse, src, dst uint32, zw int8) (Conv, error) {
	dw := l.Kind == qnet.DepthwiseConv2D
	if l.Kind != qnet.Conv2D && !dw {
		return Conv{}, fmt.Errorf("%w: kind %v", ErrUnsupported, l.Kind)
	}
	k := l.KH
	if l.KW != k || (k != 1 && k != 3) || (dw && k != 3) {
		return Conv{}, fmt.Errorf("%w: %d×%d kernel", ErrUnsupported, l.KH, l.KW)
	}
	g, err := geometry(l, f)
	if err != nil {
		return Conv{}, err
	}
	in := g.in
	if in.H > maxRows {
		return Conv{}, fmt.Errorf("%w: a %d-row KPU input (at most %d)", ErrUnsupported, in.H, maxRows)
	}
	oc := l.Out.C
	if dw && oc != in.C {
		return Conv{}, fmt.Errorf("%w: depthwise multiplier %d", ErrUnsupported, oc/max(in.C, 1))
	}
	taps := k * k * in.C
	if dw {
		taps = k * k
	}
	nw := taps * oc
	if dw {
		nw = k * k * in.C
	}
	if len(l.Weights) != nw || len(l.Requant) != oc || (l.Bias != nil && len(l.Bias) != oc) {
		return Conv{}, fmt.Errorf("%w: %d weights, %d requants, %d biases for %d output channels", kmodel.ErrFormat, len(l.Weights), len(l.Requant), len(l.Bias), oc)
	}
	lo, hi, err := clamp(l, f.Act)
	if err != nil {
		return Conv{}, err
	}

	zpIn := int64(l.InQ.ZP) + 128
	zpW := int64(zw) + 128
	inLayout := kmodel.UploadLayout(in.W, in.H)
	a := kmodel.NewArgs()
	for _, err := range []error{
		a.SetSrcAddr(src + uint32(g.skip*inLayout.RowLines)), a.SetDstAddr(dst),
		a.SetInShape(in.W, in.H, in.C), a.SetOutShape(g.outW, g.outH, oc),
		a.SetInLayout(inLayout), a.SetOutLayout(kmodel.UploadLayout(g.outW, g.outH)),
		a.SetKernel(k, dw), a.SetPool(g.pool),
		a.SetConvValue(-zpW, 0, -zpIn, 0), a.SetArgAdd(zpW * zpIn * int64(k*k)),
		setLoad(&a, k, in.C, oc, dw),
	} {
		if err != nil {
			return Conv{}, err
		}
	}
	a.SetPadValue(uint8(g.pad + 128))

	w := make([]uint16, nw)
	for i, v := range l.Weights {
		w[i] = uint16(int16(v) + 128)
	}
	weights, err := kmodel.EncodeWeights(w, k, in.C, oc, dw, true)
	if err != nil {
		return Conv{}, err
	}
	bn, p, err := batchNorm(l, int64(taps), zw)
	if err != nil {
		return Conv{}, err
	}
	bnb, err := kmodel.EncodeBN(bn)
	if err != nil {
		return Conv{}, err
	}
	actb, err := kmodel.EncodeAct(activation(int64(l.OutQ.ZP)+128, lo+128, hi+128, p))
	if err != nil {
		return Conv{}, err
	}
	return Conv{Args: a, Weights: weights, BN: bnb, Act: actb}, nil
}

// lowered is the KPU geometry of a conv with its fused layers.
type lowered struct {
	in         qi8.Shape // the KPU layer's input (unpadded with a fused Pad)
	outW, outH int       // its output after the pool or sampler
	pool       int       // pool_type
	pad        int64     // int8 value of out-of-image taps
	skip       int       // input rows to skip at the start of each channel block
}

func same(a, b qnet.Quant) bool { return a == b }

var errNotFull = fmt.Errorf("%w: Full is for a stride-2 depthwise or TF SAME stride-2 conv", ErrUnsupported)

func geometry(l *qnet.Layer, f Fuse) (lowered, error) {
	k, s := l.KH, l.StrideH
	if l.StrideW != s || (s != 1 && s != 2) {
		return lowered{}, fmt.Errorf("%w: stride %d×%d", ErrUnsupported, l.StrideH, l.StrideW)
	}
	g := lowered{in: l.In, pad: int64(l.InQ.ZP)}
	dw := l.Kind == qnet.DepthwiseConv2D
	if f.Pool != nil && dw {
		return lowered{}, fmt.Errorf("%w: a pool on a depthwise conv (the KPU does not pool in depthwise mode)", ErrUnsupported)
	}
	if f.Full && !(s == 2 && (dw || (k == 3 && l.Padding == qi8.PaddingSame))) {
		return lowered{}, errNotFull
	}
	if f.OddSample && !f.Subsample {
		return lowered{}, fmt.Errorf("%w: OddSample needs Subsample", ErrUnsupported)
	}
	if f.Subsample {
		if dw || k != 1 || s != 1 || f.Pad != nil || f.Pool != nil {
			return lowered{}, fmt.Errorf("%w: Subsample is for a 1×1 stride-1 conv without Pad or pool", ErrUnsupported)
		}
		g.in = qi8.Shape{H: 2 * l.In.H, W: 2 * l.In.W, C: l.In.C}
		g.outW, g.outH, g.pool = l.In.W, l.In.H, 5
		if f.OddSample {
			g.pool, g.skip = 6, 1 // right-top sampler, from one row down
		}
		if l.Out.H != l.In.H || l.Out.W != l.In.W {
			return lowered{}, fmt.Errorf("%w: output %v for input %v", ErrUnsupported, l.Out, l.In)
		}
		return g, nil
	}
	if s == 1 {
		if f.Pad != nil {
			return lowered{}, fmt.Errorf("%w: a fused Pad needs a stride-2 VALID conv", ErrUnsupported)
		}
		if k == 3 && l.Padding != qi8.PaddingSame {
			return lowered{}, fmt.Errorf("%w: stride-1 3×3 needs SAME padding", ErrUnsupported)
		}
		if l.Out.H != l.In.H || l.Out.W != l.In.W {
			return lowered{}, fmt.Errorf("%w: output %v for input %v", ErrUnsupported, l.Out, l.In)
		}
		g.outW, g.outH = l.In.W, l.In.H
		if f.Pool != nil {
			p := f.Pool
			pk := p.KH
			if (p.Kind != qnet.MaxPool2D && p.Kind != qnet.AvgPool2D) || p.KW != pk || p.StrideH != pk || p.StrideW != pk || (pk != 2 && pk != 4) || p.Padding != qi8.PaddingValid {
				return lowered{}, fmt.Errorf("%w: pool %v %d×%d stride %d×%d", ErrUnsupported, p.Kind, p.KH, p.KW, p.StrideH, p.StrideW)
			}
			if p.In != l.Out || l.In.H%pk != 0 || l.In.W%pk != 0 || p.Out != (qi8.Shape{H: l.In.H / pk, W: l.In.W / pk, C: l.Out.C}) || !same(p.InQ, l.OutQ) || !same(p.OutQ, l.OutQ) {
				return lowered{}, fmt.Errorf("%w: pool does not chain with the conv (%v → %v)", ErrUnsupported, p.In, p.Out)
			}
			g.pool = map[[2]int]int{{2, 0}: 1, {2, 1}: 2, {4, 0}: 3, {4, 1}: 4}[[2]int{pk, b2i(p.Kind == qnet.AvgPool2D)}]
			g.outW, g.outH = l.In.W/pk, l.In.H/pk
		}
		return g, nil
	}
	// Stride 2: the KPU's left-top or right-top sampler.
	if f.Pool != nil {
		return lowered{}, fmt.Errorf("%w: a pool after a stride-2 conv (the sampler uses the pool slot)", ErrUnsupported)
	}
	if l.Padding == qi8.PaddingSame && k == 3 {
		// TF SAME stride 2 on an even size: windows centre on odd pixels.
		if f.Pad != nil || l.In.H%2 != 0 || l.In.W%2 != 0 || l.Out.H != l.In.H/2 || l.Out.W != l.In.W/2 {
			return lowered{}, fmt.Errorf("%w: SAME stride 2 needs an even input and no fused Pad", ErrUnsupported)
		}
		if !f.Full {
			return lowered{}, fmt.Errorf("%w: a SAME stride-2 conv needs Full and an OddSample reader after it", ErrUnsupported)
		}
		g.outW, g.outH = l.In.W, l.In.H // full resolution; the reader samples the odd pixels
		return g, nil
	}
	if l.Padding != qi8.PaddingValid && k == 3 {
		return lowered{}, fmt.Errorf("%w: stride-2 padding %v", ErrUnsupported, l.Padding)
	}
	if k == 3 {
		p := f.Pad
		if p == nil || p.Kind != qnet.Pad || p.Pads != [6]int{1, 1, 1, 1, 0, 0} {
			return lowered{}, fmt.Errorf("%w: a stride-2 3×3 conv needs a fused Pad(1,1,1,1)", ErrUnsupported)
		}
		if p.Out != l.In || p.In != (qi8.Shape{H: l.In.H - 2, W: l.In.W - 2, C: l.In.C}) || !same(p.OutQ, l.InQ) {
			return lowered{}, fmt.Errorf("%w: Pad does not chain with the conv (%v → %v)", ErrUnsupported, p.In, p.Out)
		}
		g.in, g.pad = p.In, int64(p.PadValue)
	} else if f.Pad != nil {
		return lowered{}, fmt.Errorf("%w: a fused Pad before a 1×1 conv", ErrUnsupported)
	}
	if g.in.H%2 != 0 || g.in.W%2 != 0 {
		return lowered{}, fmt.Errorf("%w: stride 2 on odd size %d×%d", ErrUnsupported, g.in.H, g.in.W)
	}
	if l.Out.H != g.in.H/2 || l.Out.W != g.in.W/2 {
		return lowered{}, fmt.Errorf("%w: output %v for stride-2 input %v", ErrUnsupported, l.Out, g.in)
	}
	if dw {
		if !f.Full {
			return lowered{}, fmt.Errorf("%w: a stride-2 depthwise conv needs Full and a Subsample 1×1 conv after it (the KPU does not pool in depthwise mode)", ErrUnsupported)
		}
		g.outW, g.outH = g.in.W, g.in.H // full resolution, no pool
		return g, nil
	}
	g.outW, g.outH, g.pool = g.in.W/2, g.in.H/2, 5
	return g, nil
}

func b2i(b bool) int {
	if b {
		return 1
	}
	return 0
}

// clamp returns the conv's output clamp, tightened by a fused ReLU/ReLU6.
func clamp(l *qnet.Layer, act *qnet.Layer) (lo, hi int64, err error) {
	lo, hi = int64(l.ActMin), int64(l.ActMax)
	if lo > hi {
		return 0, 0, fmt.Errorf("%w: clamp [%d, %d]", kmodel.ErrFormat, lo, hi)
	}
	if act == nil {
		return lo, hi, nil
	}
	if act.In != l.Out || act.Out != l.Out || !same(act.InQ, l.OutQ) || !same(act.OutQ, l.OutQ) {
		return 0, 0, fmt.Errorf("%w: %v does not chain with the conv", ErrUnsupported, act.Kind)
	}
	// The second clamp [lo2, hi2] applied after the first: their
	// intersection, or the second's bound nearest the first range if they
	// do not overlap.
	lo2, hi2 := int64(l.OutQ.ZP), int64(127)
	switch act.Kind {
	case qnet.ReLU:
	case qnet.ReLU6:
		hi2 = int64(qi8.Quantize6(l.OutQ.Scale, l.OutQ.ZP))
	default:
		return 0, 0, fmt.Errorf("%w: fused activation %v", ErrUnsupported, act.Kind)
	}
	switch {
	case hi < lo2:
		return lo2, lo2, nil
	case lo > hi2:
		return hi2, hi2, nil
	}
	return max(lo, lo2), min(hi, hi2), nil
}

// setLoad plans the weight loads as ncc does: a conv loads as many whole
// output channels as fit 30 KiB of 8-bit weights; a depthwise loads all.
func setLoad(a *kmodel.Args, k, ic, oc int, dw bool) error {
	if dw {
		return a.SetLoad(uint32(k*k*ic), 1, oc)
	}
	one := k * k * ic
	per := min(oc, 30*1024/one)
	if per < 1 {
		return fmt.Errorf("%w: one output channel's weights (%d B) exceed a 30 KiB load", ErrUnsupported, one)
	}
	return a.SetLoad(uint32(one*per), (oc+per-1)/per, per)
}

// multiplier returns qnet's real requantisation multiplier M = Mult·2^(Shift−31).
func abs64(v int64) int64 { return max(v, -v) }

func multiplier(r qi8.Requant) float64 {
	return float64(r.Mult) * math.Ldexp(1, int(r.Shift)-31)
}

// batchNorm picks the layer's post-shift P and each channel's BN entry, for
// weights with zero point zw.
func batchNorm(l *qnet.Layer, taps int64, zw int8) ([]kmodel.BN, int, error) {
	oc := len(l.Requant)
	wMax := int64(128) // |q_w − z_w|; 128 bounds every symmetric int8 weight
	if zw != 0 {
		for _, v := range l.Weights {
			wMax = max(wMax, abs64(int64(v)-int64(zw)))
		}
	}
	accMax := float64(taps*wMax) * 255 // |Σ (q_w − z_w)·(q_x − zp)|
	bias := func(c int) float64 {
		if l.Bias == nil {
			return 0
		}
		return float64(l.Bias[c])
	}
	for p := maxPostShift; p >= 1; p-- {
		scale := math.Ldexp(1, p)
		bn := make([]kmodel.BN, oc)
		ok := true
		for c := 0; c < oc && ok; c++ {
			m := multiplier(l.Requant[c])
			if m == 0 {
				continue // dead channel: y = 0, the output is zp_out
			}
			b := bias(c)
			add := math.Round(b * m * scale)
			if (accMax+math.Abs(b))*m*scale >= yBound || math.Abs(add) >= 1<<31 {
				ok = false
				break
			}
			s := 15
			for s > 0 && math.Round(m*math.Ldexp(scale, s)) >= mulBound {
				s--
			}
			mul := math.Round(m * math.Ldexp(scale, s))
			if mul >= mulBound {
				ok = false
				break
			}
			bn[c] = kmodel.BN{Mul: int64(mul), Add: int64(add), Shift: uint(s)}
		}
		if ok {
			return bn, p, nil
		}
	}
	return nil, 0, fmt.Errorf("%w: requantisation multipliers out of the KPU's range", ErrUnsupported)
}

// activation is the layer's table: out = clamp(zp + round(y/2^p), lo, hi)
// in uint8, with flat segments below lo and from hi.
func activation(zp, lo, hi int64, p int) [16]kmodel.ActSegment {
	var t [16]kmodel.ActSegment
	t[0] = kmodel.ActSegment{StartX: -1 << 35, Add: lo}
	// Below xLo the linear segment would give less than lo; from xHi on, more
	// than hi (just below xHi it already rounds up to hi).
	xLo := (lo - zp) << p
	xHi := max((hi-zp)<<p, xLo+1)
	t[1] = kmodel.ActSegment{StartX: xLo, Mul: 1, Shift: uint(p), Add: lo}
	t[2] = kmodel.ActSegment{StartX: xHi, Add: hi}
	for i := 3; i < 16; i++ {
		t[i] = kmodel.ActSegment{StartX: 1<<35 - 16 + int64(i), Add: hi}
	}
	return t
}

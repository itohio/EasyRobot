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
	// VALID, directly after a stride-1 conv (or after Act).
	Pool *qnet.Layer
}

// LowerConv lowers a single qnet conv layer (see Lower) without fusion.
func LowerConv(l *qnet.Layer, src, dst uint32) (Conv, error) { return Lower(l, Fuse{}, src, dst) }

// Lower lowers a qnet Conv2D (1×1 or 3×3) or DepthwiseConv2D (3×3,
// multiplier 1), together with the layers in f, with its input at KPU line
// address src and its output at dst, both in the standard layout for their
// width (kmodel.UploadLayout).
//
// Supported: stride 1 SAME (1×1 also VALID); stride 2 as Pad(1,1) + 3×3
// VALID or as 1×1 VALID, on even H and W (the left-top sampler); a fused
// 2×2 or 4×4 max or average pool after a stride-1 conv; a fused ReLU or
// ReLU6. TF SAME stride 2 is rejected: its windows centre on odd pixels,
// which the KPU cannot sample (export with Pad(1,1) + VALID instead).
//
// The integers are exact up to requantisation (design note, "Integer
// mapping"): KPU tensors are uint8 x′ = q + 128 and weights w′ = q_w + 128;
// arg_x = −128, arg_w = −(zp_in + 128), arg_add = 128·(zp_in + 128)·k²
// (the hardware multiplies by IC for a conv) and pad_value = zp_in + 128
// (PadValue + 128 with a fused Pad), so the KPU accumulator equals qnet's
// Σ q_w·(q_x − zp_in).
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
	a := kmodel.NewArgs()
	for _, err := range []error{
		a.SetSrcAddr(src), a.SetDstAddr(dst),
		a.SetInShape(in.W, in.H, in.C), a.SetOutShape(g.outW, g.outH, oc),
		a.SetInLayout(kmodel.UploadLayout(in.W, in.H)), a.SetOutLayout(kmodel.UploadLayout(g.outW, g.outH)),
		a.SetKernel(k, dw), a.SetPool(g.pool),
		a.SetConvValue(-128, 0, -zpIn, 0), a.SetArgAdd(128 * zpIn * int64(k*k)),
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
	bn, p, err := batchNorm(l, int64(taps))
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
}

func same(a, b qnet.Quant) bool { return a == b }

func geometry(l *qnet.Layer, f Fuse) (lowered, error) {
	k, s := l.KH, l.StrideH
	if l.StrideW != s || (s != 1 && s != 2) {
		return lowered{}, fmt.Errorf("%w: stride %d×%d", ErrUnsupported, l.StrideH, l.StrideW)
	}
	g := lowered{in: l.In, pad: int64(l.InQ.ZP)}
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
	// Stride 2: the KPU's left-top sampler.
	if f.Pool != nil {
		return lowered{}, fmt.Errorf("%w: a pool after a stride-2 conv (the sampler uses the pool slot)", ErrUnsupported)
	}
	if l.Padding != qi8.PaddingValid {
		return lowered{}, fmt.Errorf("%w: stride 2 with SAME padding (windows on odd pixels); export as Pad(1,1) + VALID", ErrUnsupported)
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
	g.outW, g.outH, g.pool = g.in.W/2, g.in.H/2, 5
	if l.Out.H != g.outH || l.Out.W != g.outW {
		return lowered{}, fmt.Errorf("%w: output %v for stride-2 input %v", ErrUnsupported, l.Out, g.in)
	}
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
func multiplier(r qi8.Requant) float64 {
	return float64(r.Mult) * math.Ldexp(1, int(r.Shift)-31)
}

// batchNorm picks the layer's post-shift P and each channel's BN entry.
func batchNorm(l *qnet.Layer, taps int64) ([]kmodel.BN, int, error) {
	oc := len(l.Requant)
	accMax := float64(taps) * 128 * 255 // |Σ q_w·(q_x − zp)|
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

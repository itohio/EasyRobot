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

// LowerConv lowers a qnet Conv2D (1×1 or 3×3) or DepthwiseConv2D (3×3,
// multiplier 1), stride 1, SAME (1×1 also VALID), with its input at KPU
// line address src and its output at dst, both in the standard layout for
// their width (kmodel.UploadLayout).
//
// The integers are exact up to requantisation (design note, "Integer
// mapping"): KPU tensors are uint8 x′ = q + 128 and weights w′ = q_w + 128;
// arg_x = −128, arg_w = −(zp_in + 128), arg_add = 128·(zp_in + 128)·k²
// (the hardware multiplies by IC for a conv) and pad_value = zp_in + 128,
// so the KPU accumulator equals qnet's Σ q_w·(q_x − zp_in).
//
// Requantisation is split: per channel, BN computes y = (acc·mul >> s) +
// add ≈ (acc + bias)·M_c·2^P; the layer's activation table then rounds
// y/2^P (y_mul 1, shift P), adds zp_out + 128 and clamps to
// [ActMin, ActMax] + 128 with two flat segments. P is the largest value
// ≤ 24 that keeps every channel's y inside 2^34, its BN add inside int32 and
// its mul inside 24 bits. The two roundings (BN floors, the activation
// rounds half up) make the result differ from qnet's single rounding by at
// most 1 LSB.
func LowerConv(l *qnet.Layer, src, dst uint32) (Conv, error) {
	dw := l.Kind == qnet.DepthwiseConv2D
	if l.Kind != qnet.Conv2D && !dw {
		return Conv{}, fmt.Errorf("%w: kind %v", ErrUnsupported, l.Kind)
	}
	k := l.KH
	if l.KW != k || (k != 1 && k != 3) || (dw && k != 3) {
		return Conv{}, fmt.Errorf("%w: %d×%d kernel", ErrUnsupported, l.KH, l.KW)
	}
	if l.StrideH != 1 || l.StrideW != 1 {
		return Conv{}, fmt.Errorf("%w: stride %d×%d (stride 2 is the left-top sampler, K9c)", ErrUnsupported, l.StrideH, l.StrideW)
	}
	if k == 3 && l.Padding != qi8.PaddingSame {
		return Conv{}, fmt.Errorf("%w: 3×3 needs SAME padding", ErrUnsupported)
	}
	in := l.In
	oc := l.Out.C
	if dw && oc != in.C {
		return Conv{}, fmt.Errorf("%w: depthwise multiplier %d", ErrUnsupported, oc/max(in.C, 1))
	}
	if l.Out.H != in.H || l.Out.W != in.W {
		return Conv{}, fmt.Errorf("%w: output %v for input %v", ErrUnsupported, l.Out, in)
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
	if l.ActMin > l.ActMax {
		return Conv{}, fmt.Errorf("%w: clamp [%d, %d]", kmodel.ErrFormat, l.ActMin, l.ActMax)
	}

	zpIn := int64(l.InQ.ZP) + 128
	a := kmodel.NewArgs()
	l1 := kmodel.UploadLayout(in.W, in.H)
	for _, err := range []error{
		a.SetSrcAddr(src), a.SetDstAddr(dst),
		a.SetInShape(in.W, in.H, in.C), a.SetOutShape(in.W, in.H, oc),
		a.SetInLayout(l1), a.SetOutLayout(l1),
		a.SetKernel(k, dw), a.SetPool(0),
		a.SetConvValue(-128, 0, -zpIn, 0), a.SetArgAdd(128 * zpIn * int64(k*k)),
		setLoad(&a, k, in.C, oc, dw),
	} {
		if err != nil {
			return Conv{}, err
		}
	}
	a.SetPadValue(uint8(zpIn))

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
	actb, err := kmodel.EncodeAct(activation(int64(l.OutQ.ZP)+128, int64(l.ActMin)+128, int64(l.ActMax)+128, p))
	if err != nil {
		return Conv{}, err
	}
	return Conv{Args: a, Weights: weights, BN: bnb, Act: actb}, nil
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

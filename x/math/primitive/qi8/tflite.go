package qi8

import "math"

// Rounding selects how accumulators are requantised.
type Rounding uint8

const (
	// RoundExact (the zero value) rounds acc·M' once, exactly, half away
	// from zero (Requant.Apply).
	RoundExact Rounding = iota
	// RoundTFLite reproduces TensorFlow Lite's int8 reference kernels (default
	// build, double rounding): Requant.ApplyTFLite, and Add with TFLite's
	// left shift and three multipliers (AddRequantTFLite). Use it for models
	// imported from TFLite.
	RoundTFLite
)

func (r Rounding) valid() bool { return r <= RoundTFLite }

// ApplyTFLite returns TFLite's MultiplyByQuantizedMultiplier(acc, Mult,
// Shift): x = acc·2^max(Shift, 0) in int32 (wrapping, where C overflows),
// h = SaturatingRoundingDoublingHighMul(x, Mult) (round(x·Mult/2^31), ties
// toward +∞), then RoundingDivideByPOT(h, max(−Shift, 0)) (ties away from
// zero). Two roundings, so it can differ from Apply by 1. r must be Valid.
func (r Requant) ApplyTFLite(acc int32) int32 {
	x, right := acc, 0
	if r.Shift > 0 {
		x = acc << uint(r.Shift)
	} else {
		right = int(-r.Shift)
	}
	return roundingDivideByPOT(srdhm(x, r.Mult), right)
}

// srdhm is gemmlowp's SaturatingRoundingDoublingHighMul: round(a·b/2^31)
// with ties toward +∞ (truncating division of a·b + nudge), saturating the
// single overflow case a = b = MinInt32.
func srdhm(a, b int32) int32 {
	if a == math.MinInt32 && b == math.MinInt32 {
		return math.MaxInt32
	}
	ab := int64(a) * int64(b)
	nudge := int64(1) << 30
	if ab < 0 {
		nudge = 1 - nudge
	}
	return int32((ab + nudge) / (1 << 31)) // truncating division, as in C
}

// roundingDivideByPOT is gemmlowp's RoundingDivideByPOT: x/2^e rounded to
// nearest, ties away from zero; e in [0, 31].
func roundingDivideByPOT(x int32, e int) int32 {
	mask := int32(int64(1)<<uint(e) - 1)
	threshold := mask >> 1
	if x < 0 {
		threshold++
	}
	v := x >> uint(e)
	if x&mask > threshold {
		v++
	}
	return v
}

// requantizeTFLite is requantize with ApplyTFLite.
func requantizeTFLite(acc int32, r Requant, zp, lo, hi int64) int8 {
	v := int64(r.ApplyTFLite(acc)) + zp
	if v < lo {
		v = lo
	}
	if v > hi {
		v = hi
	}
	return int8(v)
}

// requantizeMode maps an accumulator to int8 in the given rounding mode.
func requantizeMode(acc int32, r Requant, zp, lo, hi int64, mode Rounding) int8 {
	if mode == RoundTFLite {
		return requantizeTFLite(acc, r, zp, lo, hi)
	}
	return requantize(acc, r, zp, lo, hi)
}

// tfliteAddLeftShift is TFLite's int8 Add input left shift.
const tfliteAddLeftShift = 20

// AddRequantTFLite holds TFLite's int8 Add multipliers (kernels/add.cc):
// A and B scale the shifted inputs (real scaleX/(2·max(scaleA, scaleB))),
// Out maps the sum back (real 2·max(scaleA, scaleB)/(2^20·scaleOut)). All
// three are Valid with Shift ≤ 0 (multipliers below 1).
type AddRequantTFLite struct {
	A, B, Out Requant
}

func (r *AddRequantTFLite) valid() bool {
	for _, q := range [3]Requant{r.A, r.B, r.Out} {
		if !q.Valid() || q.Shift > 0 {
			return false
		}
	}
	return true
}

// QuantizeAddTFLite computes TFLite's int8 Add multipliers from the float32
// tensor scales, in float64 as TFLite does, with QuantizeMultiplier.
// Returns ErrRequant for non-positive or non-finite scales, or a multiplier
// that is not representable below 1.
func QuantizeAddTFLite(scaleA, scaleB, scaleOut float32) (AddRequantTFLite, error) {
	for _, s := range [3]float32{scaleA, scaleB, scaleOut} {
		if !(s > 0) || math.IsInf(float64(s), 0) {
			return AddRequantTFLite{}, ErrRequant
		}
	}
	twiceMax := float64(2 * max(scaleA, scaleB))
	reals := [3]float64{
		float64(scaleA) / twiceMax,
		float64(scaleB) / twiceMax,
		twiceMax / float64(float32(1<<tfliteAddLeftShift)*scaleOut),
	}
	var q [3]Requant
	for i, real := range reals {
		r, err := QuantizeMultiplier(real)
		if err != nil || r.Shift > 0 {
			return AddRequantTFLite{}, ErrRequant
		}
		q[i] = r
	}
	return AddRequantTFLite{A: q[0], B: q[1], Out: q[2]}, nil
}

// addTFLiteGeneric is TFLite's AddElementwise for int8; equal lengths,
// p valid with Round == RoundTFLite.
func addTFLiteGeneric(dst, a, b []int8, p *AddParams) {
	n := len(dst)
	a, b = a[:n], b[:n]
	ra, rb, ro := p.TFLite.A, p.TFLite.B, p.TFLite.Out
	za, zb := int32(p.ZPA), int32(p.ZPB)
	zo, lo, hi := int32(p.ZPOut), int32(p.Min), int32(p.Max)
	for i := range dst {
		sa := ra.ApplyTFLite((int32(a[i]) - za) << tfliteAddLeftShift)
		sb := rb.ApplyTFLite((int32(b[i]) - zb) << tfliteAddLeftShift)
		v := ro.ApplyTFLite(sa+sb) + zo
		if v < lo {
			v = lo
		}
		if v > hi {
			v = hi
		}
		dst[i] = int8(v)
	}
}

// Quantize6TFLite is TFLite's quantised 6.0 for ReLU6 (kernel_util.cc
// CalculateActivationRangeQuantized): zp + roundf(6/scale) with the
// division in float32, saturated to int8. It can differ from Quantize6
// (float64 division) on a near-.5 quotient.
func Quantize6TFLite(scale float32, zp int8) int8 {
	q := float32(6) / scale
	v := float64(zp) + math.Round(float64(q))
	if !(v < 127) {
		return 127
	}
	if v < -128 {
		return -128
	}
	return int8(v)
}

// ActReLU6TFLite returns TFLite's fused ReLU6 clamp range [zp, Quantize6TFLite].
func ActReLU6TFLite(scale float32, zp int8) (min, max int8) {
	return zp, Quantize6TFLite(scale, zp)
}

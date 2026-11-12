package qi8

import (
	"errors"
	"math"
)

// Sentinel errors returned by the kernels. They are returned as-is (never
// wrapped) so that error paths do not allocate; compare with ==.
var (
	// ErrShape reports a non-positive or inconsistent dimension, stride or
	// kernel size, or a "valid" padding with a kernel larger than the input.
	ErrShape = errors.New("qi8: invalid shape")
	// ErrLength reports a slice whose length does not match the shape.
	ErrLength = errors.New("qi8: slice length mismatch")
	// ErrRequant reports a Requant outside the supported range, or a real
	// multiplier that cannot be represented.
	ErrRequant = errors.New("qi8: invalid requantisation parameters")
	// ErrActivation reports a clamp range with min > max.
	ErrActivation = errors.New("qi8: invalid activation range")
	// ErrNotPrepared reports a prepared layer whose Prepare was not called,
	// failed, or whose exported fields changed since (call Prepare again).
	ErrNotPrepared = errors.New("qi8: layer not prepared")
)

const (
	multMin = int32(1) << 30 // smallest valid Requant.Mult (Q31 0.5)

	// MinShift and MaxShift bound Requant.Shift so that the total right shift
	// 31-Shift lies in [1, 62].
	MinShift = -31
	MaxShift = 30
)

// Requant is a per-channel requantisation multiplier in Q31 form.
//
// The real multiplier it represents is M' = Mult * 2^(Shift-31), with Mult
// in [2^30, 2^31) and Shift in [MinShift, MaxShift] (total right shift
// 31-Shift in [1, 62]). Applying it to an int32 accumulator is exact:
// round_half_away_from_zero(acc*Mult / 2^(31-Shift)), computed in int64 with a
// single rounding. The zero value {0, 0} is also valid and represents M' = 0
// (TFLite's encoding of a multiplier below 2^-32, e.g. a dead channel): the
// channel outputs its zero point.
type Requant struct {
	Mult  int32
	Shift int32
}

// Valid reports whether r is inside the supported range or the zero
// multiplier.
func (r Requant) Valid() bool {
	return r == Requant{} || r.Mult >= multMin && r.Shift >= MinShift && r.Shift <= MaxShift
}

// Apply returns round_half_away_from_zero(acc * M') as int64 (no zero point,
// no clamping). r must be Valid.
//
// For Shift ≤ −1 (M' < 1/2, the usual case) the total shift s = 31−Shift is
// at least 32, so only the high word of |acc|·Mult matters: the result is
// (hi + 2^(s−33)) >> (s−32), or hi + (lo >> 31) for s = 32 — the same
// integer as the 64-bit formula (the low word cannot carry past the next
// multiple of 2^s), computed with an unsigned 32×32→64 multiply and 32-bit
// shifts, which a 32-bit core (the ESP32-S3) does faster than 64-bit
// variable shifts.
func (r Requant) Apply(acc int32) int64 {
	if r.Shift < 0 {
		a := uint32(acc)
		if acc < 0 {
			a = -a // |acc| ≤ 2^31 fits uint32
		}
		p := uint64(a) * uint64(uint32(r.Mult))
		hi := uint32(p >> 32) // < 2^30
		var v uint32
		if k := uint32(-1 - r.Shift); k == 0 {
			v = hi + uint32(p)>>31
		} else {
			v = (hi + 1<<(k-1)) >> k
		}
		if acc < 0 {
			return -int64(v)
		}
		return int64(v)
	}
	p := int64(acc) * int64(r.Mult) // |p| < 2^62
	s := uint(31 - r.Shift)         // 1..31 here
	half := int64(1) << (s - 1)
	if p >= 0 {
		return (p + half) >> s
	}
	return -((-p + half) >> s)
}

// QuantizeMultiplier converts a positive real multiplier
// (typically inScale*weightScale/outScale) into a Requant.
//
// real = frac * 2^exp with frac in [0.5, 1); Mult = round(frac * 2^31)
// (half away from zero, exact in float64), renormalised when it rounds up to
// 2^31; Shift = exp. Returns ErrRequant for non-finite, non-positive or
// out-of-range values (representable range is [2^-32, 2^30)).
func QuantizeMultiplier(real float64) (Requant, error) {
	if !(real > 0) || math.IsInf(real, 0) {
		return Requant{}, ErrRequant
	}
	frac, exp := math.Frexp(real)
	m := math.Round(frac * (1 << 31))
	if m == 1<<31 {
		m = 1 << 30
		exp++
	}
	r := Requant{Mult: int32(m), Shift: int32(exp)}
	if exp < MinShift || exp > MaxShift {
		return Requant{}, ErrRequant
	}
	return r, nil
}

// requantize maps an accumulator to int8: clamp(Apply(acc)+zp, lo, hi).
func requantize(acc int32, r Requant, zp, lo, hi int64) int8 {
	v := r.Apply(acc) + zp
	if v < lo {
		v = lo
	}
	if v > hi {
		v = hi
	}
	return int8(v)
}

// validRequants checks every multiplier and the length.
func validRequants(rq []Requant, n int) error {
	if len(rq) != n {
		return ErrLength
	}
	for i := range rq {
		if !rq[i].Valid() {
			return ErrRequant
		}
	}
	return nil
}

// validBias accepts nil/empty (no bias) or exactly n entries.
func validBias(bias []int32, n int) error {
	if len(bias) != 0 && len(bias) != n {
		return ErrLength
	}
	return nil
}

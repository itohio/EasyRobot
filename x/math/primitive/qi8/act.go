package qi8

import "math"

// ActNone returns the full int8 clamp range (no fused activation).
func ActNone() (min, max int8) { return -128, 127 }

// ActReLU returns the fused ReLU clamp range for an output tensor with zero
// point zp: [zp, 127]. scale is unused (kept for a uniform signature).
func ActReLU(scale float32, zp int8) (min, max int8) {
	_ = scale
	return zp, 127
}

// ActReLU6 returns the fused ReLU6 clamp range for an output tensor with the
// given scale (> 0) and zero point: [zp, Quantize6(scale, zp)].
func ActReLU6(scale float32, zp int8) (min, max int8) {
	return zp, Quantize6(scale, zp)
}

// Quantize6 returns the quantised value of 6.0 saturated to int8:
// clamp(zp + roundHalfAwayFromZero(6/scale), -128, 127).
func Quantize6(scale float32, zp int8) int8 {
	v := float64(zp) + math.Round(6/float64(scale))
	if !(v < 127) { // also catches +Inf/NaN from a non-positive scale
		return 127
	}
	if v < -128 {
		return -128
	}
	return int8(v)
}

// ReLU computes dst[i] = max(src[i], zp) (same scale and zero point in and
// out). dst and src may alias.
func ReLU(dst, src []int8, zp int8) error {
	if len(dst) != len(src) {
		return ErrLength
	}
	dst = dst[:len(src)]
	for i, v := range src {
		if v < zp {
			v = zp
		}
		dst[i] = v
	}
	return nil
}

// ReLU6 computes dst[i] = clamp(src[i], zp, q6), where q6 is the quantised
// value of 6.0 (see Quantize6). dst and src may alias. Returns ErrActivation
// when q6 < zp.
func ReLU6(dst, src []int8, zp, q6 int8) error {
	if len(dst) != len(src) {
		return ErrLength
	}
	if q6 < zp {
		return ErrActivation
	}
	dst = dst[:len(src)]
	for i, v := range src {
		if v < zp {
			v = zp
		}
		if v > q6 {
			v = q6
		}
		dst[i] = v
	}
	return nil
}

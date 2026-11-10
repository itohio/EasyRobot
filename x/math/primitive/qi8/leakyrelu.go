package qi8

import "math"

// LeakyReLUParams describes TFLite's int8 LEAKY_RELU with its own input and
// output quantisation. With v = x − InZP:
//
//	dst = clamp(OutZP + R(v, Identity), −128, 127)  for v ≥ 0
//	dst = clamp(OutZP + R(v, Alpha), −128, 127)     for v < 0
//
// R is Requant.ApplyTFLite (MultiplyByQuantizedMultiplier, double rounding)
// for RoundTFLite, which TFLite's reference kernel uses, or Requant.Apply
// (one exact rounding, half away from zero) for RoundExact. Identity and
// Alpha come from QuantizeLeakyReLU; either may be the zero Requant.
type LeakyReLUParams struct {
	InZP, OutZP     int8
	Identity, Alpha Requant
	Round           Rounding
}

// quantizeMultiplierTFLite is QuantizeMultiplier with TFLite's handling of
// tiny values: a real multiplier of 0, or one whose exponent falls below
// MinShift, becomes the zero Requant (TFLite's (0, 0)) instead of an error.
// Values of 2^30 or more are still ErrRequant (TFLite saturates them and
// then overflows int32).
func quantizeMultiplierTFLite(real float64) (Requant, error) {
	if real == 0 {
		return Requant{}, nil
	}
	r, err := QuantizeMultiplier(real)
	if err != nil && real > 0 && real < 1 {
		return Requant{}, nil
	}
	return r, err
}

// QuantizeLeakyReLU derives the two multipliers as TFLite's LeakyReluPrepare
// does, in float32 (then widened to float64 for QuantizeMultiplier):
//
//	identity = float32(inScale / outScale)
//	alpha    = float32(float32(inScale·alpha) / outScale)
//
// alpha = 0 gives the zero Alpha (negative inputs map to OutZP), as in
// TFLite; alpha > 1 is supported. Returns ErrRequant for non-positive or
// non-finite scales, a negative or non-finite alpha (TFLite accepts a
// negative alpha with a negative multiplier, which Requant cannot represent),
// or a multiplier of 2^30 or more.
func QuantizeLeakyReLU(inScale, outScale, alpha float32) (identity, alphaRQ Requant, err error) {
	if !(inScale > 0) || !(outScale > 0) || math.IsInf(float64(inScale), 0) || math.IsInf(float64(outScale), 0) {
		return Requant{}, Requant{}, ErrRequant
	}
	if !(alpha >= 0) || math.IsInf(float64(alpha), 0) {
		return Requant{}, Requant{}, ErrRequant
	}
	identity, err = quantizeMultiplierTFLite(float64(float32(inScale / outScale)))
	if err != nil {
		return Requant{}, Requant{}, err
	}
	alphaRQ, err = quantizeMultiplierTFLite(float64(float32(float32(inScale*alpha) / outScale)))
	if err != nil {
		return Requant{}, Requant{}, err
	}
	return identity, alphaRQ, nil
}

// LeakyReLU applies p to src (equal lengths, else ErrLength; ErrRequant for
// an invalid multiplier or Round). dst may alias src.
func LeakyReLU(dst, src []int8, p LeakyReLUParams) error {
	if len(dst) != len(src) {
		return ErrLength
	}
	if !p.Identity.Valid() || !p.Alpha.Valid() || !p.Round.valid() {
		return ErrRequant
	}
	leakyReLUGeneric(dst, src, p.InZP, p.OutZP, p.Identity, p.Alpha, p.Round)
	return nil
}

// leakyReLUGeneric is the LeakyReLU loop; equal lengths, multipliers and
// round valid. (Parameters by value: TinyGo heap-allocates p when its
// address is taken here.)
func leakyReLUGeneric(dst, src []int8, inZP, outZP int8, identity, alpha Requant, round Rounding) {
	dst = dst[:len(src)]
	zi, zo := int32(inZP), int64(outZP)
	for i, x := range src {
		v := int32(x) - zi
		r := identity
		if v < 0 {
			r = alpha
		}
		dst[i] = requantizeMode(v, r, zo, -128, 127, round)
	}
}

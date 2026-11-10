package qi8

import "math"

// RequantizeParams describes an int8 → int8 requantisation (TFLite
// QUANTIZE with an int8 input):
//
//	dst[i] = clamp(OutZP + round((src[i] − InZP)·M'), −128, 127)
//
// M' is RQ (from QuantizeRequantize). RoundExact rounds once, half away from
// zero (Requant.Apply); RoundTFLite is TFLite's MultiplyByQuantizedMultiplier
// (Requant.ApplyTFLite, double rounding), which TFLite's reference
// Requantize uses.
type RequantizeParams struct {
	InZP, OutZP int8
	RQ          Requant // from QuantizeRequantize
	Round       Rounding
}

// QuantizeRequantize derives TFLite's QUANTIZE multiplier
// inScale/outScale in float64 (kernels/quantize.cc) with QuantizeMultiplier.
// Equal scales give {2^30, 1}, which is exact in both rounding modes.
// Returns ErrRequant for non-positive or non-finite scales or an
// unrepresentable ratio.
func QuantizeRequantize(inScale, outScale float32) (Requant, error) {
	if !(inScale > 0) || !(outScale > 0) || math.IsInf(float64(inScale), 0) || math.IsInf(float64(outScale), 0) {
		return Requant{}, ErrRequant
	}
	return QuantizeMultiplier(float64(inScale) / float64(outScale))
}

// Requantize maps src to dst (equal lengths, else ErrLength; ErrRequant for
// an invalid RQ or Round). dst may alias src.
func Requantize(dst, src []int8, p RequantizeParams) error {
	if len(dst) != len(src) {
		return ErrLength
	}
	if !p.RQ.Valid() || !p.Round.valid() {
		return ErrRequant
	}
	requantizeGeneric(dst, src, &p)
	return nil
}

// requantizeGeneric is the Requantize loop; equal lengths, p valid.
func requantizeGeneric(dst, src []int8, p *RequantizeParams) {
	dst = dst[:len(src)]
	zi, zo := int32(p.InZP), int64(p.OutZP)
	for i, v := range src {
		dst[i] = requantizeMode(int32(v)-zi, p.RQ, zo, -128, 127, p.Round)
	}
}

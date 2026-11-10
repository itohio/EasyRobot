package qi8

import "math"

// MulParams describes a requantised element-wise product with cyclic
// broadcast of b:
//
//	dst[i] = clamp(ZPOut + round((a[i]−ZPA)·(b[j]−ZPB)·M'), Min, Max),  j = i mod len(b)
//
// M' is RQ (from QuantizeMul). Round selects the rounding of the product
// (|·| ≤ 255² so no overflow): RoundExact rounds once, half away from zero
// (Requant.Apply); RoundTFLite is TFLite's MultiplyByQuantizedMultiplier
// (Requant.ApplyTFLite, double rounding), which TFLite's int8 MUL uses.
// Min/Max is the fused activation clamp.
type MulParams struct {
	ZPA, ZPB, ZPOut int8
	RQ              Requant
	Min, Max        int8
	Round           Rounding
}

// QuantizeMul derives TFLite's int8 MUL multiplier: the real multiplier
// sA·sB/sOut is evaluated in float32 (float32 product, then float32
// division, as in kernels/mul.cc), widened to float64 and passed to
// QuantizeMultiplier. Returns ErrRequant for non-positive or non-finite
// scales or an unrepresentable multiplier.
func QuantizeMul(scaleA, scaleB, scaleOut float32) (Requant, error) {
	for _, s := range [3]float32{scaleA, scaleB, scaleOut} {
		if !(s > 0) || math.IsInf(float64(s), 0) {
			return Requant{}, ErrRequant
		}
	}
	real := float32(float32(scaleA*scaleB) / scaleOut)
	return QuantizeMultiplier(float64(real))
}

func (p *MulParams) valid() error {
	if !p.RQ.Valid() || !p.Round.valid() {
		return ErrRequant
	}
	if p.Min > p.Max {
		return ErrActivation
	}
	return nil
}

// Mul computes the requantised product (see MulParams). len(dst) must equal
// len(a), and len(b) must be positive and divide len(a) (b is the same
// shape as a, its trailing channel vector, or a scalar), else ErrLength.
// dst may alias a (not b, unless len(b) == len(a)).
func Mul(dst, a, b []int8, p MulParams) error {
	if len(dst) != len(a) || len(b) == 0 || len(a)%len(b) != 0 {
		return ErrLength
	}
	if err := p.valid(); err != nil {
		return err
	}
	mulGeneric(dst, a, b, &p)
	return nil
}

// mulGeneric is the Mul loop; lengths checked, p valid.
func mulGeneric(dst, a, b []int8, p *MulParams) {
	za, zb := int32(p.ZPA), int32(p.ZPB)
	zo, lo, hi := int64(p.ZPOut), int64(p.Min), int64(p.Max)
	nb := len(b)
	for off := 0; off < len(a); off += nb {
		ar, dr := a[off:off+nb], dst[off:off+nb]
		for j, bv := range b {
			acc := (int32(ar[j]) - za) * (int32(bv) - zb)
			dr[j] = requantizeMode(acc, p.RQ, zo, lo, hi, p.Round)
		}
	}
}

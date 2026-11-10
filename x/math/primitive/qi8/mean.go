package qi8

import (
	"math"
	"math/bits"
)

// maxMeanCount bounds H·W in MeanHW so that Σ(x − InZP) fits in int32, as
// TFLite's int32 accumulator requires: 255·count < 2^31.
const maxMeanCount = (1<<31 - 1) / 255

// MeanParams describes the requantised mean over H and W of an NHWC tensor
// (TFLite MEAN with axis [1, 2]). With S_c = Σ_{h,w} (x[h,w,c] − InZP) and
// n = H·W:
//
//	RoundExact:  dst[c] = clamp(OutZP + round(S_c·M'/n), −128, 127)
//
// with a single exact rounding, half away from zero (M' = RQ).
//
//	RoundTFLite: dst[c] = clamp(OutZP + MultiplyByQuantizedMultiplier(S_c, m, e), −128, 127)
//
// is TFLite's reference QuantizedMeanOrSum (what LiteRT runs for int8 MEAN,
// keep_dims or not): 1/n is folded into the multiplier in integers,
// k = min(⌊log2 n⌋, 32, 31 + RQ.Shift), m = ⌊RQ.Mult·2^k / n⌋, e = RQ.Shift − k,
// and the product is double-rounded like Requant.ApplyTFLite.
type MeanParams struct {
	InZP, OutZP int8
	RQ          Requant // from QuantizeMean
	Round       Rounding
}

// QuantizeMean derives the multiplier inScale/outScale in float64 (as
// TFLite's PrepareMeanOrSum) with QuantizeMultiplier. The 1/(H·W) factor is
// applied by MeanHW. Returns ErrRequant for non-positive or non-finite
// scales or an unrepresentable ratio.
func QuantizeMean(inScale, outScale float32) (Requant, error) {
	if !(inScale > 0) || !(outScale > 0) || math.IsInf(float64(inScale), 0) || math.IsInf(float64(outScale), 0) {
		return Requant{}, ErrRequant
	}
	return QuantizeMultiplier(float64(inScale) / float64(outScale))
}

// MeanHW computes the per-channel mean of src (shape in) into dst
// (1×1×in.C), see MeanParams. ErrShape for an invalid shape or
// H·W > (2^31−1)/255; ErrLength when len(src) != in.Len() or
// len(dst) != in.C; ErrRequant for an invalid RQ or Round.
func MeanHW(dst, src []int8, in Shape, p MeanParams) error {
	n, ok := in.size()
	if !ok {
		return ErrShape
	}
	count := in.H * in.W // ≤ n, cannot overflow
	if count > maxMeanCount {
		return ErrShape
	}
	if len(src) != n || len(dst) != in.C {
		return ErrLength
	}
	if !p.RQ.Valid() || !p.Round.valid() {
		return ErrRequant
	}
	meanGeneric(dst, src[:n], count, &p)
	return nil
}

// meanTFLiteRequant folds 1/count into r the way TFLite's reference
// QuantizedMeanOrSum does. The result may have Mult < 2^30 (it is only
// used with ApplyTFLite, which accepts any multiplier).
func meanTFLiteRequant(r Requant, count int) Requant {
	k := bits.Len64(uint64(count)) - 1
	k = min(k, 32, 31+int(r.Shift))
	m := (int64(r.Mult) << uint(k)) / int64(count)
	return Requant{Mult: int32(m), Shift: r.Shift - int32(k)}
}

// meanExact returns round(acc·M'/count), half away from zero, exactly.
// |acc| < 2^31 and Mult < 2^31, so |acc|·Mult < 2^62.
func meanExact(acc int32, r Requant, count int) int64 {
	neg := acc < 0
	num := uint64(acc)
	if neg {
		num = uint64(-int64(acc))
	}
	num *= uint64(r.Mult)
	s := uint(31 - r.Shift) // 1..62
	if int(s)+bits.Len64(uint64(count)) > 63 {
		return 0 // count·2^s ≥ 2^63 > 2·num
	}
	den := uint64(count) << s // < 2^63, even
	q := int64((num + den/2) / den)
	if neg {
		return -q
	}
	return q
}

// meanGeneric is the MeanHW loop; len(src) == count·len(dst), p valid.
func meanGeneric(dst, src []int8, count int, p *MeanParams) {
	c := len(dst)
	zi := int32(p.InZP)
	zo := int64(p.OutZP)
	tr := meanTFLiteRequant(p.RQ, count)
	for ch := range dst {
		var acc int32
		for i := ch; i < len(src); i += c {
			acc += int32(src[i]) - zi
		}
		var v int64
		if p.Round == RoundTFLite {
			v = int64(tr.ApplyTFLite(acc))
		} else {
			v = meanExact(acc, p.RQ, count)
		}
		dst[ch] = int8(min(max(v+zo, -128), 127))
	}
}

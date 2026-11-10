package qi8

import (
	"math"
	"math/bits"
)

// Softmax is TensorFlow Lite's int8 reference softmax (reference_ops::Softmax,
// the kernel TFLite Micro runs), in gemmlowp fixed point: differences to the
// row maximum are scaled to Q5.26, exponentiated with exp_on_negative_values,
// summed in Q12.19, and divided with a Newton–Raphson reciprocal. The output
// is quantised with scale 1/256 and zero point −128, as TFLite requires for
// int8 softmax. TFLite's default (non-reference) CPU kernel uses a lookup
// table and can differ by 1.

// SoftmaxParams is the prepared input scaling (TFLite SoftmaxParams):
// Mult/LeftShift represent beta·inputScale·2^26 (QuantizeMultiplier with a
// non-negative shift); differences below DiffMin give 0 (output −128).
type SoftmaxParams struct {
	Mult      int32
	LeftShift int32
	DiffMin   int32
}

// softmaxIntBits is TFLite's kScaledDiffIntegerBits; softmaxAccBits is
// kAccumulationIntegerBits.
const (
	softmaxIntBits = 5
	softmaxAccBits = 12
)

// QuantizeSoftmax computes SoftmaxParams from the softmax beta and the input
// scale (both float32, widened to float64 as TFLite's
// PreprocessSoftmaxScaling and CalculateInputRadius do). Returns ErrRequant
// for non-positive or non-finite values or a multiplier below 1.
func QuantizeSoftmax(beta, inScale float32) (SoftmaxParams, error) {
	if !(beta > 0) || !(inScale > 0) || math.IsInf(float64(beta), 0) || math.IsInf(float64(inScale), 0) {
		return SoftmaxParams{}, ErrRequant
	}
	real := math.Min(float64(beta)*float64(inScale)*(1<<(31-softmaxIntBits)), (1<<31)-1.0)
	r, err := QuantizeMultiplier(real)
	if err != nil || r.Shift < 0 {
		return SoftmaxParams{}, ErrRequant
	}
	maxInput := float64((1<<softmaxIntBits)-1) * float64(int64(1)<<(31-softmaxIntBits)) / float64(int64(1)<<uint(r.Shift))
	return SoftmaxParams{Mult: r.Mult, LeftShift: r.Shift, DiffMin: -int32(math.Floor(maxInput))}, nil
}

// valid checks the multiplier range and that (x − max)·2^LeftShift cannot
// overflow for x − max ≥ DiffMin.
func (p SoftmaxParams) valid() bool {
	if p.Mult < multMin || p.LeftShift < 0 || p.LeftShift > 30 || p.DiffMin > 0 {
		return false
	}
	return int64(p.DiffMin)<<uint(p.LeftShift) >= math.MinInt32
}

// Softmax computes the softmax of each row of depth elements of src into dst
// (equal lengths, a multiple of depth; dst may alias src).
func Softmax(dst, src []int8, depth int, p SoftmaxParams) error {
	if len(dst) != len(src) {
		return ErrLength
	}
	if depth <= 0 || len(src)%depth != 0 {
		return ErrShape
	}
	if !p.valid() {
		return ErrRequant
	}
	for r := 0; r < len(src); r += depth {
		softmaxRow(dst[r:r+depth], src[r:r+depth], &p)
	}
	return nil
}

func softmaxRow(dst, src []int8, p *SoftmaxParams) {
	mx := int32(src[0])
	for _, v := range src[1:] {
		mx = max(mx, int32(v))
	}
	var sum int32 // Q12.19
	for _, v := range src {
		d := int32(v) - mx
		if d >= p.DiffMin {
			e := expOnNegativeValues(srdhm(d<<uint(p.LeftShift), p.Mult))
			sum += roundingDivideByPOT(e, softmaxAccBits)
		}
	}
	scale, nbits := reciprocal(sum, softmaxAccBits)
	for i, v := range src {
		d := int32(v) - mx
		out := int32(-128)
		if d >= p.DiffMin {
			e := expOnNegativeValues(srdhm(d<<uint(p.LeftShift), p.Mult))
			out = roundingDivideByPOT(srdhm(scale, e), nbits+31-8) - 128
			out = min(max(out, -128), 127)
		}
		dst[i] = int8(out)
	}
}

// satMulPOT is gemmlowp's SaturatingRoundingMultiplyByPOT.
func satMulPOT(x int32, e int) int32 {
	switch {
	case e == 0:
		return x
	case e < 0:
		return roundingDivideByPOT(x, -e)
	}
	threshold := int32(1)<<uint(31-e) - 1
	if x > threshold {
		return math.MaxInt32
	}
	if x < -threshold {
		return math.MinInt32
	}
	return x << uint(e)
}

// expOnIntervalNegQuarterTo0 is gemmlowp's
// exp_on_interval_between_negative_one_quarter_and_0_excl (Q0.31 in and out):
// a Taylor expansion around −1/8.
func expOnIntervalNegQuarterTo0(a int32) int32 {
	const (
		constantTerm = 1895147668 // exp(−1/8)
		oneOver3     = 715827883
	)
	x := a + 1<<28
	x2 := srdhm(x, x)
	x3 := srdhm(x2, x)
	x4 := srdhm(x2, x2)
	x4Over4 := satMulPOT(x4, -2)
	t := satMulPOT(srdhm(x4Over4+x3, oneOver3)+x2, -1)
	return constantTerm + srdhm(constantTerm, x+t)
}

// expBarrel are exp(−2^k) in Q0.31 for k = −2…4 (gemmlowp's barrel shifter).
var expBarrel = [7]int32{1672461947, 1302514674, 790015084, 290630308, 39332535, 720401, 242}

// expOnNegativeValues is gemmlowp's exp_on_negative_values for a Q5.26
// input ≤ 0, result Q0.31.
func expOnNegativeValues(a int32) int32 {
	const frac = 31 - softmaxIntBits
	const quarter = int32(1) << (frac - 2)
	aMod := a&(quarter-1) - quarter
	result := expOnIntervalNegQuarterTo0(satMulPOT(aMod, softmaxIntBits))
	rem := aMod - a
	for k, m := range expBarrel {
		if rem&(int32(1)<<uint(frac-2+k)) != 0 {
			result = srdhm(result, m)
		}
	}
	if a == 0 {
		result = math.MaxInt32
	}
	return result
}

// oneOverOnePlusX is gemmlowp's one_over_one_plus_x_for_x_in_0_1 (Q0.31 in
// and out): three Newton–Raphson steps in Q2.29.
func oneOverOnePlusX(a int32) int32 {
	const (
		c48Over17    = 1515870810
		cNeg32Over17 = -1010580540
		oneQ2        = int32(1) << 29
	)
	s := int64(a) + math.MaxInt32
	sign := int64(1)
	if s < 0 {
		sign = -1
	}
	half := int32((s + sign) / 2) // RoundingHalfSum(a, 1)
	x := c48Over17 + srdhm(half, cNeg32Over17)
	for range 3 {
		x += satMulPOT(srdhm(x, oneQ2-srdhm(half, x)), 2)
	}
	return satMulPOT(x, 1)
}

// reciprocal is TFLite's GetReciprocal: 1/x for x in Q(intDigits) as a Q0.31
// scale and the number of bits x has above 1.0.
func reciprocal(x int32, intDigits int) (scale int32, bitsOverUnit int) {
	headroom := bits.LeadingZeros32(uint32(x))
	bitsOverUnit = intDigits - headroom
	shiftedMinusOne := int32(uint32(x)<<uint(headroom) - 1<<31)
	return oneOverOnePlusX(shiftedMinusOne), bitsOverUnit
}

package qi8

import "math"

// LogisticParams is TensorFlow Lite's int8 LOGISTIC lookup table:
// Table[uint8(x)] is the output for input x. The output quantisation is
// fixed at scale 1/256, zero point −128 (TFLite requires it).
type LogisticParams struct {
	Table [256]int8
}

// QuantizeLogistic builds the table exactly as TFLite's reference and
// generic-optimised int8 LOGISTIC kernels do (LUTPopulate with
// 1/(1+exp(−x)) in float32): for every input x,
//
//	d = float32(inScale · float32(x − inZP))
//	t = 1 / (1 + expf(−d))                    (float32)
//	q = clamp(roundf(t · 256) − 128, −128, 127)
//
// with roundf half away from zero (t·256 is exact). expf is taken as
// float32(math.Exp(float64(−d))), which equals the correctly rounded
// float32 exponential except in astronomically rare double-rounding cases.
// Returns ErrRequant for a non-positive or non-finite scale.
func QuantizeLogistic(inScale float32, inZP int8) (LogisticParams, error) {
	if !(inScale > 0) || math.IsInf(float64(inScale), 0) {
		return LogisticParams{}, ErrRequant
	}
	var p LogisticParams
	for x := -128; x <= 127; x++ {
		// Explicit float32 conversions round every step to float32 and stop
		// the compiler from fusing operations (Go spec: conversions round).
		d := float32(inScale * float32(x-int(inZP)))
		e := float32(math.Exp(float64(-d)))
		t := float32(1 / float32(1+e))
		q := math.Round(float64(t)*256) - 128
		p.Table[uint8(int8(x))] = int8(min(max(q, -128), 127))
	}
	return p, nil
}

// Logistic computes dst[i] = p.Table[uint8(src[i])] (equal lengths, else
// ErrLength); dst may alias src.
func Logistic(dst, src []int8, p LogisticParams) error {
	if len(dst) != len(src) {
		return ErrLength
	}
	logisticGeneric(dst, src, &p.Table)
	return nil
}

// logisticGeneric is the table lookup loop; len(dst) == len(src).
func logisticGeneric(dst, src []int8, table *[256]int8) {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = table[uint8(v)]
	}
}

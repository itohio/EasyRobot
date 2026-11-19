package qi8

import "math"

// Concat concatenates int8 tensors along one axis (TFLite CONCATENATION).
// Each srcs[k] is viewed as outer rows of len(srcs[k])/outer bytes, and dst
// receives row r of srcs[0], row r of srcs[1], … for r = 0 … outer−1. For an
// NHWC tensor with batch 1, outer is the product of the dimensions before
// the axis: H·W for C, H for W, 1 for H.
//
// It is a pure copy: TFLite's int8 CONCATENATION requires every input to
// share the output's scale and zero point (and no fused activation), so no
// requantisation is involved.
//
// ErrShape when outer ≤ 0; ErrLength when a source length is not a multiple
// of outer or the lengths do not add up to len(dst). dst must not overlap
// any source.
func Concat(dst []int8, srcs [][]int8, outer int) error {
	if outer <= 0 {
		return ErrShape
	}
	total := 0
	for _, s := range srcs {
		if len(s)%outer != 0 || len(s) > len(dst)-total {
			return ErrLength
		}
		total += len(s)
	}
	if total != len(dst) {
		return ErrLength
	}
	concatGeneric(dst, srcs, outer)
	return nil
}

// concatGeneric is the Concat copy loop; lengths checked.
func concatGeneric(dst []int8, srcs [][]int8, outer int) {
	off := 0
	for r := range outer {
		for _, s := range srcs {
			w := len(s) / outer
			off += copy(dst[off:], s[r*w:r*w+w])
		}
	}
}

// ConcatRescale is Concat for TFLite's uint8 CONCATENATION of inputs that
// do not share the output's quantisation: each value of srcs[k] goes
// through maps[k] (from QuantizeConcatRescale), or is copied when maps[k]
// is nil. len(maps) must equal len(srcs) (else ErrLength); the other
// errors are Concat's.
func ConcatRescale(dst []int8, srcs [][]int8, outer int, maps []*[256]int8) error {
	if len(maps) != len(srcs) {
		return ErrLength
	}
	if err := Concat(dst, srcs, outer); err != nil {
		return err
	}
	off := 0
	for range outer {
		for k, s := range srcs {
			w := len(s) / outer
			if m := maps[k]; m != nil {
				row := dst[off : off+w]
				for i, v := range row {
					row[i] = m[uint8(v)]
				}
			}
			off += w
		}
	}
	return nil
}

// QuantizeConcatRescale builds the map TFLite's uint8 CONCATENATION applies
// to an input quantised (inScale, inZP) for an output (outScale, outZP)
// (ConcatenationWithScaling, in float32), with the uint8 values held as
// int8 (value − 128, zero points too): for every input u (as uint8),
//
//	s = float32(inScale · float32(1/outScale))
//	b = float32(−float32(inZP + 128) · s)
//	y = clamp(roundf(fmaf(u, s, b)) + outZP + 128, 0, 255)
//
// with roundf half away from zero; the entry for u − 128 is y − 128. u·s + b
// rounds once, as one fused multiply-add: LiteRT's arm64 builds contract it
// (checked against the interpreter; an unfused build rounds the product
// first and can differ by one). Returns ErrRequant for a non-positive or
// non-finite scale.
func QuantizeConcatRescale(inScale float32, inZP int8, outScale float32, outZP int8) (*[256]int8, error) {
	bad := func(s float32) bool { return !(s > 0) || math.IsInf(float64(s), 0) }
	if bad(inScale) || bad(outScale) {
		return nil, ErrRequant
	}
	// Explicit float32 conversions round every step and stop fusion.
	s := float32(inScale * float32(1/outScale))
	b := float32(-float32(int(inZP)+128) * s)
	var m [256]int8
	for u := range 256 {
		// u has 8 bits and s 24, so the float64 product and sum are exact:
		// the conversion to float32 is fmaf's single rounding.
		v := float32(float64(float64(u)*float64(s)) + float64(b))
		y := int(math.Round(float64(v))) + int(outZP) + 128
		m[uint8(int8(u-128))] = int8(min(max(y, 0), 255) - 128)
	}
	return &m, nil
}

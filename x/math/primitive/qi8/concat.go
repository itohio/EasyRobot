package qi8

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

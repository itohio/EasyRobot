package helpers

import "math"

const MAX_DIMS = 16

// Numeric types that can be used with generic operations
type Numeric interface {
	~float64 | ~int64 | ~float32 | ~int | ~int32 | ~int16 | ~int8
}

// Types that need clamping when converting to int8 (sorted by type size)
type ClampableToInt8 interface {
	~float64 | ~int64 | ~float32 | ~int | ~int32 | ~int16
}

// Types that need clamping when converting to int16 (sorted by type size)
type ClampableToInt16 interface {
	~float64 | ~int64 | ~float32 | ~int | ~int32
}

// Types that need clamping when converting to int32 (sorted by type size)
type ClampableToInt32 interface {
	~float64 | ~int64 | ~float32 | ~int
}

// Types that need clamping when converting to int64 (sorted by type size)
type ClampableToInt64 interface {
	~float64 | ~float32
}

// ClampToInt8Value clamps a single value to int8 range [-128, 127].
func ClampToInt8Value[U ClampableToInt8](v U) int8 {
	if v > U(math.MaxInt8) {
		return math.MaxInt8
	}
	if v < U(math.MinInt8) {
		return math.MinInt8
	}
	return int8(v)
}

// ClampToInt16Value clamps a single value to int16 range [-32768, 32767].
func ClampToInt16Value[U ClampableToInt16](v U) int16 {
	if v > U(math.MaxInt16) {
		return math.MaxInt16
	}
	if v < U(math.MinInt16) {
		return math.MinInt16
	}
	return int16(v)
}

// ClampToInt32Value clamps a single value to int32 range.
func ClampToInt32Value[U ClampableToInt32](v U) int32 {
	if v > U(math.MaxInt32) {
		return math.MaxInt32
	}
	if v < U(math.MinInt32) {
		return math.MinInt32
	}
	return int32(v)
}

// ClampToInt64Value clamps a single value to int64 range.
func ClampToInt64Value[U ClampableToInt64](v U) int64 {
	if v > U(int64(math.MaxInt64)) {
		return math.MaxInt64
	}
	if v < U(int64(math.MinInt64)) {
		return math.MinInt64
	}
	return int64(v)
}

// ComputeStridesRank returns the rank (number of dimensions) of the shape.
// This is a lightweight function that avoids computing full strides when only the rank is needed.
func ComputeStridesRank(shape []int) int {
	return len(shape)
}

// ComputeStrides computes the canonical row-major strides for the given shape into dst.
// If dst is nil or has insufficient capacity, a stack-allocated array is used.
// Example: shape [2,3,4] -> strides [12,4,1].
// Returns the slice containing the computed strides.
// Note: Shapes are constrained to MAX_DIMS (16) dimensions, so stack allocation is always used.
func ComputeStrides(dst []int, shape []int) []int {
	if len(shape) == 0 {
		return nil
	}

	rank := len(shape)
	if dst == nil || len(dst) < rank {
		// Always use stack-allocated array (shapes are constrained to MAX_DIMS)
		var static [MAX_DIMS]int
		dst = static[:rank]
	} else {
		dst = dst[:rank]
	}

	stride := 1
	for i := rank - 1; i >= 0; i-- {
		dst[i] = stride
		stride *= shape[i]
	}

	return dst
}

// SizeFromShape computes the total number of elements described by the shape.
func SizeFromShape(shape []int) int {
	size := 1
	if len(shape) == 0 {
		return 0
	}
	for _, dim := range shape {
		if dim <= 0 {
			return 0
		}
		size *= dim
	}
	return size
}

// EnsureStrides ensures strides are valid for the shape, computing canonical strides into dst if needed.
// If the provided strides match the shape rank, they are returned as-is.
// Otherwise, canonical row-major strides are computed into dst (or stack-allocated if dst is nil).
// Returns the slice containing the valid strides.
func EnsureStrides(dst []int, strides []int, shape []int) []int {
	rank := ComputeStridesRank(shape)
	if rank == 0 {
		return nil
	}
	if len(strides) != rank {
		return ComputeStrides(dst, shape)
	}
	return strides
}

// IsContiguous reports whether the strides describe a dense row-major layout for the shape.
// Uses stack-allocated array for comparison to avoid heap allocation.
// Note: Shapes are constrained to MAX_DIMS (16) dimensions.
func IsContiguous(strides []int, shape []int) bool {
	rank := ComputeStridesRank(shape)
	if rank == 0 {
		return true
	}
	// Early exit if rank doesn't match
	if len(strides) != rank {
		return false
	}
	// Always use stack-allocated array (shapes are constrained to MAX_DIMS)
	var static [MAX_DIMS]int
	canonical := ComputeStrides(static[:rank], shape)
	for i := range canonical {
		if strides[i] != canonical[i] {
			return false
		}
	}
	return true
}

// AdvanceOffsets advances the multi-dimensional indices/offsets tuple.
// Accepts two stride slices (stridesDst and stridesSrc) instead of a slice of slices
// to avoid allocation and reduce bounds checks.
// Returns true if the iteration should continue, false when the final
// element has been processed.
func AdvanceOffsets(shape []int, indices []int, offsets []int, stridesDst, stridesSrc []int) bool {
	if len(shape) == 0 {
		return false
	}

	for dim := len(shape) - 1; dim >= 0; dim-- {
		indices[dim]++
		// Pre-compute stride values to reduce nested array access
		strideDst := stridesDst[dim]
		strideSrc := stridesSrc[dim]
		offsets[0] += strideDst
		offsets[1] += strideSrc

		if indices[dim] < shape[dim] {
			return true
		}

		// Reset offsets when dimension wraps
		offsets[0] -= strideDst * shape[dim]
		offsets[1] -= strideSrc * shape[dim]
		indices[dim] = 0
	}

	return false
}

// AdvanceOffsets3 advances offsets for 3 arrays incrementally (like AdvanceOffsets but for 3 strides).
// Updates offsets[0] using stridesDst, offsets[1] using stridesA, and offsets[2] using stridesB.
// Returns true if the iteration should continue, false when the final element has been processed.
func AdvanceOffsets3(shape []int, indices []int, offsets []int, stridesDst, stridesA, stridesB []int) bool {
	if len(shape) == 0 {
		return false
	}

	for dim := len(shape) - 1; dim >= 0; dim-- {
		indices[dim]++
		// Pre-compute stride values to reduce nested array access
		strideDst := stridesDst[dim]
		strideA := stridesA[dim]
		strideB := stridesB[dim]
		offsets[0] += strideDst
		offsets[1] += strideA
		offsets[2] += strideB

		if indices[dim] < shape[dim] {
			return true
		}

		// Reset offsets when dimension wraps
		offsets[0] -= strideDst * shape[dim]
		offsets[1] -= strideA * shape[dim]
		offsets[2] -= strideB * shape[dim]
		indices[dim] = 0
	}

	return false
}

// AdvanceOffsets4 advances offsets for 4 arrays incrementally (like AdvanceOffsets but for 4 strides).
// Updates offsets[0] using stridesDst, offsets[1] using stridesCond, offsets[2] using stridesA, and offsets[3] using stridesB.
// Returns true if the iteration should continue, false when the final element has been processed.
func AdvanceOffsets4(shape []int, indices []int, offsets []int, stridesDst, stridesCond, stridesA, stridesB []int) bool {
	if len(shape) == 0 {
		return false
	}

	for dim := len(shape) - 1; dim >= 0; dim-- {
		indices[dim]++
		// Pre-compute stride values to reduce nested array access
		strideDst := stridesDst[dim]
		strideCond := stridesCond[dim]
		strideA := stridesA[dim]
		strideB := stridesB[dim]
		offsets[0] += strideDst
		offsets[1] += strideCond
		offsets[2] += strideA
		offsets[3] += strideB

		if indices[dim] < shape[dim] {
			return true
		}

		// Reset offsets when dimension wraps
		offsets[0] -= strideDst * shape[dim]
		offsets[1] -= strideCond * shape[dim]
		offsets[2] -= strideA * shape[dim]
		offsets[3] -= strideB * shape[dim]
		indices[dim] = 0
	}

	return false
}

// IterateOffsets iterates over all multi-dimensional indices in the given shape,
// computing linear offsets for each set of strides, and calls the callback with offsets.
// This is a convenience wrapper around AdvanceOffsets.
// Uses stack-allocated arrays for indices and offsets (shapes are constrained to MAX_DIMS).
func IterateOffsets(shape []int, stridesDst, stridesSrc []int, callback func(offsets []int)) {
	if len(shape) == 0 {
		return
	}

	rank := len(shape)
	// Always use stack allocation (shapes are constrained to MAX_DIMS)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]

	for {
		callback(offsets)
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// IterateOffsetsWithIndices iterates over all multi-dimensional indices in the given shape,
// computing linear offsets for each set of strides, and calls the callback with both indices and offsets.
// This is a convenience wrapper around AdvanceOffsets.
// Uses stack-allocated arrays for indices and offsets (shapes are constrained to MAX_DIMS).
func IterateOffsetsWithIndices(shape []int, stridesDst, stridesSrc []int, callback func(indices []int, offsets []int)) {
	if len(shape) == 0 {
		return
	}

	rank := len(shape)
	// Always use stack allocation (shapes are constrained to MAX_DIMS)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]

	for {
		callback(indices, offsets)
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ComputeStrideOffset computes the linear offset from multi-dimensional indices and strides.
// Uses bound check elimination trick to improve performance for short, critical loops.
func ComputeStrideOffset(indices []int, strides []int) int {
	offset := 0
	n := len(indices)
	if n > 0 {
		_ = indices[n-1]
		_ = strides[n-1]
	}
	for i := range n {
		offset += indices[i] * strides[i]
	}
	return offset
}

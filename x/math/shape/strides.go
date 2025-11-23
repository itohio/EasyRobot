package shape

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

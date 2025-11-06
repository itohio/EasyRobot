package generics

// ComputeStridesRank returns the rank (number of dimensions) of the shape.
// This is a lightweight function that avoids computing full strides when only the rank is needed.
func ComputeStridesRank(shape []int) int {
	return len(shape)
}

// ComputeStrides returns the canonical row-major strides for the given shape.
// Example: shape [2,3,4] -> strides [12,4,1].
func ComputeStrides(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}

	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return strides
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

// EnsureStrides returns the provided strides if they match the shape; otherwise it falls back to canonical row-major strides.
func EnsureStrides(strides []int, shape []int) []int {
	rank := ComputeStridesRank(shape)
	if rank == 0 {
		return nil
	}
	if len(strides) != rank {
		return ComputeStrides(shape)
	}
	return strides
}

// IsContiguous reports whether the strides describe a dense row-major layout for the shape.
func IsContiguous(strides []int, shape []int) bool {
	rank := ComputeStridesRank(shape)
	if rank == 0 {
		return true
	}
	// Early exit if rank doesn't match
	if len(strides) != rank {
		return false
	}
	// Only compute full strides if rank matches
	canonical := ComputeStrides(shape)
	for i := range canonical {
		if strides[i] != canonical[i] {
			return false
		}
	}
	return true
}

// AdvanceOffsets advances the multi-dimensional indices/offsets tuple.
// Returns true if the iteration should continue, false when the final
// element has been processed.
func AdvanceOffsets(shape []int, indices []int, offsets []int, strides [][]int) bool {
	if len(shape) == 0 {
		return false
	}

	for dim := len(shape) - 1; dim >= 0; dim-- {
		indices[dim]++
		for buf := range offsets {
			offsets[buf] += strides[buf][dim]
		}

		if indices[dim] < shape[dim] {
			return true
		}

		for buf := range offsets {
			offsets[buf] -= strides[buf][dim] * shape[dim]
		}
		indices[dim] = 0
	}

	return false
}

// IterateOffsets iterates over all multi-dimensional indices in the given shape,
// computing linear offsets for each set of strides, and calls the callback with offsets.
// This is a convenience wrapper around AdvanceOffsets.
func IterateOffsets(shape []int, strides [][]int, callback func(offsets []int)) {
	if len(shape) == 0 {
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, len(strides))

	for {
		callback(offsets)
		if !AdvanceOffsets(shape, indices, offsets, strides) {
			break
		}
	}
}

// IterateOffsetsWithIndices iterates over all multi-dimensional indices in the given shape,
// computing linear offsets for each set of strides, and calls the callback with both indices and offsets.
// This is a convenience wrapper around AdvanceOffsets.
func IterateOffsetsWithIndices(shape []int, strides [][]int, callback func(indices []int, offsets []int)) {
	if len(shape) == 0 {
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, len(strides))

	for {
		callback(indices, offsets)
		if !AdvanceOffsets(shape, indices, offsets, strides) {
			break
		}
	}
}

// ComputeStrideOffset computes the linear offset from multi-dimensional indices and strides.
func ComputeStrideOffset(indices []int, strides []int) int {
	offset := 0
	for i := range indices {
		offset += indices[i] * strides[i]
	}
	return offset
}

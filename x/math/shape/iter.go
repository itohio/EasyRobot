package shape

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

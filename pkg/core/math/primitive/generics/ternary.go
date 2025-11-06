package generics

// ElemWhere writes elements from a where condition is true, otherwise from b.
// condition, a, b must have compatible shapes/strides.
// condition > 0 means true, otherwise false.
func ElemWhere[T Numeric](dst, condition, a, b []T, shape []int, stridesDst, stridesCond, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesCond = EnsureStrides(stridesCond, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesCond, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		for i := 0; i < size; i++ {
			if condition[i] > 0 {
				dst[i] = a[i]
			} else {
				dst[i] = b[i]
			}
		}
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 4)
	strideSet := [][]int{stridesDst, stridesCond, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		cIdx := offsets[1]
		aIdx := offsets[2]
		bIdx := offsets[3]
		if condition[cIdx] > 0 {
			dst[dIdx] = a[aIdx]
		} else {
			dst[dIdx] = b[bIdx]
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}


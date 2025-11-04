package fp32

// ElemAdd writes element-wise sum of a and b into dst for the provided shape/strides.
func ElemAdd(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		return av + bv
	})
}

// ElemSub writes element-wise difference of a and b into dst (dst = a - b).
func ElemSub(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		return av - bv
	})
}

// ElemMul writes element-wise product of a and b into dst.
func ElemMul(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		return av * bv
	})
}

// ElemDiv writes element-wise division of a by b into dst, skipping positions where b == 0.
// When the divisor is zero, the destination retains its previous value to match existing tensor semantics.
func ElemDiv(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		for i := 0; i < size; i++ {
			bv := b[i]
			if bv != 0 {
				dst[i] = a[i] / bv
			}
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 3)
	strideSet := [][]int{stridesDst, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		bv := b[bIdx]
		if bv != 0 {
			dst[dIdx] = a[aIdx] / bv
		}
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemScale multiplies dst by the given scalar (in-place) for the provided shape/strides.
func ElemScale(dst []float32, scalar float32, shape []int, stridesDst []int) {
	if scalar == 1.0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}
	if IsContiguous(stridesDst, shape) {
		Scal(dst, 1, size, scalar)
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 1)
	strideSet := [][]int{stridesDst}
	for {
		dIdx := offsets[0]
		dst[dIdx] *= scalar
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemCopy copies src into dst respecting the supplied shape/strides.
func ElemCopy(dst, src []float32, shape []int, stridesDst, stridesSrc []int) {
	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		Copy(dst, src, 1, 1, size)
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = src[sIdx]
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemWhere writes elements from a where condition is true, otherwise from b.
// condition, a, b must have compatible shapes/strides.
func ElemWhere(dst, condition, a, b []float32, shape []int, stridesDst, stridesCond, stridesA, stridesB []int) {
	applyElemTernary(dst, condition, a, b, shape, stridesDst, stridesCond, stridesA, stridesB, func(cv, av, bv float32) float32 {
		if cv > 0 { // condition > 0 means true
			return av
		}
		return bv
	})
}

// ElemGreaterThan writes 1.0 where a > b, 0.0 otherwise.
func ElemGreaterThan(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int) {
	applyElemBinary(dst, a, b, shape, stridesDst, stridesA, stridesB, func(av, bv float32) float32 {
		if av > bv {
			return 1.0
		}
		return 0.0
	})
}

func applyElemBinary(dst, a, b []float32, shape []int, stridesDst, stridesA, stridesB []int, op func(float32, float32) float32) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		for i := 0; i < size; i++ {
			dst[i] = op(a[i], b[i])
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 3)
	strideSet := [][]int{stridesDst, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		dst[dIdx] = op(a[aIdx], b[bIdx])
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

func applyElemTernary(dst, condition, a, b []float32, shape []int, stridesDst, stridesCond, stridesA, stridesB []int, op func(float32, float32, float32) float32) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesCond = EnsureStrides(stridesCond, shape)
	stridesA = EnsureStrides(stridesA, shape)
	stridesB = EnsureStrides(stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesCond, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		for i := 0; i < size; i++ {
			dst[i] = op(condition[i], a[i], b[i])
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 4)
	strideSet := [][]int{stridesDst, stridesCond, stridesA, stridesB}
	for {
		dIdx := offsets[0]
		cIdx := offsets[1]
		aIdx := offsets[2]
		bIdx := offsets[3]
		dst[dIdx] = op(condition[cIdx], a[aIdx], b[bIdx])
		if !advanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

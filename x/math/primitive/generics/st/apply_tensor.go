package st

import (
	. "github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
)

// ElemApplyBinary applies a binary function element-wise to contiguous arrays: dst[i] = op(a[i], b[i]).
// Optimized for the common case of contiguous memory.
func ElemApplyBinary[T Numeric](dst, a, b []T, n int, op func(T, T) T) {
	if n == 0 {
		return
	}
	a = a[:n]
	b = b[:n]
	for i := range n {
		dst[i] = op(a[i], b[i])
	}
}

// ElemApplyBinaryStrided applies a binary function element-wise with strides: dst[i] = op(a[i], b[i]).
// Handles both contiguous and strided cases with optimized paths.
func ElemApplyBinaryStrided[T Numeric](dst, a, b []T, shape []int, stridesDst, stridesA, stridesB []int, op func(T, T) T) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	var bStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)
	stridesB = EnsureStrides(bStridesStatic[:len(shape)], stridesB, shape)

	rank := len(shape)
	isContiguous := IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape)

	// Use optimized Vec/Mat variants for rank 1-2, or contiguous cases
	if isContiguous {
		// Contiguous: use Vec variant (works for any rank as 1D)
		ElemVecApplyBinaryStrided(dst, a, b, size, 1, 1, 1, op)
		return
	}

	// Strided path: delegate to Vec/Mat for rank 1-2, otherwise use AdvanceOffsets3
	switch rank {
	case 1:
		// 1D case: delegate to optimized vector function
		ElemVecApplyBinaryStrided(dst, a, b, shape[0], stridesDst[0], stridesA[0], stridesB[0], op)
		return
	case 2:
		// 2D case: delegate to optimized matrix function
		// Leading dimension is stride[0], inner dimension is stride[1]
		ElemMatApplyBinaryStrided(dst, a, b, shape[0], shape[1], stridesDst[0], stridesA[0], stridesB[0], op)
		return
	default:
		// Higher ranks: recursively decompose to leverage optimized lower-dim operations
		// Iterate over first dimension, recursively process remaining dimensions
		n0 := shape[0]
		dStride0 := stridesDst[0]
		aStride0 := stridesA[0]
		bStride0 := stridesB[0]

		remainingShape := shape[1:]
		remainingStridesDst := stridesDst[1:]
		remainingStridesA := stridesA[1:]
		remainingStridesB := stridesB[1:]

		for i0 := range n0 {
			dBase := i0 * dStride0
			aBase := i0 * aStride0
			bBase := i0 * bStride0

			// Recursively process remaining dimensions
			ElemApplyBinaryStrided(
				dst[dBase:],
				a[aBase:],
				b[bBase:],
				remainingShape,
				remainingStridesDst,
				remainingStridesA,
				remainingStridesB,
				op,
			)
		}
	}
}

// ElemApplyUnary applies a unary function element-wise to contiguous arrays: dst[i] = op(src[i]).
// Optimized for the common case of contiguous memory.
func ElemApplyUnary[T Numeric](dst, src []T, n int, op func(T) T) {
	if n == 0 {
		return
	}
	dst = dst[:n]
	src = src[:n]
	for i := range n {
		dst[i] = op(src[i])
	}
}

// ElemApplyUnaryStrided applies a unary function element-wise with strides: dst[i] = op(src[i]).
// Handles both contiguous and strided cases with optimized paths.
func ElemApplyUnaryStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int, op func(T) T) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		dst = dst[:size]
		src = src[:size]
		for i := range size {
			dst[i] = op(src[i])
		}
		return
	}

	// Strided path: delegate to Vec/Mat for rank 1-2, otherwise recursively decompose
	rank := len(shape)
	switch rank {
	case 1:
		// 1D case: delegate to optimized vector function
		ElemVecApplyUnaryStrided(dst, src, shape[0], stridesDst[0], stridesSrc[0], op)
		return
	case 2:
		// 2D case: delegate to optimized matrix function
		ElemMatApplyUnaryStrided(dst, src, shape[0], shape[1], stridesDst[0], stridesSrc[0], op)
		return
	default:
		// Higher ranks: recursively decompose
		n0 := shape[0]
		dStride0 := stridesDst[0]
		sStride0 := stridesSrc[0]

		remainingShape := shape[1:]
		remainingStridesDst := stridesDst[1:]
		remainingStridesSrc := stridesSrc[1:]

		for i0 := range n0 {
			dBase := i0 * dStride0
			sBase := i0 * sStride0

			ElemApplyUnaryStrided(
				dst[dBase:],
				src[sBase:],
				remainingShape,
				remainingStridesDst,
				remainingStridesSrc,
				op,
			)
		}
	}
}

// ElemApplyTernary applies a ternary function element-wise to contiguous arrays: dst[i] = op(condition[i], a[i], b[i]).
// Optimized for the common case of contiguous memory.
func ElemApplyTernary[T Numeric](dst, condition, a, b []T, n int, op func(T, T, T) T) {
	if n == 0 {
		return
	}
	dst = dst[:n]
	condition = condition[:n]
	a = a[:n]
	b = b[:n]
	for i := range n {
		dst[i] = op(condition[i], a[i], b[i])
	}
}

// ElemApplyTernaryStrided applies a ternary function element-wise with strides: dst[i] = op(condition[i], a[i], b[i]).
// Handles both contiguous and strided cases with optimized paths.
func ElemApplyTernaryStrided[T Numeric](dst, condition, a, b []T, shape []int, stridesDst, stridesCond, stridesA, stridesB []int, op func(T, T, T) T) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var condStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	var bStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesCond = EnsureStrides(condStridesStatic[:len(shape)], stridesCond, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)
	stridesB = EnsureStrides(bStridesStatic[:len(shape)], stridesB, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesCond, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		dst = dst[:size]
		condition = condition[:size]
		a = a[:size]
		b = b[:size]
		for i := range size {
			dst[i] = op(condition[i], a[i], b[i])
		}
		return
	}

	// Strided path: delegate to Vec/Mat for rank 1-2, otherwise recursively decompose
	rank := len(shape)
	switch rank {
	case 1:
		// 1D case: delegate to optimized vector function
		ElemVecApplyTernaryStrided(dst, condition, a, b, shape[0], stridesDst[0], stridesCond[0], stridesA[0], stridesB[0], op)
		return
	case 2:
		// 2D case: delegate to optimized matrix function
		ElemMatApplyTernaryStrided(dst, condition, a, b, shape[0], shape[1], stridesDst[0], stridesCond[0], stridesA[0], stridesB[0], op)
		return
	default:
		// Higher ranks: recursively decompose
		n0 := shape[0]
		dStride0 := stridesDst[0]
		cStride0 := stridesCond[0]
		aStride0 := stridesA[0]
		bStride0 := stridesB[0]

		remainingShape := shape[1:]
		remainingStridesDst := stridesDst[1:]
		remainingStridesCond := stridesCond[1:]
		remainingStridesA := stridesA[1:]
		remainingStridesB := stridesB[1:]

		for i0 := range n0 {
			dBase := i0 * dStride0
			cBase := i0 * cStride0
			aBase := i0 * aStride0
			bBase := i0 * bStride0

			ElemApplyTernaryStrided(
				dst[dBase:],
				condition[cBase:],
				a[aBase:],
				b[bBase:],
				remainingShape,
				remainingStridesDst,
				remainingStridesCond,
				remainingStridesA,
				remainingStridesB,
				op,
			)
		}
	}
}

// ElemApplyUnaryScalar applies a unary function with a scalar parameter to contiguous arrays: dst[i] = op(src[i], scalar).
// Optimized for the common case of contiguous memory.
func ElemApplyUnaryScalar[T Numeric](dst, src []T, scalar T, n int, op func(T, T) T) {
	if n == 0 {
		return
	}
	dst = dst[:n]
	src = src[:n]
	for i := range n {
		dst[i] = op(src[i], scalar)
	}
}

// ElemApplyUnaryScalarStrided applies a unary function with a scalar parameter with strides: dst[i] = op(src[i], scalar).
// Handles both contiguous and strided cases with optimized paths.
func ElemApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int, op func(T, T) T) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		dst = dst[:size]
		src = src[:size]
		for i := range size {
			dst[i] = op(src[i], scalar)
		}
		return
	}

	// Strided path: delegate to Vec/Mat for rank 1-2, otherwise recursively decompose
	rank := len(shape)
	switch rank {
	case 1:
		// 1D case: delegate to optimized vector function
		ElemVecApplyUnaryScalarStrided(dst, src, scalar, shape[0], stridesDst[0], stridesSrc[0], op)
		return
	case 2:
		// 2D case: delegate to optimized matrix function
		ElemMatApplyUnaryScalarStrided(dst, src, scalar, shape[0], shape[1], stridesDst[0], stridesSrc[0], op)
		return
	default:
		// Higher ranks: recursively decompose
		n0 := shape[0]
		dStride0 := stridesDst[0]
		sStride0 := stridesSrc[0]

		remainingShape := shape[1:]
		remainingStridesDst := stridesDst[1:]
		remainingStridesSrc := stridesSrc[1:]

		for i0 := range n0 {
			dBase := i0 * dStride0
			sBase := i0 * sStride0

			ElemApplyUnaryScalarStrided(
				dst[dBase:],
				src[sBase:],
				scalar,
				remainingShape,
				remainingStridesDst,
				remainingStridesSrc,
				op,
			)
		}
	}
}

// ElemApplyBinaryScalar applies a binary function with a scalar parameter to contiguous arrays: dst[i] = op(a[i], scalar).
// Optimized for the common case of contiguous memory.
func ElemApplyBinaryScalar[T Numeric](dst, a []T, scalar T, n int, op func(T, T) T) {
	if n == 0 {
		return
	}
	dst = dst[:n]
	a = a[:n]
	for i := range n {
		dst[i] = op(a[i], scalar)
	}
}

// ElemApplyBinaryScalarStrided applies a binary function with a scalar parameter with strides: dst[i] = op(a[i], scalar).
// Handles both contiguous and strided cases with optimized paths.
func ElemApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, shape []int, stridesDst, stridesA []int, op func(T, T) T) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		dst = dst[:size]
		a = a[:size]
		for i := range size {
			dst[i] = op(a[i], scalar)
		}
		return
	}

	// Strided path: delegate to Vec/Mat for rank 1-2, otherwise recursively decompose
	rank := len(shape)
	switch rank {
	case 1:
		// 1D case: delegate to optimized vector function
		ElemVecApplyBinaryScalarStrided(dst, a, scalar, shape[0], stridesDst[0], stridesA[0], op)
		return
	case 2:
		// 2D case: delegate to optimized matrix function
		ElemMatApplyBinaryScalarStrided(dst, a, scalar, shape[0], shape[1], stridesDst[0], stridesA[0], op)
		return
	default:
		// Higher ranks: recursively decompose
		n0 := shape[0]
		dStride0 := stridesDst[0]
		aStride0 := stridesA[0]

		remainingShape := shape[1:]
		remainingStridesDst := stridesDst[1:]
		remainingStridesA := stridesA[1:]

		for i0 := range n0 {
			dBase := i0 * dStride0
			aBase := i0 * aStride0

			ElemApplyBinaryScalarStrided(
				dst[dBase:],
				a[aBase:],
				scalar,
				remainingShape,
				remainingStridesDst,
				remainingStridesA,
				op,
			)
		}
	}
}

// ElemApplyTernaryScalar applies a ternary function with a scalar parameter to contiguous arrays: dst[i] = op(condition[i], a[i], scalar).
// Optimized for the common case of contiguous memory.
func ElemApplyTernaryScalar[T Numeric](dst, condition, a []T, scalar T, n int, op func(T, T, T) T) {
	if n == 0 {
		return
	}
	condition = condition[:n]
	a = a[:n]
	for i := range n {
		dst[i] = op(condition[i], a[i], scalar)
	}
}

// ElemApplyTernaryScalarStrided applies a ternary function with a scalar parameter with strides: dst[i] = op(condition[i], a[i], scalar).
// Handles both contiguous and strided cases with optimized paths.
func ElemApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, shape []int, stridesDst, stridesCond, stridesA []int, op func(T, T, T) T) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var condStridesStatic [MAX_DIMS]int
	var aStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesCond = EnsureStrides(condStridesStatic[:len(shape)], stridesCond, shape)
	stridesA = EnsureStrides(aStridesStatic[:len(shape)], stridesA, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesCond, shape) && IsContiguous(stridesA, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		dst = dst[:size]
		condition = condition[:size]
		a = a[:size]
		for i := range size {
			dst[i] = op(condition[i], a[i], scalar)
		}
		return
	}

	// Strided path: delegate to Vec/Mat for rank 1-2, otherwise recursively decompose
	rank := len(shape)
	switch rank {
	case 1:
		// 1D case: delegate to optimized vector function
		ElemVecApplyTernaryScalarStrided(dst, condition, a, scalar, shape[0], stridesDst[0], stridesCond[0], stridesA[0], op)
		return
	case 2:
		// 2D case: delegate to optimized matrix function
		ElemMatApplyTernaryScalarStrided(dst, condition, a, scalar, shape[0], shape[1], stridesDst[0], stridesCond[0], stridesA[0], op)
		return
	default:
		// Higher ranks: recursively decompose
		n0 := shape[0]
		dStride0 := stridesDst[0]
		cStride0 := stridesCond[0]
		aStride0 := stridesA[0]

		remainingShape := shape[1:]
		remainingStridesDst := stridesDst[1:]
		remainingStridesCond := stridesCond[1:]
		remainingStridesA := stridesA[1:]

		for i0 := range n0 {
			dBase := i0 * dStride0
			cBase := i0 * cStride0
			aBase := i0 * aStride0

			ElemApplyTernaryScalarStrided(
				dst[dBase:],
				condition[cBase:],
				a[aBase:],
				scalar,
				remainingShape,
				remainingStridesDst,
				remainingStridesCond,
				remainingStridesA,
				op,
			)
		}
	}
}

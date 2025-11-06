package generics

// ElemVecApplyUnaryStrided applies a unary function to a vector: dst[i] = op(src[i]).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyUnaryStrided[T Numeric](dst, src []T, n int, strideDst, strideSrc int, op func(T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		// Fast path: contiguous vectors
		for i := 0; i < n; i++ {
			dst[i] = op(src[i])
		}
		return
	}

	// Strided path
	dIdx := 0
	sIdx := 0
	for i := 0; i < n; i++ {
		dst[dIdx] = op(src[sIdx])
		dIdx += strideDst
		sIdx += strideSrc
	}
}

// ElemVecApplyBinaryStrided applies a binary function to vectors: dst[i] = op(a[i], b[i]).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyBinaryStrided[T Numeric](dst, a, b []T, n int, strideDst, strideA, strideB int, op func(T, T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideA == 1 && strideB == 1 {
		// Fast path: contiguous vectors
		for i := 0; i < n; i++ {
			dst[i] = op(a[i], b[i])
		}
		return
	}

	// Strided path
	dIdx := 0
	aIdx := 0
	bIdx := 0
	for i := 0; i < n; i++ {
		dst[dIdx] = op(a[aIdx], b[bIdx])
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
	}
}

// ElemVecApplyTernaryStrided applies a ternary function to vectors: dst[i] = op(condition[i], a[i], b[i]).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyTernaryStrided[T Numeric](dst, condition, a, b []T, n int, strideDst, strideCond, strideA, strideB int, op func(T, T, T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideCond == 1 && strideA == 1 && strideB == 1 {
		// Fast path: contiguous vectors
		for i := 0; i < n; i++ {
			dst[i] = op(condition[i], a[i], b[i])
		}
		return
	}

	// Strided path
	dIdx := 0
	cIdx := 0
	aIdx := 0
	bIdx := 0
	for i := 0; i < n; i++ {
		dst[dIdx] = op(condition[cIdx], a[aIdx], b[bIdx])
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
		bIdx += strideB
	}
}

// ElemVecApplyUnaryScalarStrided applies a unary function with a scalar to a vector: dst[i] = op(src[i], scalar).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, n int, strideDst, strideSrc int, op func(T, T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		// Fast path: contiguous vectors
		for i := 0; i < n; i++ {
			dst[i] = op(src[i], scalar)
		}
		return
	}

	// Strided path
	dIdx := 0
	sIdx := 0
	for i := 0; i < n; i++ {
		dst[dIdx] = op(src[sIdx], scalar)
		dIdx += strideDst
		sIdx += strideSrc
	}
}

// ElemVecApplyBinaryScalarStrided applies a binary function with a scalar to a vector: dst[i] = op(a[i], scalar).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, n int, strideDst, strideA int, op func(T, T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideA == 1 {
		// Fast path: contiguous vectors
		for i := 0; i < n; i++ {
			dst[i] = op(a[i], scalar)
		}
		return
	}

	// Strided path
	dIdx := 0
	aIdx := 0
	for i := 0; i < n; i++ {
		dst[dIdx] = op(a[aIdx], scalar)
		dIdx += strideDst
		aIdx += strideA
	}
}

// ElemVecApplyTernaryScalarStrided applies a ternary function with a scalar to a vector: dst[i] = op(condition[i], a[i], scalar).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, n int, strideDst, strideCond, strideA int, op func(T, T, T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideCond == 1 && strideA == 1 {
		// Fast path: contiguous vectors
		for i := 0; i < n; i++ {
			dst[i] = op(condition[i], a[i], scalar)
		}
		return
	}

	// Strided path
	dIdx := 0
	cIdx := 0
	aIdx := 0
	for i := 0; i < n; i++ {
		dst[dIdx] = op(condition[cIdx], a[aIdx], scalar)
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
	}
}

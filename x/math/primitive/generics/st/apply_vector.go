package st

import (
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

func applyUnaryContiguous[T Numeric](dst, src []T, n int, op func(T) T) {
	dst = dst[:n]
	src = src[:n]
	i := 0
	limit := n &^ 3
	for ; i < limit; i += 4 {
		dst[i] = op(src[i])
		dst[i+1] = op(src[i+1])
		dst[i+2] = op(src[i+2])
		dst[i+3] = op(src[i+3])
	}
	for ; i < n; i++ {
		dst[i] = op(src[i])
	}
}

func applyUnaryStrided[T Numeric](dst, src []T, n, strideDst, strideSrc int, op func(T) T) {
	dIdx := 0
	sIdx := 0
	limit := n &^ 3
	for i := 0; i < limit; i += 4 {
		dst[dIdx] = op(src[sIdx])
		dIdx += strideDst
		sIdx += strideSrc
		dst[dIdx] = op(src[sIdx])
		dIdx += strideDst
		sIdx += strideSrc
		dst[dIdx] = op(src[sIdx])
		dIdx += strideDst
		sIdx += strideSrc
		dst[dIdx] = op(src[sIdx])
		dIdx += strideDst
		sIdx += strideSrc
	}
	for i := limit; i < n; i++ {
		dst[dIdx] = op(src[sIdx])
		dIdx += strideDst
		sIdx += strideSrc
	}
}

func applyBinaryContiguous[T Numeric](dst, a, b []T, n int, op func(T, T) T) {
	dst = dst[:n]
	a = a[:n]
	b = b[:n]
	i := 0
	limit := n &^ 3
	for ; i < limit; i += 4 {
		dst[i] = op(a[i], b[i])
		dst[i+1] = op(a[i+1], b[i+1])
		dst[i+2] = op(a[i+2], b[i+2])
		dst[i+3] = op(a[i+3], b[i+3])
	}
	for ; i < n; i++ {
		dst[i] = op(a[i], b[i])
	}
}

func applyBinaryStrided[T Numeric](dst, a, b []T, n, strideDst, strideA, strideB int, op func(T, T) T) {
	dIdx := 0
	aIdx := 0
	bIdx := 0
	limit := n &^ 3
	for i := 0; i < limit; i += 4 {
		dst[dIdx] = op(a[aIdx], b[bIdx])
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
		dst[dIdx] = op(a[aIdx], b[bIdx])
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
		dst[dIdx] = op(a[aIdx], b[bIdx])
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
		dst[dIdx] = op(a[aIdx], b[bIdx])
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
	}
	for i := limit; i < n; i++ {
		dst[dIdx] = op(a[aIdx], b[bIdx])
		dIdx += strideDst
		aIdx += strideA
		bIdx += strideB
	}
}

func applyTernaryContiguous[T Numeric](dst, condition, a, b []T, n int, op func(T, T, T) T) {
	dst = dst[:n]
	condition = condition[:n]
	a = a[:n]
	b = b[:n]
	i := 0
	limit := n &^ 3
	for ; i < limit; i += 4 {
		dst[i] = op(condition[i], a[i], b[i])
		dst[i+1] = op(condition[i+1], a[i+1], b[i+1])
		dst[i+2] = op(condition[i+2], a[i+2], b[i+2])
		dst[i+3] = op(condition[i+3], a[i+3], b[i+3])
	}
	for ; i < n; i++ {
		dst[i] = op(condition[i], a[i], b[i])
	}
}

func applyTernaryStrided[T Numeric](dst, condition, a, b []T, n, strideDst, strideCond, strideA, strideB int, op func(T, T, T) T) {
	dIdx := 0
	cIdx := 0
	aIdx := 0
	bIdx := 0
	limit := n &^ 3
	for i := 0; i < limit; i += 4 {
		dst[dIdx] = op(condition[cIdx], a[aIdx], b[bIdx])
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
		bIdx += strideB
		dst[dIdx] = op(condition[cIdx], a[aIdx], b[bIdx])
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
		bIdx += strideB
		dst[dIdx] = op(condition[cIdx], a[aIdx], b[bIdx])
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
		bIdx += strideB
		dst[dIdx] = op(condition[cIdx], a[aIdx], b[bIdx])
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
		bIdx += strideB
	}
	for i := limit; i < n; i++ {
		dst[dIdx] = op(condition[cIdx], a[aIdx], b[bIdx])
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
		bIdx += strideB
	}
}

func applyUnaryScalarContiguous[T Numeric](dst, src []T, scalar T, n int, op func(T, T) T) {
	dst = dst[:n]
	src = src[:n]
	i := 0
	limit := n &^ 3
	for ; i < limit; i += 4 {
		dst[i] = op(src[i], scalar)
		dst[i+1] = op(src[i+1], scalar)
		dst[i+2] = op(src[i+2], scalar)
		dst[i+3] = op(src[i+3], scalar)
	}
	for ; i < n; i++ {
		dst[i] = op(src[i], scalar)
	}
}

func applyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, n, strideDst, strideSrc int, op func(T, T) T) {
	dIdx := 0
	sIdx := 0
	limit := n &^ 3
	for i := 0; i < limit; i += 4 {
		dst[dIdx] = op(src[sIdx], scalar)
		dIdx += strideDst
		sIdx += strideSrc
		dst[dIdx] = op(src[sIdx], scalar)
		dIdx += strideDst
		sIdx += strideSrc
		dst[dIdx] = op(src[sIdx], scalar)
		dIdx += strideDst
		sIdx += strideSrc
		dst[dIdx] = op(src[sIdx], scalar)
		dIdx += strideDst
		sIdx += strideSrc
	}
	for i := limit; i < n; i++ {
		dst[dIdx] = op(src[sIdx], scalar)
		dIdx += strideDst
		sIdx += strideSrc
	}
}

func applyBinaryScalarContiguous[T Numeric](dst, a []T, scalar T, n int, op func(T, T) T) {
	dst = dst[:n]
	a = a[:n]
	i := 0
	limit := n &^ 3
	for ; i < limit; i += 4 {
		dst[i] = op(a[i], scalar)
		dst[i+1] = op(a[i+1], scalar)
		dst[i+2] = op(a[i+2], scalar)
		dst[i+3] = op(a[i+3], scalar)
	}
	for ; i < n; i++ {
		dst[i] = op(a[i], scalar)
	}
}

func applyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, n, strideDst, strideA int, op func(T, T) T) {
	dIdx := 0
	aIdx := 0
	limit := n &^ 3
	for i := 0; i < limit; i += 4 {
		dst[dIdx] = op(a[aIdx], scalar)
		dIdx += strideDst
		aIdx += strideA
		dst[dIdx] = op(a[aIdx], scalar)
		dIdx += strideDst
		aIdx += strideA
		dst[dIdx] = op(a[aIdx], scalar)
		dIdx += strideDst
		aIdx += strideA
		dst[dIdx] = op(a[aIdx], scalar)
		dIdx += strideDst
		aIdx += strideA
	}
	for i := limit; i < n; i++ {
		dst[dIdx] = op(a[aIdx], scalar)
		dIdx += strideDst
		aIdx += strideA
	}
}

func applyTernaryScalarContiguous[T Numeric](dst, condition, a []T, scalar T, n int, op func(T, T, T) T) {
	dst = dst[:n]
	condition = condition[:n]
	a = a[:n]
	i := 0
	limit := n &^ 3
	for ; i < limit; i += 4 {
		dst[i] = op(condition[i], a[i], scalar)
		dst[i+1] = op(condition[i+1], a[i+1], scalar)
		dst[i+2] = op(condition[i+2], a[i+2], scalar)
		dst[i+3] = op(condition[i+3], a[i+3], scalar)
	}
	for ; i < n; i++ {
		dst[i] = op(condition[i], a[i], scalar)
	}
}

func applyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, n, strideDst, strideCond, strideA int, op func(T, T, T) T) {
	dIdx := 0
	cIdx := 0
	aIdx := 0
	limit := n &^ 3
	for i := 0; i < limit; i += 4 {
		dst[dIdx] = op(condition[cIdx], a[aIdx], scalar)
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
		dst[dIdx] = op(condition[cIdx], a[aIdx], scalar)
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
		dst[dIdx] = op(condition[cIdx], a[aIdx], scalar)
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
		dst[dIdx] = op(condition[cIdx], a[aIdx], scalar)
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
	}
	for i := limit; i < n; i++ {
		dst[dIdx] = op(condition[cIdx], a[aIdx], scalar)
		dIdx += strideDst
		cIdx += strideCond
		aIdx += strideA
	}
}

// ElemVecApplyUnaryStrided applies a unary function to a vector: dst[i] = op(src[i]).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyUnaryStrided[T Numeric](dst, src []T, n int, strideDst, strideSrc int, op func(T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		applyUnaryContiguous(dst, src, n, op)
		return
	}

	applyUnaryStrided(dst, src, n, strideDst, strideSrc, op)
}

// ElemVecApplyBinaryStrided applies a binary function to vectors: dst[i] = op(a[i], b[i]).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyBinaryStrided[T Numeric](dst, a, b []T, n int, strideDst, strideA, strideB int, op func(T, T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideA == 1 && strideB == 1 {
		applyBinaryContiguous(dst, a, b, n, op)
		return
	}

	applyBinaryStrided(dst, a, b, n, strideDst, strideA, strideB, op)
}

// ElemVecApplyTernaryStrided applies a ternary function to vectors: dst[i] = op(condition[i], a[i], b[i]).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyTernaryStrided[T Numeric](dst, condition, a, b []T, n int, strideDst, strideCond, strideA, strideB int, op func(T, T, T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideCond == 1 && strideA == 1 && strideB == 1 {
		applyTernaryContiguous(dst, condition, a, b, n, op)
		return
	}

	applyTernaryStrided(dst, condition, a, b, n, strideDst, strideCond, strideA, strideB, op)
}

// ElemVecApplyUnaryScalarStrided applies a unary function with a scalar to a vector: dst[i] = op(src[i], scalar).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyUnaryScalarStrided[T Numeric](dst, src []T, scalar T, n int, strideDst, strideSrc int, op func(T, T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideSrc == 1 {
		applyUnaryScalarContiguous(dst, src, scalar, n, op)
		return
	}

	applyUnaryScalarStrided(dst, src, scalar, n, strideDst, strideSrc, op)
}

// ElemVecApplyBinaryScalarStrided applies a binary function with a scalar to a vector: dst[i] = op(a[i], scalar).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyBinaryScalarStrided[T Numeric](dst, a []T, scalar T, n int, strideDst, strideA int, op func(T, T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideA == 1 {
		applyBinaryScalarContiguous(dst, a, scalar, n, op)
		return
	}

	applyBinaryScalarStrided(dst, a, scalar, n, strideDst, strideA, op)
}

// ElemVecApplyTernaryScalarStrided applies a ternary function with a scalar to a vector: dst[i] = op(condition[i], a[i], scalar).
// Optimized for 1D vector operations with stride support.
func ElemVecApplyTernaryScalarStrided[T Numeric](dst, condition, a []T, scalar T, n int, strideDst, strideCond, strideA int, op func(T, T, T) T) {
	if n == 0 {
		return
	}

	if strideDst == 1 && strideCond == 1 && strideA == 1 {
		applyTernaryScalarContiguous(dst, condition, a, scalar, n, op)
		return
	}

	applyTernaryScalarStrided(dst, condition, a, scalar, n, strideDst, strideCond, strideA, op)
}

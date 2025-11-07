package st

import . "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

// ElemApplyBinary applies a binary function element-wise to contiguous arrays: dst[i] = op(a[i], b[i]).
// Optimized for the common case of contiguous memory.
func ElemApplyBinary[T Numeric](dst, a, b []T, n int, op func(T, T) T) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
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

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesA, shape) && IsContiguous(stridesB, shape) {
		// Fast path: contiguous arrays
		// Boundary check elimination hint
		if size > 0 {
			_ = dst[size-1]
			_ = a[size-1]
			_ = b[size-1]
		}
		for i := 0; i < size; i++ {
			dst[i] = op(a[i], b[i])
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [3]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:3]
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		bIdx := offsets[2]
		dst[dIdx] = op(a[aIdx], b[bIdx])
		if !AdvanceOffsets3(shape, indices, offsets, stridesDst, stridesA, stridesB) {
			break
		}
	}
}

// ElemApplyUnary applies a unary function element-wise to contiguous arrays: dst[i] = op(src[i]).
// Optimized for the common case of contiguous memory.
func ElemApplyUnary[T Numeric](dst, src []T, n int, op func(T) T) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
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
		if size > 0 {
			_ = dst[size-1]
			_ = src[size-1]
		}
		for i := 0; i < size; i++ {
			dst[i] = op(src[i])
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = op(src[sIdx])
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemApplyTernary applies a ternary function element-wise to contiguous arrays: dst[i] = op(condition[i], a[i], b[i]).
// Optimized for the common case of contiguous memory.
func ElemApplyTernary[T Numeric](dst, condition, a, b []T, n int, op func(T, T, T) T) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
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
		if size > 0 {
			_ = dst[size-1]
			_ = condition[size-1]
			_ = a[size-1]
			_ = b[size-1]
		}
		for i := 0; i < size; i++ {
			dst[i] = op(condition[i], a[i], b[i])
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [4]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:4]
	for {
		dIdx := offsets[0]
		cIdx := offsets[1]
		aIdx := offsets[2]
		bIdx := offsets[3]
		dst[dIdx] = op(condition[cIdx], a[aIdx], b[bIdx])
		if !AdvanceOffsets4(shape, indices, offsets, stridesDst, stridesCond, stridesA, stridesB) {
			break
		}
	}
}

// ElemApplyUnaryScalar applies a unary function with a scalar parameter to contiguous arrays: dst[i] = op(src[i], scalar).
// Optimized for the common case of contiguous memory.
func ElemApplyUnaryScalar[T Numeric](dst, src []T, scalar T, n int, op func(T, T) T) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
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
		if size > 0 {
			_ = dst[size-1]
			_ = src[size-1]
		}
		for i := 0; i < size; i++ {
			dst[i] = op(src[i], scalar)
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		dst[dIdx] = op(src[sIdx], scalar)
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemApplyBinaryScalar applies a binary function with a scalar parameter to contiguous arrays: dst[i] = op(a[i], scalar).
// Optimized for the common case of contiguous memory.
func ElemApplyBinaryScalar[T Numeric](dst, a []T, scalar T, n int, op func(T, T) T) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
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
		if size > 0 {
			_ = dst[size-1]
			_ = a[size-1]
		}
		for i := 0; i < size; i++ {
			dst[i] = op(a[i], scalar)
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [2]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:2]
	for {
		dIdx := offsets[0]
		aIdx := offsets[1]
		dst[dIdx] = op(a[aIdx], scalar)
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesA) {
			break
		}
	}
}

// ElemApplyTernaryScalar applies a ternary function with a scalar parameter to contiguous arrays: dst[i] = op(condition[i], a[i], scalar).
// Optimized for the common case of contiguous memory.
func ElemApplyTernaryScalar[T Numeric](dst, condition, a []T, scalar T, n int, op func(T, T, T) T) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
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
		if size > 0 {
			_ = dst[size-1]
			_ = condition[size-1]
			_ = a[size-1]
		}
		for i := 0; i < size; i++ {
			dst[i] = op(condition[i], a[i], scalar)
		}
		return
	}

	// Strided path: iterate with strides using stack-allocated arrays
	rank := len(shape)
	var indicesStatic [MAX_DIMS]int
	var offsetsStatic [3]int
	indices := indicesStatic[:rank]
	offsets := offsetsStatic[:3]
	for {
		dIdx := offsets[0]
		cIdx := offsets[1]
		aIdx := offsets[2]
		dst[dIdx] = op(condition[cIdx], a[aIdx], scalar)
		if !AdvanceOffsets3(shape, indices, offsets, stridesDst, stridesCond, stridesA) {
			break
		}
	}
}

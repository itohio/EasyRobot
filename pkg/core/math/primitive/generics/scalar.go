package generics

import . "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

// ElemFill writes constant value to dst for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemFill[T Numeric](dst []T, value T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		dst[i] = value
	}
}

// ElemFillStrided writes constant value to dst with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemFillStrided[T Numeric](dst []T, value T, shape []int, stridesDst []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)

	if IsContiguous(stridesDst, shape) {
		// Fast path: contiguous arrays
		for i := 0; i < size; i++ {
			dst[i] = value
		}
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 1)
	strideSet := [][]int{stridesDst}
	for {
		dIdx := offsets[0]
		dst[dIdx] = value
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemEqualScalar writes 1 where src == scalar, 0 otherwise for contiguous arrays.
// Optimized for the common case of contiguous memory.
func ElemEqualScalar[T Numeric](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if src[i] == scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemEqualScalarStrided writes 1 where src == scalar, 0 otherwise with strides.
// Handles both contiguous and strided cases with optimized paths.
func ElemEqualScalarStrided[T Numeric](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays
		for i := 0; i < size; i++ {
			if src[i] == scalar {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	// Strided path: iterate with strides
	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		if src[sIdx] == scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemGreaterScalar writes 1 where src > scalar, 0 otherwise for contiguous arrays.
func ElemGreaterScalar[T Numeric](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if src[i] > scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemGreaterScalarStrided writes 1 where src > scalar, 0 otherwise with strides.
func ElemGreaterScalarStrided[T Numeric](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		for i := 0; i < size; i++ {
			if src[i] > scalar {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		if src[sIdx] > scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemLessScalar writes 1 where src < scalar, 0 otherwise for contiguous arrays.
func ElemLessScalar[T Numeric](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if src[i] < scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemLessScalarStrided writes 1 where src < scalar, 0 otherwise with strides.
func ElemLessScalarStrided[T Numeric](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		for i := 0; i < size; i++ {
			if src[i] < scalar {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		if src[sIdx] < scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemNotEqualScalar writes 1 where src != scalar, 0 otherwise for contiguous arrays.
func ElemNotEqualScalar[T Numeric](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if src[i] != scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemNotEqualScalarStrided writes 1 where src != scalar, 0 otherwise with strides.
func ElemNotEqualScalarStrided[T Numeric](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		for i := 0; i < size; i++ {
			if src[i] != scalar {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		if src[sIdx] != scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemLessEqualScalar writes 1 where src <= scalar, 0 otherwise for contiguous arrays.
func ElemLessEqualScalar[T Numeric](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if src[i] <= scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemLessEqualScalarStrided writes 1 where src <= scalar, 0 otherwise with strides.
func ElemLessEqualScalarStrided[T Numeric](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		for i := 0; i < size; i++ {
			if src[i] <= scalar {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		if src[sIdx] <= scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}

// ElemGreaterEqualScalar writes 1 where src >= scalar, 0 otherwise for contiguous arrays.
func ElemGreaterEqualScalar[T Numeric](dst, src []T, scalar T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		if src[i] >= scalar {
			dst[i] = 1
		} else {
			dst[i] = 0
		}
	}
}

// ElemGreaterEqualScalarStrided writes 1 where src >= scalar, 0 otherwise with strides.
func ElemGreaterEqualScalarStrided[T Numeric](dst, src []T, scalar T, shape []int, stridesDst, stridesSrc []int) {
	size := SizeFromShape(shape)
	if len(shape) == 0 || size == 0 {
		return
	}

	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		for i := 0; i < size; i++ {
			if src[i] >= scalar {
				dst[i] = 1
			} else {
				dst[i] = 0
			}
		}
		return
	}

	indices := make([]int, len(shape))
	offsets := make([]int, 2)
	strideSet := [][]int{stridesDst, stridesSrc}
	for {
		dIdx := offsets[0]
		sIdx := offsets[1]
		if src[sIdx] >= scalar {
			dst[dIdx] = 1
		} else {
			dst[dIdx] = 0
		}
		if !AdvanceOffsets(shape, indices, offsets, strideSet) {
			break
		}
	}
}


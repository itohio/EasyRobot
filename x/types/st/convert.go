package st

import (
	. "github.com/itohio/EasyRobot/x/math/shape"
	. "github.com/itohio/EasyRobot/x/types/internal"
)

// Convert converts src to dst with proper scaling between type ranges.
// For downconversions, applies dithering to reduce quantization artifacts.
// For same-type conversions, uses copy (fast path).
func Convert[T, U Number](dst []T, src []U, n int) {
	if n == 0 {
		return
	}
	// Use minimum length to avoid bounds checks
	if len(src) < n {
		n = len(src)
	}
	if len(dst) < n {
		n = len(dst)
	}

	// Optimized dispatch: single type assertion, combined same-type check and dispatch
	// The compiler should optimize this when types are known at compile time
	switch d := any(dst).(type) {
	case []int8:
		if s, ok := any(src).([]int8); ok {
			// Same type fast path
			copy(d[:n], s[:n])
			return
		}
		convertToInt8(d, src, n)
	case []int16:
		if s, ok := any(src).([]int16); ok {
			// Same type fast path
			copy(d[:n], s[:n])
			return
		}
		convertToInt16(d, src, n)
	case []int32:
		if s, ok := any(src).([]int32); ok {
			// Same type fast path
			copy(d[:n], s[:n])
			return
		}
		convertToInt32(d, src, n)
	case []int:
		if s, ok := any(src).([]int); ok {
			// Same type fast path
			copy(d[:n], s[:n])
			return
		}
		convertToInt(d, src, n)
	case []float32:
		if s, ok := any(src).([]float32); ok {
			// Same type fast path
			copy(d[:n], s[:n])
			return
		}
		convertToFloat32(d, src, n)
	case []complex64:
		if s, ok := any(src).([]complex64); ok {
			// Same type fast path
			copy(d[:n], s[:n])
			return
		}
		convertToComplex64(d, src, n)
	}
}

// ConvertStrided handles strided/non-contiguous arrays
func ConvertStrided[T, U Number](dst []T, src []U, shape []int, stridesDst, stridesSrc []int) {
	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	// Check if types match for fast path (same type, contiguous)
	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		Convert(dst, src, size)
		return
	}

	// Different types or strided: use conversion with strides
	convertStrided(dst, src, shape, stridesSrc, stridesDst)
}




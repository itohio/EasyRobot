package primitive

import (
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

// CopyWithConversion copies data from src to dst with type conversion.
// Supports conversion between float32, float64, int16, and int8.
// If srcData and dstData have the same type, performs direct copy.
// Both slices must have the same length.
// Returns dstData on success, nil on error.
func CopyWithConversion(dstData, srcData any) any {
	if srcData == nil || dstData == nil {
		return nil
	}

	// Check if types match for fast path
	switch src := srcData.(type) {
	case []float32:
		if dst, ok := dstData.([]float32); ok {
			copy(dst, src)
			return dst
		}
	case []float64:
		if dst, ok := dstData.([]float64); ok {
			copy(dst, src)
			return dst
		}
	case []int16:
		if dst, ok := dstData.([]int16); ok {
			copy(dst, src)
			return dst
		}
	case []int8:
		if dst, ok := dstData.([]int8); ok {
			copy(dst, src)
			return dst
		}
	}

	// Different types: perform conversion
	return copyWithConversion(dstData, srcData)
}

// Numeric types that can be converted
type numeric interface {
	~float32 | ~float64 | ~int16 | ~int8
}

// Types that need clamping when converting to int8
type clampableToInt8 interface {
	~float32 | ~float64 | ~int16
}

// Types that need clamping when converting to int16
type clampableToInt16 interface {
	~float32 | ~float64
}

// copyWithConversion performs type conversion between different types.
// Optimizes by copying the minimum of the two slice lengths to avoid bounds checks.
// Type switches happen once at dispatch, then generic functions handle the hot path.
func copyWithConversion(dstData, srcData any) any {
	switch dst := dstData.(type) {
	case []float32:
		return copyWithConversionInner(dst, srcData)
	case []float64:
		return copyWithConversionInner(dst, srcData)
	case []int16:
		return copyWithConversionInner(dst, srcData)
	case []int8:
		return copyWithConversionInner(dst, srcData)
	}
	return nil
}

// copyWithConversionInner dispatches on source type and calls specialized conversion functions.
// This eliminates all type switches from the hot path.
func copyWithConversionInner[T numeric](dst []T, srcData any) any {
	switch src := srcData.(type) {
	case []float32:
		return copyConvertNumeric(dst, src)
	case []float64:
		return copyConvertNumeric(dst, src)
	case []int16:
		return copyConvertNumeric(dst, src)
	case []int8:
		return copyConvertNumeric(dst, src)
	}
	return nil
}

// copyConvertNumeric dispatches to specialized functions only for down-conversions that need clamping.
// For up-conversions and same-type, uses generic function with direct T(src) conversion.
func copyConvertNumeric[T, U numeric](dst []T, src []U) any {
	// Check if this is a down-conversion that needs clamping
	// Down-conversions: float32/float64 -> int16/int8, int16 -> int8
	switch d := any(dst).(type) {
	case []int16:
		// Check if source is larger (float32/float64) - needs clamping
		switch any(src).(type) {
		case []float32, []float64:
			return copyConvertToInt16(d, src)
		default:
			// int16->int16 or int8->int16: direct conversion, no clamping
			return copyConvertGeneric(dst, src)
		}
	case []int8:
		// Check if source is larger (float32/float64/int16) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int16:
			return copyConvertToInt8(d, src)
		default:
			// int8->int8: direct conversion, no clamping
			return copyConvertGeneric(dst, src)
		}
	default:
		// For float32/float64: always direct conversion (up-conversion or same-type)
		return copyConvertGeneric(dst, src)
	}
}

// copyConvertGeneric handles up-conversions and same-type conversions.
// Uses direct T(src) conversion - no clamping needed, no switches in hot path.
func copyConvertGeneric[T, U numeric](dst []T, src []U) []T {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	// Fast path: same type - use type assertion to check and use copy
	switch d := any(dst).(type) {
	case []float32:
		if s, ok := any(src).([]float32); ok {
			copy(d, s[:n])
			return dst
		}
	case []float64:
		if s, ok := any(src).([]float64); ok {
			copy(d, s[:n])
			return dst
		}
	}
	// Generic conversion: compiler optimizes T(src[i]) for each type combination
	for i := 0; i < n; i++ {
		dst[i] = T(src[i])
	}
	return dst
}

// copyConvertToInt16 handles conversions to int16 with clamping.
// Handles: float32/float64 -> int16 (needs clamping)
//
//	int16 -> int16 (same type, fast path)
//	int8 -> int16 (up-conversion, no clamping)
func copyConvertToInt16(dst []int16, src any) []int16 {
	switch s := src.(type) {
	case []float32:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: generic clamp function avoids bounds checks
			clampToInt16(dst, s, n)
		}
	case []float64:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: generic clamp function avoids bounds checks
			clampToInt16(dst, s, n)
		}
	case []int16:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		copy(dst, s[:n])
	case []int8:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: direct conversion, no clamping needed
			int8ToInt16Loop(dst, s, n)
		}
	}
	return dst
}

// copyConvertToInt8 handles conversions to int8 with clamping.
// Handles: float32/float64/int16 -> int8 (needs clamping)
//
//	int8 -> int8 (same type, fast path)
func copyConvertToInt8(dst []int8, src any) []int8 {
	switch s := src.(type) {
	case []float32:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: generic clamp function avoids bounds checks
			clampToInt8(dst, s, n)
		}
	case []float64:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: generic clamp function avoids bounds checks
			clampToInt8(dst, s, n)
		}
	case []int16:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: generic clamp function avoids bounds checks
			clampToInt8(dst, s, n)
		}
	case []int8:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		copy(dst, s[:n])
	}
	return dst
}

// clampToInt8 implements the hot path inner loop for clamping to int8 range [-128, 127].
// Generic over source types that need clamping (float32, float64, int16).
// n is the number of elements to process (bounds checks eliminated by explicit length).
// The inner loop is implemented directly here to avoid function call overhead.
func clampToInt8[U clampableToInt8](dst []int8, src []U, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1] // Bounds check elimination hint
	_ = src[n-1] // Bounds check elimination hint
	for i := 0; i < n; i++ {
		val := src[i]
		valFloat := float64(val)
		if valFloat > 127 {
			dst[i] = 127
		} else if valFloat < -128 {
			dst[i] = -128
		} else {
			dst[i] = int8(val)
		}
	}
}

// clampToInt16 implements the hot path inner loop for clamping to int16 range [-32768, 32767].
// Generic over source types that need clamping (float32, float64).
// n is the number of elements to process (bounds checks eliminated by explicit length).
// The inner loop is implemented directly here to avoid function call overhead.
func clampToInt16[U clampableToInt16](dst []int16, src []U, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1] // Bounds check elimination hint
	_ = src[n-1] // Bounds check elimination hint
	for i := 0; i < n; i++ {
		val := src[i]
		valFloat := float64(val)
		if valFloat > 32767 {
			dst[i] = 32767
		} else if valFloat < -32768 {
			dst[i] = -32768
		} else {
			dst[i] = int16(val)
		}
	}
}

// clampToInt8Value clamps a single value to int8 range [-128, 127].
// Used by stride-based copying where we process one element at a time.
func clampToInt8Value[U clampableToInt8](v U) int8 {
	valFloat := float64(v)
	if valFloat > 127 {
		return 127
	}
	if valFloat < -128 {
		return -128
	}
	return int8(v)
}

// clampToInt16Value clamps a single value to int16 range [-32768, 32767].
// Used by stride-based copying where we process one element at a time.
func clampToInt16Value[U clampableToInt16](v U) int16 {
	valFloat := float64(v)
	if valFloat > 32767 {
		return 32767
	}
	if valFloat < -32768 {
		return -32768
	}
	return int16(v)
}

// int8ToInt16Loop converts int8 to int16 (up-conversion, no clamping needed).
// n is the number of elements to process (bounds checks eliminated by explicit length).
func int8ToInt16Loop(dst []int16, src []int8, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1] // Bounds check elimination hint
	_ = src[n-1] // Bounds check elimination hint
	for i := 0; i < n; i++ {
		dst[i] = int16(src[i])
	}
}

// CopyWithStrides copies data from src to dst element-by-element respecting strides.
// Supports data type conversion between different types.
// If srcData and dstData have the same type, performs direct copy.
// If types differ, uses CopyWithConversion for conversion.
func CopyWithStrides(srcData, dstData any, shape []int, srcStrides, dstStrides []int) {
	if srcData == nil || dstData == nil {
		return
	}

	if len(shape) == 0 {
		return
	}

	// Normalize strides
	srcStrides = fp32.EnsureStrides(srcStrides, shape)
	dstStrides = fp32.EnsureStrides(dstStrides, shape)

	// Check if types match for fast path
	switch src := srcData.(type) {
	case []float32:
		if dst, ok := dstData.([]float32); ok {
			// Same type: use fp32.ElemCopy
			fp32.ElemCopy(dst, src, shape, dstStrides, srcStrides)
			return
		}
	case []float64:
		if dst, ok := dstData.([]float64); ok {
			// Same type: direct copy with strides
			copyWithStridesSameType(dst, src, shape, dstStrides, srcStrides)
			return
		}
	case []int16:
		if dst, ok := dstData.([]int16); ok {
			// Same type: direct copy with strides
			copyWithStridesSameType(dst, src, shape, dstStrides, srcStrides)
			return
		}
	case []int8:
		if dst, ok := dstData.([]int8); ok {
			// Same type: direct copy with strides
			copyWithStridesSameType(dst, src, shape, dstStrides, srcStrides)
			return
		}
	}

	// Different types: use CopyWithConversion for conversion with strides
	copyWithStridesAndConversion(srcData, dstData, shape, srcStrides, dstStrides)
}

// copyWithStridesSameType copies data element-by-element with strides for same type.
// Uses iterative approach instead of recursion for better performance.
func copyWithStridesSameType[T any](dst, src []T, shape []int, dstStrides, srcStrides []int) {
	ndims := len(shape)
	if ndims == 0 {
		return
	}

	indices := make([]int, ndims)
	dim := 0

	for {
		// Process current element if we've reached the leaf
		if dim == ndims {
			// Hot path: compute offsets and copy
			sIdx := computeStrideOffset(indices, srcStrides)
			dIdx := computeStrideOffset(indices, dstStrides)
			dst[dIdx] = src[sIdx]

			// Backtrack to previous dimension
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		// Check if we've exhausted current dimension
		if indices[dim] >= shape[dim] {
			indices[dim] = 0
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		// Move to next dimension
		dim++
	}
}

// copyWithStridesAndConversion copies data element-by-element with strides,
// performing type conversion with clamping only for down-conversions.
func copyWithStridesAndConversion(srcData, dstData any, shape []int, srcStrides, dstStrides []int) {
	// Dispatch based on whether clamping is needed (down-conversion)
	switch dst := dstData.(type) {
	case []int16:
		// Check if source is larger (float32/float64) - needs clamping
		switch src := srcData.(type) {
		case []float32:
			clampToInt16Strided(dst, src, shape, srcStrides, dstStrides)
		case []float64:
			clampToInt16Strided(dst, src, shape, srcStrides, dstStrides)
		case []int16:
			// Same type: direct copy
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		case []int8:
			// Up-conversion: direct conversion
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []int8:
		// Check if source is larger (float32/float64/int16) - needs clamping
		switch src := srcData.(type) {
		case []float32:
			clampToInt8Strided(dst, src, shape, srcStrides, dstStrides)
		case []float64:
			clampToInt8Strided(dst, src, shape, srcStrides, dstStrides)
		case []int16:
			clampToInt8Strided(dst, src, shape, srcStrides, dstStrides)
		case []int8:
			// Same type: direct copy
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []float32:
		// For float32: always direct conversion
		switch src := srcData.(type) {
		case []float32:
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		case []float64:
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		case []int16:
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		case []int8:
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []float64:
		// For float64: always direct conversion
		switch src := srcData.(type) {
		case []float32:
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		case []float64:
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		case []int16:
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		case []int8:
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	}
}

// copyWithStridesGeneric copies with strides using direct conversion - no clamping needed.
// Handles up-conversions and same-type conversions.
// Generic over source and destination types to eliminate type switches in hot path.
func copyWithStridesGeneric[T, U numeric](dst []T, src []U, shape []int, srcStrides, dstStrides []int) {
	// Iterative approach using a stack to avoid recursion overhead
	ndims := len(shape)
	if ndims == 0 {
		return
	}

	indices := make([]int, ndims)
	dim := 0

	for {
		// Process current element if we've reached the leaf
		if dim == ndims {
			sIdx := computeStrideOffset(indices, srcStrides)
			dIdx := computeStrideOffset(indices, dstStrides)
			// Hot path: direct conversion, no type switches
			dst[dIdx] = T(src[sIdx])

			// Backtrack to previous dimension
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		// Check if we've exhausted current dimension
		if indices[dim] >= shape[dim] {
			indices[dim] = 0
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		// Move to next dimension
		dim++
	}
}

// Helper to compute linear offset from multi-dimensional indices
func computeStrideOffset(indices []int, strides []int) int {
	offset := 0
	for i := range indices {
		offset += indices[i] * strides[i]
	}
	return offset
}

// clampToInt8Strided implements the hot path for strided copying with clamping to int8.
// The entire iteration is implemented inside this function to avoid function call overhead.
// Uses iterative approach instead of recursion for better performance.
// Generic over source types that need clamping (float32, float64, int16).
func clampToInt8Strided[U clampableToInt8](dst []int8, src []U, shape []int, srcStrides, dstStrides []int) {
	ndims := len(shape)
	if ndims == 0 {
		return
	}

	indices := make([]int, ndims)
	dim := 0

	for {
		// Process current element if we've reached the leaf
		if dim == ndims {
			// Hot path: compute offsets and clamp
			sIdx := computeStrideOffset(indices, srcStrides)
			dIdx := computeStrideOffset(indices, dstStrides)
			val := src[sIdx]
			valFloat := float64(val)
			if valFloat > 127 {
				dst[dIdx] = 127
			} else if valFloat < -128 {
				dst[dIdx] = -128
			} else {
				dst[dIdx] = int8(val)
			}

			// Backtrack to previous dimension
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		// Check if we've exhausted current dimension
		if indices[dim] >= shape[dim] {
			indices[dim] = 0
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		// Move to next dimension
		dim++
	}
}

// clampToInt16Strided implements the hot path for strided copying with clamping to int16.
// The entire iteration is implemented inside this function to avoid function call overhead.
// Uses iterative approach instead of recursion for better performance.
// Generic over source types that need clamping (float32, float64).
func clampToInt16Strided[U clampableToInt16](dst []int16, src []U, shape []int, srcStrides, dstStrides []int) {
	ndims := len(shape)
	if ndims == 0 {
		return
	}

	indices := make([]int, ndims)
	dim := 0

	for {
		// Process current element if we've reached the leaf
		if dim == ndims {
			// Hot path: compute offsets and clamp
			sIdx := computeStrideOffset(indices, srcStrides)
			dIdx := computeStrideOffset(indices, dstStrides)
			val := src[sIdx]
			valFloat := float64(val)
			if valFloat > 32767 {
				dst[dIdx] = 32767
			} else if valFloat < -32768 {
				dst[dIdx] = -32768
			} else {
				dst[dIdx] = int16(val)
			}

			// Backtrack to previous dimension
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		// Check if we've exhausted current dimension
		if indices[dim] >= shape[dim] {
			indices[dim] = 0
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		// Move to next dimension
		dim++
	}
}

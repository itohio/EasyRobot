package primitive

import (
	"math"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

// CopyWithConversion copies data from src to dst with type conversion.
// Supports conversion between float32, float64, int64, int32, int16, and int8.
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
	case []int:
		if dst, ok := dstData.([]int); ok {
			copy(dst, src)
			return dst
		}
	case []int64:
		if dst, ok := dstData.([]int64); ok {
			copy(dst, src)
			return dst
		}
	case []int32:
		if dst, ok := dstData.([]int32); ok {
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

// Numeric types that can be converted, sorted by type size (largest to smallest)
type numeric interface {
	~float64 | ~int64 | ~float32 | ~int | ~int32 | ~int16 | ~int8
}

// Types that need clamping when converting to int8 (sorted by type size)
type clampableToInt8 interface {
	~float64 | ~int64 | ~float32 | ~int | ~int32 | ~int16
}

// Types that need clamping when converting to int16 (sorted by type size)
type clampableToInt16 interface {
	~float64 | ~int64 | ~float32 | ~int | ~int32
}

// Types that need clamping when converting to int32 (sorted by type size)
type clampableToInt32 interface {
	~float64 | ~int64 | ~float32 | ~int
}

// Types that need clamping when converting to int64 (sorted by type size)
type clampableToInt64 interface {
	~float64 | ~float32
}

// copyWithConversion performs type conversion between different types.
// Optimizes by copying the minimum of the two slice lengths to avoid bounds checks.
// Type switches happen once at dispatch, then generic functions handle the hot path.
func copyWithConversion(dstData, srcData any) any {
	switch dst := dstData.(type) {
	case []float64:
		return copyWithConversionInner(dst, srcData)
	case []float32:
		return copyWithConversionInner(dst, srcData)
	case []int64:
		return copyWithConversionInner(dst, srcData)
	case []int:
		return copyWithConversionInner(dst, srcData)
	case []int32:
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
	case []float64:
		return copyConvertNumeric(dst, src)
	case []float32:
		return copyConvertNumeric(dst, src)
	case []int64:
		return copyConvertNumeric(dst, src)
	case []int:
		return copyConvertNumeric(dst, src)
	case []int32:
		return copyConvertNumeric(dst, src)
	case []int16:
		return copyConvertNumeric(dst, src)
	case []int8:
		return copyConvertNumeric(dst, src)
	}
	return nil
}

// copyGeneric handles up-conversions and same-type conversions.
// Uses direct T(src) conversion - no clamping needed, no switches in hot path.
// Optimized to detect same type at compile time when possible.
func copyGeneric[T, U numeric](dst []T, src []U) []T {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	// Fast path: same type - use type assertion to check and use copy
	// This is optimized by the compiler when T == U
	switch d := any(dst).(type) {
	case []float64:
		if s, ok := any(src).([]float64); ok {
			copy(d, s[:n])
			return dst
		}
	case []float32:
		if s, ok := any(src).([]float32); ok {
			copy(d, s[:n])
			return dst
		}
	case []int64:
		if s, ok := any(src).([]int64); ok {
			copy(d, s[:n])
			return dst
		}
	case []int:
		if s, ok := any(src).([]int); ok {
			copy(d, s[:n])
			return dst
		}
	case []int32:
		if s, ok := any(src).([]int32); ok {
			copy(d, s[:n])
			return dst
		}
	case []int16:
		if s, ok := any(src).([]int16); ok {
			copy(d, s[:n])
			return dst
		}
	case []int8:
		if s, ok := any(src).([]int8); ok {
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

// copyConvertToInt64 handles conversions to int64 with clamping.
// Handles: float32/float64 -> int64 (needs clamping)
// int64 -> int64 (same type, fast path)
// int32/int16/int8 -> int64 (up-conversion, no clamping)
func copyConvertToInt64(dst []int64, src any) []int64 {
	switch s := src.(type) {
	case []float32:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt64(dst, s, n)
		}
	case []float64:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt64(dst, s, n)
		}
	}
	return dst
}

// copyConvertToInt16 handles conversions to int16 with clamping.
// Handles: float32/float64/int64/int32 -> int16 (needs clamping)
// int16 -> int16 (same type, fast path)
// int8 -> int16 (up-conversion, no clamping)
func copyConvertToInt16(dst []int16, src any) []int16 {
	switch s := src.(type) {
	case []float64:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: generic clamp function avoids bounds checks
			clampToInt16(dst, s, n)
		}
	case []float32:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: generic clamp function avoids bounds checks
			clampToInt16(dst, s, n)
		}
	case []int64:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt16(dst, s, n)
		}
	case []int:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt16(dst, s, n)
		}
	case []int32:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt16(dst, s, n)
		}
	}
	return dst
}

// copyConvertToInt8 handles conversions to int8 with clamping.
// Handles: float32/float64/int64/int32/int16 -> int8 (needs clamping)
// int8 -> int8 (same type, fast path)
func copyConvertToInt8(dst []int8, src any) []int8 {
	switch s := src.(type) {
	case []float64:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: generic clamp function avoids bounds checks
			clampToInt8(dst, s, n)
		}
	case []float32:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: generic clamp function avoids bounds checks
			clampToInt8(dst, s, n)
		}
	case []int:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: optimized integer-only clamp
			clampToInt8(dst, s, n)
		}
	case []int64:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: optimized integer-only clamp
			clampToInt8(dst, s, n)
		}
	case []int32:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: optimized integer-only clamp
			clampToInt8(dst, s, n)
		}
	case []int16:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			// Hot path: optimized integer-only clamp (no float64 conversion)
			clampToInt8(dst, s, n)
		}
	}
	return dst
}

// clampToInt8 implements the hot path inner loop for clamping to int8 range [-128, 127].
// Generic over source types that need clamping (float32, float64, int16).
// n is the number of elements to process (bounds checks eliminated by explicit length).
// The inner loop is implemented directly here to avoid function call overhead.
// Uses branchless min/max for better performance.
func clampToInt8[U clampableToInt8](dst []int8, src []U, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1] // Bounds check elimination hint
	_ = src[n-1] // Bounds check elimination hint
	for i := 0; i < n; i++ {
		val := src[i]
		if val < U(math.MinInt8) {
			val = math.MinInt8
		}
		if val > U(math.MaxInt8) {
			val = math.MaxInt8
		}
		dst[i] = int8(val)
	}
}

// clampToInt64 implements the hot path inner loop for clamping to int64 range.
// Generic over source types that need clamping (float32, float64).
func clampToInt64[U clampableToInt64](dst []int64, src []U, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1]
	_ = src[n-1]
	for i := 0; i < n; i++ {
		val := src[i]
		// For int64, we clamp to int64 max/min
		if val < U(int64(math.MinInt64)) {
			val = math.MinInt64
		}
		if val > U(int64(math.MaxInt64)) {
			val = math.MaxInt64
		}
		dst[i] = int64(val)
	}
}

// clampToInt implements the hot path inner loop for clamping to int range.
// Generic over source types that need clamping (float32, float64, int64).
func clampToInt[U clampableToInt](dst []int, src []U, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1]
	_ = src[n-1]
	for i := 0; i < n; i++ {
		val := src[i]
		if val < U(math.MinInt) {
			val = math.MinInt
		}
		if val > U(math.MaxInt) {
			val = math.MaxInt
		}
		dst[i] = int(val)
	}
}

// clampToInt32 implements the hot path inner loop for clamping to int32 range [-2147483648, 2147483647].
// Generic over source types that need clamping (float32, float64, int64).
func clampToInt32[U clampableToInt32](dst []int32, src []U, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1]
	_ = src[n-1]
	for i := 0; i < n; i++ {
		val := src[i]
		if val < U(math.MinInt32) {
			val = math.MinInt32
		}
		if val > U(math.MaxInt32) {
			val = math.MaxInt32
		}
		dst[i] = int32(val)
	}
}

// clampToInt16 implements the hot path inner loop for clamping to int16 range [-32768, 32767].
// Generic over source types that need clamping (float32, float64).
// n is the number of elements to process (bounds checks eliminated by explicit length).
// The inner loop is implemented directly here to avoid function call overhead.
// Uses branchless min/max for better performance.
func clampToInt16[U clampableToInt16](dst []int16, src []U, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1] // Bounds check elimination hint
	_ = src[n-1] // Bounds check elimination hint
	for i := 0; i < n; i++ {
		val := src[i]
		// Min/max pattern: compiler optimizes to conditional moves (CMOV)
		// First clamp to minimum, then to maximum
		if val < U(math.MinInt16) {
			val = math.MinInt16
		}
		if val > U(math.MaxInt16) {
			val = math.MaxInt16
		}
		dst[i] = int16(val)
	}
}

// clampToInt8Value clamps a single value to int8 range [-128, 127].
// Used by stride-based copying where we process one element at a time.
func clampToInt8Value[U clampableToInt8](v U) int8 {
	if v > U(math.MaxInt8) {
		return math.MaxInt8
	}
	if v < U(math.MinInt8) {
		return math.MinInt8
	}
	return int8(v)
}

// clampToInt16Value clamps a single value to int16 range [-32768, 32767].
// Used by stride-based copying where we process one element at a time.
func clampToInt16Value[U clampableToInt16](v U) int16 {
	if v > U(math.MaxInt16) {
		return math.MaxInt16
	}
	if v < U(math.MinInt16) {
		return math.MinInt16
	}
	return int16(v)
}

// clampToInt16Value clamps a single value to int16 range [-32768, 32767].
// Used by stride-based copying where we process one element at a time.
func clampToInt32Value[U clampableToInt32](v U) int32 {
	if v > U(math.MaxInt32) {
		return math.MaxInt32
	}
	if v < U(math.MinInt32) {
		return math.MinInt32
	}
	return int32(v)
}

// clampToInt16Value clamps a single value to int16 range [-32768, 32767].
// Used by stride-based copying where we process one element at a time.
func clampToInt64Value[U clampableToInt64](v U) int64 {
	if v > U(int64(math.MaxInt64)) {
		return math.MaxInt64
	}
	if v < U(int64(math.MinInt64)) {
		return math.MinInt64
	}
	return int64(v)
}

// clampToIntValue clamps a single value to int range.
// Used by stride-based copying where we process one element at a time.
func clampToIntValue[U clampableToInt](v U) int {
	if v > U(math.MaxInt) {
		return math.MaxInt
	}
	if v < U(math.MinInt) {
		return math.MinInt
	}
	return int(v)
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
	case []int64:
		if dst, ok := dstData.([]int64); ok {
			// Same type: direct copy with strides
			copyWithStridesSameType(dst, src, shape, dstStrides, srcStrides)
			return
		}
	case []int32:
		if dst, ok := dstData.([]int32); ok {
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

// copyConvertNumericStrided dispatches to specialized functions only for down-conversions that need clamping.
// For up-conversions and same-type, uses generic function with direct T(src) conversion.
func copyConvertNumericStrided[T, U numeric](dst []T, src []U, shape []int, srcStrides, dstStrides []int) {
	// Check if this is a down-conversion that needs clamping
	switch d := any(dst).(type) {
	case []int64:
		// Check if source is larger (float32/float64) - needs clamping
		switch any(src).(type) {
		case []float32, []float64:
			copyConvertToInt64Strided(d, src, shape, srcStrides, dstStrides)
		default:
			// int64->int64 or smaller->int64: direct conversion, no clamping
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []int:
		// Check if source is larger - needs clamping
		switch any(src).(type) {
		case []float32, []float64:
			copyConvertToIntStrided(d, src, shape, srcStrides, dstStrides)
		default:
			// int->int or smaller->int: direct conversion, no clamping
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []int32:
		// Check if source is larger (float32/float64/int64/int) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int:
			copyConvertToInt32Strided(d, src, shape, srcStrides, dstStrides)
		default:
			// int32->int32 or smaller->int32: direct conversion, no clamping
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []int16:
		// Check if source is larger (float32/float64/int64/int/int32) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int, []int32:
			copyConvertToInt16Strided(d, src, shape, srcStrides, dstStrides)
		default:
			// int16->int16 or int8->int16: direct conversion, no clamping
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []int8:
		// Check if source is larger (float32/float64/int64/int/int32/int16) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int, []int32, []int16:
			copyConvertToInt8Strided(d, src, shape, srcStrides, dstStrides)
		default:
			// int8->int8: direct conversion, no clamping
			copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	default:
		// For float32/float64: always direct conversion (up-conversion or same-type)
		copyWithStridesGeneric(dst, src, shape, srcStrides, dstStrides)
	}
}

// copyWithStridesAndConversionInner dispatches on source type and calls specialized conversion functions.
// This eliminates all type switches from the hot path.
func copyWithStridesAndConversionInner[T numeric](dst []T, srcData any, shape []int, srcStrides, dstStrides []int) {
	switch src := srcData.(type) {
	case []float64:
		copyConvertNumericStrided(dst, src, shape, srcStrides, dstStrides)
	case []float32:
		copyConvertNumericStrided(dst, src, shape, srcStrides, dstStrides)
	case []int64:
		copyConvertNumericStrided(dst, src, shape, srcStrides, dstStrides)
	case []int:
		copyConvertNumericStrided(dst, src, shape, srcStrides, dstStrides)
	case []int32:
		copyConvertNumericStrided(dst, src, shape, srcStrides, dstStrides)
	case []int16:
		copyConvertNumericStrided(dst, src, shape, srcStrides, dstStrides)
	case []int8:
		copyConvertNumericStrided(dst, src, shape, srcStrides, dstStrides)
	}
}

// copyWithStridesAndConversion copies data element-by-element with strides,
// performing type conversion with clamping only for down-conversions.
func copyWithStridesAndConversion(srcData, dstData any, shape []int, srcStrides, dstStrides []int) {
	// Dispatch based on destination type
	switch dst := dstData.(type) {
	case []float64:
		copyWithStridesAndConversionInner(dst, srcData, shape, srcStrides, dstStrides)
	case []float32:
		copyWithStridesAndConversionInner(dst, srcData, shape, srcStrides, dstStrides)
	case []int64:
		copyWithStridesAndConversionInner(dst, srcData, shape, srcStrides, dstStrides)
	case []int:
		copyWithStridesAndConversionInner(dst, srcData, shape, srcStrides, dstStrides)
	case []int32:
		copyWithStridesAndConversionInner(dst, srcData, shape, srcStrides, dstStrides)
	case []int16:
		copyWithStridesAndConversionInner(dst, srcData, shape, srcStrides, dstStrides)
	case []int8:
		copyWithStridesAndConversionInner(dst, srcData, shape, srcStrides, dstStrides)
	}
}

// copyConvertToInt64Strided handles conversions to int64 with clamping.
func copyConvertToInt64Strided(dst []int64, src any, shape []int, srcStrides, dstStrides []int) {
	switch s := src.(type) {
	case []float32:
		clampToInt64Strided(dst, s, shape, srcStrides, dstStrides)
	case []float64:
		clampToInt64Strided(dst, s, shape, srcStrides, dstStrides)
	}
}

// copyConvertToIntStrided handles conversions to int with clamping.
func copyConvertToIntStrided(dst []int, src any, shape []int, srcStrides, dstStrides []int) {
	switch s := src.(type) {
	case []float32:
		clampToIntStrided(dst, s, shape, srcStrides, dstStrides)
	case []float64:
		clampToIntStrided(dst, s, shape, srcStrides, dstStrides)
	}
}

// copyConvertToInt32Strided handles conversions to int32 with clamping.
func copyConvertToInt32Strided(dst []int32, src any, shape []int, srcStrides, dstStrides []int) {
	switch s := src.(type) {
	case []float32:
		clampToInt32Strided(dst, s, shape, srcStrides, dstStrides)
	case []float64:
		clampToInt32Strided(dst, s, shape, srcStrides, dstStrides)
	case []int64:
		clampToInt32Strided(dst, s, shape, srcStrides, dstStrides)
	case []int:
		clampToInt32Strided(dst, s, shape, srcStrides, dstStrides)
	}
}

// copyConvertToInt16Strided handles conversions to int16 with clamping.
func copyConvertToInt16Strided(dst []int16, src any, shape []int, srcStrides, dstStrides []int) {
	switch s := src.(type) {
	case []float32:
		clampToInt16Strided(dst, s, shape, srcStrides, dstStrides)
	case []float64:
		clampToInt16Strided(dst, s, shape, srcStrides, dstStrides)
	case []int64:
		clampToInt16Strided(dst, s, shape, srcStrides, dstStrides)
	case []int:
		clampToInt16Strided(dst, s, shape, srcStrides, dstStrides)
	case []int32:
		clampToInt16Strided(dst, s, shape, srcStrides, dstStrides)
	}
}

// copyConvertToInt8Strided handles conversions to int8 with clamping.
func copyConvertToInt8Strided(dst []int8, src any, shape []int, srcStrides, dstStrides []int) {
	switch s := src.(type) {
	case []float32:
		clampToInt8Strided(dst, s, shape, srcStrides, dstStrides)
	case []float64:
		clampToInt8Strided(dst, s, shape, srcStrides, dstStrides)
	case []int64:
		clampToInt8Strided(dst, s, shape, srcStrides, dstStrides)
	case []int:
		clampToInt8Strided(dst, s, shape, srcStrides, dstStrides)
	case []int32:
		clampToInt8Strided(dst, s, shape, srcStrides, dstStrides)
	case []int16:
		clampToInt8Strided(dst, s, shape, srcStrides, dstStrides)
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
			if val > U(math.MaxInt8) {
				dst[dIdx] = math.MaxInt8
			} else if val < U(math.MinInt8) {
				dst[dIdx] = math.MinInt8
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
			if val > U(math.MaxInt16) {
				dst[dIdx] = math.MaxInt16
			} else if val < U(math.MinInt16) {
				dst[dIdx] = math.MinInt16
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

// clampToInt64Strided implements the hot path for strided copying with clamping to int64.
// Generic over source types that need clamping (float32, float64).
func clampToInt64Strided[U clampableToInt64](dst []int64, src []U, shape []int, srcStrides, dstStrides []int) {
	ndims := len(shape)
	if ndims == 0 {
		return
	}

	indices := make([]int, ndims)
	dim := 0

	for {
		if dim == ndims {
			sIdx := computeStrideOffset(indices, srcStrides)
			dIdx := computeStrideOffset(indices, dstStrides)
			val := src[sIdx]
			if val > U(int64(math.MaxInt64)) {
				dst[dIdx] = math.MaxInt64
			} else if val < U(int64(math.MinInt64)) {
				dst[dIdx] = math.MinInt64
			} else {
				dst[dIdx] = int64(val)
			}

			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		if indices[dim] >= shape[dim] {
			indices[dim] = 0
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		dim++
	}
}

// clampToIntStrided implements the hot path for strided copying with clamping to int.
// Generic over source types that need clamping (float32, float64).
func clampToIntStrided[U clampableToInt](dst []int, src []U, shape []int, srcStrides, dstStrides []int) {
	ndims := len(shape)
	if ndims == 0 {
		return
	}

	indices := make([]int, ndims)
	dim := 0

	for {
		if dim == ndims {
			sIdx := computeStrideOffset(indices, srcStrides)
			dIdx := computeStrideOffset(indices, dstStrides)
			val := src[sIdx]
			if val > U(math.MaxInt) {
				dst[dIdx] = math.MaxInt
			} else if val < U(math.MinInt) {
				dst[dIdx] = math.MinInt
			} else {
				dst[dIdx] = int(val)
			}

			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		if indices[dim] >= shape[dim] {
			indices[dim] = 0
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		dim++
	}
}

// clampToInt32Strided implements the hot path for strided copying with clamping to int32.
// Generic over source types that need clamping (float32, float64).
func clampToInt32Strided[U clampableToInt32](dst []int32, src []U, shape []int, srcStrides, dstStrides []int) {
	ndims := len(shape)
	if ndims == 0 {
		return
	}

	indices := make([]int, ndims)
	dim := 0

	for {
		if dim == ndims {
			sIdx := computeStrideOffset(indices, srcStrides)
			dIdx := computeStrideOffset(indices, dstStrides)
			val := src[sIdx]
			if val > U(math.MaxInt32) {
				dst[dIdx] = math.MaxInt32
			} else if val < U(math.MinInt32) {
				dst[dIdx] = math.MinInt32
			} else {
				dst[dIdx] = int32(val)
			}

			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		if indices[dim] >= shape[dim] {
			indices[dim] = 0
			dim--
			if dim < 0 {
				break
			}
			indices[dim]++
			continue
		}

		dim++
	}
}

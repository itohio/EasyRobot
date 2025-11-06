package mt

import (
	"math"

	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
	st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"
)

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

// clampableToInt is platform-specific - see convert_i32.go and convert_i64.go

// ElemConvert converts src to dst for contiguous arrays.
// Multi-threaded version that parallelizes the conversion across CPU cores.
// Falls back to single-threaded implementation for small arrays.
// Supports conversion between float32, float64, int64, int32, int16, and int8.
func ElemConvert[T, U Numeric](dst []T, src []U, n int) {
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

	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(n) {
		st.ElemConvert(dst, src, n)
		return
	}

	// Parallelize conversion using chunks - reuse st function for each chunk
	parallelChunks(n, func(start, end int) {
		st.ElemConvert(dst[start:end], src[start:end], end-start)
	})
}

// elemConvertNumeric dispatches to specialized functions only for down-conversions that need clamping.
// For up-conversions and same-type, uses generic function with direct T(src) conversion.
func elemConvertNumeric[T, U Numeric](dst []T, src []U) []T {
	// Check if this is a down-conversion that needs clamping
	switch d := any(dst).(type) {
	case []int64:
		// Check if source is larger (float32/float64) - needs clamping
		switch any(src).(type) {
		case []float32, []float64:
			elemConvertToInt64(d, src)
			return dst
		default:
			// int64->int64 or smaller->int64: direct conversion, no clamping
			return elemConvertGeneric(dst, src)
		}
	case []int:
		// Check if source is larger - needs clamping (platform-specific)
		switch any(src).(type) {
		case []float32, []float64:
			elemConvertToInt(d, src)
			return dst
		default:
			// int->int or smaller->int: direct conversion, no clamping
			return elemConvertGeneric(dst, src)
		}
	case []int32:
		// Check if source is larger (float32/float64/int64/int) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int:
			elemConvertToInt32(d, src)
			return dst
		default:
			// int32->int32 or smaller->int32: direct conversion, no clamping
			return elemConvertGeneric(dst, src)
		}
	case []int16:
		// Check if source is larger (float32/float64/int64/int32/int) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int32, []int:
			elemConvertToInt16(d, src)
			return dst
		default:
			// int16->int16 or int8->int16: direct conversion, no clamping
			return elemConvertGeneric(dst, src)
		}
	case []int8:
		// Check if source is larger (float32/float64/int64/int32/int16/int) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int32, []int16, []int:
			elemConvertToInt8(d, src)
			return dst
		default:
			// int8->int8: direct conversion, no clamping
			return elemConvertGeneric(dst, src)
		}
	default:
		// For float32/float64: always direct conversion (up-conversion or same-type)
		return elemConvertGeneric(dst, src)
	}
}

// elemConvertGeneric handles up-conversions and same-type conversions.
// Uses direct T(src) conversion - no clamping needed, no switches in hot path.
// Optimized to detect same type at compile time when possible.
func elemConvertGeneric[T, U Numeric](dst []T, src []U) []T {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	// Fast path: same type - use type assertion to check and use copy
	// This is optimized by the compiler when T == U
	switch d := any(dst).(type) {
	case []float64:
		if s, ok := any(src).([]float64); ok {
			copy(d[:n], s[:n])
			return dst
		}
	case []float32:
		if s, ok := any(src).([]float32); ok {
			copy(d[:n], s[:n])
			return dst
		}
	case []int64:
		if s, ok := any(src).([]int64); ok {
			copy(d[:n], s[:n])
			return dst
		}
	case []int:
		if s, ok := any(src).([]int); ok {
			copy(d[:n], s[:n])
			return dst
		}
	case []int32:
		if s, ok := any(src).([]int32); ok {
			copy(d[:n], s[:n])
			return dst
		}
	case []int16:
		if s, ok := any(src).([]int16); ok {
			copy(d[:n], s[:n])
			return dst
		}
	case []int8:
		if s, ok := any(src).([]int8); ok {
			copy(d[:n], s[:n])
			return dst
		}
	}
	// Generic conversion: compiler optimizes T(src[i]) for each type combination
	// Bounds check elimination
	if n > 0 {
		_ = dst[n-1]
		_ = src[n-1]
	}
	for i := 0; i < n; i++ {
		dst[i] = T(src[i])
	}
	return dst
}

// ElemConvertStrided converts src to dst respecting the supplied shape/strides.
// Multi-threaded version that parallelizes strided conversion across CPU cores.
// Falls back to single-threaded implementation for small arrays.
// Supports conversion between float32, float64, int64, int32, int16, and int8.
func ElemConvertStrided[T, U Numeric](dst []T, src []U, shape []int, stridesDst, stridesSrc []int) {
	stridesDst = EnsureStrides(stridesDst, shape)
	stridesSrc = EnsureStrides(stridesSrc, shape)
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	// Check if types match for fast path (same type, contiguous)
	// For same type contiguous, just use copy() - no parallelization needed (copy is already optimized)
	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		switch d := any(dst).(type) {
		case []float32:
			if s, ok := any(src).([]float32); ok {
				copy(d[:size], s[:size])
				return
			}
		case []float64:
			if s, ok := any(src).([]float64); ok {
				copy(d[:size], s[:size])
				return
			}
		case []int64:
			if s, ok := any(src).([]int64); ok {
				copy(d[:size], s[:size])
				return
			}
		case []int:
			if s, ok := any(src).([]int); ok {
				copy(d[:size], s[:size])
				return
			}
		case []int32:
			if s, ok := any(src).([]int32); ok {
				copy(d[:size], s[:size])
				return
			}
		case []int16:
			if s, ok := any(src).([]int16); ok {
				copy(d[:size], s[:size])
				return
			}
		case []int8:
			if s, ok := any(src).([]int8); ok {
				copy(d[:size], s[:size])
				return
			}
		}
	}

	// Different types or strided: use conversion with strides
	// For small arrays, fallback to single-threaded implementation
	if !shouldParallelize(size) || len(shape) == 0 {
		st.ElemConvertStrided(dst, src, shape, stridesDst, stridesSrc)
		return
	}

	// Parallelize along first dimension - reuse st function for each chunk
	parallelTensorChunks(shape, func(startDim0, endDim0 int) {
		// Process chunk of first dimension [startDim0, endDim0)
		// Create adjusted shape for this chunk
		chunkShape := make([]int, len(shape))
		copy(chunkShape, shape)
		chunkShape[0] = endDim0 - startDim0

		// Calculate base offsets for this chunk
		baseOffsetDst := startDim0 * stridesDst[0]
		baseOffsetSrc := startDim0 * stridesSrc[0]

		// Reuse st function for this chunk - it will handle the strided iteration correctly
		st.ElemConvertStrided(dst[baseOffsetDst:], src[baseOffsetSrc:], chunkShape, stridesDst, stridesSrc)
	})
}

// elemConvertNumericStrided dispatches to specialized functions only for down-conversions that need clamping.
// For up-conversions and same-type, uses generic function with direct T(src) conversion.
func elemConvertNumericStrided[T, U Numeric](dst []T, src []U, shape []int, srcStrides, dstStrides []int) {
	// Check if this is a down-conversion that needs clamping
	switch d := any(dst).(type) {
	case []int64:
		// Check if source is larger (float32/float64) - needs clamping
		switch any(src).(type) {
		case []float32, []float64:
			elemConvertToInt64Strided(d, src, shape, srcStrides, dstStrides)
		default:
			// int64->int64 or smaller->int64: direct conversion, no clamping
			elemConvertStridedGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []int:
		// Check if source is larger - needs clamping (platform-specific)
		switch any(src).(type) {
		case []float32, []float64:
			elemConvertToIntStrided(d, src, shape, srcStrides, dstStrides)
		default:
			// int->int or smaller->int: direct conversion, no clamping
			elemConvertStridedGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []int32:
		// Check if source is larger (float32/float64/int64/int) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int:
			elemConvertToInt32Strided(d, src, shape, srcStrides, dstStrides)
		default:
			// int32->int32 or smaller->int32: direct conversion, no clamping
			elemConvertStridedGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []int16:
		// Check if source is larger (float32/float64/int64/int32/int) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int32, []int:
			elemConvertToInt16Strided(d, src, shape, srcStrides, dstStrides)
		default:
			// int16->int16 or int8->int16: direct conversion, no clamping
			elemConvertStridedGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	case []int8:
		// Check if source is larger (float32/float64/int64/int32/int16/int) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int32, []int16, []int:
			elemConvertToInt8Strided(d, src, shape, srcStrides, dstStrides)
		default:
			// int8->int8: direct conversion, no clamping
			elemConvertStridedGeneric(dst, src, shape, srcStrides, dstStrides)
		}
	default:
		// For float32/float64: always direct conversion (up-conversion or same-type)
		elemConvertStridedGeneric(dst, src, shape, srcStrides, dstStrides)
	}
}

// elemConvertStridedGeneric copies with strides using direct conversion - no clamping needed.
// Handles up-conversions and same-type conversions.
// Generic over source and destination types to eliminate type switches in hot path.
func elemConvertStridedGeneric[T, U Numeric](dst []T, src []U, shape []int, srcStrides, dstStrides []int) {
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
			sIdx := ComputeStrideOffset(indices, srcStrides)
			dIdx := ComputeStrideOffset(indices, dstStrides)
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

// elemConvertToInt64 handles conversions to int64 with clamping.
// Handles: float32/float64 -> int64 (needs clamping)
// int64 -> int64 (same type, fast path)
// int32/int16/int8 -> int64 (up-conversion, no clamping)
func elemConvertToInt64[U Numeric](dst []int64, src []U) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n > 0 {
		switch s := any(src).(type) {
		case []float32:
			clampToInt64(dst, s, n)
		case []float64:
			clampToInt64(dst, s, n)
		default:
			// Up-conversion: direct conversion
			for i := 0; i < n; i++ {
				dst[i] = int64(src[i])
			}
		}
	}
}

// elemConvertToInt32 handles conversions to int32 with clamping.
// Handles: float32/float64/int64 -> int32 (needs clamping)
// int32 -> int32 (same type, fast path)
// int16/int8 -> int32 (up-conversion, no clamping)
func elemConvertToInt32[U Numeric](dst []int32, src []U) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n > 0 {
		switch s := any(src).(type) {
		case []float32:
			clampToInt32(dst, s, n)
		case []float64:
			clampToInt32(dst, s, n)
		case []int64:
			clampToInt32(dst, s, n)
		case []int:
			clampToInt32(dst, s, n)
		default:
			// Up-conversion: direct conversion
			for i := 0; i < n; i++ {
				dst[i] = int32(src[i])
			}
		}
	}
}

// elemConvertToInt16 handles conversions to int16 with clamping.
// Handles: float32/float64/int64/int32 -> int16 (needs clamping)
// int16 -> int16 (same type, fast path)
// int8 -> int16 (up-conversion, no clamping)
func elemConvertToInt16[U Numeric](dst []int16, src []U) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n > 0 {
		switch s := any(src).(type) {
		case []float64:
			clampToInt16(dst, s, n)
		case []float32:
			clampToInt16(dst, s, n)
		case []int64:
			clampToInt16(dst, s, n)
		case []int:
			clampToInt16(dst, s, n)
		case []int32:
			clampToInt16(dst, s, n)
		default:
			// Up-conversion: direct conversion
			for i := 0; i < n; i++ {
				dst[i] = int16(src[i])
			}
		}
	}
}

// elemConvertToInt8 handles conversions to int8 with clamping.
// Handles: float32/float64/int64/int32/int16 -> int8 (needs clamping)
// int8 -> int8 (same type, fast path)
func elemConvertToInt8[U Numeric](dst []int8, src []U) {
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	if n > 0 {
		switch s := any(src).(type) {
		case []float64:
			clampToInt8(dst, s, n)
		case []float32:
			clampToInt8(dst, s, n)
		case []int:
			clampToInt8(dst, s, n)
		case []int64:
			clampToInt8(dst, s, n)
		case []int32:
			clampToInt8(dst, s, n)
		case []int16:
			clampToInt8(dst, s, n)
		default:
			// Same type: direct conversion
			for i := 0; i < n; i++ {
				dst[i] = int8(src[i])
			}
		}
	}
}

// clampToInt8 implements the hot path inner loop for clamping to int8 range [-128, 127].
// Generic over source types that need clamping (float32, float64, int16).
// n is the number of elements to process (bounds checks eliminated by explicit length).
func clampToInt8[U clampableToInt8](dst []int8, src []U, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1] // Bounds check elimination hint
	_ = src[n-1] // Bounds check elimination hint
	for i := 0; i < n; i++ {
		val := src[i]
		// Direct assignment to avoid type conversion issues and improve performance
		if val > U(math.MaxInt8) {
			dst[i] = math.MaxInt8
		} else if val < U(math.MinInt8) {
			dst[i] = math.MinInt8
		} else {
			dst[i] = int8(val)
		}
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
		// Direct assignment to avoid type conversion issues and improve performance
		if val > U(int64(math.MaxInt64)) {
			dst[i] = math.MaxInt64
		} else if val < U(int64(math.MinInt64)) {
			dst[i] = math.MinInt64
		} else {
			dst[i] = int64(val)
		}
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
		// Direct assignment to avoid float32 precision issues when val is float32
		if val > U(math.MaxInt32) {
			dst[i] = math.MaxInt32
		} else if val < U(math.MinInt32) {
			dst[i] = math.MinInt32
		} else {
			dst[i] = int32(val)
		}
	}
}

// clampToInt16 implements the hot path inner loop for clamping to int16 range [-32768, 32767].
// Generic over source types that need clamping (float32, float64).
func clampToInt16[U clampableToInt16](dst []int16, src []U, n int) {
	if n == 0 {
		return
	}
	_ = dst[n-1] // Bounds check elimination hint
	_ = src[n-1] // Bounds check elimination hint
	for i := 0; i < n; i++ {
		val := src[i]
		// Direct assignment to avoid type conversion issues and improve performance
		if val > U(math.MaxInt16) {
			dst[i] = math.MaxInt16
		} else if val < U(math.MinInt16) {
			dst[i] = math.MinInt16
		} else {
			dst[i] = int16(val)
		}
	}
}

// elemConvertToInt64Strided handles conversions to int64 with clamping.
func elemConvertToInt64Strided[U Numeric](dst []int64, src []U, shape []int, srcStrides, dstStrides []int) {
	switch s := any(src).(type) {
	case []float32:
		clampToInt64Strided(dst, s, shape, srcStrides, dstStrides)
	case []float64:
		clampToInt64Strided(dst, s, shape, srcStrides, dstStrides)
	}
}

// elemConvertToInt32Strided handles conversions to int32 with clamping.
func elemConvertToInt32Strided[U Numeric](dst []int32, src []U, shape []int, srcStrides, dstStrides []int) {
	switch s := any(src).(type) {
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

// elemConvertToInt16Strided handles conversions to int16 with clamping.
func elemConvertToInt16Strided[U Numeric](dst []int16, src []U, shape []int, srcStrides, dstStrides []int) {
	switch s := any(src).(type) {
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

// elemConvertToInt8Strided handles conversions to int8 with clamping.
func elemConvertToInt8Strided[U Numeric](dst []int8, src []U, shape []int, srcStrides, dstStrides []int) {
	switch s := any(src).(type) {
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
			sIdx := ComputeStrideOffset(indices, srcStrides)
			dIdx := ComputeStrideOffset(indices, dstStrides)
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
			sIdx := ComputeStrideOffset(indices, srcStrides)
			dIdx := ComputeStrideOffset(indices, dstStrides)
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
			sIdx := ComputeStrideOffset(indices, srcStrides)
			dIdx := ComputeStrideOffset(indices, dstStrides)
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
			sIdx := ComputeStrideOffset(indices, srcStrides)
			dIdx := ComputeStrideOffset(indices, dstStrides)
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

package generics

import . "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

// ElemCopy copies src into dst for contiguous arrays.
// This is a simple, optimized version for the common case of contiguous memory.
// For strided or non-contiguous arrays, use ElemCopyStrided.
func ElemCopy[T Numeric](dst, src []T, n int) {
	if n == 0 {
		return
	}
	copy(dst[:n], src[:n])
}

// ElemCopyStrided copies src into dst respecting the supplied shape/strides.
// This function handles both contiguous and strided cases with optimized paths.
func ElemCopyStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int) {
	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays - use direct copy
		copy(dst[:size], src[:size])
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
		dst[dIdx] = src[sIdx]
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemSwap swaps elements between dst and src for contiguous arrays.
// This is a simple, optimized version for the common case of contiguous memory.
// For strided or non-contiguous arrays, use ElemSwapStrided.
func ElemSwap[T Numeric](dst, src []T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		dst[i], src[i] = src[i], dst[i]
	}
}

// ElemSwapStrided swaps elements between dst and src respecting the supplied shape/strides.
// This function handles both contiguous and strided cases with optimized paths.
func ElemSwapStrided[T Numeric](dst, src []T, shape []int, stridesDst, stridesSrc []int) {
	// Use stack-allocated arrays for stride computation
	var dstStridesStatic [MAX_DIMS]int
	var srcStridesStatic [MAX_DIMS]int
	stridesDst = EnsureStrides(dstStridesStatic[:len(shape)], stridesDst, shape)
	stridesSrc = EnsureStrides(srcStridesStatic[:len(shape)], stridesSrc, shape)
	size := SizeFromShape(shape)
	if size == 0 {
		return
	}

	if IsContiguous(stridesDst, shape) && IsContiguous(stridesSrc, shape) {
		// Fast path: contiguous arrays - use direct swap
		for i := 0; i < size; i++ {
			dst[i], src[i] = src[i], dst[i]
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
		dst[dIdx], src[sIdx] = src[sIdx], dst[dIdx]
		if !AdvanceOffsets(shape, indices, offsets, stridesDst, stridesSrc) {
			break
		}
	}
}

// ElemCopyAny copies src into dst for contiguous arrays, accepting any numeric types.
// If src and dst have the same type, uses ElemCopy for optimal performance.
// If types differ, uses ElemConvert to perform type conversion.
// This is a convenience wrapper when both src and dst are of type any.
func ElemCopyAny(dst, src any, n int) {
	if n == 0 {
		return
	}

	// Type switch on source to determine types
	switch s := src.(type) {
	case []float32:
		if d, ok := dst.([]float32); ok {
			ElemCopy(d, s, n)
			return
		}
		// Different types: use conversion
		switch d := dst.(type) {
		case []float64:
			ElemConvert(d, s, n)
		case []int64:
			ElemConvert(d, s, n)
		case []int:
			ElemConvert(d, s, n)
		case []int32:
			ElemConvert(d, s, n)
		case []int16:
			ElemConvert(d, s, n)
		case []int8:
			ElemConvert(d, s, n)
		}
	case []float64:
		if d, ok := dst.([]float64); ok {
			ElemCopy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvert(d, s, n)
		case []int64:
			ElemConvert(d, s, n)
		case []int:
			ElemConvert(d, s, n)
		case []int32:
			ElemConvert(d, s, n)
		case []int16:
			ElemConvert(d, s, n)
		case []int8:
			ElemConvert(d, s, n)
		}
	case []int64:
		if d, ok := dst.([]int64); ok {
			ElemCopy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvert(d, s, n)
		case []float64:
			ElemConvert(d, s, n)
		case []int:
			ElemConvert(d, s, n)
		case []int32:
			ElemConvert(d, s, n)
		case []int16:
			ElemConvert(d, s, n)
		case []int8:
			ElemConvert(d, s, n)
		}
	case []int:
		if d, ok := dst.([]int); ok {
			ElemCopy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvert(d, s, n)
		case []float64:
			ElemConvert(d, s, n)
		case []int64:
			ElemConvert(d, s, n)
		case []int32:
			ElemConvert(d, s, n)
		case []int16:
			ElemConvert(d, s, n)
		case []int8:
			ElemConvert(d, s, n)
		}
	case []int32:
		if d, ok := dst.([]int32); ok {
			ElemCopy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvert(d, s, n)
		case []float64:
			ElemConvert(d, s, n)
		case []int64:
			ElemConvert(d, s, n)
		case []int:
			ElemConvert(d, s, n)
		case []int16:
			ElemConvert(d, s, n)
		case []int8:
			ElemConvert(d, s, n)
		}
	case []int16:
		if d, ok := dst.([]int16); ok {
			ElemCopy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvert(d, s, n)
		case []float64:
			ElemConvert(d, s, n)
		case []int64:
			ElemConvert(d, s, n)
		case []int:
			ElemConvert(d, s, n)
		case []int32:
			ElemConvert(d, s, n)
		case []int8:
			ElemConvert(d, s, n)
		}
	case []int8:
		if d, ok := dst.([]int8); ok {
			ElemCopy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvert(d, s, n)
		case []float64:
			ElemConvert(d, s, n)
		case []int64:
			ElemConvert(d, s, n)
		case []int:
			ElemConvert(d, s, n)
		case []int32:
			ElemConvert(d, s, n)
		case []int16:
			ElemConvert(d, s, n)
		}
	}
}

// ElemCopyStridedAny copies src into dst respecting the supplied shape/strides, accepting any numeric types.
// If src and dst have the same type, uses ElemCopyStrided for optimal performance.
// If types differ, uses ElemConvertStrided to perform type conversion.
// This is a convenience wrapper when both src and dst are of type any.
func ElemCopyStridedAny(dst, src any, shape []int, stridesDst, stridesSrc []int) {
	if len(shape) == 0 {
		return
	}

	// Type switch on source to determine types
	switch s := src.(type) {
	case []float32:
		if d, ok := dst.([]float32); ok {
			ElemCopyStrided(d, s, shape, stridesDst, stridesSrc)
			return
		}
		// Different types: use conversion
		switch d := dst.(type) {
		case []float64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []float64:
		if d, ok := dst.([]float64); ok {
			ElemCopyStrided(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []int64:
		if d, ok := dst.([]int64); ok {
			ElemCopyStrided(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []float64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []int:
		if d, ok := dst.([]int); ok {
			ElemCopyStrided(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []float64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []int32:
		if d, ok := dst.([]int32); ok {
			ElemCopyStrided(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []float64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []int16:
		if d, ok := dst.([]int16); ok {
			ElemCopyStrided(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []float64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []int8:
		if d, ok := dst.([]int8); ok {
			ElemCopyStrided(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []float64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			ElemConvertStrided(d, s, shape, stridesDst, stridesSrc)
		}
	}
}

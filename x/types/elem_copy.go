package types

// CopyClampedAny copies src into dst for contiguous arrays, accepting any numeric types.
// If src and dst have the same type, uses ElemCopy for optimal performance.
// If types differ, uses ElemConvert to perform type conversion.
// This is a convenience wrapper when both src and dst are of type any.
func CopyClampedAny(dst, src any, n int) {
	if n == 0 {
		return
	}

	// Type switch on source to determine types
	switch s := src.(type) {
	case []float32:
		if d, ok := dst.([]float32); ok {
			Copy(d, s, n)
			return
		}
		// Different types: use conversion
		switch d := dst.(type) {
		case []float64:
			CastClamped(d, s, n)
		case []int64:
			CastClamped(d, s, n)
		case []int:
			CastClamped(d, s, n)
		case []int32:
			CastClamped(d, s, n)
		case []int16:
			CastClamped(d, s, n)
		case []int8:
			CastClamped(d, s, n)
		}
	case []float64:
		if d, ok := dst.([]float64); ok {
			Copy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClamped(d, s, n)
		case []int64:
			CastClamped(d, s, n)
		case []int:
			CastClamped(d, s, n)
		case []int32:
			CastClamped(d, s, n)
		case []int16:
			CastClamped(d, s, n)
		case []int8:
			CastClamped(d, s, n)
		}
	case []int64:
		if d, ok := dst.([]int64); ok {
			Copy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClamped(d, s, n)
		case []float64:
			CastClamped(d, s, n)
		case []int:
			CastClamped(d, s, n)
		case []int32:
			CastClamped(d, s, n)
		case []int16:
			CastClamped(d, s, n)
		case []int8:
			CastClamped(d, s, n)
		}
	case []int:
		if d, ok := dst.([]int); ok {
			Copy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClamped(d, s, n)
		case []float64:
			CastClamped(d, s, n)
		case []int64:
			CastClamped(d, s, n)
		case []int32:
			CastClamped(d, s, n)
		case []int16:
			CastClamped(d, s, n)
		case []int8:
			CastClamped(d, s, n)
		}
	case []int32:
		if d, ok := dst.([]int32); ok {
			Copy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClamped(d, s, n)
		case []float64:
			CastClamped(d, s, n)
		case []int64:
			CastClamped(d, s, n)
		case []int:
			CastClamped(d, s, n)
		case []int16:
			CastClamped(d, s, n)
		case []int8:
			CastClamped(d, s, n)
		}
	case []int16:
		if d, ok := dst.([]int16); ok {
			Copy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClamped(d, s, n)
		case []float64:
			CastClamped(d, s, n)
		case []int64:
			CastClamped(d, s, n)
		case []int:
			CastClamped(d, s, n)
		case []int32:
			CastClamped(d, s, n)
		case []int8:
			CastClamped(d, s, n)
		}
	case []int8:
		if d, ok := dst.([]int8); ok {
			Copy(d, s, n)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClamped(d, s, n)
		case []float64:
			CastClamped(d, s, n)
		case []int64:
			CastClamped(d, s, n)
		case []int:
			CastClamped(d, s, n)
		case []int32:
			CastClamped(d, s, n)
		case []int16:
			CastClamped(d, s, n)
		}
	}
}

// CopyClampedStridedAny copies src into dst respecting the supplied shape/strides, accepting any numeric types.
// If src and dst have the same type, uses ElemCopyStrided for optimal performance.
// If types differ, uses ElemConvertStrided to perform type conversion.
// This is a convenience wrapper when both src and dst are of type any.
func CopyClampedStridedAny(dst, src any, shape []int, stridesDst, stridesSrc []int) {
	if len(shape) == 0 {
		return
	}

	// Type switch on source to determine types
	switch s := src.(type) {
	case []float32:
		if d, ok := dst.([]float32); ok {
			CopyStridedShaped(d, s, shape, stridesDst, stridesSrc)
			return
		}
		// Different types: use conversion
		switch d := dst.(type) {
		case []float64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []float64:
		if d, ok := dst.([]float64); ok {
			CopyStridedShaped(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []int64:
		if d, ok := dst.([]int64); ok {
			CopyStridedShaped(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []float64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []int:
		if d, ok := dst.([]int); ok {
			CopyStridedShaped(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []float64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []int32:
		if d, ok := dst.([]int32); ok {
			CopyStridedShaped(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []float64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []int16:
		if d, ok := dst.([]int16); ok {
			CopyStridedShaped(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []float64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int8:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		}
	case []int8:
		if d, ok := dst.([]int8); ok {
			CopyStridedShaped(d, s, shape, stridesDst, stridesSrc)
			return
		}
		switch d := dst.(type) {
		case []float32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []float64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int64:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int32:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		case []int16:
			CastClampedStrided(d, s, shape, stridesDst, stridesSrc)
		}
	}
}

package types

import (
	shp "github.com/itohio/EasyRobot/x/math/shape"
)

// ConvertAny converts src to dst with proper scaling between type ranges.
// For downconversions, applies dithering to reduce quantization artifacts.
// For same-type conversions, uses copy (fast path).
// Accepts any type and returns the destination slice.
func ConvertAny(dstData, srcData any) any {
	if srcData == nil || dstData == nil {
		return nil
	}

	// Check if types match for fast path
	switch src := srcData.(type) {
	case []float32:
		if dst, ok := dstData.([]float32); ok {
			n := min(len(dst), len(src))
			copy(dst[:n], src[:n])
			return dst
		}
	case []int:
		if dst, ok := dstData.([]int); ok {
			n := min(len(dst), len(src))
			copy(dst[:n], src[:n])
			return dst
		}
	case []int64:
		if dst, ok := dstData.([]int64); ok {
			n := min(len(dst), len(src))
			copy(dst[:n], src[:n])
			return dst
		}
	case []int32:
		if dst, ok := dstData.([]int32); ok {
			n := min(len(dst), len(src))
			copy(dst[:n], src[:n])
			return dst
		}
	case []int16:
		if dst, ok := dstData.([]int16); ok {
			n := min(len(dst), len(src))
			copy(dst[:n], src[:n])
			return dst
		}
	case []int8:
		if dst, ok := dstData.([]int8); ok {
			n := min(len(dst), len(src))
			copy(dst[:n], src[:n])
			return dst
		}
	case []complex64:
		if dst, ok := dstData.([]complex64); ok {
			n := min(len(dst), len(src))
			copy(dst[:n], src[:n])
			return dst
		}
	}

	// Different types: perform conversion with scaling and dithering
	return convertWithScaling(dstData, srcData)
}

// convertWithScaling performs type conversion with proper scaling between type ranges.
func convertWithScaling(dstData, srcData any) any {
	switch dst := dstData.(type) {
	case []float64:
		return convertWithScalingInner(dst, srcData)
	case []float32:
		return convertWithScalingInner(dst, srcData)
	case []int64:
		return convertWithScalingInner(dst, srcData)
	case []int:
		return convertWithScalingInner(dst, srcData)
	case []int32:
		return convertWithScalingInner(dst, srcData)
	case []int16:
		return convertWithScalingInner(dst, srcData)
	case []int8:
		return convertWithScalingInner(dst, srcData)
	case []complex64:
		return convertWithScalingInner(dst, srcData)
	}
	return nil
}

// convertWithScalingInner dispatches on source type and calls specialized Convert functions.
// This eliminates all type switches from the hot path.
func convertWithScalingInner[T Number](dst []T, srcData any) any {
	n := min(len(dst), getSliceLength(srcData))
	if n == 0 {
		return dst
	}

	switch src := srcData.(type) {
	case []float64:
		Convert(dst, src, n)
		return dst
	case []float32:
		Convert(dst, src, n)
		return dst
	case []int64:
		Convert(dst, src, n)
		return dst
	case []int:
		Convert(dst, src, n)
		return dst
	case []int32:
		Convert(dst, src, n)
		return dst
	case []int16:
		Convert(dst, src, n)
		return dst
	case []int8:
		Convert(dst, src, n)
		return dst
	case []complex64:
		Convert(dst, src, n)
		return dst
	}
	return nil
}

// getSliceLength returns the length of a slice stored in any, or 0 if not a slice
func getSliceLength(data any) int {
	switch s := data.(type) {
	case []float64:
		return len(s)
	case []float32:
		return len(s)
	case []int64:
		return len(s)
	case []int:
		return len(s)
	case []int32:
		return len(s)
	case []int16:
		return len(s)
	case []int8:
		return len(s)
	case []complex64:
		return len(s)
	}
	return 0
}

// ConvertStridedAny handles strided/non-contiguous arrays with any types
func ConvertStridedAny(dstData, srcData any, shape []int, stridesDst, stridesSrc []int) any {
	if srcData == nil || dstData == nil {
		return nil
	}

	if len(shape) == 0 {
		return nil
	}

	// Normalize strides
	var dstStridesBuf [shp.MAX_DIMS]int
	dstStrides := shp.EnsureStrides(dstStridesBuf[:len(shape)], stridesDst, shape)
	var srcStridesBuf [shp.MAX_DIMS]int
	srcStrides := shp.EnsureStrides(srcStridesBuf[:len(shape)], stridesSrc, shape)

	// Check if types match for fast path
	switch src := srcData.(type) {
	case []float32:
		if dst, ok := dstData.([]float32); ok {
			if shp.IsContiguous(dstStrides, shape) && shp.IsContiguous(srcStrides, shape) {
				size := shp.SizeFromShape(shape)
				Convert(dst, src, size)
				return dst
			}
		}
	case []int:
		if dst, ok := dstData.([]int); ok {
			if shp.IsContiguous(dstStrides, shape) && shp.IsContiguous(srcStrides, shape) {
				size := shp.SizeFromShape(shape)
				Convert(dst, src, size)
				return dst
			}
		}
	case []int64:
		if dst, ok := dstData.([]int64); ok {
			if shp.IsContiguous(dstStrides, shape) && shp.IsContiguous(srcStrides, shape) {
				size := shp.SizeFromShape(shape)
				Convert(dst, src, size)
				return dst
			}
		}
	case []int32:
		if dst, ok := dstData.([]int32); ok {
			if shp.IsContiguous(dstStrides, shape) && shp.IsContiguous(srcStrides, shape) {
				size := shp.SizeFromShape(shape)
				Convert(dst, src, size)
				return dst
			}
		}
	case []int16:
		if dst, ok := dstData.([]int16); ok {
			if shp.IsContiguous(dstStrides, shape) && shp.IsContiguous(srcStrides, shape) {
				size := shp.SizeFromShape(shape)
				Convert(dst, src, size)
				return dst
			}
		}
	case []int8:
		if dst, ok := dstData.([]int8); ok {
			if shp.IsContiguous(dstStrides, shape) && shp.IsContiguous(srcStrides, shape) {
				size := shp.SizeFromShape(shape)
				Convert(dst, src, size)
				return dst
			}
		}
	case []complex64:
		if dst, ok := dstData.([]complex64); ok {
			if shp.IsContiguous(dstStrides, shape) && shp.IsContiguous(srcStrides, shape) {
				size := shp.SizeFromShape(shape)
				Convert(dst, src, size)
				return dst
			}
		}
	}

	// Different types or strided: use conversion with strides
	return convertStridedWithScaling(dstData, srcData, shape, dstStrides, srcStrides)
}

// convertStridedWithScaling performs strided conversion with proper scaling
func convertStridedWithScaling(dstData, srcData any, shape []int, dstStrides, srcStrides []int) any {
	switch dst := dstData.(type) {
	case []float64:
		return convertStridedAnyInner(dst, srcData, shape, dstStrides, srcStrides)
	case []float32:
		return convertStridedAnyInner(dst, srcData, shape, dstStrides, srcStrides)
	case []int64:
		return convertStridedAnyInner(dst, srcData, shape, dstStrides, srcStrides)
	case []int:
		return convertStridedAnyInner(dst, srcData, shape, dstStrides, srcStrides)
	case []int32:
		return convertStridedAnyInner(dst, srcData, shape, dstStrides, srcStrides)
	case []int16:
		return convertStridedAnyInner(dst, srcData, shape, dstStrides, srcStrides)
	case []int8:
		return convertStridedAnyInner(dst, srcData, shape, dstStrides, srcStrides)
	case []complex64:
		return convertStridedAnyInner(dst, srcData, shape, dstStrides, srcStrides)
	}
	return nil
}

// convertStridedAnyInner dispatches on source type and calls ConvertStrided
func convertStridedAnyInner[T Number](dst []T, srcData any, shape []int, stridesDst, stridesSrc []int) any {
	switch src := srcData.(type) {
	case []float64:
		ConvertStrided(dst, src, shape, stridesDst, stridesSrc)
		return dst
	case []float32:
		ConvertStrided(dst, src, shape, stridesDst, stridesSrc)
		return dst
	case []int64:
		ConvertStrided(dst, src, shape, stridesDst, stridesSrc)
		return dst
	case []int:
		ConvertStrided(dst, src, shape, stridesDst, stridesSrc)
		return dst
	case []int32:
		ConvertStrided(dst, src, shape, stridesDst, stridesSrc)
		return dst
	case []int16:
		ConvertStrided(dst, src, shape, stridesDst, stridesSrc)
		return dst
	case []int8:
		ConvertStrided(dst, src, shape, stridesDst, stridesSrc)
		return dst
	case []complex64:
		ConvertStrided(dst, src, shape, stridesDst, stridesSrc)
		return dst
	}
	return nil
}

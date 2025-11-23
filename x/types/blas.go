package types

import (
	. "github.com/itohio/EasyRobot/x/math/shape"
)

// Copy copies x into y for contiguous arrays.
// This is BLAS COPY operation optimized for the common case of contiguous memory.
func Copy[T Number](y, x []T, n int) {
	if n == 0 {
		return
	}
	copy(y[:n], x[:n])
}

// CopyStrided copies x into y with stride support.
// This is BLAS COPY operation with stride parameters.
func CopyStrided[T Number](y, x []T, strideY, strideX, n int) {
	if n == 0 {
		return
	}

	if strideY == 1 && strideX == 1 {
		// Fast path: contiguous arrays
		copy(y[:n], x[:n])
		return
	}

	// Strided path
	py := 0
	px := 0
	for i := 0; i < n; i++ {
		y[py] = x[px]
		py += strideY
		px += strideX
	}
}

// CopyStridedShaped copies dst into src respecting the supplied shape/strides (multi-dimensional version).
// This function handles both contiguous and strided cases with optimized paths.
func CopyStridedShaped[T Number](dst, src []T, shape []int, stridesDst, stridesSrc []int) {
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

// Swap swaps x and y for contiguous arrays.
// This is BLAS SWAP operation optimized for the common case of contiguous memory.
func Swap[T Number](x, y []T, n int) {
	if n == 0 {
		return
	}
	for i := 0; i < n; i++ {
		x[i], y[i] = y[i], x[i]
	}
}

// SwapStrided swaps x and y with stride support.
// This is BLAS SWAP operation with stride parameters.
func SwapStrided[T Number](x, y []T, strideX, strideY, n int) {
	if n == 0 {
		return
	}

	if strideX == 1 && strideY == 1 {
		// Fast path: contiguous arrays
		for i := 0; i < n; i++ {
			x[i], y[i] = y[i], x[i]
		}
		return
	}

	// Strided path
	px := 0
	py := 0
	for i := 0; i < n; i++ {
		x[px], y[py] = y[py], x[px]
		px += strideX
		py += strideY
	}
}

// SwapStridedShaped swaps dst and src respecting the supplied shape/strides (multi-dimensional version).
// This function handles both contiguous and strided cases with optimized paths.
func SwapStridedShaped[T Number](dst, src []T, shape []int, stridesDst, stridesSrc []int) {
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

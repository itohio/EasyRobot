package fp32

import "github.com/chewxy/math32"

func ReduceSum(dst []float32, dstShape []int, dstStrides []int, src []float32, srcShape []int, srcStrides []int, axes []int) {
	sizeDst := SizeFromShape(dstShape)
	if len(srcShape) == 0 || len(src) == 0 || sizeDst == 0 {
		return
	}

	for i := 0; i < sizeDst; i++ {
		dst[i] = 0
	}

	dstStrides = EnsureStrides(dstStrides, dstShape)
	srcStrides = EnsureStrides(srcStrides, srcShape)
	reduceMask := makeReduceMask(len(srcShape), axes)
	mapped := mapStridesToSource(srcShape, reduceMask, dstStrides)

	indices := make([]int, len(srcShape))
	offsets := make([]int, 2) // src, dst
	strideSet := [][]int{srcStrides, mapped}
	for {
		dstOffset := offsets[1]
		dst[dstOffset] += src[offsets[0]]
		if !advanceOffsets(srcShape, indices, offsets, strideSet) {
			break
		}
	}
}

func ReduceMean(dst []float32, dstShape []int, dstStrides []int, src []float32, srcShape []int, srcStrides []int, axes []int) {
	ReduceSum(dst, dstShape, dstStrides, src, srcShape, srcStrides, axes)

	if len(axes) == 0 {
		return
	}

	count := 1
	for _, axis := range axes {
		count *= srcShape[axis]
	}
	if count == 0 {
		return
	}

	inv := 1.0 / float32(count)
	sizeDst := SizeFromShape(dstShape)
	for i := 0; i < sizeDst; i++ {
		dst[i] *= inv
	}
}

func ReduceMax(dst []float32, dstShape []int, dstStrides []int, src []float32, srcShape []int, srcStrides []int, axes []int) {
	sizeDst := SizeFromShape(dstShape)
	if len(srcShape) == 0 || len(src) == 0 || sizeDst == 0 {
		return
	}

	dstStrides = EnsureStrides(dstStrides, dstShape)
	srcStrides = EnsureStrides(srcStrides, srcShape)
	reduceMask := makeReduceMask(len(srcShape), axes)
	mapped := mapStridesToSource(srcShape, reduceMask, dstStrides)

	for i := 0; i < sizeDst; i++ {
		dst[i] = -math32.MaxFloat32
	}

	indices := make([]int, len(srcShape))
	offsets := make([]int, 2)
	strideSet := [][]int{srcStrides, mapped}
	for {
		dstOffset := offsets[1]
		val := src[offsets[0]]
		if val > dst[dstOffset] {
			dst[dstOffset] = val
		}
		if !advanceOffsets(srcShape, indices, offsets, strideSet) {
			break
		}
	}
}

func ReduceMin(dst []float32, dstShape []int, dstStrides []int, src []float32, srcShape []int, srcStrides []int, axes []int) {
	sizeDst := SizeFromShape(dstShape)
	if len(srcShape) == 0 || len(src) == 0 || sizeDst == 0 {
		return
	}

	dstStrides = EnsureStrides(dstStrides, dstShape)
	srcStrides = EnsureStrides(srcStrides, srcShape)
	reduceMask := makeReduceMask(len(srcShape), axes)
	mapped := mapStridesToSource(srcShape, reduceMask, dstStrides)

	for i := 0; i < sizeDst; i++ {
		dst[i] = math32.MaxFloat32
	}

	indices := make([]int, len(srcShape))
	offsets := make([]int, 2)
	strideSet := [][]int{srcStrides, mapped}
	for {
		dstOffset := offsets[1]
		val := src[offsets[0]]
		if val < dst[dstOffset] {
			dst[dstOffset] = val
		}
		if !advanceOffsets(srcShape, indices, offsets, strideSet) {
			break
		}
	}
}

func Argmax(dst []float32, dstShape []int, dstStrides []int, src []float32, srcShape []int, srcStrides []int, axis int) {
	sizeDst := SizeFromShape(dstShape)
	if len(srcShape) == 0 || len(src) == 0 || sizeDst == 0 {
		return
	}

	dstStrides = EnsureStrides(dstStrides, dstShape)
	srcStrides = EnsureStrides(srcStrides, srcShape)
	reduceMask := makeReduceMask(len(srcShape), []int{axis})
	mapped := mapStridesToSource(srcShape, reduceMask, dstStrides)
	maxVals := make([]float32, sizeDst)
	for i := range maxVals {
		maxVals[i] = -math32.MaxFloat32
		dst[i] = 0
	}

	indices := make([]int, len(srcShape))
	offsets := make([]int, 2)
	strideSet := [][]int{srcStrides, mapped}
	for {
		dstOffset := offsets[1]
		val := src[offsets[0]]
		if val > maxVals[dstOffset] {
			maxVals[dstOffset] = val
			dst[dstOffset] = float32(indices[axis])
		}
		if !advanceOffsets(srcShape, indices, offsets, strideSet) {
			break
		}
	}
}

// Argmin finds index of minimum element along specified axis.
// dst stores the indices as float32 values (can be cast to int32 if needed).
func Argmin(dst []int32, dstShape, dstStrides []int, src []float32, srcShape, srcStrides []int, axis int) {
	sizeDst := SizeFromShape(dstShape)
	if len(srcShape) == 0 || len(src) == 0 || sizeDst == 0 {
		return
	}

	dstStrides = EnsureStrides(dstStrides, dstShape)
	srcStrides = EnsureStrides(srcStrides, srcShape)
	reduceMask := makeReduceMask(len(srcShape), []int{axis})
	mapped := mapStridesToSource(srcShape, reduceMask, dstStrides)
	minVals := make([]float32, sizeDst)
	for i := range minVals {
		minVals[i] = math32.MaxFloat32
		dst[i] = 0
	}

	indices := make([]int, len(srcShape))
	offsets := make([]int, 2)
	strideSet := [][]int{srcStrides, mapped}
	for {
		dstOffset := offsets[1]
		val := src[offsets[0]]
		if val < minVals[dstOffset] {
			minVals[dstOffset] = val
			dst[dstOffset] = int32(indices[axis])
		}
		if !advanceOffsets(srcShape, indices, offsets, strideSet) {
			break
		}
	}
}

func makeReduceMask(rank int, axes []int) []bool {
	mask := make([]bool, rank)
	for _, axis := range axes {
		if axis >= 0 && axis < rank {
			mask[axis] = true
		}
	}
	return mask
}

func mapStridesToSource(srcShape []int, mask []bool, dstStrides []int) []int {
	mapped := make([]int, len(srcShape))
	dstIdx := 0
	for dim := range srcShape {
		if mask[dim] {
			mapped[dim] = 0
			continue
		}
		if dstIdx < len(dstStrides) {
			mapped[dim] = dstStrides[dstIdx]
		}
		dstIdx++
	}
	return mapped
}

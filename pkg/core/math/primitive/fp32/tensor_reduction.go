package fp32

import (
	"github.com/chewxy/math32"
	helpers "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

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
	var mappedBuf [helpers.MAX_DIMS]int
	mapped := mapStridesToSource(mappedBuf[:], srcShape, reduceMask, dstStrides)

	strideSet := [][]int{srcStrides, mapped}
	var offsetsArr [2]int
	process := func(indices []int, offsets []int) {
		for {
			dstOffset := offsets[1]
			dst[dstOffset] += src[offsets[0]]
			if !advanceOffsets(srcShape, indices, offsets, strideSet) {
				break
			}
		}
	}

	if len(srcShape) <= helpers.MAX_DIMS {
		var indicesArr [helpers.MAX_DIMS]int
		process(indicesArr[:len(srcShape)], offsetsArr[:len(strideSet)])
		return
	}

	process(make([]int, len(srcShape)), offsetsArr[:len(strideSet)])
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
	var mappedBuf [helpers.MAX_DIMS]int
	mapped := mapStridesToSource(mappedBuf[:], srcShape, reduceMask, dstStrides)

	for i := 0; i < sizeDst; i++ {
		dst[i] = -math32.MaxFloat32
	}

	strideSet := [][]int{srcStrides, mapped}
	var offsetsArr [2]int
	process := func(indices []int, offsets []int) {
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

	if len(srcShape) <= helpers.MAX_DIMS {
		var indicesArr [helpers.MAX_DIMS]int
		process(indicesArr[:len(srcShape)], offsetsArr[:len(strideSet)])
		return
	}

	process(make([]int, len(srcShape)), offsetsArr[:len(strideSet)])
}

func ReduceMin(dst []float32, dstShape []int, dstStrides []int, src []float32, srcShape []int, srcStrides []int, axes []int) {
	sizeDst := SizeFromShape(dstShape)
	if len(srcShape) == 0 || len(src) == 0 || sizeDst == 0 {
		return
	}

	dstStrides = EnsureStrides(dstStrides, dstShape)
	srcStrides = EnsureStrides(srcStrides, srcShape)
	reduceMask := makeReduceMask(len(srcShape), axes)
	var mappedBuf [helpers.MAX_DIMS]int
	mapped := mapStridesToSource(mappedBuf[:], srcShape, reduceMask, dstStrides)

	for i := 0; i < sizeDst; i++ {
		dst[i] = math32.MaxFloat32
	}

	strideSet := [][]int{srcStrides, mapped}
	var offsetsArr [2]int
	process := func(indices []int, offsets []int) {
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

	if len(srcShape) <= helpers.MAX_DIMS {
		var indicesArr [helpers.MAX_DIMS]int
		process(indicesArr[:len(srcShape)], offsetsArr[:len(strideSet)])
		return
	}

	process(make([]int, len(srcShape)), offsetsArr[:len(strideSet)])
}

func Argmax(dst []float32, dstShape []int, dstStrides []int, src []float32, srcShape []int, srcStrides []int, axis int) {
	sizeDst := SizeFromShape(dstShape)
	if len(srcShape) == 0 || len(src) == 0 || sizeDst == 0 {
		return
	}

	dstStrides = EnsureStrides(dstStrides, dstShape)
	srcStrides = EnsureStrides(srcStrides, srcShape)
	reduceMask := makeReduceMask(len(srcShape), []int{axis})
	var mappedBuf [helpers.MAX_DIMS]int
	mapped := mapStridesToSource(mappedBuf[:], srcShape, reduceMask, dstStrides)
	maxVals := Pool.Get(sizeDst)
	defer Pool.Put(maxVals)
	for i := 0; i < sizeDst; i++ {
		maxVals[i] = -math32.MaxFloat32
		dst[i] = 0
	}

	strideSet := [][]int{srcStrides, mapped}
	var offsetsArr [2]int
	process := func(indices []int, offsets []int) {
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

	if len(srcShape) <= helpers.MAX_DIMS {
		var indicesArr [helpers.MAX_DIMS]int
		process(indicesArr[:len(srcShape)], offsetsArr[:len(strideSet)])
		return
	}

	process(make([]int, len(srcShape)), offsetsArr[:len(strideSet)])
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
	var mappedBuf [helpers.MAX_DIMS]int
	mapped := mapStridesToSource(mappedBuf[:], srcShape, reduceMask, dstStrides)
	minVals := Pool.Get(sizeDst)
	defer Pool.Put(minVals)
	for i := 0; i < sizeDst; i++ {
		minVals[i] = math32.MaxFloat32
		dst[i] = 0
	}

	strideSet := [][]int{srcStrides, mapped}
	var offsetsArr [2]int
	process := func(indices []int, offsets []int) {
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

	if len(srcShape) <= helpers.MAX_DIMS {
		var indicesArr [helpers.MAX_DIMS]int
		process(indicesArr[:len(srcShape)], offsetsArr[:len(strideSet)])
		return
	}

	process(make([]int, len(srcShape)), offsetsArr[:len(strideSet)])
}

func makeReduceMask(rank int, axes []int) []bool {
	maskStatic := [helpers.MAX_DIMS]bool{}
	var mask []bool
	if rank > len(maskStatic) {
		mask = make([]bool, rank)
	} else {
		mask = maskStatic[:rank]
	}
	for _, axis := range axes {
		if axis >= 0 && axis < rank {
			mask[axis] = true
		}
	}
	return mask
}

func mapStridesToSource(buf []int, srcShape []int, mask []bool, dstStrides []int) []int {
	if len(srcShape) > len(buf) {
		buf = make([]int, len(srcShape))
	}
	mapped := buf[:len(srcShape)]
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

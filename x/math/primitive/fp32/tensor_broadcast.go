package fp32

import (
	"fmt"

	helpers "github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
)

// BroadcastStrides computes effective strides when broadcasting a tensor with the given shape/strides to the target shape.
func BroadcastStrides(shape []int, strides []int, target []int) ([]int, error) {
	return BroadcastStridesTo(nil, shape, strides, target)
}

// BroadcastStridesTo computes effective strides when broadcasting and writes them into buf when provided.
// If buf is nil or too small, a new slice is allocated.
func BroadcastStridesTo(buf []int, shape []int, strides []int, target []int) ([]int, error) {
	if len(target) < len(shape) {
		return nil, fmt.Errorf("fp32: cannot broadcast rank %d tensor to rank %d", len(shape), len(target))
	}

	strides = EnsureStrides(strides, shape)
	if len(target) > len(buf) {
		buf = make([]int, len(target))
	}
	result := buf[:len(target)]

	baseIdx := len(shape) - 1
	for targetIdx := len(target) - 1; targetIdx >= 0; targetIdx-- {
		if baseIdx >= 0 {
			baseDim := shape[baseIdx]
			targetDim := target[targetIdx]
			switch {
			case baseDim == targetDim:
				result[targetIdx] = strides[baseIdx]
			case baseDim == 1:
				result[targetIdx] = 0
			default:
				return nil, fmt.Errorf("fp32: cannot broadcast dimension %d (size %d) to target size %d", baseIdx, baseDim, targetDim)
			}
			baseIdx--
			continue
		}

		if target[targetIdx] != 1 {
			return nil, fmt.Errorf("fp32: cannot broadcast implicit dimension to size %d", target[targetIdx])
		}
		result[targetIdx] = 0
	}

	return result, nil
}

// ExpandTo writes a broadcasted view of src into dst.
func ExpandTo(dst, src []float32, dstShape, srcShape []int, dstStrides, srcStrides []int) error {
	sizeDst := SizeFromShape(dstShape)
	if sizeDst == 0 {
		return nil
	}

	dstStrides = EnsureStrides(dstStrides, dstShape)
	srcStrides = EnsureStrides(srcStrides, srcShape)
	var effectiveSrcBuf [helpers.MAX_DIMS]int
	effectiveSrc, err := BroadcastStridesTo(effectiveSrcBuf[:], srcShape, srcStrides, dstShape)
	if err != nil {
		return err
	}

	strideSet := [][]int{dstStrides, effectiveSrc}
	var offsetsArr [2]int
	process := func(indices []int, offsets []int) {
		for {
			dIdx := offsets[0]
			sIdx := offsets[1]
			dst[dIdx] = src[sIdx]
			if !advanceOffsets(dstShape, indices, offsets, strideSet) {
				break
			}
		}
	}

	if len(dstShape) <= helpers.MAX_DIMS {
		var indicesArr [helpers.MAX_DIMS]int
		process(indicesArr[:len(dstShape)], offsetsArr[:len(strideSet)])
		return nil
	}

	process(make([]int, len(dstShape)), offsetsArr[:len(strideSet)])

	return nil
}

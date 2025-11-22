package audio

import (
	"fmt"

	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// extractTensorChunk extracts a chunk of frames from tensor data.
func extractTensorChunk(tensorData any, dtype tensortypes.DataType, frameOffset, chunkFrames, channels int) (any, error) {
	switch data := tensorData.(type) {
	case []int16:
		chunk := make([]int16, chunkFrames*channels)
		for i := 0; i < chunkFrames; i++ {
			srcIdx := (frameOffset + i) * channels
			dstIdx := i * channels
			copy(chunk[dstIdx:dstIdx+channels], data[srcIdx:srcIdx+channels])
		}
		return chunk, nil

	case []int32:
		chunk := make([]int32, chunkFrames*channels)
		for i := 0; i < chunkFrames; i++ {
			srcIdx := (frameOffset + i) * channels
			dstIdx := i * channels
			copy(chunk[dstIdx:dstIdx+channels], data[srcIdx:srcIdx+channels])
		}
		return chunk, nil

	case []float32:
		chunk := make([]float32, chunkFrames*channels)
		for i := 0; i < chunkFrames; i++ {
			srcIdx := (frameOffset + i) * channels
			dstIdx := i * channels
			copy(chunk[dstIdx:dstIdx+channels], data[srcIdx:srcIdx+channels])
		}
		return chunk, nil

	case []uint8:
		chunk := make([]uint8, chunkFrames*channels)
		for i := 0; i < chunkFrames; i++ {
			srcIdx := (frameOffset + i) * channels
			dstIdx := i * channels
			copy(chunk[dstIdx:dstIdx+channels], data[srcIdx:srcIdx+channels])
		}
		return chunk, nil

	default:
		return nil, fmt.Errorf("unsupported tensor data type: %T", tensorData)
	}
}


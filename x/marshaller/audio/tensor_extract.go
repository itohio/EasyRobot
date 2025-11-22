package audio

import (
	"fmt"

	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// extractDeviceChannels extracts specific channels from tensor data for a device.
func extractDeviceChannels(tensorData any, dtype tensortypes.DataType, frameOffset, chunkFrames, channelOffset, deviceChannels, totalChannels int) (any, error) {
	switch data := tensorData.(type) {
	case []int16:
		chunk := make([]int16, chunkFrames*deviceChannels)
		for i := 0; i < chunkFrames; i++ {
			for ch := 0; ch < deviceChannels; ch++ {
				srcIdx := (frameOffset+i)*totalChannels + channelOffset + ch
				dstIdx := i*deviceChannels + ch
				if srcIdx < len(data) {
					chunk[dstIdx] = data[srcIdx]
				}
			}
		}
		return chunk, nil

	case []int32:
		chunk := make([]int32, chunkFrames*deviceChannels)
		for i := 0; i < chunkFrames; i++ {
			for ch := 0; ch < deviceChannels; ch++ {
				srcIdx := (frameOffset+i)*totalChannels + channelOffset + ch
				dstIdx := i*deviceChannels + ch
				if srcIdx < len(data) {
					chunk[dstIdx] = data[srcIdx]
				}
			}
		}
		return chunk, nil

	case []float32:
		chunk := make([]float32, chunkFrames*deviceChannels)
		for i := 0; i < chunkFrames; i++ {
			for ch := 0; ch < deviceChannels; ch++ {
				srcIdx := (frameOffset+i)*totalChannels + channelOffset + ch
				dstIdx := i*deviceChannels + ch
				if srcIdx < len(data) {
					chunk[dstIdx] = data[srcIdx]
				}
			}
		}
		return chunk, nil

	case []uint8:
		chunk := make([]uint8, chunkFrames*deviceChannels)
		for i := 0; i < chunkFrames; i++ {
			for ch := 0; ch < deviceChannels; ch++ {
				srcIdx := (frameOffset+i)*totalChannels + channelOffset + ch
				dstIdx := i*deviceChannels + ch
				if srcIdx < len(data) {
					chunk[dstIdx] = data[srcIdx]
				}
			}
		}
		return chunk, nil

	default:
		return nil, fmt.Errorf("unsupported tensor data type: %T", tensorData)
	}
}


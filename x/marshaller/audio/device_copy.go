package audio

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// copyDeviceChannelsToTensor copies device channel data to specific channels in combined tensor.
func copyDeviceChannelsToTensor(t types.Tensor, tensorData any, dtype tensortypes.DataType, frameOffset, channelOffset, channels, totalChannels int) error {
	tData := t.Data()
	if tData == nil {
		return fmt.Errorf("tensor data is nil")
	}

	switch dtype {
	case tensortypes.INT16:
		dst := tData.([]int16)
		src := tensorData.([]int16)
		for i, val := range src {
			frame := i / channels
			ch := i % channels
			idx := (frameOffset+frame)*totalChannels + channelOffset + ch
			if idx < len(dst) {
				dst[idx] = val
			}
		}

	case tensortypes.INT32:
		dst := tData.([]int32)
		src := tensorData.([]int32)
		for i, val := range src {
			frame := i / channels
			ch := i % channels
			idx := (frameOffset+frame)*totalChannels + channelOffset + ch
			if idx < len(dst) {
				dst[idx] = val
			}
		}

	case tensortypes.FP32:
		dst := tData.([]float32)
		src := tensorData.([]float32)
		for i, val := range src {
			frame := i / channels
			ch := i % channels
			idx := (frameOffset+frame)*totalChannels + channelOffset + ch
			if idx < len(dst) {
				dst[idx] = val
			}
		}

	case tensortypes.UINT8:
		dst := tData.([]uint8)
		src := tensorData.([]uint8)
		for i, val := range src {
			frame := i / channels
			ch := i % channels
			idx := (frameOffset+frame)*totalChannels + channelOffset + ch
			if idx < len(dst) {
				dst[idx] = val
			}
		}

	default:
		return fmt.Errorf("unsupported dtype: %v", dtype)
	}

	return nil
}


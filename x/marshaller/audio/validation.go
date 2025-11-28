package audio

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// validateTensorDtype checks if tensor dtype matches device format.
// Returns error if mismatch.
func validateTensorDtype(t types.Tensor, deviceDtype tensortypes.DataType) error {
	tensorDtype := t.DataType()
	if tensorDtype != deviceDtype {
		return fmt.Errorf("tensor dtype mismatch: tensor has %v, device expects %v", tensorDtype, deviceDtype)
	}
	return nil
}

// validateTensorChannels checks if total number of output device channels equals tensor channels.
// Returns error if mismatch.
func validateTensorChannels(t types.Tensor, totalDeviceChannels int) error {
	shape := t.Shape()
	if len(shape) != 2 {
		return fmt.Errorf("tensor must be 2D [frames, channels] for audio")
	}
	tensorChannels := shape[1]
	if tensorChannels != totalDeviceChannels {
		return fmt.Errorf("channel count mismatch: tensor has %d channels, devices total %d", tensorChannels, totalDeviceChannels)
	}
	return nil
}

// validateFrameTensorDtype checks if frame tensor dtype matches device format.
// Returns error if mismatch.
func validateFrameTensorDtype(frame types.Frame, deviceDtype tensortypes.DataType) error {
	if len(frame.Tensors) == 0 {
		return fmt.Errorf("frame has no tensors")
	}
	
	// Check all tensors in frame
	for i, t := range frame.Tensors {
		if err := validateTensorDtype(t, deviceDtype); err != nil {
			return fmt.Errorf("tensor %d in frame: %w", i, err)
		}
	}
	return nil
}

// validateFrameTensorChannels checks if total number of output device channels equals frame tensor channels.
// Returns error if mismatch.
func validateFrameTensorChannels(frame types.Frame, totalDeviceChannels int) error {
	if len(frame.Tensors) == 0 {
		return fmt.Errorf("frame has no tensors")
	}
	
	// Sum channels from all tensors in frame
	totalTensorChannels := 0
	for i, t := range frame.Tensors {
		shape := t.Shape()
		if len(shape) != 2 {
			return fmt.Errorf("tensor %d in frame must be 2D [frames, channels] for audio", i)
		}
		totalTensorChannels += shape[1]
	}
	
	if totalTensorChannels != totalDeviceChannels {
		return fmt.Errorf("channel count mismatch: frame tensors total %d channels, devices total %d", totalTensorChannels, totalDeviceChannels)
	}
	return nil
}


package audio

import (
	"fmt"
	"time"

	"github.com/itohio/EasyRobot/x/marshaller/types"
)

// copyToTensor copies data into tensor.
func copyToTensor(t types.Tensor, data any) error {
	tData := t.Data()
	if tData == nil {
		return fmt.Errorf("tensor data is nil")
	}

	switch dst := tData.(type) {
	case []int16:
		src, ok := data.([]int16)
		if !ok {
			return fmt.Errorf("type mismatch: expected []int16, got %T", data)
		}
		copy(dst, src)

	case []int32:
		src, ok := data.([]int32)
		if !ok {
			return fmt.Errorf("type mismatch: expected []int32, got %T", data)
		}
		copy(dst, src)

	case []float32:
		src, ok := data.([]float32)
		if !ok {
			return fmt.Errorf("type mismatch: expected []float32, got %T", data)
		}
		copy(dst, src)

	case []uint8:
		src, ok := data.([]uint8)
		if !ok {
			return fmt.Errorf("type mismatch: expected []uint8, got %T", data)
		}
		copy(dst, src)

	default:
		return fmt.Errorf("unsupported tensor data type: %T", tData)
	}

	return nil
}

// getDeviceFormat returns the format string from device.
func getDeviceFormat(dev deviceInterface) string {
	if md, ok := dev.(*malgoDevice); ok {
		return md.format
	}
	return "S16" // Default
}

// getDeviceChannels returns the channel count from device.
func getDeviceChannels(dev deviceInterface) int {
	if md, ok := dev.(*malgoDevice); ok {
		return md.channels
	}
	return 2 // Default
}

// getDeviceSampleRate returns the sample rate from device.
func getDeviceSampleRate(dev deviceInterface) int {
	if md, ok := dev.(*malgoDevice); ok {
		return md.sampleRate
	}
	return 48000 // Default
}

// deviceInterface defines the interface for audio devices.
type deviceInterface interface {
	start() error
	stop() error
	health() map[string]int64
	readFrame(timeout time.Duration) ([]byte, error)
	writeFrame(buf []byte) error
}


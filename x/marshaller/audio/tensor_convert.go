package audio

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/marshaller/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
	tensortypes "github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/chewxy/math32"
)

// convertTensorDtype converts a tensor to target dtype with proper scaling.
// Handles: 24bit, float32 -> 16bit, 24bit with proper scaling.
// Returns a new tensor with converted dtype.
func convertTensorDtype(t types.Tensor, targetDtype tensortypes.DataType) (types.Tensor, error) {
	if t.Empty() {
		return nil, fmt.Errorf("empty tensor")
	}
	
	sourceDtype := t.DataType()
	if sourceDtype == targetDtype {
		return t, nil // No conversion needed
	}
	
	shape := t.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("tensor must be 2D [frames, channels]")
	}
	
	frames := shape[0]
	channels := shape[1]
	
	// Create output tensor
	outputShape := tensor.NewShape(frames, channels)
	outputTensor := tensor.New(targetDtype, outputShape)
	if outputTensor.Empty() {
		return nil, fmt.Errorf("failed to allocate output tensor")
	}
	
	sourceData := t.Data()
	outputData := outputTensor.Data()
	
	// Type dispatch for conversion
	var err error
	switch sourceDtype {
	case tensortypes.FP32:
		_, err = convertFromFloat32(sourceData.([]float32), outputData, targetDtype, frames*channels)
	case tensortypes.INT32:
		_, err = convertFromInt32(sourceData.([]int32), outputData, targetDtype, frames*channels)
	case tensortypes.INT16:
		_, err = convertFromInt16(sourceData.([]int16), outputData, targetDtype, frames*channels)
	case tensortypes.UINT8:
		_, err = convertFromUint8(sourceData.([]uint8), outputData, targetDtype, frames*channels)
	default:
		return nil, fmt.Errorf("unsupported source dtype: %v", sourceDtype)
	}
	
	if err != nil {
		var tensorInterface types.Tensor = outputTensor
		if releaser, ok := tensorInterface.(types.Releaser); ok {
			releaser.Release()
		}
		return nil, err
	}
	
	return outputTensor, nil
}

// convertFromFloat32 converts float32 samples to target dtype with proper scaling.
func convertFromFloat32(src []float32, dst any, targetDtype tensortypes.DataType, samples int) (types.Tensor, error) {
	switch targetDtype {
	case tensortypes.INT16:
		dstData := dst.([]int16)
		for i := range samples {
			// Clamp to [-1, 1] and scale to int16 range
			val := math32.Max(-1.0, math32.Min(1.0, src[i]))
			dstData[i] = int16(val * 32767.0)
		}
	case tensortypes.INT32:
		dstData := dst.([]int32)
		for i := range samples {
			// Clamp to [-1, 1] and scale to int32 range
			val := math32.Max(-1.0, math32.Min(1.0, src[i]))
			dstData[i] = int32(val * 2147483647.0)
		}
	case tensortypes.FP32:
		dstData := dst.([]float32)
		copy(dstData, src)
	default:
		return nil, fmt.Errorf("unsupported target dtype for float32: %v", targetDtype)
	}
	return nil, nil // Output tensor is already created
}

// convertFromInt32 converts int32 samples to target dtype with proper scaling.
func convertFromInt32(src []int32, dst any, targetDtype tensortypes.DataType, samples int) (types.Tensor, error) {
	switch targetDtype {
	case tensortypes.INT16:
		dstData := dst.([]int16)
		for i := range samples {
			// Scale from int32 to int16 (divide by 65536)
			dstData[i] = int16(src[i] >> 16)
		}
	case tensortypes.INT32:
		dstData := dst.([]int32)
		copy(dstData, src)
	case tensortypes.FP32:
		dstData := dst.([]float32)
		for i := range samples {
			// Normalize to [-1, 1]
			dstData[i] = float32(src[i]) / 2147483647.0
		}
	default:
		return nil, fmt.Errorf("unsupported target dtype for int32: %v", targetDtype)
	}
	return nil, nil
}

// convertFromInt16 converts int16 samples to target dtype with proper scaling.
func convertFromInt16(src []int16, dst any, targetDtype tensortypes.DataType, samples int) (types.Tensor, error) {
	switch targetDtype {
	case tensortypes.INT16:
		dstData := dst.([]int16)
		copy(dstData, src)
	case tensortypes.INT32:
		dstData := dst.([]int32)
		for i := range samples {
			// Scale from int16 to int32 (multiply by 65536)
			dstData[i] = int32(src[i]) << 16
		}
	case tensortypes.FP32:
		dstData := dst.([]float32)
		for i := range samples {
			// Normalize to [-1, 1]
			dstData[i] = float32(src[i]) / 32767.0
		}
	default:
		return nil, fmt.Errorf("unsupported target dtype for int16: %v", targetDtype)
	}
	return nil, nil
}

// convertFromUint8 converts uint8 samples to target dtype with proper scaling.
func convertFromUint8(src []uint8, dst any, targetDtype tensortypes.DataType, samples int) (types.Tensor, error) {
	switch targetDtype {
	case tensortypes.UINT8:
		dstData := dst.([]uint8)
		copy(dstData, src)
	case tensortypes.INT16:
		dstData := dst.([]int16)
		for i := range samples {
			// Convert from unsigned to signed and scale
			dstData[i] = int16(int(src[i]) - 128) << 8
		}
	case tensortypes.INT32:
		dstData := dst.([]int32)
		for i := range samples {
			// Convert from unsigned to signed and scale
			dstData[i] = int32(int(src[i])-128) << 24
		}
	case tensortypes.FP32:
		dstData := dst.([]float32)
		for i := range samples {
			// Normalize to [-1, 1]
			dstData[i] = (float32(src[i]) - 128.0) / 128.0
		}
	default:
		return nil, fmt.Errorf("unsupported target dtype for uint8: %v", targetDtype)
	}
	return nil, nil
}

// extractChannelsAndPad extracts first N channels from tensor and zero-pads if needed.
// Returns a new tensor with exactly outputChannels channels.
// Reuses output tensor if provided (for efficiency).
func extractChannelsAndPad(t types.Tensor, outputChannels int, outputTensor types.Tensor) (types.Tensor, error) {
	if t.Empty() {
		return nil, fmt.Errorf("empty tensor")
	}
	
	shape := t.Shape()
	if len(shape) != 2 {
		return nil, fmt.Errorf("tensor must be 2D [frames, channels]")
	}
	
	frames := shape[0]
	inputChannels := shape[1]
	
	if outputChannels <= 0 {
		outputChannels = inputChannels
	}
	
	// Reuse output tensor if provided and compatible
	var output types.Tensor
	if outputTensor != nil && !outputTensor.Empty() {
		outputShape := outputTensor.Shape()
		if len(outputShape) == 2 && outputShape[0] == frames && outputShape[1] == outputChannels {
			output = outputTensor
		}
	}
	
	// Allocate new tensor if needed
	if output == nil {
		outputShape := tensor.NewShape(frames, outputChannels)
		output = tensor.New(t.DataType(), outputShape)
		if output.Empty() {
			return nil, fmt.Errorf("failed to allocate output tensor")
		}
	}
	
	sourceData := t.Data()
	outputData := output.Data()
	
	// Extract channels and zero-pad
	switch data := sourceData.(type) {
	case []int16:
		dst := outputData.([]int16)
		for i := range frames {
			for ch := 0; ch < outputChannels; ch++ {
				idx := i*outputChannels + ch
				if ch < inputChannels {
					srcIdx := i*inputChannels + ch
					dst[idx] = data[srcIdx]
				} else {
					dst[idx] = 0 // Zero-pad
				}
			}
		}
	case []int32:
		dst := outputData.([]int32)
		for i := range frames {
			for ch := 0; ch < outputChannels; ch++ {
				idx := i*outputChannels + ch
				if ch < inputChannels {
					srcIdx := i*inputChannels + ch
					dst[idx] = data[srcIdx]
				} else {
					dst[idx] = 0 // Zero-pad
				}
			}
		}
	case []float32:
		dst := outputData.([]float32)
		for i := range frames {
			for ch := 0; ch < outputChannels; ch++ {
				idx := i*outputChannels + ch
				if ch < inputChannels {
					srcIdx := i*inputChannels + ch
					dst[idx] = data[srcIdx]
				} else {
					dst[idx] = 0.0 // Zero-pad
				}
			}
		}
	case []uint8:
		dst := outputData.([]uint8)
		for i := range frames {
			for ch := 0; ch < outputChannels; ch++ {
				idx := i*outputChannels + ch
				if ch < inputChannels {
					srcIdx := i*inputChannels + ch
					dst[idx] = data[srcIdx]
				} else {
					dst[idx] = 128 // Zero-pad (midpoint for uint8)
				}
			}
		}
	default:
		return nil, fmt.Errorf("unsupported tensor data type: %T", sourceData)
	}
	
	return output, nil
}

// matchSampleRate performs simple sample rate conversion (linear interpolation).
// For now, only handles exact matches and simple integer ratios.
// Returns a new tensor with converted sample rate.
func matchSampleRate(t types.Tensor, sourceRate, targetRate int) (types.Tensor, error) {
	if sourceRate == targetRate {
		return t, nil // No conversion needed
	}
	
	// For now, only handle exact integer ratios (upsample/downsample)
	// TODO: Implement proper resampling with anti-aliasing
	if sourceRate > targetRate && sourceRate%targetRate == 0 {
		// Downsample by integer factor
		factor := sourceRate / targetRate
		return downsampleTensor(t, factor)
	} else if targetRate > sourceRate && targetRate%sourceRate == 0 {
		// Upsample by integer factor
		factor := targetRate / sourceRate
		return upsampleTensor(t, factor)
	}
	
	// For non-integer ratios, use simple linear interpolation
	return resampleTensorLinear(t, sourceRate, targetRate)
}

// downsampleTensor downsamples by integer factor (simple decimation).
func downsampleTensor(t types.Tensor, factor int) (types.Tensor, error) {
	shape := t.Shape()
	frames := shape[0]
	channels := shape[1]
	
	outputFrames := frames / factor
	if outputFrames == 0 {
		outputFrames = 1
	}
	
	outputShape := tensor.NewShape(outputFrames, channels)
	output := tensor.New(t.DataType(), outputShape)
	if output.Empty() {
		return nil, fmt.Errorf("failed to allocate output tensor")
	}
	
	sourceData := t.Data()
	outputData := output.Data()
	
	// Simple decimation - take every Nth sample
	switch data := sourceData.(type) {
	case []int16:
		dst := outputData.([]int16)
		for i := range outputFrames {
			for ch := 0; ch < channels; ch++ {
				srcIdx := (i * factor) * channels + ch
				if srcIdx < len(data) {
					dst[i*channels+ch] = data[srcIdx]
				}
			}
		}
	case []int32:
		dst := outputData.([]int32)
		for i := range outputFrames {
			for ch := 0; ch < channels; ch++ {
				srcIdx := (i * factor) * channels + ch
				if srcIdx < len(data) {
					dst[i*channels+ch] = data[srcIdx]
				}
			}
		}
	case []float32:
		dst := outputData.([]float32)
		for i := range outputFrames {
			for ch := 0; ch < channels; ch++ {
				srcIdx := (i * factor) * channels + ch
				if srcIdx < len(data) {
					dst[i*channels+ch] = data[srcIdx]
				}
			}
		}
	case []uint8:
		dst := outputData.([]uint8)
		for i := range outputFrames {
			for ch := 0; ch < channels; ch++ {
				srcIdx := (i * factor) * channels + ch
				if srcIdx < len(data) {
					dst[i*channels+ch] = data[srcIdx]
				}
			}
		}
	default:
		return nil, fmt.Errorf("unsupported tensor data type: %T", sourceData)
	}
	
	return output, nil
}

// upsampleTensor upsamples by integer factor (zero-order hold).
func upsampleTensor(t types.Tensor, factor int) (types.Tensor, error) {
	shape := t.Shape()
	frames := shape[0]
	channels := shape[1]
	
	outputFrames := frames * factor
	outputShape := tensor.NewShape(outputFrames, channels)
	output := tensor.New(t.DataType(), outputShape)
	if output.Empty() {
		return nil, fmt.Errorf("failed to allocate output tensor")
	}
	
	sourceData := t.Data()
	outputData := output.Data()
	
	// Zero-order hold - repeat each sample N times
	switch data := sourceData.(type) {
	case []int16:
		dst := outputData.([]int16)
		for i := range frames {
			for ch := 0; ch < channels; ch++ {
				val := data[i*channels+ch]
				for j := 0; j < factor; j++ {
					dst[(i*factor+j)*channels+ch] = val
				}
			}
		}
	case []int32:
		dst := outputData.([]int32)
		for i := range frames {
			for ch := 0; ch < channels; ch++ {
				val := data[i*channels+ch]
				for j := 0; j < factor; j++ {
					dst[(i*factor+j)*channels+ch] = val
				}
			}
		}
	case []float32:
		dst := outputData.([]float32)
		for i := range frames {
			for ch := 0; ch < channels; ch++ {
				val := data[i*channels+ch]
				for j := 0; j < factor; j++ {
					dst[(i*factor+j)*channels+ch] = val
				}
			}
		}
	case []uint8:
		dst := outputData.([]uint8)
		for i := range frames {
			for ch := 0; ch < channels; ch++ {
				val := data[i*channels+ch]
				for j := 0; j < factor; j++ {
					dst[(i*factor+j)*channels+ch] = val
				}
			}
		}
	default:
		return nil, fmt.Errorf("unsupported tensor data type: %T", sourceData)
	}
	
	return output, nil
}

// resampleTensorLinear performs linear interpolation resampling.
func resampleTensorLinear(t types.Tensor, sourceRate, targetRate int) (types.Tensor, error) {
	shape := t.Shape()
	sourceFrames := shape[0]
	channels := shape[1]
	
	// Calculate output frames
	outputFrames := int(float64(sourceFrames) * float64(targetRate) / float64(sourceRate))
	if outputFrames == 0 {
		outputFrames = 1
	}
	
	outputShape := tensor.NewShape(outputFrames, channels)
	output := tensor.New(t.DataType(), outputShape)
	if output.Empty() {
		return nil, fmt.Errorf("failed to allocate output tensor")
	}
	
	sourceData := t.Data()
	outputData := output.Data()
	
	ratio := float64(sourceRate) / float64(targetRate)
	
	// Linear interpolation
	switch data := sourceData.(type) {
	case []float32:
		dst := outputData.([]float32)
		for i := range outputFrames {
			pos := float64(i) * ratio
			idx0 := int(pos)
			idx1 := idx0 + 1
			frac := pos - float64(idx0)
			
			if idx1 >= sourceFrames {
				idx1 = sourceFrames - 1
			}
			
			for ch := 0; ch < channels; ch++ {
				val0 := data[idx0*channels+ch]
				val1 := data[idx1*channels+ch]
				dst[i*channels+ch] = float32(float64(val0)*(1.0-frac) + float64(val1)*frac)
			}
		}
	default:
		// For integer types, convert to float32, interpolate, then convert back
		// This is a simplified approach - proper implementation would handle each type
		return nil, fmt.Errorf("linear resampling currently only supports float32, got %T", sourceData)
	}
	
	return output, nil
}


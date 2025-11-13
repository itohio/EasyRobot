package eager_tensor

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/primitive/fp32"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// BatchNormForward performs batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
// Normalizes across batch dimension (axis 0). Assumes input shape is [batch, ...].
// gamma and beta are learnable parameters with shape matching the non-batch dimensions.
// If gamma/beta are nil, uses gamma=1, beta=0.
// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
func (t Tensor) BatchNormForward(dst types.Tensor, gamma, beta types.Tensor, eps float64) types.Tensor {
	if t.shape == nil {
		return t
	}
	if IsNil(dst) {
		dst = NewAs(t)
	}
	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.BatchNormForward: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	shape := t.Shape().ToSlice()
	rank := t.Rank()
	if rank < 2 {
		// Copy input if shape is too small for batch norm
		t.Copy(dst)
		return dst
	}

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](dst)
		var gammaData []float32
		if gamma != nil {
			gammaData = types.GetTensorData[[]float32](gamma)
		}
		var betaData []float32
		if beta != nil {
			betaData = types.GetTensorData[[]float32](beta)
		}

		fp32.BatchNormForward(dstData, tData, gammaData, betaData, shape, float32(eps))
	default:
		panic(fmt.Sprintf("tensor.BatchNormForward: unsupported data type: %T", tData))
	}
	return dst
}

// BatchNormGrad computes gradients for batch normalization.
// gradInputDst: destination for input gradient [batch, ...] (can be nil)
// gradGammaDst: destination for gamma gradient [...] (can be nil)
// gradBetaDst: destination for beta gradient [...] (can be nil)
// gradOutput: gradient w.r.t. output [batch, ...]
// input: original input [batch, ...]
// gamma: scale parameter [...]
// eps: epsilon for numerical stability
// Returns: (gradInput, gradGamma, gradBeta) - new tensors if dst was nil, otherwise returns dst tensors
func (t Tensor) BatchNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	if t.shape == nil {
		return gradInputDst, gradGammaDst, gradBetaDst
	}
	if IsNil(gradOutput) || IsNil(input) {
		return gradInputDst, gradGammaDst, gradBetaDst
	}
	if !t.Shape().Equal(gradOutput.Shape()) || !t.Shape().Equal(input.Shape()) {
		panic(fmt.Sprintf("tensor.BatchNormGrad: shape mismatch: t=%v, gradOutput=%v, input=%v", t.shape, gradOutput.Shape(), input.Shape()))
	}

	shape := t.Shape().ToSlice()
	rank := t.Rank()
	if rank < 2 {
		// Return zero gradients for invalid shapes
		if IsNil(gradInputDst) {
			gradInputDst = NewAs(t)
		}
		return gradInputDst, gradGammaDst, gradBetaDst
	}

	featureSize := 1
	for i := 1; i < rank; i++ {
		featureSize *= shape[i]
	}

	if IsNil(gradInputDst) {
		gradInputDst = New(t.DataType(), t.Shape())
	} else if !t.Shape().Equal(gradInputDst.Shape()) {
		panic(fmt.Sprintf("tensor.BatchNormGrad: gradInputDst shape mismatch: expected %v, got %v", t.shape, gradInputDst.Shape()))
	}

	if gamma != nil {
		gammaShape := types.NewShape(featureSize)
		if IsNil(gradGammaDst) {
			gradGammaDst = New(t.DataType(), gammaShape)
		} else if !gammaShape.Equal(gradGammaDst.Shape()) {
			panic(fmt.Sprintf("tensor.BatchNormGrad: gradGammaDst shape mismatch: expected %v, got %v", gammaShape, gradGammaDst.Shape()))
		}
		if IsNil(gradBetaDst) {
			gradBetaDst = New(t.DataType(), gammaShape)
		} else if !gammaShape.Equal(gradBetaDst.Shape()) {
			panic(fmt.Sprintf("tensor.BatchNormGrad: gradBetaDst shape mismatch: expected %v, got %v", gammaShape, gradBetaDst.Shape()))
		}
	}

	switch gradOutputData := gradOutput.Data().(type) {
	case []float32:
		inputData := types.GetTensorData[[]float32](input)
		gradInputData := types.GetTensorData[[]float32](gradInputDst)
		var gammaData []float32
		if gamma != nil {
			gammaData = types.GetTensorData[[]float32](gamma)
		}
		var gradGammaData []float32
		var gradBetaData []float32
		if gradGammaDst != nil {
			gradGammaData = types.GetTensorData[[]float32](gradGammaDst)
			gradBetaData = types.GetTensorData[[]float32](gradBetaDst)
		}

		fp32.BatchNormGrad(gradInputData, gradGammaData, gradBetaData, gradOutputData, inputData, gammaData, shape, float32(eps))
	default:
		panic(fmt.Sprintf("tensor.BatchNormGrad: unsupported data type: %T", gradOutputData))
	}
	return gradInputDst, gradGammaDst, gradBetaDst
}

// LayerNormForward performs layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
// Normalizes across the last dimension (feature dimension). Assumes input is contiguous.
// gamma and beta are learnable parameters with shape matching the last dimension.
// If gamma/beta are nil, uses gamma=1, beta=0.
// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
func (t Tensor) LayerNormForward(dst types.Tensor, gamma, beta types.Tensor, eps float64) types.Tensor {
	if t.shape == nil {
		return t
	}
	if IsNil(dst) {
		dst = NewAs(t)
	}
	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.LayerNormForward: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	shape := t.Shape().ToSlice()

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](dst)
		var gammaData []float32
		if gamma != nil {
			gammaData = types.GetTensorData[[]float32](gamma)
		}
		var betaData []float32
		if beta != nil {
			betaData = types.GetTensorData[[]float32](beta)
		}

		fp32.LayerNormForward(dstData, tData, gammaData, betaData, shape, float32(eps))
	default:
		panic(fmt.Sprintf("tensor.LayerNormForward: unsupported data type: %T", tData))
	}
	return dst
}

// LayerNormGrad computes gradients for layer normalization.
// gradInputDst: destination for input gradient [...] (can be nil)
// gradGammaDst: destination for gamma gradient [last_dim] (can be nil)
// gradBetaDst: destination for beta gradient [last_dim] (can be nil)
// gradOutput: gradient w.r.t. output [...]
// input: original input [...]
// gamma: scale parameter [last_dim]
// eps: epsilon for numerical stability
// Returns: (gradInput, gradGamma, gradBeta) - new tensors if dst was nil, otherwise returns dst tensors
func (t Tensor) LayerNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	if t.shape == nil {
		return gradInputDst, gradGammaDst, gradBetaDst
	}
	if IsNil(gradOutput) || IsNil(input) {
		return gradInputDst, gradGammaDst, gradBetaDst
	}
	if !t.Shape().Equal(gradOutput.Shape()) || !t.Shape().Equal(input.Shape()) {
		panic(fmt.Sprintf("tensor.LayerNormGrad: shape mismatch: t=%v, gradOutput=%v, input=%v", t.shape, gradOutput.Shape(), input.Shape()))
	}

	shape := t.Shape().ToSlice()
	lastDim := shape[len(shape)-1]

	if IsNil(gradInputDst) {
		gradInputDst = New(t.DataType(), t.Shape())
	} else if !t.Shape().Equal(gradInputDst.Shape()) {
		panic(fmt.Sprintf("tensor.LayerNormGrad: gradInputDst shape mismatch: expected %v, got %v", t.shape, gradInputDst.Shape()))
	}

	if gamma != nil {
		gammaShape := types.NewShape(lastDim)
		if IsNil(gradGammaDst) {
			gradGammaDst = New(t.DataType(), gammaShape)
		} else if !gammaShape.Equal(gradGammaDst.Shape()) {
			panic(fmt.Sprintf("tensor.LayerNormGrad: gradGammaDst shape mismatch: expected %v, got %v", gammaShape, gradGammaDst.Shape()))
		}
		if IsNil(gradBetaDst) {
			gradBetaDst = New(t.DataType(), gammaShape)
		} else if !gammaShape.Equal(gradBetaDst.Shape()) {
			panic(fmt.Sprintf("tensor.LayerNormGrad: gradBetaDst shape mismatch: expected %v, got %v", gammaShape, gradBetaDst.Shape()))
		}
	}

	switch gradOutputData := gradOutput.Data().(type) {
	case []float32:
		inputData := types.GetTensorData[[]float32](input)
		gradInputData := types.GetTensorData[[]float32](gradInputDst)
		var gammaData []float32
		if gamma != nil {
			gammaData = types.GetTensorData[[]float32](gamma)
		}
		var gradGammaData []float32
		var gradBetaData []float32
		if gradGammaDst != nil {
			gradGammaData = types.GetTensorData[[]float32](gradGammaDst)
			gradBetaData = types.GetTensorData[[]float32](gradBetaDst)
		}

		fp32.LayerNormGrad(gradInputData, gradGammaData, gradBetaData, gradOutputData, inputData, gammaData, shape, float32(eps))
	default:
		panic(fmt.Sprintf("tensor.LayerNormGrad: unsupported data type: %T", gradOutputData))
	}
	return gradInputDst, gradGammaDst, gradBetaDst
}

// RMSNormForward performs RMS normalization: x / sqrt(mean(x^2) + eps) * gamma
// Simpler than layer norm - only scales, no centering. Often used in transformers.
// gamma is a learnable parameter with shape matching the last dimension.
// If gamma is nil, uses gamma=1.
// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
func (t Tensor) RMSNormForward(dst types.Tensor, gamma types.Tensor, eps float64) types.Tensor {
	if t.shape == nil {
		return t
	}
	if IsNil(dst) {
		dst = NewAs(t)
	}
	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.RMSNormForward: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	shape := t.Shape().ToSlice()

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](dst)
		var gammaData []float32
		if gamma != nil {
			gammaData = types.GetTensorData[[]float32](gamma)
		}

		fp32.RMSNormForward(dstData, tData, gammaData, shape, float32(eps))
	default:
		panic(fmt.Sprintf("tensor.RMSNormForward: unsupported data type: %T", tData))
	}
	return dst
}

// RMSNormGrad computes gradients for RMS normalization.
// gradInputDst: destination for input gradient [...] (can be nil)
// gradGammaDst: destination for gamma gradient [last_dim] (can be nil)
// gradOutput: gradient w.r.t. output [...]
// input: original input [...]
// gamma: scale parameter [last_dim]
// eps: epsilon for numerical stability
// Returns: (gradInput, gradGamma) - new tensors if dst was nil, otherwise returns dst tensors
func (t Tensor) RMSNormGrad(gradInputDst, gradGammaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor) {
	if t.shape == nil {
		return gradInputDst, gradGammaDst
	}
	if IsNil(gradOutput) || IsNil(input) {
		return gradInputDst, gradGammaDst
	}
	if !t.Shape().Equal(gradOutput.Shape()) || !t.Shape().Equal(input.Shape()) {
		panic(fmt.Sprintf("tensor.RMSNormGrad: shape mismatch: t=%v, gradOutput=%v, input=%v", t.shape, gradOutput.Shape(), input.Shape()))
	}

	shape := t.Shape().ToSlice()
	lastDim := shape[len(shape)-1]

	if IsNil(gradInputDst) {
		gradInputDst = New(t.DataType(), t.Shape())
	} else if !t.Shape().Equal(gradInputDst.Shape()) {
		panic(fmt.Sprintf("tensor.RMSNormGrad: gradInputDst shape mismatch: expected %v, got %v", t.shape, gradInputDst.Shape()))
	}

	if gamma != nil {
		gammaShape := types.NewShape(lastDim)
		if IsNil(gradGammaDst) {
			gradGammaDst = New(t.DataType(), gammaShape)
		} else if !gammaShape.Equal(gradGammaDst.Shape()) {
			panic(fmt.Sprintf("tensor.RMSNormGrad: gradGammaDst shape mismatch: expected %v, got %v", gammaShape, gradGammaDst.Shape()))
		}
	}

	switch gradOutputData := gradOutput.Data().(type) {
	case []float32:
		inputData := types.GetTensorData[[]float32](input)
		gradInputData := types.GetTensorData[[]float32](gradInputDst)
		var gammaData []float32
		if gamma != nil {
			gammaData = types.GetTensorData[[]float32](gamma)
		}
		var gradGammaData []float32
		if gradGammaDst != nil {
			gradGammaData = types.GetTensorData[[]float32](gradGammaDst)
		}

		fp32.RMSNormGrad(gradInputData, gradGammaData, gradOutputData, inputData, gammaData, shape, float32(eps))
	default:
		panic(fmt.Sprintf("tensor.RMSNormGrad: unsupported data type: %T", gradOutputData))
	}
	return gradInputDst, gradGammaDst
}

// InstanceNorm2D performs instance normalization for 2D feature maps.
// Normalizes across spatial dimensions (H, W) for each instance and channel.
// Input shape: [batch, channels, height, width]
// gamma/beta shape: [channels] (one per channel)
// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
func (t Tensor) InstanceNorm2D(dst types.Tensor, gamma, beta types.Tensor, eps float64) types.Tensor {
	if t.shape == nil {
		return t
	}
	if IsNil(dst) {
		dst = NewAs(t)
	}
	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.InstanceNorm2D: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	shape := t.Shape().ToSlice()
	if len(shape) != 4 {
		panic(fmt.Sprintf("tensor.InstanceNorm2D: expected 4D tensor, got shape %v", shape))
	}

	batchSize, channels, height, width := shape[0], shape[1], shape[2], shape[3]

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](dst)
		var gammaData []float32
		if gamma != nil {
			gammaData = types.GetTensorData[[]float32](gamma)
		}
		var betaData []float32
		if beta != nil {
			betaData = types.GetTensorData[[]float32](beta)
		}

		fp32.InstanceNorm2D(dstData, tData, gammaData, betaData, batchSize, channels, height, width, float32(eps))
	default:
		panic(fmt.Sprintf("tensor.InstanceNorm2D: unsupported data type: %T", tData))
	}
	return dst
}

// InstanceNorm2DGrad computes gradients for 2D instance normalization.
// gradInputDst: destination for input gradient [batch, channels, height, width] (can be nil)
// gradGammaDst: destination for gamma gradient [channels] (can be nil)
// gradBetaDst: destination for beta gradient [channels] (can be nil)
// gradOutput: gradient w.r.t. output [batch, channels, height, width]
// input: original input [batch, channels, height, width]
// gamma: scale parameter [channels]
// eps: epsilon for numerical stability
// Returns: (gradInput, gradGamma, gradBeta) - new tensors if dst was nil, otherwise returns dst tensors
func (t Tensor) InstanceNorm2DGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	if t.shape == nil {
		return gradInputDst, gradGammaDst, gradBetaDst
	}
	if IsNil(gradOutput) || IsNil(input) {
		return gradInputDst, gradGammaDst, gradBetaDst
	}
	if !t.Shape().Equal(gradOutput.Shape()) || !t.Shape().Equal(input.Shape()) {
		panic(fmt.Sprintf("tensor.InstanceNorm2DGrad: shape mismatch: t=%v, gradOutput=%v, input=%v", t.shape, gradOutput.Shape(), input.Shape()))
	}

	shape := t.Shape().ToSlice()
	if len(shape) != 4 {
		panic(fmt.Sprintf("tensor.InstanceNorm2DGrad: expected 4D tensor, got shape %v", shape))
	}

	batchSize, channels, height, width := shape[0], shape[1], shape[2], shape[3]

	if IsNil(gradInputDst) {
		gradInputDst = New(t.DataType(), t.Shape())
	} else if !t.Shape().Equal(gradInputDst.Shape()) {
		panic(fmt.Sprintf("tensor.InstanceNorm2DGrad: gradInputDst shape mismatch: expected %v, got %v", t.shape, gradInputDst.Shape()))
	}

	if gamma != nil {
		channelsShape := types.NewShape(channels)
		if IsNil(gradGammaDst) {
			gradGammaDst = New(t.DataType(), channelsShape)
		} else if !channelsShape.Equal(gradGammaDst.Shape()) {
			panic(fmt.Sprintf("tensor.InstanceNorm2DGrad: gradGammaDst shape mismatch: expected %v, got %v", channelsShape, gradGammaDst.Shape()))
		}
		if IsNil(gradBetaDst) {
			gradBetaDst = New(t.DataType(), channelsShape)
		} else if !channelsShape.Equal(gradBetaDst.Shape()) {
			panic(fmt.Sprintf("tensor.InstanceNorm2DGrad: gradBetaDst shape mismatch: expected %v, got %v", channelsShape, gradBetaDst.Shape()))
		}
	}

	switch gradOutputData := gradOutput.Data().(type) {
	case []float32:
		inputData := types.GetTensorData[[]float32](input)
		gradInputData := types.GetTensorData[[]float32](gradInputDst)
		var gammaData []float32
		if gamma != nil {
			gammaData = types.GetTensorData[[]float32](gamma)
		}
		var gradGammaData []float32
		var gradBetaData []float32
		if gradGammaDst != nil {
			gradGammaData = types.GetTensorData[[]float32](gradGammaDst)
			gradBetaData = types.GetTensorData[[]float32](gradBetaDst)
		}

		fp32.InstanceNorm2DGrad(gradInputData, gradGammaData, gradBetaData, gradOutputData, inputData, gammaData, batchSize, channels, height, width, float32(eps))
	default:
		panic(fmt.Sprintf("tensor.InstanceNorm2DGrad: unsupported data type: %T", gradOutputData))
	}
	return gradInputDst, gradGammaDst, gradBetaDst
}

// GroupNormForward performs group normalization.
// Divides channels into groups and normalizes within each group.
// Input shape: [batch, channels, ...] where channels must be divisible by numGroups.
// gamma/beta shape: [channels] (one per channel)
// If dst is nil, creates a new tensor. If dst is provided, writes result to dst and returns dst.
func (t Tensor) GroupNormForward(dst types.Tensor, gamma, beta types.Tensor, numGroups int, eps float64) types.Tensor {
	if t.shape == nil {
		return t
	}
	if IsNil(dst) {
		dst = NewAs(t)
	}
	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.GroupNormForward: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	shape := t.Shape().ToSlice()
	if len(shape) < 2 {
		panic(fmt.Sprintf("tensor.GroupNormForward: expected at least 2D tensor, got shape %v", shape))
	}

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](dst)
		var gammaData []float32
		if gamma != nil {
			gammaData = types.GetTensorData[[]float32](gamma)
		}
		var betaData []float32
		if beta != nil {
			betaData = types.GetTensorData[[]float32](beta)
		}

		fp32.GroupNormForward(dstData, tData, gammaData, betaData, shape, numGroups, float32(eps))
	default:
		panic(fmt.Sprintf("tensor.GroupNormForward: unsupported data type: %T", tData))
	}
	return dst
}

// GroupNormGrad computes gradients for group normalization.
// gradInputDst: destination for input gradient [batch, channels, ...] (can be nil)
// gradGammaDst: destination for gamma gradient [channels] (can be nil)
// gradBetaDst: destination for beta gradient [channels] (can be nil)
// gradOutput: gradient w.r.t. output [batch, channels, ...]
// input: original input [batch, channels, ...]
// gamma: scale parameter [channels]
// numGroups: number of groups used in forward pass
// eps: epsilon for numerical stability
// Returns: (gradInput, gradGamma, gradBeta) - new tensors if dst was nil, otherwise returns dst tensors
func (t Tensor) GroupNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, numGroups int, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	if t.shape == nil {
		return gradInputDst, gradGammaDst, gradBetaDst
	}
	if IsNil(gradOutput) || IsNil(input) {
		return gradInputDst, gradGammaDst, gradBetaDst
	}
	if !t.Shape().Equal(gradOutput.Shape()) || !t.Shape().Equal(input.Shape()) {
		panic(fmt.Sprintf("tensor.GroupNormGrad: shape mismatch: t=%v, gradOutput=%v, input=%v", t.shape, gradOutput.Shape(), input.Shape()))
	}

	shape := t.Shape().ToSlice()
	if len(shape) < 2 {
		panic(fmt.Sprintf("tensor.GroupNormGrad: expected at least 2D tensor, got shape %v", shape))
	}

	channels := shape[1]
	if IsNil(gradInputDst) {
		gradInputDst = New(t.DataType(), t.Shape())
	} else if !t.Shape().Equal(gradInputDst.Shape()) {
		panic(fmt.Sprintf("tensor.GroupNormGrad: gradInputDst shape mismatch: expected %v, got %v", t.shape, gradInputDst.Shape()))
	}

	if gamma != nil {
		channelsShape := types.NewShape(channels)
		if IsNil(gradGammaDst) {
			gradGammaDst = New(t.DataType(), channelsShape)
		} else if !channelsShape.Equal(gradGammaDst.Shape()) {
			panic(fmt.Sprintf("tensor.GroupNormGrad: gradGammaDst shape mismatch: expected %v, got %v", channelsShape, gradGammaDst.Shape()))
		}
		if IsNil(gradBetaDst) {
			gradBetaDst = New(t.DataType(), channelsShape)
		} else if !channelsShape.Equal(gradBetaDst.Shape()) {
			panic(fmt.Sprintf("tensor.GroupNormGrad: gradBetaDst shape mismatch: expected %v, got %v", channelsShape, gradBetaDst.Shape()))
		}
	}

	switch gradOutputData := gradOutput.Data().(type) {
	case []float32:
		inputData := types.GetTensorData[[]float32](input)
		gradInputData := types.GetTensorData[[]float32](gradInputDst)
		var gammaData []float32
		if gamma != nil {
			gammaData = types.GetTensorData[[]float32](gamma)
		}
		var gradGammaData []float32
		var gradBetaData []float32
		if gradGammaDst != nil {
			gradGammaData = types.GetTensorData[[]float32](gradGammaDst)
			gradBetaData = types.GetTensorData[[]float32](gradBetaDst)
		}

		fp32.GroupNormGrad(gradInputData, gradGammaData, gradBetaData, gradOutputData, inputData, gammaData, shape, numGroups, float32(eps))
	default:
		panic(fmt.Sprintf("tensor.GroupNormGrad: unsupported data type: %T", gradOutputData))
	}
	return gradInputDst, gradGammaDst, gradBetaDst
}

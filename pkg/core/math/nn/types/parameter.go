package types

import (
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// ParamIndex represents a typed parameter index for layers
type ParamIndex int

// Standard parameter indices
const (
	ParamWeights ParamIndex = 1
	ParamBiases  ParamIndex = 2
	ParamKernels ParamIndex = 3
	ParamCustom  ParamIndex = 100
)

// LSTM parameter indices
// LSTM uses concatenated weights for all 4 gates (input, forget, cell, output)
// Order: input gate, forget gate, cell gate, output gate
const (
	ParamLSTMWeightIH ParamIndex = 101 // Input-to-hidden weights [4*hidden_size, input_size]
	ParamLSTMWeightHH ParamIndex = 102 // Hidden-to-hidden weights [4*hidden_size, hidden_size]
	ParamLSTMBias     ParamIndex = 103 // Bias [4*hidden_size] (optional)
)

// Parameter represents a trainable parameter (weight or bias).
type Parameter struct {
	Data         types.Tensor // Parameter values
	Grad         types.Tensor // Gradients (lazy allocation)
	RequiresGrad bool         // Whether to compute gradients (deprecated, use layer CanLearn)
}

// Init initializes the parameter if Data is not already set.
// If Data is already set (via WithWeights/WithBiases/WithKernels options), it is preserved.
// Otherwise, initializes Data using XavierUniform initialization with the given parameters.
//
// Parameters:
//   - dtype: Data type for the parameter tensor
//   - shape: Shape of the parameter tensor
//   - paramIdx: Parameter index (ParamWeights, ParamBiases, ParamKernels) used to infer fanIn/fanOut
//   - fanIn: Input fan size for Xavier initialization (if 0, inferred from paramIdx and shape)
//   - fanOut: Output fan size for Xavier initialization (if 0, inferred from paramIdx and shape)
//   - rng: Random number generator for initialization
//   - requiresGrad: Whether gradients should be computed for this parameter
func (p *Parameter) Init(dtype tensor.DataType, shape tensor.Shape, paramIdx ParamIndex, fanIn, fanOut int, rng tensor.RNG, requiresGrad bool) {
	if p == nil {
		return
	}

	// If Data is already set, preserve it (provided via options)
	if !tensor.IsNil(p.Data) {
		p.RequiresGrad = requiresGrad
		return
	}

	// Infer fanIn and fanOut from paramIdx and shape if not provided
	if fanIn == 0 || fanOut == 0 {
		fanIn, fanOut = inferFanSizes(paramIdx, shape)
	}

	// Create and initialize tensor with XavierUniform
	p.Data = tensor.XavierUniform(dtype, shape, fanIn, fanOut, rng)
	p.RequiresGrad = requiresGrad
}

// inferFanSizes infers fanIn and fanOut from parameter index and shape.
// This provides sensible defaults for Xavier initialization.
func inferFanSizes(paramIdx ParamIndex, shape tensor.Shape) (fanIn, fanOut int) {
	if shape == nil || len(shape) == 0 {
		return 1, 1
	}

	switch paramIdx {
	case ParamBiases:
		// Biases: fanIn=1, fanOut=output size
		return 1, shape[0]
	case ParamWeights:
		// Dense weights: shape [inFeatures, outFeatures]
		if len(shape) == 2 {
			return shape[0], shape[1]
		}
		// Default: use first and last dimension
		if len(shape) >= 2 {
			return shape[0], shape[len(shape)-1]
		}
		return shape[0], shape[0]
	case ParamKernels:
		// Conv kernels: shape [outChannels, inChannels, ...]
		// fanIn = inChannels * kernel_size, fanOut = outChannels
		if len(shape) >= 2 {
			outChannels := shape[0]
			inChannels := shape[1]
			// Calculate kernel size (product of remaining dimensions)
			kernelSize := 1
			for i := 2; i < len(shape); i++ {
				kernelSize *= shape[i]
			}
			return inChannels * kernelSize, outChannels
		}
		return shape[0], shape[0]
	default:
		// Default: use first and last dimension
		if len(shape) >= 2 {
			return shape[0], shape[len(shape)-1]
		}
		return shape[0], shape[0]
	}
}

// ZeroGrad zeros the gradient tensor.
// Allocates gradient tensor if needed and zeroes all values.
func (p *Parameter) ZeroGrad() {
	if p == nil || !p.RequiresGrad {
		return
	}

	if tensor.IsNil(p.Grad) {
		// Lazy allocation: create gradient tensor with same shape as data
		// Use data's data type instead of hardcoding DTFP32
		p.Grad = tensor.New(p.Data.DataType(), p.Data.Shape())
	}

	// Zero all gradient values
	p.Grad.Fill(nil, 0.0)
}

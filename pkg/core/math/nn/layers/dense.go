package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/nn"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Dense parameter indices (use standard ParamWeights, ParamBiases from base)

// Dense represents a fully connected (linear) layer.
type Dense struct {
	Base
	inFeatures  int
	outFeatures int
	hasBias     bool
}

// DenseOption represents an option for configuring a Dense layer.
type DenseOption func(*Dense)

// WithDenseBias sets whether to use a bias term.
func WithDenseBias(useBias bool) DenseOption {
	return func(d *Dense) {
		d.hasBias = useBias
	}
}

// NewDense creates a new Dense layer with the given input and output features.
// Weight initialization is not handled here - weights must be set externally.
// Accepts both DenseOption and BaseOption.
func NewDense(inFeatures, outFeatures int, opts ...interface{}) (*Dense, error) {
	if inFeatures <= 0 {
		return nil, fmt.Errorf("Dense: inFeatures must be positive, got %d", inFeatures)
	}
	if outFeatures <= 0 {
		return nil, fmt.Errorf("Dense: outFeatures must be positive, got %d", outFeatures)
	}

	// Collect base options
	var baseOpts []Option
	denseOpts := []DenseOption{}
	dense := &Dense{
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
		hasBias:     true, // Default to having bias
	}

	// Separate options into base options and dense options
	for _, opt := range opts {
		switch v := opt.(type) {
		case Option:
			baseOpts = append(baseOpts, v)
		case DenseOption:
			denseOpts = append(denseOpts, v)
		default:
			return nil, fmt.Errorf("Dense: invalid option type")
		}
	}

	// Create Base with options
	dense.Base = NewBase(baseOpts...)

	// Apply dense-specific options
	for _, opt := range denseOpts {
		opt(dense)
	}

	// Initialize weight parameter in map
	weightData := tensor.Tensor{
		Dim:  []int{inFeatures, outFeatures},
		Data: make([]float32, inFeatures*outFeatures),
	}
	dense.Base.SetParam(ParamWeights, &nn.Parameter{
		Data:         weightData,
		RequiresGrad: dense.Base.CanLearn(),
	})

	// Create bias parameter if needed
	if dense.hasBias {
		biasData := tensor.Tensor{
			Dim:  []int{outFeatures},
			Data: make([]float32, outFeatures),
		}
		dense.Base.SetParam(ParamBiases, &nn.Parameter{
			Data:         biasData,
			RequiresGrad: dense.Base.CanLearn(),
		})
	}

	return dense, nil
}

// Init initializes the layer, creating internal computation tensors.
func (d *Dense) Init(inputShape []int) error {
	if d == nil {
		return fmt.Errorf("Dense.Init: nil layer")
	}

	// Validate input shape matches inFeatures
	if len(inputShape) == 1 {
		if inputShape[0] != d.inFeatures {
			return fmt.Errorf("Dense.Init: input shape %v incompatible with inFeatures %d", inputShape, d.inFeatures)
		}
	} else if len(inputShape) == 2 {
		if inputShape[1] != d.inFeatures {
			return fmt.Errorf("Dense.Init: input shape %v incompatible with inFeatures %d", inputShape, d.inFeatures)
		}
	} else {
		return fmt.Errorf("Dense.Init: input must be 1D or 2D, got %dD", len(inputShape))
	}

	// Compute output shape
	var outputShape []int
	if len(inputShape) == 1 {
		outputShape = []int{d.outFeatures}
	} else {
		outputShape = []int{inputShape[0], d.outFeatures}
	}

	// Allocate output tensor
	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}
	d.Base.AllocOutput(outputShape, outputSize)

	// Set default name if not explicitly set
	d.Base.SetDefaultName("Dense", outputShape)

	return nil
}

// Forward computes the forward pass: output = input @ weight + bias.
// Computes directly into pre-allocated output tensor.
func (d *Dense) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if d == nil {
		return tensor.Tensor{}, fmt.Errorf("Dense.Forward: nil layer")
	}

	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Dense.Forward: empty input")
	}

	// Store input
	d.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := d.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Dense.Forward: output not allocated, must call Init first")
	}

	// Compute linear transformation directly into output tensor
	// output = input @ weight + bias
	weightParam := d.Base.GetParam(ParamWeights)
	var biasParam *tensor.Tensor
	if d.hasBias {
		biasParam = &d.Base.GetParam(ParamBiases).Data
	}
	if err := computeLinear(&input, &weightParam.Data, biasParam, &output); err != nil {
		return tensor.Tensor{}, fmt.Errorf("Dense.Forward: Linear operation failed: %w", err)
	}

	// Store output
	d.Base.StoreOutput(output)
	return output, nil
}

// computeLinear computes output = input @ weight + bias directly into output tensor.
func computeLinear(input *tensor.Tensor, weight *tensor.Tensor, bias *tensor.Tensor, output *tensor.Tensor) error {
	inputShape := input.Shape()
	weightShape := weight.Shape()
	inFeatures := weightShape[0]
	outFeatures := weightShape[1]

	if len(inputShape) == 1 {
		// Single sample: [inFeatures] → [outFeatures]
		// output = input @ weight = weight^T @ input
		inputTensor := &tensor.Tensor{Dim: []int{inFeatures}, Data: input.Data}
		weightTensor := &tensor.Tensor{Dim: []int{inFeatures, outFeatures}, Data: weight.Data}
		outputTensor := &tensor.Tensor{Dim: []int{outFeatures}, Data: output.Data}
		outputTensor.MatVecMulTransposed(weightTensor, inputTensor, 1.0, 0.0)

		if bias != nil {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == outFeatures {
				biasTensor := &tensor.Tensor{Dim: []int{outFeatures}, Data: bias.Data}
				outputTensor.AddScaled(biasTensor, 1.0)
			}
		}
	} else if len(inputShape) == 2 {
		// Batch: [batch, inFeatures] → [batch, outFeatures]
		batchSize := inputShape[0]

		// Use MatMul: output = input @ weight
		inputTensor := &tensor.Tensor{Dim: []int{batchSize, inFeatures}, Data: input.Data}
		weightTensor := &tensor.Tensor{Dim: []int{inFeatures, outFeatures}, Data: weight.Data}
		outputTensor := &tensor.Tensor{Dim: []int{batchSize, outFeatures}, Data: output.Data}
		inputTensor.MatMulTo(weightTensor, outputTensor)

		if bias != nil {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == outFeatures {
				biasTensor := &tensor.Tensor{Dim: []int{outFeatures}, Data: bias.Data}
				// Add bias to each batch element
				for b := 0; b < batchSize; b++ {
					batchOutput := &tensor.Tensor{
						Dim:  []int{outFeatures},
						Data: output.Data[b*outFeatures : (b+1)*outFeatures],
					}
					batchOutput.AddScaled(biasTensor, 1.0)
				}
			}
		}
	}

	return nil
}

// Backward computes gradients w.r.t. input, weight, and bias.
// Uses stored input/output from Forward pass.
func (d *Dense) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if d == nil {
		return tensor.Tensor{}, fmt.Errorf("Dense.Backward: nil layer")
	}

	if len(gradOutput.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Dense.Backward: empty gradOutput")
	}

	input := d.Base.Input()
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Dense.Backward: input not stored, must call Forward first")
	}

	output := d.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Dense.Backward: output not stored, must call Forward first")
	}

	inputPtr := &input

	// Compute gradient w.r.t. weight: gradWeight = gradOutput^T @ input
	// For batch: gradWeight = sum over batch of (input^T @ gradOutput)
	// For single sample: gradWeight = input^T @ gradOutput
	inputShape := inputPtr.Shape()
	if len(inputShape) == 1 {
		// Single sample: [inFeatures]
		// gradWeight shape: [inFeatures, outFeatures]
		// gradWeight = input^T @ gradOutput
		// Only compute gradients if CanLearn is true
		weightParam := d.Base.GetParam(ParamWeights)
		if d.Base.CanLearn() && weightParam != nil && weightParam.RequiresGrad {
			if len(weightParam.Grad.Dim) == 0 {
				weightParam.Grad = tensor.Tensor{
					Dim:  []int{d.inFeatures, d.outFeatures},
					Data: make([]float32, d.inFeatures*d.outFeatures),
				}
			}
			// Use MatMulTransposed: gradWeight = input^T @ gradOutput
			// Treat input [inFeatures] as [1, inFeatures] and gradOutput [outFeatures] as [1, outFeatures]
			// Outer product: input^T @ gradOutput = [inFeatures, 1] @ [1, outFeatures]
			inputTensor := &tensor.Tensor{Dim: []int{1, d.inFeatures}, Data: input.Data}
			gradTensor := &tensor.Tensor{Dim: []int{1, d.outFeatures}, Data: gradOutput.Data}
			gradWeightTensor := &tensor.Tensor{Dim: []int{d.inFeatures, d.outFeatures}, Data: weightParam.Grad.Data}
			inputTensor.MatMulTransposed(gradTensor, true, false, gradWeightTensor)
		}

		// Compute gradient w.r.t. bias: gradBias = gradOutput
		// Only compute gradients if CanLearn is true
		if d.hasBias {
			biasParam := d.Base.GetParam(ParamBiases)
			if d.Base.CanLearn() && biasParam != nil && biasParam.RequiresGrad {
				if len(biasParam.Grad.Dim) == 0 {
					biasParam.Grad = tensor.Tensor{
						Dim:  []int{d.outFeatures},
						Data: make([]float32, d.outFeatures),
					}
				}
				biasGradTensor := &tensor.Tensor{Dim: []int{d.outFeatures}, Data: biasParam.Grad.Data}
				gradTensor := &tensor.Tensor{Dim: []int{d.outFeatures}, Data: gradOutput.Data}
				biasGradTensor.AddScaled(gradTensor, 1.0)
			}
		}

		// Compute gradient w.r.t. input: gradInput = gradOutput @ weight^T
		gradInput := tensor.Tensor{
			Dim:  []int{d.inFeatures},
			Data: make([]float32, d.inFeatures),
		}
		// Treat gradOutput [outFeatures] as [1, outFeatures] and weight^T
		gradTensor := &tensor.Tensor{Dim: []int{1, d.outFeatures}, Data: gradOutput.Data}
		weightTensor := &tensor.Tensor{Dim: []int{d.inFeatures, d.outFeatures}, Data: weightParam.Data.Data}
		gradInputTensor := &tensor.Tensor{Dim: []int{1, d.inFeatures}, Data: gradInput.Data}
		gradTensor.MatMulTransposed(weightTensor, false, true, gradInputTensor)
		return gradInput, nil
	} else if len(inputShape) == 2 {
		// Batch case: [batch, inFeatures]
		batchSize := inputShape[0]

		// Compute gradient w.r.t. weight: sum over batch of (input^T @ gradOutput)
		// Use MatMulTransposed: gradWeight = input^T @ gradOutput (accumulated over batch)
		// Only compute gradients if CanLearn is true
		weightParam := d.Base.GetParam(ParamWeights)
		if d.Base.CanLearn() && weightParam != nil && weightParam.RequiresGrad {
			if len(weightParam.Grad.Dim) == 0 {
				weightParam.Grad = tensor.Tensor{
					Dim:  []int{d.inFeatures, d.outFeatures},
					Data: make([]float32, d.inFeatures*d.outFeatures),
				}
			}
			// input^T @ gradOutput: transpose input, no transpose on gradOutput
			inputTensor := &tensor.Tensor{Dim: []int{batchSize, d.inFeatures}, Data: inputPtr.Data}
			gradTensor := &tensor.Tensor{Dim: []int{batchSize, d.outFeatures}, Data: gradOutput.Data}
			gradWeightTensor := &tensor.Tensor{Dim: []int{d.inFeatures, d.outFeatures}, Data: weightParam.Grad.Data}
			inputTensor.MatMulTransposed(gradTensor, true, false, gradWeightTensor)
		}

		// Compute gradient w.r.t. bias: sum over batch of gradOutput
		// Only compute gradients if CanLearn is true
		if d.hasBias {
			biasParam := d.Base.GetParam(ParamBiases)
			if d.Base.CanLearn() && biasParam != nil && biasParam.RequiresGrad {
				if len(biasParam.Grad.Dim) == 0 {
					biasParam.Grad = tensor.Tensor{
						Dim:  []int{d.outFeatures},
						Data: make([]float32, d.outFeatures),
					}
				}
				// Sum gradOutput over batch dimension
				biasGradTensor := &tensor.Tensor{Dim: []int{d.outFeatures}, Data: biasParam.Grad.Data}
				for b := 0; b < batchSize; b++ {
					batchGrad := &tensor.Tensor{
						Dim:  []int{d.outFeatures},
						Data: gradOutput.Data[b*d.outFeatures : (b+1)*d.outFeatures],
					}
					biasGradTensor.AddScaled(batchGrad, 1.0)
				}
			}
		}

		// Compute gradient w.r.t. input: gradInput = gradOutput @ weight^T
		gradInput := tensor.Tensor{
			Dim:  []int{batchSize, d.inFeatures},
			Data: make([]float32, batchSize*d.inFeatures),
		}
		// gradOutput @ weight^T: no transpose on gradOutput, transpose weight
		gradTensor := &tensor.Tensor{Dim: []int{batchSize, d.outFeatures}, Data: gradOutput.Data}
		weightTensor := &tensor.Tensor{Dim: []int{d.inFeatures, d.outFeatures}, Data: weightParam.Data.Data}
		gradInputTensor := &tensor.Tensor{Dim: []int{batchSize, d.inFeatures}, Data: gradInput.Data}
		gradTensor.MatMulTransposed(weightTensor, false, true, gradInputTensor)
		d.Base.StoreGrad(gradInput)
		return gradInput, nil
	}

	return tensor.Tensor{}, fmt.Errorf("Dense.Backward: unsupported input shape: %v", inputPtr.Shape())
}

// OutputShape returns the output shape for given input shape.
func (d *Dense) OutputShape(inputShape []int) ([]int, error) {
	if d == nil {
		return nil, fmt.Errorf("Dense.OutputShape: nil layer")
	}

	if len(inputShape) == 1 {
		if inputShape[0] != d.inFeatures {
			return nil, fmt.Errorf("Dense.OutputShape: input shape %v incompatible with inFeatures %d", inputShape, d.inFeatures)
		}
		return []int{d.outFeatures}, nil
	} else if len(inputShape) == 2 {
		if inputShape[1] != d.inFeatures {
			return nil, fmt.Errorf("Dense.OutputShape: input shape %v incompatible with inFeatures %d", inputShape, d.inFeatures)
		}
		return []int{inputShape[0], d.outFeatures}, nil
	}
	return nil, fmt.Errorf("Dense.OutputShape: input must be 1D or 2D, got %dD", len(inputShape))
}

// Weight returns the weight parameter tensor.
func (d *Dense) Weight() tensor.Tensor {
	if d == nil {
		return tensor.Tensor{}
	}
	weightParam := d.Base.GetParam(ParamWeights)
	if weightParam == nil {
		return tensor.Tensor{}
	}
	return weightParam.Data
}

// Bias returns the bias parameter tensor.
func (d *Dense) Bias() tensor.Tensor {
	if d == nil || !d.hasBias {
		return tensor.Tensor{}
	}
	biasParam := d.Base.GetParam(ParamBiases)
	if biasParam == nil {
		return tensor.Tensor{}
	}
	return biasParam.Data
}

// SetWeight sets the weight parameter tensor.
func (d *Dense) SetWeight(weight tensor.Tensor) error {
	if d == nil {
		return fmt.Errorf("Dense.SetWeight: nil layer")
	}
	if len(weight.Dim) == 0 {
		return fmt.Errorf("Dense.SetWeight: empty weight tensor")
	}
	// Validate shape matches expected
	if len(weight.Dim) != 2 {
		return fmt.Errorf("Dense.SetWeight: weight must be 2D, got %dD", len(weight.Dim))
	}
	if weight.Dim[0] != d.inFeatures || weight.Dim[1] != d.outFeatures {
		return fmt.Errorf("Dense.SetWeight: weight shape %v doesn't match expected [%d, %d]",
			weight.Dim, d.inFeatures, d.outFeatures)
	}
	weightParam := d.Base.GetParam(ParamWeights)
	if weightParam == nil {
		return fmt.Errorf("Dense.SetWeight: weight parameter not initialized")
	}
	weightParam.Data = weight
	return nil
}

// SetBias sets the bias parameter tensor.
func (d *Dense) SetBias(bias tensor.Tensor) error {
	if d == nil {
		return fmt.Errorf("Dense.SetBias: nil layer")
	}
	if !d.hasBias {
		return fmt.Errorf("Dense.SetBias: layer has no bias")
	}
	if len(bias.Dim) == 0 {
		return fmt.Errorf("Dense.SetBias: empty bias tensor")
	}
	// Validate shape matches expected
	if len(bias.Dim) != 1 || bias.Dim[0] != d.outFeatures {
		return fmt.Errorf("Dense.SetBias: bias shape %v doesn't match expected [%d]",
			bias.Dim, d.outFeatures)
	}
	biasParam := d.Base.GetParam(ParamBiases)
	if biasParam == nil {
		return fmt.Errorf("Dense.SetBias: bias parameter not initialized")
	}
	biasParam.Data = bias
	return nil
}

package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Dense parameter indices (use standard ParamWeights, ParamBiases from base)

// Dense represents a fully connected (linear) layer.
type Dense struct {
	Base
	inFeatures  int
	outFeatures int
	hasBias     bool
}

// NewDense creates a new Dense layer with the given input and output features.
// Weight initialization is not handled here - weights must be set externally.
// Accepts Base Option types.
func NewDense(inFeatures, outFeatures int, opts ...Option) (*Dense, error) {
	if inFeatures <= 0 {
		return nil, fmt.Errorf("Dense: inFeatures must be positive, got %d", inFeatures)
	}
	if outFeatures <= 0 {
		return nil, fmt.Errorf("Dense: outFeatures must be positive, got %d", outFeatures)
	}

	// Create Base with all options
	base := NewBase("dense", opts...)

	// Check if bias parameter was set via options (e.g., WithBiases)
	_, hasBiasParam := base.Parameter(ParamBiases)

	dense := &Dense{
		Base:        base,
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
		hasBias:     true, // Always create bias by default
	}

	// Initialize weight parameter in map
	weightData := tensor.New(tensor.DTFP32, tensor.NewShape(inFeatures, outFeatures))
	dense.Base.SetParam(ParamWeights, Parameter{
		Data:         weightData,
		RequiresGrad: dense.Base.CanLearn(),
	})

	// Create bias parameter if not already set via options
	if !hasBiasParam {
		biasData := tensor.New(tensor.DTFP32, tensor.NewShape(outFeatures))
		dense.Base.SetParam(ParamBiases, Parameter{
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

	return nil
}

// Forward computes the forward pass: output = input @ weight + bias.
// Computes directly into pre-allocated output tensor.
func (d *Dense) Forward(input types.Tensor) (types.Tensor, error) {
	if d == nil {
		return nil, fmt.Errorf("Dense.Forward: nil layer")
	}

	if input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Dense.Forward: empty input")
	}

	// Store input
	d.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := d.Base.Output()
	if output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Dense.Forward: output not allocated, must call Init first")
	}

	// Compute linear transformation directly into output tensor
	// output = input @ weight + bias
	weightParam, ok := d.Base.Parameter(ParamWeights)
	if !ok {
		return nil, fmt.Errorf("Dense.Forward: weight parameter not initialized")
	}
	var biasParam *types.Tensor
	if d.hasBias {
		biasParamVal, ok := d.Base.Parameter(ParamBiases)
		if ok {
			biasParam = &biasParamVal.Data
		}
	}
	if err := computeLinear(input, weightParam.Data, biasParam, output); err != nil {
		return nil, fmt.Errorf("Dense.Forward: Linear operation failed: %w", err)
	}

	// Store output
	d.Base.StoreOutput(output)
	return output, nil
}

// computeLinear computes output = input @ weight + bias directly into output tensor.
func computeLinear(input types.Tensor, weight types.Tensor, bias *types.Tensor, output types.Tensor) error {
	inputShape := input.Shape()
	weightShape := weight.Shape()
	inFeatures := weightShape[0]
	outFeatures := weightShape[1]

	if len(inputShape) == 1 {
		// Single sample: [inFeatures] → [outFeatures]
		// output = input @ weight = weight^T @ input
		inputTensor := tensor.FromFloat32(tensor.NewShape(inFeatures), input.Data().([]float32))
		weightTensor := tensor.FromFloat32(tensor.NewShape(inFeatures, outFeatures), weight.Data().([]float32))
		outputTensor := tensor.FromFloat32(tensor.NewShape(outFeatures), output.Data().([]float32))
		outputTensor.MatVecMulTransposed(weightTensor, inputTensor, 1.0, 0.0)

		if bias != nil {
			biasShape := (*bias).Shape()
			if len(biasShape) == 1 && biasShape[0] == outFeatures {
				biasTensor := tensor.FromFloat32(tensor.NewShape(outFeatures), (*bias).Data().([]float32))
				outputTensor.AddScaled(biasTensor, 1.0)
			}
		}
	} else if len(inputShape) == 2 {
		// Batch: [batch, inFeatures] → [batch, outFeatures]
		batchSize := inputShape[0]

		// Use MatMul: output = input @ weight
		inputTensor := tensor.FromFloat32(tensor.NewShape(batchSize, inFeatures), input.Data().([]float32))
		weightTensor := tensor.FromFloat32(tensor.NewShape(inFeatures, outFeatures), weight.Data().([]float32))
		outputTensor := tensor.FromFloat32(tensor.NewShape(batchSize, outFeatures), output.Data().([]float32))
		inputTensor.MatMulTo(weightTensor, &outputTensor)

		if bias != nil {
			biasShape := (*bias).Shape()
			if len(biasShape) == 1 && biasShape[0] == outFeatures {
				biasTensor := tensor.FromFloat32(tensor.NewShape(outFeatures), (*bias).Data().([]float32))
				// Add bias to each batch element
				outputData := output.Data().([]float32)
				for b := 0; b < batchSize; b++ {
					batchOutput := tensor.FromFloat32(tensor.NewShape(outFeatures), outputData[b*outFeatures:(b+1)*outFeatures])
					batchOutput.AddScaled(biasTensor, 1.0)
				}
			}
		}
	}

	return nil
}

// Backward computes gradients w.r.t. input, weight, and bias.
// Uses stored input/output from Forward pass.
func (d *Dense) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if d == nil {
		return nil, fmt.Errorf("Dense.Backward: nil layer")
	}

	if gradOutput.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Dense.Backward: empty gradOutput")
	}

	input := d.Base.Input()
	if input == nil || input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Dense.Backward: input not stored, must call Forward first")
	}

	output := d.Base.Output()
	if output == nil || output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("Dense.Backward: output not stored, must call Forward first")
	}

	inputPtr := input

	// Compute gradient w.r.t. weight: gradWeight = gradOutput^T @ input
	// For batch: gradWeight = sum over batch of (input^T @ gradOutput)
	// For single sample: gradWeight = input^T @ gradOutput
	inputShape := inputPtr.Shape()
	if len(inputShape) == 1 {
		// Single sample: [inFeatures]
		// gradWeight shape: [inFeatures, outFeatures]
		// gradWeight = input^T @ gradOutput
		// Only compute gradients if CanLearn is true
		weightParam, ok := d.Base.Parameter(ParamWeights)
		if d.Base.CanLearn() && ok && weightParam.RequiresGrad {
			if weightParam.Grad == nil || weightParam.Grad.Shape().Rank() == 0 {
				weightParam.Grad = tensor.New(tensor.DTFP32, tensor.NewShape(d.inFeatures, d.outFeatures))
			}
			// Use MatMulTransposed: gradWeight = input^T @ gradOutput
			// Treat input [inFeatures] as [1, inFeatures] and gradOutput [outFeatures] as [1, outFeatures]
			// Outer product: input^T @ gradOutput = [inFeatures, 1] @ [1, outFeatures]
			inputTensor := tensor.FromFloat32(tensor.NewShape(1, d.inFeatures), input.Data().([]float32))
			gradTensor := tensor.FromFloat32(tensor.NewShape(1, d.outFeatures), gradOutput.Data().([]float32))
			gradWeightTensor := tensor.FromFloat32(tensor.NewShape(d.inFeatures, d.outFeatures), weightParam.Grad.Data().([]float32))
			inputTensor.MatMulTransposed(gradTensor, true, false, &gradWeightTensor)
		}

		// Compute gradient w.r.t. bias: gradBias = gradOutput
		// Only compute gradients if CanLearn is true
		if d.hasBias {
			biasParam, ok := d.Base.Parameter(ParamBiases)
			if d.Base.CanLearn() && ok && biasParam.RequiresGrad {
				if biasParam.Grad == nil || biasParam.Grad.Shape().Rank() == 0 {
					biasParam.Grad = tensor.New(tensor.DTFP32, tensor.NewShape(d.outFeatures))
				}
				biasGradTensor := tensor.FromFloat32(tensor.NewShape(d.outFeatures), biasParam.Grad.Data().([]float32))
				gradTensor := tensor.FromFloat32(tensor.NewShape(d.outFeatures), gradOutput.Data().([]float32))
				biasGradTensor.AddScaled(gradTensor, 1.0)
				d.Base.SetParam(ParamBiases, biasParam)
			}
		}

		// Compute gradient w.r.t. input: gradInput = gradOutput @ weight^T
		gradInput := tensor.New(tensor.DTFP32, tensor.NewShape(d.inFeatures))
		// Treat gradOutput [outFeatures] as [1, outFeatures] and weight^T
		gradTensor := tensor.FromFloat32(tensor.NewShape(1, d.outFeatures), gradOutput.Data().([]float32))
		weightTensor := tensor.FromFloat32(tensor.NewShape(d.inFeatures, d.outFeatures), weightParam.Data.Data().([]float32))
		gradInputTensor := tensor.FromFloat32(tensor.NewShape(1, d.inFeatures), gradInput.Data().([]float32))
		gradTensor.MatMulTransposed(weightTensor, false, true, &gradInputTensor)
		if d.Base.CanLearn() && ok {
			d.Base.SetParam(ParamWeights, weightParam)
		}
		return gradInput, nil
	} else if len(inputShape) == 2 {
		// Batch case: [batch, inFeatures]
		batchSize := inputShape[0]

		// Compute gradient w.r.t. weight: sum over batch of (input^T @ gradOutput)
		// Use MatMulTransposed: gradWeight = input^T @ gradOutput (accumulated over batch)
		// Only compute gradients if CanLearn is true
		weightParam, ok := d.Base.Parameter(ParamWeights)
		if d.Base.CanLearn() && ok && weightParam.RequiresGrad {
			if weightParam.Grad == nil || weightParam.Grad.Shape().Rank() == 0 {
				weightParam.Grad = tensor.New(tensor.DTFP32, tensor.NewShape(d.inFeatures, d.outFeatures))
			}
			// input^T @ gradOutput: transpose input, no transpose on gradOutput
			inputTensor := tensor.FromFloat32(tensor.NewShape(batchSize, d.inFeatures), inputPtr.Data().([]float32))
			gradTensor := tensor.FromFloat32(tensor.NewShape(batchSize, d.outFeatures), gradOutput.Data().([]float32))
			gradWeightTensor := tensor.FromFloat32(tensor.NewShape(d.inFeatures, d.outFeatures), weightParam.Grad.Data().([]float32))
			inputTensor.MatMulTransposed(gradTensor, true, false, &gradWeightTensor)
			d.Base.SetParam(ParamWeights, weightParam)
		}

		// Compute gradient w.r.t. bias: sum over batch of gradOutput
		// Only compute gradients if CanLearn is true
		if d.hasBias {
			biasParam, ok := d.Base.Parameter(ParamBiases)
			if d.Base.CanLearn() && ok && biasParam.RequiresGrad {
				if biasParam.Grad == nil || biasParam.Grad.Shape().Rank() == 0 {
					biasParam.Grad = tensor.New(tensor.DTFP32, tensor.NewShape(d.outFeatures))
				}
				// Sum gradOutput over batch dimension
				biasGradTensor := tensor.FromFloat32(tensor.NewShape(d.outFeatures), biasParam.Grad.Data().([]float32))
				gradOutputData := gradOutput.Data().([]float32)
				for b := 0; b < batchSize; b++ {
					batchGrad := tensor.FromFloat32(tensor.NewShape(d.outFeatures), gradOutputData[b*d.outFeatures:(b+1)*d.outFeatures])
					biasGradTensor.AddScaled(batchGrad, 1.0)
				}
				d.Base.SetParam(ParamBiases, biasParam)
			}
		}

		// Compute gradient w.r.t. input: gradInput = gradOutput @ weight^T
		gradInput := tensor.New(tensor.DTFP32, tensor.NewShape(batchSize, d.inFeatures))
		// gradOutput @ weight^T: no transpose on gradOutput, transpose weight
		gradTensor := tensor.FromFloat32(tensor.NewShape(batchSize, d.outFeatures), gradOutput.Data().([]float32))
		weightTensor := tensor.FromFloat32(tensor.NewShape(d.inFeatures, d.outFeatures), weightParam.Data.Data().([]float32))
		gradInputTensor := tensor.FromFloat32(tensor.NewShape(batchSize, d.inFeatures), gradInput.Data().([]float32))
		gradTensor.MatMulTransposed(weightTensor, false, true, &gradInputTensor)
		d.Base.StoreGrad(gradInput)
		return gradInput, nil
	}

	return nil, fmt.Errorf("Dense.Backward: unsupported input shape: %v", inputPtr.Shape())
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
func (d *Dense) Weight() types.Tensor {
	if d == nil {
		return nil
	}
	return d.Base.Weights().Data
}

// Bias returns the bias parameter tensor.
func (d *Dense) Bias() types.Tensor {
	if d == nil || !d.hasBias {
		return nil
	}
	return d.Base.Biases().Data
}

// SetWeight sets the weight parameter tensor.
func (d *Dense) SetWeight(weight types.Tensor) error {
	if d == nil {
		return fmt.Errorf("Dense.SetWeight: nil layer")
	}
	if weight.Shape().Rank() == 0 {
		return fmt.Errorf("Dense.SetWeight: empty weight tensor")
	}
	// Validate shape matches expected
	if weight.Shape().Rank() != 2 {
		return fmt.Errorf("Dense.SetWeight: weight must be 2D, got %dD", weight.Shape().Rank())
	}
	weightShape := weight.Shape()
	if weightShape[0] != d.inFeatures || weightShape[1] != d.outFeatures {
		return fmt.Errorf("Dense.SetWeight: weight shape %v doesn't match expected [%d, %d]",
			weightShape, d.inFeatures, d.outFeatures)
	}
	weightParam, ok := d.Base.Parameter(ParamWeights)
	if !ok {
		return fmt.Errorf("Dense.SetWeight: weight parameter not initialized")
	}
	weightParam.Data = weight
	d.Base.SetParam(ParamWeights, weightParam)
	return nil
}

// SetBias sets the bias parameter tensor.
func (d *Dense) SetBias(bias types.Tensor) error {
	if d == nil {
		return fmt.Errorf("Dense.SetBias: nil layer")
	}
	if !d.hasBias {
		return fmt.Errorf("Dense.SetBias: layer has no bias")
	}
	if bias.Shape().Rank() == 0 {
		return fmt.Errorf("Dense.SetBias: empty bias tensor")
	}
	// Validate shape matches expected
	biasShape := bias.Shape()
	if biasShape.Rank() != 1 || biasShape[0] != d.outFeatures {
		return fmt.Errorf("Dense.SetBias: bias shape %v doesn't match expected [%d]",
			biasShape, d.outFeatures)
	}
	biasParam, ok := d.Base.Parameter(ParamBiases)
	if !ok {
		return fmt.Errorf("Dense.SetBias: bias parameter not initialized")
	}
	biasParam.Data = bias
	d.Base.SetParam(ParamBiases, biasParam)
	return nil
}

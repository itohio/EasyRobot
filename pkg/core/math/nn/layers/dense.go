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

	// Create Base without options first
	base := NewBase("dense")

	dense := &Dense{
		Base:        base,
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
		hasBias:     true, // Always create bias by default
	}

	// Parse options first to get data type (if specified)
	dense.Base.ParseOptions(opts...)

	// Set defaults: initialize weight and bias parameters using layer's data type
	dtype := dense.Base.DataType()
	weightData := tensor.New(dtype, tensor.NewShape(inFeatures, outFeatures))
	dense.Base.SetParam(ParamWeights, Parameter{
		Data:         weightData,
		RequiresGrad: dense.Base.CanLearn(),
	})

	// Bias data type must match weight data type
	biasData := tensor.New(dtype, tensor.NewShape(outFeatures))
	dense.Base.SetParam(ParamBiases, Parameter{
		Data:         biasData,
		RequiresGrad: dense.Base.CanLearn(),
	})

	// Update RequiresGrad on parameters after options are parsed
	// This ensures WithCanLearn option takes effect
	if dense.Base.CanLearn() {
		weightParam, ok := dense.Base.Parameter(ParamWeights)
		if ok {
			weightParam.RequiresGrad = true
			dense.Base.SetParam(ParamWeights, weightParam)
		}
		biasParam, ok := dense.Base.Parameter(ParamBiases)
		if ok {
			biasParam.RequiresGrad = true
			dense.Base.SetParam(ParamBiases, biasParam)
		}
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

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Dense.Forward: empty input")
	}

	// Store input
	d.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := d.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Dense.Forward: output not allocated, must call Init first")
	}

	// Compute linear transformation directly into output tensor
	// output = input @ weight + bias
	weightParam, ok := d.Base.Parameter(ParamWeights)
	if !ok {
		return nil, fmt.Errorf("Dense.Forward: weight parameter not initialized")
	}
	var biasParam types.Tensor
	if d.hasBias {
		biasParamVal, ok := d.Base.Parameter(ParamBiases)
		if ok {
			biasParam = biasParamVal.Data
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
// Uses weight's data type for all operations. Input and output must have compatible data types.
func computeLinear(input types.Tensor, weight types.Tensor, bias types.Tensor, output types.Tensor) error {
	inputShape := input.Shape()
	weightShape := weight.Shape()
	outFeatures := weightShape[1]

	if len(inputShape) == 1 {
		// Single sample: [inFeatures] → [outFeatures]
		// output = input @ weight = weight^T @ input
		// Use tensors directly - they already have the correct shapes
		output.MatVecMulTransposed(weight, input, 1.0, 0.0)

		if bias.Shape().Rank() > 0 {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == outFeatures {
				output.AddScaled(bias, 1.0)
			}
		}
	} else if len(inputShape) == 2 {
		// Batch: [batch, inFeatures] → [batch, outFeatures]
		// Use MatMul: output = input @ weight
		input.MatMulTo(weight, output)

		if bias.Shape().Rank() > 0 {
			biasShape := bias.Shape()
			if len(biasShape) == 1 && biasShape[0] == outFeatures {
				// Add bias using tensor operations - broadcast and add
				// Reshape bias to [1, outFeatures] for broadcasting
				biasBroadcast := bias.Reshape(tensor.NewShape(1, outFeatures))
				// Broadcast bias to match output shape [batch, outFeatures]
				biasBroadcastFull, err := biasBroadcast.BroadcastTo(output.Shape())
				if err == nil {
					output.Add(biasBroadcastFull)
				} else {
					// Fallback: Add bias element-wise using iteration
					// For each batch element, add the bias
					batchSize := inputShape[0]
					for b := 0; b < batchSize; b++ {
						// Get output slice for this batch
						outputBatch := output.Reshape(tensor.NewShape(batchSize, outFeatures))
						outputBatchSlice := outputBatch.Reshape(tensor.NewShape(outFeatures))
						// Fix batch dimension and iterate over features
						outputBatchSlice.AddScaled(bias, 1.0)
					}
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

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Dense.Backward: empty gradOutput")
	}

	input := d.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Dense.Backward: input not stored, must call Forward first")
	}

	output := d.Base.Output()
	if tensor.IsNil(output) {
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
		if !ok || weightParam.Data == nil {
			return nil, fmt.Errorf("Dense.Backward: weight parameter not initialized")
		}
		weightDtype := weightParam.Data.DataType()
		if d.Base.CanLearn() && weightParam.RequiresGrad {
			if tensor.IsNil(weightParam.Grad) {
				weightParam.Grad = tensor.New(weightDtype, tensor.NewShape(d.inFeatures, d.outFeatures))
			}
			// Use MatMulTransposed: gradWeight = input^T @ gradOutput
			// Reshape input and gradOutput to 2D for matrix operations
			inputReshaped := input.Reshape(tensor.NewShape(1, d.inFeatures))
			gradReshaped := gradOutput.Reshape(tensor.NewShape(1, d.outFeatures))
			inputReshaped.MatMulTransposed(gradReshaped, true, false, weightParam.Grad)
			d.Base.SetParam(ParamWeights, weightParam)
		}

		// Compute gradient w.r.t. bias: gradBias = gradOutput
		// Only compute gradients if CanLearn is true
		if d.hasBias {
			biasParam, ok := d.Base.Parameter(ParamBiases)
			if d.Base.CanLearn() && ok && biasParam.RequiresGrad {
				if tensor.IsNil(biasParam.Grad) {
					biasParam.Grad = tensor.New(weightDtype, tensor.NewShape(d.outFeatures))
				}
				// Copy gradOutput directly (same shape)
				biasParam.Grad.Copy(gradOutput)
				d.Base.SetParam(ParamBiases, biasParam)
			}
		}

		// Compute gradient w.r.t. input: gradInput = gradOutput @ weight^T
		// Use gradOutput's data type to match incoming gradient
		// Reshape gradOutput to [1, outFeatures] for matrix multiplication
		gradReshaped := gradOutput.Reshape(tensor.NewShape(1, d.outFeatures))
		// Result will be [1, inFeatures], need to reshape to [inFeatures]
		gradInput2D := tensor.New(gradOutput.DataType(), tensor.NewShape(1, d.inFeatures))
		result := gradReshaped.MatMulTransposed(weightParam.Data, false, true, gradInput2D)
		if result == nil {
			return nil, fmt.Errorf("Dense.Backward: MatMulTransposed returned nil")
		}
		// Reshape back to 1D
		gradInput := result.Reshape(tensor.NewShape(d.inFeatures))
		if d.Base.CanLearn() {
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
		weightDtype := weightParam.Data.DataType()
		if d.Base.CanLearn() && ok && weightParam.RequiresGrad {
			if tensor.IsNil(weightParam.Grad) {
				weightParam.Grad = tensor.New(weightDtype, tensor.NewShape(d.inFeatures, d.outFeatures))
			}
			// input^T @ gradOutput: transpose input, no transpose on gradOutput
			// Use tensors directly - they already have the correct shapes
			inputPtr.MatMulTransposed(gradOutput, true, false, weightParam.Grad)
			d.Base.SetParam(ParamWeights, weightParam)
		}

		// Compute gradient w.r.t. bias: sum over batch of gradOutput
		// Only compute gradients if CanLearn is true
		if d.hasBias {
			biasParam, ok := d.Base.Parameter(ParamBiases)
			if d.Base.CanLearn() && ok && biasParam.RequiresGrad {
				if tensor.IsNil(biasParam.Grad) {
					biasParam.Grad = tensor.New(weightDtype, tensor.NewShape(d.outFeatures))
				}
				// Sum gradOutput over batch dimension using tensor operations
				summed := gradOutput.Sum(0) // Sum over batch dimension (axis 0)
				// Copy summed values to bias gradient
				biasParam.Grad.Copy(summed)
				d.Base.SetParam(ParamBiases, biasParam)
			}
		}

		// Compute gradient w.r.t. input: gradInput = gradOutput @ weight^T
		// Use gradOutput's data type to match incoming gradient
		gradInput := tensor.New(gradOutput.DataType(), tensor.NewShape(batchSize, d.inFeatures))
		// gradOutput @ weight^T: no transpose on gradOutput, transpose weight
		gradOutput.MatMulTransposed(weightParam.Data, false, true, gradInput)
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
		// Return empty tensor instead of nil to match test expectations
		return tensor.Empty(tensor.DTFP32)
	}
	weightParam := d.Base.Weights()
	if tensor.IsNil(weightParam.Data) {
		return tensor.Empty(tensor.DTFP32)
	}
	return weightParam.Data
}

// Bias returns the bias parameter tensor.
func (d *Dense) Bias() types.Tensor {
	if d == nil || !d.hasBias {
		// Return empty tensor instead of nil to match test expectations
		return tensor.Empty(tensor.DTFP32)
	}
	biasParam := d.Base.Biases()
	if tensor.IsNil(biasParam.Data) {
		return tensor.Empty(tensor.DTFP32)
	}
	return biasParam.Data
}

// SetWeight sets the weight parameter tensor.
func (d *Dense) SetWeight(weight types.Tensor) error {
	if d == nil {
		return fmt.Errorf("Dense.SetWeight: nil layer")
	}
	if tensor.IsNil(weight) {
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
	// Preserve Grad when setting new weight, but update RequiresGrad based on CanLearn
	weightParam.Data = weight
	weightParam.RequiresGrad = d.Base.CanLearn()
	d.Base.SetParam(ParamWeights, weightParam)

	// Ensure bias data type matches weight data type
	if d.hasBias {
		biasParam, ok := d.Base.Parameter(ParamBiases)
		if ok && biasParam.Data != nil && biasParam.Data.Shape() != nil && biasParam.Data.DataType() != weight.DataType() {
			// Recreate bias with matching data type
			biasShape := biasParam.Data.Shape()
			biasParam.Data = tensor.New(weight.DataType(), biasShape)
			d.Base.SetParam(ParamBiases, biasParam)
		}
	}
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
	if tensor.IsNil(bias) {
		return fmt.Errorf("Dense.SetBias: empty bias tensor")
	}
	// Validate shape matches expected
	biasShape := bias.Shape()
	if biasShape.Rank() != 1 || biasShape[0] != d.outFeatures {
		return fmt.Errorf("Dense.SetBias: bias shape %v doesn't match expected [%d]",
			biasShape, d.outFeatures)
	}
	// Validate bias data type matches weight data type
	weightParam, ok := d.Base.Parameter(ParamWeights)
	if !ok {
		return fmt.Errorf("Dense.SetBias: weight parameter not initialized")
	}
	if weightParam.Data == nil {
		return fmt.Errorf("Dense.SetBias: weight data not initialized")
	}
	if bias.DataType() != weightParam.Data.DataType() {
		return fmt.Errorf("Dense.SetBias: bias data type %v doesn't match weight data type %v",
			bias.DataType(), weightParam.Data.DataType())
	}
	biasParam, ok := d.Base.Parameter(ParamBiases)
	if !ok {
		return fmt.Errorf("Dense.SetBias: bias parameter not initialized")
	}
	// Preserve Grad when setting new bias, but update RequiresGrad based on CanLearn
	biasParam.Data = bias
	biasParam.RequiresGrad = d.Base.CanLearn()
	d.Base.SetParam(ParamBiases, biasParam)
	return nil
}

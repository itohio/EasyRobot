package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	tensorTypes "github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Conv1D represents a 1D convolution layer.
type Conv1D struct {
	Base
	inChannels  int
	outChannels int
	kernelLen   int
	stride      int
	pad         int
	hasBias     bool
}

// NewConv1D creates a new Conv1D layer.
func NewConv1D(
	inChannels, outChannels, kernelLen, stride, pad int,
	opts ...Option,
) (*Conv1D, error) {
	if inChannels <= 0 {
		return nil, fmt.Errorf("Conv1D: inChannels must be positive, got %d", inChannels)
	}
	if outChannels <= 0 {
		return nil, fmt.Errorf("Conv1D: outChannels must be positive, got %d", outChannels)
	}
	if kernelLen <= 0 {
		return nil, fmt.Errorf("Conv1D: kernelLen must be positive, got %d", kernelLen)
	}
	if stride <= 0 {
		return nil, fmt.Errorf("Conv1D: stride must be positive, got %d", stride)
	}

	// Create Base without options first
	base := NewBase("conv1d")

	// Parse options first to get configuration hints (like BiasHint)
	base.ParseOptions(opts...)

	// Determine if bias should be created based on hint
	hasBias := false // Default to no bias
	if hint := base.BiasHint(); hint != nil {
		hasBias = *hint
	}

	conv := &Conv1D{
		Base:        base,
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelLen:   kernelLen,
		stride:      stride,
		pad:         pad,
		hasBias:     hasBias,
	}

	// Set defaults: create kernel parameter using layer's data type
	_, hasKernel := conv.Base.Parameter(types.ParamKernels)
	if !hasKernel {
		dtype := conv.Base.DataType()
		kernelData := tensor.New(dtype, tensor.NewShape(outChannels, inChannels, kernelLen))
		conv.Base.SetParam(types.ParamKernels, types.Parameter{
			Data:         kernelData,
			RequiresGrad: conv.Base.CanLearn(),
		})
	}

	// Create bias parameter if needed and not already set via options
	if conv.hasBias {
		_, hasBiasParam := conv.Base.Parameter(types.ParamBiases)
		if !hasBiasParam {
			// Use layer's data type for bias
			dtype := conv.Base.DataType()
			biasData := tensor.New(dtype, tensor.NewShape(outChannels))
			conv.Base.SetParam(types.ParamBiases, types.Parameter{
				Data:         biasData,
				RequiresGrad: conv.Base.CanLearn(),
			})
		}
	}

	// Parse options again after defaults to allow overriding
	conv.Base.ParseOptions(opts...)

	// Update RequiresGrad on parameters after options are parsed
	// This ensures WithCanLearn option takes effect
	if conv.Base.CanLearn() {
		kernelParam, ok := conv.Base.Parameter(types.ParamKernels)
		if ok {
			kernelParam.RequiresGrad = true
			conv.Base.SetParam(types.ParamKernels, kernelParam)
		}
		if conv.hasBias {
			biasParam, ok := conv.Base.Parameter(types.ParamBiases)
			if ok {
				biasParam.RequiresGrad = true
				conv.Base.SetParam(types.ParamBiases, biasParam)
			}
		}
	}

	return conv, nil
}

// Init initializes the layer, creating internal computation tensors.
func (c *Conv1D) Init(inputShape tensor.Shape) error {
	if c == nil {
		return fmt.Errorf("Conv1D.Init: nil layer")
	}

	// Validate input shape: [batch, inChannels, length]
	if len(inputShape) != 3 {
		return fmt.Errorf("Conv1D.Init: input must be 3D [batch, inChannels, length], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	if batchSize <= 0 {
		return fmt.Errorf("Conv1D.Init: batch size must be positive, got %d", batchSize)
	}

	inChannels := inputShape[1]
	if inChannels != c.inChannels {
		return fmt.Errorf("Conv1D.Init: input channels %d don't match layer inChannels %d", inChannels, c.inChannels)
	}

	length := inputShape[2]

	// Compute output shape: [batch, outChannels, outLen]
	outLen := (length+2*c.pad-c.kernelLen)/c.stride + 1

	if outLen <= 0 {
		return fmt.Errorf("Conv1D.Init: invalid output length %d (input length %d, kernel %d, pad %d, stride %d)",
			outLen, length, c.kernelLen, c.pad, c.stride)
	}

	outputShape := tensor.NewShape(batchSize, c.outChannels, outLen)
	outputSize := batchSize * c.outChannels * outLen
	c.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass using tensor.Conv1D.
func (c *Conv1D) Forward(input tensorTypes.Tensor) (tensorTypes.Tensor, error) {
	if c == nil {
		return nil, fmt.Errorf("Conv1D.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Conv1D.Forward: empty input")
	}

	// Store input
	c.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := c.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Conv1D.Forward: output not allocated, must call Init first")
	}

	// Compute convolution using tensor.Conv1DTo with pre-allocated output
	kernelParam, ok := c.Base.Parameter(types.ParamKernels)
	if !ok {
		return nil, fmt.Errorf("Conv1D.Forward: kernel parameter not initialized")
	}
	var biasTensor tensorTypes.Tensor
	if c.hasBias {
		biasParamVal, ok := c.Base.Parameter(types.ParamBiases)
		if ok {
			biasTensor = biasParamVal.Data
		}
	}
	output = input.Conv1DTo(kernelParam.Data, biasTensor, output, c.stride, c.pad)

	// Store output
	c.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input, weight, and bias.
func (c *Conv1D) Backward(gradOutput tensorTypes.Tensor) (tensorTypes.Tensor, error) {
	if c == nil {
		return nil, fmt.Errorf("Conv1D.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Conv1D.Backward: empty gradOutput")
	}

	input := c.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Conv1D.Backward: input not stored, must call Forward first")
	}

	output := c.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Conv1D.Backward: output not stored, must call Forward first")
	}

	// Get kernel parameter
	kernelParam, ok := c.Base.Parameter(types.ParamKernels)
	if !ok {
		return nil, fmt.Errorf("Conv1D.Backward: kernel parameter not initialized")
	}

	// Compute bias gradient: sum gradOutput over spatial dimensions and batch
	if c.hasBias && c.Base.CanLearn() {
		biasParam, ok := c.Base.Parameter(types.ParamBiases)
		if ok && biasParam.RequiresGrad {
			if tensor.IsNil(biasParam.Grad) {
				// Use parameter's data type for gradient (matches layer's data type)
				biasParam.Grad = tensor.New(biasParam.Data.DataType(), tensor.NewShape(c.outChannels))
			}

			// Sum over batch and length dimensions for each output channel
			// gradOutput shape: [batch, outChannels, outLength]
			summed := gradOutput.Sum(0, 2) // Sum over batch, length -> [outChannels]
			// Copy summed values to bias gradient using optimized Tensor.Copy method
			biasParam.Grad.Copy(summed)
			c.Base.SetParam(types.ParamBiases, biasParam)
		}
	}

	// Compute kernel gradient using primitive composition
	if c.Base.CanLearn() && kernelParam.RequiresGrad {
		if tensor.IsNil(kernelParam.Grad) {
			// Use parameter's data type for gradient (matches layer's data type)
			kernelParam.Grad = tensor.New(kernelParam.Data.DataType(), kernelParam.Data.Shape())
		}

		// Conv1D kernel gradient computation using primitives
		// For 1D conv: output[b,oc,ol] = sum_ic,k: input[b,ic,il+k] * kernel[oc,ic,k]
		// Kernel gradient: dL/dkernel[oc,ic,k] = sum_b,ol: gradOutput[b,oc,ol] * input[b,ic,il+k]

		// Reshape for matrix operations:
		// input: [batch, inChannels, inLength] -> [batch*inChannels, inLength]
		// gradOutput: [batch, outChannels, outLength] -> [batch*outChannels, outLength]

		inputShape := input.Shape()
		gradOutputShape := gradOutput.Shape()
		batchSize := inputShape[0]
		inChannels := inputShape[1]
		outChannels := gradOutputShape[1]
		inLength := inputShape[2]
		outLength := gradOutputShape[2]

		// Flatten batch and channel dimensions for matrix operations
		inputFlat := input.Reshape([]int{batchSize * inChannels, inLength})
		gradOutputFlat := gradOutput.Reshape([]int{batchSize * outChannels, outLength})

		// Create toeplitz-like matrices for convolution
		// For each position, we need to correlate gradOutput with input patches
		// This is equivalent to: kernelGrad = gradOutput_patches^T @ input_patches

		// Create the kernel gradient by computing correlations
		// Result shape: [outChannels, inChannels * kernelLen]
		// Use layer's data type for intermediate tensor
		dtype := c.Base.DataType()
		kernelGradFlat := tensor.New(dtype, tensor.NewShape(outChannels, inChannels*c.kernelLen))

		// For each output channel and kernel position, compute the correlation
		for oc := 0; oc < outChannels; oc++ {
			for ic := 0; ic < inChannels; ic++ {
				for k := 0; k < c.kernelLen; k++ {
					sum := float32(0.0)

					// Correlate gradOutput[oc] with input[ic] at each valid position
					for b := 0; b < batchSize; b++ {
						for ol := 0; ol < outLength; ol++ {
							// Corresponding input position: ol*stride + k - pad
							il := ol*c.stride + k - c.pad
							if il >= 0 && il < inLength {
								gradIdx := b*outChannels + oc
								inputIdx := b*inChannels + ic
								// At() returns float64, convert to float32 for arithmetic
								sum += float32(gradOutputFlat.At(gradIdx, ol) * inputFlat.At(inputIdx, il))
							}
						}
					}

					// Set the value using SetAt with computed multi-dimensional indices
					// For a 2D tensor [outChannels, inChannels*kernelLen], indices are [oc, ic*kernelLen + k]
					// Convert float32 to float64 for SetAt interface
					kernelGradFlat.SetAt(float64(sum), oc, ic*c.kernelLen+k)
				}
			}
		}

		// Reshape to final kernel shape: [outChannels, inChannels, kernelLen]
		kernelGradReshaped := kernelGradFlat.Reshape(tensor.NewShape(c.outChannels, c.inChannels, c.kernelLen))
		// Copy using optimized Tensor.Copy method
		kernelParam.Grad.Copy(kernelGradReshaped)
		c.Base.SetParam(types.ParamKernels, kernelParam)
	}

	// Compute input gradient using transposed convolution
	// Input gradient = Conv2DTransposed(gradOutput_4D, kernel_4D, stride, padding)
	// Use Conv2DTransposed with width=1 dimension, similar to how Conv1D uses Conv2D
	gradOutputShape := gradOutput.Shape()
	batchSize := gradOutputShape[0]

	// Reshape gradOutput from [batch, outChannels, outLength] to [batch, outChannels, outLength, 1]
	gradOutput4D := gradOutput.Reshape(tensor.NewShape(batchSize, c.outChannels, gradOutputShape[2], 1))

	// Reshape kernel from [outChannels, inChannels, kernelLen] to [outChannels, inChannels, kernelLen, 1]
	kernel4D := kernelParam.Data.Reshape(tensor.NewShape(c.outChannels, c.inChannels, c.kernelLen, 1))

	// Use Conv2DTransposed with width=1
	var emptyBias tensorTypes.Tensor
	gradInput4D := gradOutput4D.Conv2DTransposed(kernel4D, emptyBias, []int{c.stride, 1}, []int{c.pad, 0})

	// Reshape back to 3D: [batch, inChannels, inLength, 1] -> [batch, inChannels, inLength]
	gradInputShape := gradInput4D.Shape()
	gradInput := gradInput4D.Reshape(tensor.NewShape(gradInputShape[0], gradInputShape[1], gradInputShape[2]))

	c.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (c *Conv1D) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if c == nil {
		return nil, fmt.Errorf("Conv1D.OutputShape: nil layer")
	}

	if len(inputShape) != 3 {
		return nil, fmt.Errorf("Conv1D.OutputShape: input must be 3D [batch, inChannels, length], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	inChannels := inputShape[1]
	length := inputShape[2]

	if inChannels != c.inChannels {
		return nil, fmt.Errorf("Conv1D.OutputShape: input channels %d don't match layer inChannels %d", inChannels, c.inChannels)
	}

	outLen := (length+2*c.pad-c.kernelLen)/c.stride + 1

	return tensor.NewShape(batchSize, c.outChannels, outLen), nil
}

// Weight returns the kernel parameter tensor.
func (c *Conv1D) Weight() tensorTypes.Tensor {
	if c == nil {
		// Return empty tensor instead of nil to match test expectations
		return tensor.Empty(tensor.DTFP32)
	}
	kernelParam := c.Base.Kernels()
	if tensor.IsNil(kernelParam.Data) {
		return tensor.Empty(tensor.DTFP32)
	}
	return kernelParam.Data
}

// Bias returns the bias parameter tensor.
func (c *Conv1D) Bias() tensorTypes.Tensor {
	if c == nil || !c.hasBias {
		// Return empty tensor instead of nil to match test expectations
		return tensor.Empty(tensor.DTFP32)
	}
	biasParam := c.Base.Biases()
	if tensor.IsNil(biasParam.Data) {
		return tensor.Empty(tensor.DTFP32)
	}
	return biasParam.Data
}

// SetWeight sets the kernel parameter tensor.
func (c *Conv1D) SetWeight(weight tensorTypes.Tensor) error {
	if c == nil {
		return fmt.Errorf("Conv1D.SetWeight: nil layer")
	}
	if tensor.IsNil(weight) {
		return fmt.Errorf("Conv1D.SetWeight: empty weight tensor")
	}
	// Validate shape matches expected
	weightShape := weight.Shape()
	if weightShape.Rank() != 3 {
		return fmt.Errorf("Conv1D.SetWeight: weight must be 3D, got %dD", weightShape.Rank())
	}
	expectedShape := tensor.NewShape(c.outChannels, c.inChannels, c.kernelLen)
	for i, dim := range expectedShape {
		if i >= len(weightShape) || weightShape[i] != dim {
			return fmt.Errorf("Conv1D.SetWeight: weight shape %v doesn't match expected %v",
				weightShape, expectedShape)
		}
	}
	kernelParam, ok := c.Base.Parameter(types.ParamKernels)
	if !ok {
		return fmt.Errorf("Conv1D.SetWeight: kernel parameter not initialized")
	}
	kernelParam.Data = weight
	c.Base.SetParam(types.ParamKernels, kernelParam)
	return nil
}

// SetBias sets the bias parameter tensor.
func (c *Conv1D) SetBias(bias tensorTypes.Tensor) error {
	if c == nil {
		return fmt.Errorf("Conv1D.SetBias: nil layer")
	}
	if !c.hasBias {
		return fmt.Errorf("Conv1D.SetBias: layer has no bias")
	}
	if tensor.IsNil(bias) {
		return fmt.Errorf("Conv1D.SetBias: empty bias tensor")
	}
	// Validate shape matches expected
	biasShape := bias.Shape()
	if biasShape.Rank() != 1 || biasShape[0] != c.outChannels {
		return fmt.Errorf("Conv1D.SetBias: bias shape %v doesn't match expected [%d]",
			biasShape, c.outChannels)
	}
	biasParam, ok := c.Base.Parameter(types.ParamBiases)
	if !ok {
		return fmt.Errorf("Conv1D.SetBias: bias parameter not initialized")
	}
	biasParam.Data = bias
	c.Base.SetParam(types.ParamBiases, biasParam)
	return nil
}

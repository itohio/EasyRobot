package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
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

	base := NewBase("conv1d", opts...)
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

	// Create kernel parameter: [outChannels, inChannels, kernelLen]
	kernelData := *tensor.New(tensor.DTFP32, tensor.NewShape(outChannels, inChannels, kernelLen))
	conv.Base.SetParam(ParamKernels, Parameter{
		Data:         kernelData,
		RequiresGrad: conv.Base.CanLearn(),
	})

	// Create bias parameter if needed: [outChannels]
	if conv.hasBias {
		biasData := *tensor.New(tensor.DTFP32, tensor.NewShape(outChannels))
		conv.Base.SetParam(ParamBiases, Parameter{
			Data:         biasData,
			RequiresGrad: conv.Base.CanLearn(),
		})
	}

	return conv, nil
}

// Init initializes the layer, creating internal computation tensors.
func (c *Conv1D) Init(inputShape []int) error {
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

	outputShape := []int{batchSize, c.outChannels, outLen}
	outputSize := batchSize * c.outChannels * outLen
	c.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass using tensor.Conv1D.
func (c *Conv1D) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if c == nil {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Forward: nil layer")
	}

	if input.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Forward: empty input")
	}

	// Store input
	c.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := c.Base.Output()
	if output.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Forward: output not allocated, must call Init first")
	}

	// Compute convolution using tensor.Conv1DTo with pre-allocated output
	kernelParam, ok := c.Base.Parameter(ParamKernels)
	if !ok {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Forward: kernel parameter not initialized")
	}
	var biasParam *tensor.Tensor
	if c.hasBias {
		biasParamVal, ok := c.Base.Parameter(ParamBiases)
		if ok {
			biasParam = &biasParamVal.Data
		}
	}

	input.Conv1DTo(&kernelParam.Data, biasParam, &output, c.stride, c.pad)

	// Store output
	c.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input, weight, and bias.
func (c *Conv1D) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if c == nil {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Backward: nil layer")
	}

	if gradOutput.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Backward: empty gradOutput")
	}

	input := c.Base.Input()
	if input.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Backward: input not stored, must call Forward first")
	}

	output := c.Base.Output()
	if output.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Backward: output not stored, must call Forward first")
	}

	// Get kernel parameter
	kernelParam, ok := c.Base.Parameter(ParamKernels)
	if !ok {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Backward: kernel parameter not initialized")
	}

	// Compute bias gradient: sum gradOutput over spatial dimensions and batch
	if c.hasBias && c.Base.CanLearn() {
		biasParam, ok := c.Base.Parameter(ParamBiases)
		if ok && biasParam.RequiresGrad {
			if biasParam.Grad.Shape().Rank() == 0 {
				biasParam.Grad = *tensor.New(tensor.DTFP32, tensor.NewShape(c.outChannels))
			}

			// Sum over batch and length dimensions for each output channel
			// gradOutput shape: [batch, outChannels, outLength]
			summed := gradOutput.Sum(0, 2) // Sum over batch, length -> [outChannels]
			copy(biasParam.Grad.Data(), summed.Data())
			c.Base.SetParam(ParamBiases, biasParam)
		}
	}

	// Compute kernel gradient efficiently using Tensor API
	if c.Base.CanLearn() && kernelParam.RequiresGrad {
		if kernelParam.Grad.Shape().Rank() == 0 {
			kernelParam.Grad = *tensor.New(tensor.DTFP32, kernelParam.Data.Shape())
		}
		// Compute kernel gradients using efficient Conv1DKernelGrad
		kernelGrad := input.Conv1DKernelGrad(&gradOutput, &kernelParam.Data, c.stride, c.pad)
		copy(kernelParam.Grad.Data(), kernelGrad.Data())
		c.Base.SetParam(ParamKernels, kernelParam)
	}

	// Compute input gradient using transposed convolution
	// The forward kernel [outChannels, inChannels, kernelLen] is used as-is for transposed conv
	// Conv1DTransposed internally handles the reshaping
	gradInput := gradOutput.Conv1DTransposed(&kernelParam.Data, nil, c.stride, c.pad)

	c.Base.StoreGrad(*gradInput)
	return *gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (c *Conv1D) OutputShape(inputShape []int) ([]int, error) {
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

	return []int{batchSize, c.outChannels, outLen}, nil
}

// Weight returns the kernel parameter tensor.
func (c *Conv1D) Weight() tensor.Tensor {
	if c == nil {
		return tensor.Tensor{}
	}
	return c.Base.Kernels().Data
}

// Bias returns the bias parameter tensor.
func (c *Conv1D) Bias() tensor.Tensor {
	if c == nil || !c.hasBias {
		return tensor.Tensor{}
	}
	return c.Base.Biases().Data
}

// SetWeight sets the kernel parameter tensor.
func (c *Conv1D) SetWeight(weight tensor.Tensor) error {
	if c == nil {
		return fmt.Errorf("Conv1D.SetWeight: nil layer")
	}
	if weight.Shape().Rank() == 0 {
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
	kernelParam, ok := c.Base.Parameter(ParamKernels)
	if !ok {
		return fmt.Errorf("Conv1D.SetWeight: kernel parameter not initialized")
	}
	kernelParam.Data = weight
	c.Base.SetParam(ParamKernels, kernelParam)
	return nil
}

// SetBias sets the bias parameter tensor.
func (c *Conv1D) SetBias(bias tensor.Tensor) error {
	if c == nil {
		return fmt.Errorf("Conv1D.SetBias: nil layer")
	}
	if !c.hasBias {
		return fmt.Errorf("Conv1D.SetBias: layer has no bias")
	}
	if bias.Shape().Rank() == 0 {
		return fmt.Errorf("Conv1D.SetBias: empty bias tensor")
	}
	// Validate shape matches expected
	biasShape := bias.Shape()
	if biasShape.Rank() != 1 || biasShape[0] != c.outChannels {
		return fmt.Errorf("Conv1D.SetBias: bias shape %v doesn't match expected [%d]",
			biasShape, c.outChannels)
	}
	biasParam, ok := c.Base.Parameter(ParamBiases)
	if !ok {
		return fmt.Errorf("Conv1D.SetBias: bias parameter not initialized")
	}
	biasParam.Data = bias
	c.Base.SetParam(ParamBiases, biasParam)
	return nil
}

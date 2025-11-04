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

	// Compute convolution using tensor.Conv1D
	// Note: tensor.Conv1D expects weight shape [outChannels, inChannels, kernelLen]
	// which matches our parameter shape
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

	result := input.Conv1D(&kernelParam.Data, biasParam, c.stride, c.pad)

	// Copy result to pre-allocated output
	if len(result.Data()) != len(output.Data()) {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Forward: result size %d doesn't match output size %d",
			len(result.Data()), len(output.Data()))
	}
	copy(output.Data(), result.Data())

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

	// Get shapes
	inputShape := input.Shape()
	gradOutputShape := gradOutput.Shape()
	batchSize := inputShape[0]
	inLength := inputShape[2]
	outLength := gradOutputShape[2]

	// Get kernel parameter
	kernelParam, ok := c.Base.Parameter(ParamKernels)
	if !ok {
		return tensor.Tensor{}, fmt.Errorf("Conv1D.Backward: kernel parameter not initialized")
	}

	// Compute kernel gradient: sum over batch of (correlation of input with gradOutput)
	if c.Base.CanLearn() && kernelParam.RequiresGrad {
		if kernelParam.Grad.Shape().Rank() == 0 {
			kernelParam.Grad = *tensor.New(tensor.DTFP32, tensor.NewShape(c.outChannels, c.inChannels, c.kernelLen))
		}

		// For each output channel, input channel, and kernel position
		for oc := 0; oc < c.outChannels; oc++ {
			for ic := 0; ic < c.inChannels; ic++ {
				for k := 0; k < c.kernelLen; k++ {
					weightGradIdx := oc*c.inChannels*c.kernelLen + ic*c.kernelLen + k

					// Accumulate gradient over batch and output positions
					var gradSum float32
					for b := 0; b < batchSize; b++ {
						inputChanOffset := b*c.inChannels*inLength + ic*inLength
						gradChanOffset := b*c.outChannels*outLength + oc*outLength

						for outPos := 0; outPos < outLength; outPos++ {
							// Calculate corresponding input position
							inPos := outPos*c.stride + k - c.pad

							if inPos >= 0 && inPos < inLength {
								inputIdx := inputChanOffset + inPos
								gradIdx := gradChanOffset + outPos
								if inputIdx < len(input.Data()) && gradIdx < len(gradOutput.Data()) {
									gradSum += input.Data()[inputIdx] * gradOutput.Data()[gradIdx]
								}
							}
						}
					}
					kernelParam.Grad.Data()[weightGradIdx] += gradSum
				}
			}
		}
		c.Base.SetParam(ParamKernels, kernelParam)
	}

	// Compute bias gradient: sum gradOutput over spatial dimensions and batch
	if c.hasBias && c.Base.CanLearn() {
		biasParam, ok := c.Base.Parameter(ParamBiases)
		if ok && biasParam.RequiresGrad {
			if biasParam.Grad.Shape().Rank() == 0 {
				biasParam.Grad = *tensor.New(tensor.DTFP32, tensor.NewShape(c.outChannels))
			}

			for oc := 0; oc < c.outChannels; oc++ {
				var gradSum float32
				for b := 0; b < batchSize; b++ {
					gradChanOffset := b*c.outChannels*outLength + oc*outLength
					for outPos := 0; outPos < outLength; outPos++ {
						gradIdx := gradChanOffset + outPos
						if gradIdx < len(gradOutput.Data()) {
							gradSum += gradOutput.Data()[gradIdx]
						}
					}
				}
				biasParam.Grad.Data()[oc] += gradSum
			}
			c.Base.SetParam(ParamBiases, biasParam)
		}
	}

	// Compute input gradient: transposed convolution of gradOutput with flipped weights
	gradInput := *tensor.New(tensor.DTFP32, tensor.NewShape(batchSize, c.inChannels, inLength))

	// For each input channel, accumulate gradients from all output channels
	for ic := 0; ic < c.inChannels; ic++ {
		for oc := 0; oc < c.outChannels; oc++ {
			// Get weight kernel: [outChannels, inChannels, kernelLen]
			weightOffset := oc*c.inChannels*c.kernelLen + ic*c.kernelLen

			for b := 0; b < batchSize; b++ {
				inputChanOffset := b*c.inChannels*inLength + ic*inLength
				gradChanOffset := b*c.outChannels*outLength + oc*outLength

				for outPos := 0; outPos < outLength; outPos++ {
					gradIdx := gradChanOffset + outPos
					if gradIdx >= len(gradOutput.Data()) {
						continue // Skip if out of bounds (shouldn't happen with correct shapes)
					}
					gradVal := gradOutput.Data()[gradIdx]

					// For each kernel position, apply to input gradient
					for k := 0; k < c.kernelLen; k++ {
						// Calculate input position (reverse of forward)
						inPos := outPos*c.stride + k - c.pad

						if inPos >= 0 && inPos < inLength {
							// Weight is flipped: weight[kernelLen-1-k]
							flippedK := c.kernelLen - 1 - k
							weightIdx := weightOffset + flippedK
							inputIdx := inputChanOffset + inPos
							gradInput.Data()[inputIdx] += gradVal * kernelParam.Data.Data()[weightIdx]
						}
					}
				}
			}
		}
	}

	c.Base.StoreGrad(gradInput)
	return gradInput, nil
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

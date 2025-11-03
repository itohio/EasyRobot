package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Conv2D parameter indices
const (
	CPARAM_WEIGHT = 0
	CPARAM_BIAS   = 1
)

// Conv2D represents a 2D convolution layer.
type Conv2D struct {
	*Base
	inChannels  int
	outChannels int
	kernelH     int
	kernelW     int
	strideH     int
	strideW     int
	padH        int
	padW        int
	hasBias     bool
}

// Conv2DOption represents an option for configuring a Conv2D layer.
type Conv2DOption func(*Conv2D)

// WithConv2DName sets the name of the Conv2D layer.
func WithConv2DName(name string) Conv2DOption {
	return func(c *Conv2D) {
		if c.Base != nil {
			c.Base.SetName(name)
		}
	}
}

// WithConv2DBias sets whether to use a bias term.
func WithConv2DBias(useBias bool) Conv2DOption {
	return func(c *Conv2D) {
		c.hasBias = useBias
	}
}

// WithConv2DCanLearn sets whether this layer computes gradients.
func WithConv2DCanLearn(canLearn bool) Conv2DOption {
	return func(c *Conv2D) {
		if c.Base != nil {
			c.Base.SetCanLearn(canLearn)
		}
	}
}

// NewConv2D creates a new Conv2D layer.
func NewConv2D(
	inChannels, outChannels, kernelH, kernelW, strideH, strideW, padH, padW int,
	opts ...Conv2DOption,
) (*Conv2D, error) {
	if inChannels <= 0 {
		return nil, fmt.Errorf("Conv2D: inChannels must be positive, got %d", inChannels)
	}
	if outChannels <= 0 {
		return nil, fmt.Errorf("Conv2D: outChannels must be positive, got %d", outChannels)
	}
	if kernelH <= 0 {
		return nil, fmt.Errorf("Conv2D: kernelH must be positive, got %d", kernelH)
	}
	if kernelW <= 0 {
		return nil, fmt.Errorf("Conv2D: kernelW must be positive, got %d", kernelW)
	}
	if strideH <= 0 {
		return nil, fmt.Errorf("Conv2D: strideH must be positive, got %d", strideH)
	}
	if strideW <= 0 {
		return nil, fmt.Errorf("Conv2D: strideW must be positive, got %d", strideW)
	}

	conv := &Conv2D{
		Base:        NewBase(""),
		inChannels:  inChannels,
		outChannels: outChannels,
		kernelH:     kernelH,
		kernelW:     kernelW,
		strideH:     strideH,
		strideW:     strideW,
		padH:        padH,
		padW:        padW,
		hasBias:     true, // Default to having bias
	}

	// Apply options before initializing parameters
	for _, opt := range opts {
		opt(conv)
	}

	// Initialize Base parameters based on hasBias flag
	numParams := 1 // weight
	if conv.hasBias {
		numParams = 2 // weight + bias
	}
	conv.Base.InitParams(numParams)

	// Create weight parameter: [outChannels, inChannels, kernelH, kernelW]
	weightSize := outChannels * inChannels * kernelH * kernelW
	weightData := tensor.Tensor{
		Dim:  []int{outChannels, inChannels, kernelH, kernelW},
		Data: make([]float32, weightSize),
	}
	conv.Base.Parameter(CPARAM_WEIGHT).Data = weightData
	conv.Base.Parameter(CPARAM_WEIGHT).RequiresGrad = conv.Base.CanLearn()

	// Create bias parameter if needed: [outChannels]
	if conv.hasBias {
		biasData := tensor.Tensor{
			Dim:  []int{outChannels},
			Data: make([]float32, outChannels),
		}
		conv.Base.Parameter(CPARAM_BIAS).Data = biasData
		conv.Base.Parameter(CPARAM_BIAS).RequiresGrad = conv.Base.CanLearn()
	}

	return conv, nil
}

// Init initializes the layer, creating internal computation tensors.
func (c *Conv2D) Init(inputShape []int) error {
	if c == nil {
		return fmt.Errorf("Conv2D.Init: nil layer")
	}

	// Validate input shape: [batch, inChannels, height, width]
	if len(inputShape) != 4 {
		return fmt.Errorf("Conv2D.Init: input must be 4D [batch, inChannels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	if batchSize <= 0 {
		return fmt.Errorf("Conv2D.Init: batch size must be positive, got %d", batchSize)
	}

	inChannels := inputShape[1]
	if inChannels != c.inChannels {
		return fmt.Errorf("Conv2D.Init: input channels %d don't match layer inChannels %d", inChannels, c.inChannels)
	}

	inHeight := inputShape[2]
	inWidth := inputShape[3]

	// Compute output shape: [batch, outChannels, outHeight, outWidth]
	outHeight := (inHeight+2*c.padH-c.kernelH)/c.strideH + 1
	outWidth := (inWidth+2*c.padW-c.kernelW)/c.strideW + 1

	if outHeight <= 0 {
		return fmt.Errorf("Conv2D.Init: invalid output height %d (input height %d, kernel %d, pad %d, stride %d)",
			outHeight, inHeight, c.kernelH, c.padH, c.strideH)
	}
	if outWidth <= 0 {
		return fmt.Errorf("Conv2D.Init: invalid output width %d (input width %d, kernel %d, pad %d, stride %d)",
			outWidth, inWidth, c.kernelW, c.padW, c.strideW)
	}

	outputShape := []int{batchSize, c.outChannels, outHeight, outWidth}
	outputSize := batchSize * c.outChannels * outHeight * outWidth
	c.Base.AllocOutput(outputShape, outputSize)

	// Set default name if not explicitly set
	c.Base.SetDefaultName("Conv2D", outputShape)

	return nil
}

// Forward computes the forward pass using tensor.Conv2D.
func (c *Conv2D) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if c == nil {
		return tensor.Tensor{}, fmt.Errorf("Conv2D.Forward: nil layer")
	}

	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Conv2D.Forward: empty input")
	}

	// Store input
	c.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := c.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Conv2D.Forward: output not allocated, must call Init first")
	}

	// Compute convolution using tensor.Conv2D
	// Note: tensor.Conv2D expects weight shape [outChannels, inChannels, kernelH, kernelW]
	// which matches our parameter shape
	weightParam := c.Base.Parameter(CPARAM_WEIGHT)
	var biasParam *tensor.Tensor
	if c.hasBias {
		biasParam = &c.Base.Parameter(CPARAM_BIAS).Data
	}

	result := input.Conv2D(&weightParam.Data, biasParam, []int{c.strideH, c.strideW}, []int{c.padH, c.padW})

	// Copy result to pre-allocated output
	if len(result.Data) != len(output.Data) {
		return tensor.Tensor{}, fmt.Errorf("Conv2D.Forward: result size %d doesn't match output size %d",
			len(result.Data), len(output.Data))
	}
	copy(output.Data, result.Data)

	// Store output
	c.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input, weight, and bias.
func (c *Conv2D) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if c == nil {
		return tensor.Tensor{}, fmt.Errorf("Conv2D.Backward: nil layer")
	}

	if len(gradOutput.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Conv2D.Backward: empty gradOutput")
	}

	input := c.Base.Input()
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Conv2D.Backward: input not stored, must call Forward first")
	}

	output := c.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Conv2D.Backward: output not stored, must call Forward first")
	}

	// Get shapes
	inputShape := input.Shape()
	gradOutputShape := gradOutput.Shape()
	batchSize := inputShape[0]
	inHeight := inputShape[2]
	inWidth := inputShape[3]
	outHeight := gradOutputShape[2]
	outWidth := gradOutputShape[3]

	// Get weight parameter
	weightParam := c.Base.Parameter(CPARAM_WEIGHT)

	// Compute weight gradient: sum over batch of (correlation of input with gradOutput)
	if c.Base.CanLearn() && weightParam.RequiresGrad {
		if len(weightParam.Grad.Dim) == 0 {
			weightParam.Grad = tensor.Tensor{
				Dim:  []int{c.outChannels, c.inChannels, c.kernelH, c.kernelW},
				Data: make([]float32, c.outChannels*c.inChannels*c.kernelH*c.kernelW),
			}
		}

		// For each output channel, input channel, and kernel position
		for oc := 0; oc < c.outChannels; oc++ {
			for ic := 0; ic < c.inChannels; ic++ {
				for kh := 0; kh < c.kernelH; kh++ {
					for kw := 0; kw < c.kernelW; kw++ {
						weightGradIdx := oc*c.inChannels*c.kernelH*c.kernelW +
							ic*c.kernelH*c.kernelW + kh*c.kernelW + kw

						// Accumulate gradient over batch and output positions
						var gradSum float32
						for b := 0; b < batchSize; b++ {
							inputChanOffset := b*c.inChannels*inHeight*inWidth + ic*inHeight*inWidth
							gradChanOffset := b*c.outChannels*outHeight*outWidth + oc*outHeight*outWidth

							for outH := 0; outH < outHeight; outH++ {
								for outW := 0; outW < outWidth; outW++ {
									// Calculate corresponding input position
									inH := outH*c.strideH + kh - c.padH
									inW := outW*c.strideW + kw - c.padW

									if inH >= 0 && inH < inHeight && inW >= 0 && inW < inWidth {
										inputIdx := inputChanOffset + inH*inWidth + inW
										gradIdx := gradChanOffset + outH*outWidth + outW
										if inputIdx < len(input.Data) && gradIdx < len(gradOutput.Data) {
											gradSum += input.Data[inputIdx] * gradOutput.Data[gradIdx]
										}
									}
								}
							}
						}
						weightParam.Grad.Data[weightGradIdx] += gradSum
					}
				}
			}
		}
	}

	// Compute bias gradient: sum gradOutput over spatial dimensions and batch
	if c.hasBias && c.Base.CanLearn() {
		biasParam := c.Base.Parameter(CPARAM_BIAS)
		if biasParam.RequiresGrad {
			if len(biasParam.Grad.Dim) == 0 {
				biasParam.Grad = tensor.Tensor{
					Dim:  []int{c.outChannels},
					Data: make([]float32, c.outChannels),
				}
			}

			for oc := 0; oc < c.outChannels; oc++ {
				var gradSum float32
				for b := 0; b < batchSize; b++ {
					gradChanOffset := b*c.outChannels*outHeight*outWidth + oc*outHeight*outWidth
					for outH := 0; outH < outHeight; outH++ {
						for outW := 0; outW < outWidth; outW++ {
							gradIdx := gradChanOffset + outH*outWidth + outW
							if gradIdx < len(gradOutput.Data) {
								gradSum += gradOutput.Data[gradIdx]
							}
						}
					}
				}
				biasParam.Grad.Data[oc] += gradSum
			}
		}
	}

	// Compute input gradient: transposed convolution of gradOutput with flipped weights
	gradInput := tensor.Tensor{
		Dim:  []int{batchSize, c.inChannels, inHeight, inWidth},
		Data: make([]float32, batchSize*c.inChannels*inHeight*inWidth),
	}

	// For each input channel, accumulate gradients from all output channels
	for ic := 0; ic < c.inChannels; ic++ {
		for oc := 0; oc < c.outChannels; oc++ {
			// Get weight kernel: [outChannels, inChannels, kernelH, kernelW]
			weightOffset := oc*c.inChannels*c.kernelH*c.kernelW +
				ic*c.kernelH*c.kernelW

			for b := 0; b < batchSize; b++ {
				inputChanOffset := b*c.inChannels*inHeight*inWidth + ic*inHeight*inWidth
				gradChanOffset := b*c.outChannels*outHeight*outWidth + oc*outHeight*outWidth

				for outH := 0; outH < outHeight; outH++ {
					for outW := 0; outW < outWidth; outW++ {
						gradIdx := gradChanOffset + outH*outWidth + outW
						if gradIdx >= len(gradOutput.Data) {
							continue // Skip if out of bounds (shouldn't happen with correct shapes)
						}
						gradVal := gradOutput.Data[gradIdx]

						// For each kernel position, apply to input gradient
						for kh := 0; kh < c.kernelH; kh++ {
							for kw := 0; kw < c.kernelW; kw++ {
								// Calculate input position (reverse of forward)
								inH := outH*c.strideH + kh - c.padH
								inW := outW*c.strideW + kw - c.padW

								if inH >= 0 && inH < inHeight && inW >= 0 && inW < inWidth {
									// Weight is flipped: weight[kernelH-1-kh][kernelW-1-kw]
									flippedKh := c.kernelH - 1 - kh
									flippedKw := c.kernelW - 1 - kw
									weightIdx := weightOffset + flippedKh*c.kernelW + flippedKw
									inputIdx := inputChanOffset + inH*inWidth + inW
									gradInput.Data[inputIdx] += gradVal * weightParam.Data.Data[weightIdx]
								}
							}
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
func (c *Conv2D) OutputShape(inputShape []int) ([]int, error) {
	if c == nil {
		return nil, fmt.Errorf("Conv2D.OutputShape: nil layer")
	}

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("Conv2D.OutputShape: input must be 4D [batch, inChannels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	inChannels := inputShape[1]
	inHeight := inputShape[2]
	inWidth := inputShape[3]

	if inChannels != c.inChannels {
		return nil, fmt.Errorf("Conv2D.OutputShape: input channels %d don't match layer inChannels %d", inChannels, c.inChannels)
	}

	outHeight := (inHeight+2*c.padH-c.kernelH)/c.strideH + 1
	outWidth := (inWidth+2*c.padW-c.kernelW)/c.strideW + 1

	return []int{batchSize, c.outChannels, outHeight, outWidth}, nil
}

// Weight returns the weight parameter tensor.
func (c *Conv2D) Weight() tensor.Tensor {
	if c == nil {
		return tensor.Tensor{}
	}
	weightParam := c.Base.Parameter(CPARAM_WEIGHT)
	if weightParam == nil {
		return tensor.Tensor{}
	}
	return weightParam.Data
}

// Bias returns the bias parameter tensor.
func (c *Conv2D) Bias() tensor.Tensor {
	if c == nil || !c.hasBias {
		return tensor.Tensor{}
	}
	biasParam := c.Base.Parameter(CPARAM_BIAS)
	if biasParam == nil {
		return tensor.Tensor{}
	}
	return biasParam.Data
}

// SetWeight sets the weight parameter tensor.
func (c *Conv2D) SetWeight(weight tensor.Tensor) error {
	if c == nil {
		return fmt.Errorf("Conv2D.SetWeight: nil layer")
	}
	if len(weight.Dim) == 0 {
		return fmt.Errorf("Conv2D.SetWeight: empty weight tensor")
	}
	// Validate shape matches expected
	if len(weight.Dim) != 4 {
		return fmt.Errorf("Conv2D.SetWeight: weight must be 4D, got %dD", len(weight.Dim))
	}
	expectedShape := []int{c.outChannels, c.inChannels, c.kernelH, c.kernelW}
	for i, dim := range expectedShape {
		if i >= len(weight.Dim) || weight.Dim[i] != dim {
			return fmt.Errorf("Conv2D.SetWeight: weight shape %v doesn't match expected %v",
				weight.Dim, expectedShape)
		}
	}
	weightParam := c.Base.Parameter(CPARAM_WEIGHT)
	if weightParam == nil {
		return fmt.Errorf("Conv2D.SetWeight: weight parameter not initialized")
	}
	weightParam.Data = weight
	return nil
}

// SetBias sets the bias parameter tensor.
func (c *Conv2D) SetBias(bias tensor.Tensor) error {
	if c == nil {
		return fmt.Errorf("Conv2D.SetBias: nil layer")
	}
	if !c.hasBias {
		return fmt.Errorf("Conv2D.SetBias: layer has no bias")
	}
	if len(bias.Dim) == 0 {
		return fmt.Errorf("Conv2D.SetBias: empty bias tensor")
	}
	// Validate shape matches expected
	if len(bias.Dim) != 1 || bias.Dim[0] != c.outChannels {
		return fmt.Errorf("Conv2D.SetBias: bias shape %v doesn't match expected [%d]",
			bias.Dim, c.outChannels)
	}
	biasParam := c.Base.Parameter(CPARAM_BIAS)
	if biasParam == nil {
		return fmt.Errorf("Conv2D.SetBias: bias parameter not initialized")
	}
	biasParam.Data = bias
	return nil
}

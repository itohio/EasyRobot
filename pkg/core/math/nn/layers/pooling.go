package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// MaxPool2D represents a 2D max pooling layer.
type MaxPool2D struct {
	Base
	kernelH int
	kernelW int
	strideH int
	strideW int
	padH    int
	padW    int
}

// MaxPool2DOption represents an option for configuring a MaxPool2D layer.
type MaxPool2DOption func(*MaxPool2D)

// WithMaxPool2DName sets the name of the MaxPool2D layer.
func WithMaxPool2DName(name string) MaxPool2DOption {
	return func(m *MaxPool2D) {
		m.Base.SetName(name)
	}
}

// NewMaxPool2D creates a new MaxPool2D layer.
func NewMaxPool2D(kernelH, kernelW, strideH, strideW, padH, padW int) (*MaxPool2D, error) {
	if kernelH <= 0 {
		return nil, fmt.Errorf("MaxPool2D: kernelH must be positive, got %d", kernelH)
	}
	if kernelW <= 0 {
		return nil, fmt.Errorf("MaxPool2D: kernelW must be positive, got %d", kernelW)
	}
	if strideH <= 0 {
		return nil, fmt.Errorf("MaxPool2D: strideH must be positive, got %d", strideH)
	}
	if strideW <= 0 {
		return nil, fmt.Errorf("MaxPool2D: strideW must be positive, got %d", strideW)
	}

	return &MaxPool2D{
		Base:    NewBase("maxpool2d"),
		kernelH: kernelH,
		kernelW: kernelW,
		strideH: strideH,
		strideW: strideW,
		padH:    padH,
		padW:    padW,
	}, nil
}

// Init initializes the layer, creating internal computation tensors.
func (m *MaxPool2D) Init(inputShape []int) error {
	if m == nil {
		return fmt.Errorf("MaxPool2D.Init: nil layer")
	}

	// Validate input shape: [batch, channels, height, width]
	if len(inputShape) != 4 {
		return fmt.Errorf("MaxPool2D.Init: input must be 4D [batch, channels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	channels := inputShape[1]
	inHeight := inputShape[2]
	inWidth := inputShape[3]

	// Compute output shape: [batch, channels, outHeight, outWidth]
	outHeight := (inHeight+2*m.padH-m.kernelH)/m.strideH + 1
	outWidth := (inWidth+2*m.padW-m.kernelW)/m.strideW + 1

	if outHeight <= 0 {
		return fmt.Errorf("MaxPool2D.Init: invalid output height %d (input height %d, kernel %d, pad %d, stride %d)",
			outHeight, inHeight, m.kernelH, m.padH, m.strideH)
	}
	if outWidth <= 0 {
		return fmt.Errorf("MaxPool2D.Init: invalid output width %d (input width %d, kernel %d, pad %d, stride %d)",
			outWidth, inWidth, m.kernelW, m.padW, m.strideW)
	}

	outputShape := []int{batchSize, channels, outHeight, outWidth}
	outputSize := batchSize * channels * outHeight * outWidth
	m.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass using tensor.MaxPool2D.
func (m *MaxPool2D) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if m == nil {
		return tensor.Tensor{}, fmt.Errorf("MaxPool2D.Forward: nil layer")
	}

	if input.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("MaxPool2D.Forward: empty input")
	}

	// Store input
	m.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := m.Base.Output()
	if output.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("MaxPool2D.Forward: output not allocated, must call Init first")
	}

	// Compute max pooling using tensor.MaxPool2D
	result := input.MaxPool2D([]int{m.kernelH, m.kernelW}, []int{m.strideH, m.strideW}, []int{m.padH, m.padW})

	// Copy result to pre-allocated output
	if len(result.Data()) != len(output.Data()) {
		return tensor.Tensor{}, fmt.Errorf("MaxPool2D.Forward: result size %d doesn't match output size %d",
			len(result.Data()), len(output.Data()))
	}
	copy(output.Data(), result.Data())

	// Store output
	m.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
// For each output position, routes the gradient back to the input positions
// that produced the maximum value during forward pass.
// If multiple positions had the same max value, the gradient is divided equally among them.
func (m *MaxPool2D) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if m == nil {
		return tensor.Tensor{}, fmt.Errorf("MaxPool2D.Backward: nil layer")
	}

	if gradOutput.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("MaxPool2D.Backward: empty gradOutput")
	}

	input := m.Base.Input()
	if input.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("MaxPool2D.Backward: input not stored, must call Forward first")
	}

	output := m.Base.Output()
	if output.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("MaxPool2D.Backward: output not stored, must call Forward first")
	}

	inputShape := input.Shape()
	gradOutputShape := gradOutput.Shape()
	outputShape := output.Shape()

	// Validate shapes
	if len(inputShape) != 4 || len(gradOutputShape) != 4 || len(outputShape) != 4 {
		return tensor.Tensor{}, fmt.Errorf("MaxPool2D.Backward: expected 4D tensors, got input %v, gradOutput %v, output %v", inputShape, gradOutputShape, outputShape)
	}

	batchSize := inputShape[0]
	channels := inputShape[1]
	inHeight := inputShape[2]
	inWidth := inputShape[3]
	outHeight := outputShape[2]
	outWidth := outputShape[3]

	// Initialize gradient input with zeros
	gradInput := tensor.New(tensor.DTFP32, tensor.NewShape(batchSize, channels, inHeight, inWidth))
	inputData := input.Data()
	outputData := output.Data()
	gradOutputData := gradOutput.Data()
	gradInputData := gradInput.Data()

	// For each output position, route gradient back to input positions that produced the max
	for b := 0; b < batchSize; b++ {
		batchOffset := b * channels * inHeight * inWidth
		outputBatchOffset := b * channels * outHeight * outWidth
		gradOutputBatchOffset := b * channels * outHeight * outWidth

		for c := 0; c < channels; c++ {
			channelOffset := batchOffset + c*inHeight*inWidth
			outputChannelOffset := outputBatchOffset + c*outHeight*outWidth
			gradOutputChannelOffset := gradOutputBatchOffset + c*outHeight*outWidth

			for outH := 0; outH < outHeight; outH++ {
				for outW := 0; outW < outWidth; outW++ {
					// Calculate input window position
					startH := outH*m.strideH - m.padH
					startW := outW*m.strideW - m.padW

					// Get output value (max value from forward pass)
					outputIdx := outputChannelOffset + outH*outWidth + outW
					maxVal := outputData[outputIdx]

					// Get gradient for this output position
					gradOutputIdx := gradOutputChannelOffset + outH*outWidth + outW
					gradVal := gradOutputData[gradOutputIdx]

					// Count how many input positions had the max value
					maxCount := 0
					for kh := 0; kh < m.kernelH; kh++ {
						for kw := 0; kw < m.kernelW; kw++ {
							inH := startH + kh
							inW := startW + kw

							if inH >= 0 && inH < inHeight && inW >= 0 && inW < inWidth {
								inputIdx := channelOffset + inH*inWidth + inW
								// Use epsilon to handle floating point comparison
								epsilon := float32(1e-6)
								diff := inputData[inputIdx] - maxVal
								if diff > -epsilon && diff < epsilon {
									maxCount++
								}
							}
						}
					}

					// Route gradient equally to all positions that had the max value
					if maxCount > 0 {
						gradPerPosition := gradVal / float32(maxCount)
						for kh := 0; kh < m.kernelH; kh++ {
							for kw := 0; kw < m.kernelW; kw++ {
								inH := startH + kh
								inW := startW + kw

								if inH >= 0 && inH < inHeight && inW >= 0 && inW < inWidth {
									inputIdx := channelOffset + inH*inWidth + inW
									// Use epsilon to handle floating point comparison
									epsilon := float32(1e-6)
									diff := inputData[inputIdx] - maxVal
									if diff > -epsilon && diff < epsilon {
										gradInputData[inputIdx] += gradPerPosition
									}
								}
							}
						}
					}
				}
			}
		}
	}

	m.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (m *MaxPool2D) OutputShape(inputShape []int) ([]int, error) {
	if m == nil {
		return nil, fmt.Errorf("MaxPool2D.OutputShape: nil layer")
	}

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("MaxPool2D.OutputShape: input must be 4D [batch, channels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	channels := inputShape[1]
	inHeight := inputShape[2]
	inWidth := inputShape[3]

	outHeight := (inHeight+2*m.padH-m.kernelH)/m.strideH + 1
	outWidth := (inWidth+2*m.padW-m.kernelW)/m.strideW + 1

	return []int{batchSize, channels, outHeight, outWidth}, nil
}

// AvgPool2D represents a 2D average pooling layer.
type AvgPool2D struct {
	Base
	kernelH int
	kernelW int
	strideH int
	strideW int
	padH    int
	padW    int
}

// AvgPool2DOption represents an option for configuring an AvgPool2D layer.
type AvgPool2DOption func(*AvgPool2D)

// WithAvgPool2DName sets the name of the AvgPool2D layer.
func WithAvgPool2DName(name string) AvgPool2DOption {
	return func(a *AvgPool2D) {
		a.Base.SetName(name)
	}
}

// NewAvgPool2D creates a new AvgPool2D layer.
func NewAvgPool2D(kernelH, kernelW, strideH, strideW, padH, padW int) (*AvgPool2D, error) {
	if kernelH <= 0 {
		return nil, fmt.Errorf("AvgPool2D: kernelH must be positive, got %d", kernelH)
	}
	if kernelW <= 0 {
		return nil, fmt.Errorf("AvgPool2D: kernelW must be positive, got %d", kernelW)
	}
	if strideH <= 0 {
		return nil, fmt.Errorf("AvgPool2D: strideH must be positive, got %d", strideH)
	}
	if strideW <= 0 {
		return nil, fmt.Errorf("AvgPool2D: strideW must be positive, got %d", strideW)
	}

	return &AvgPool2D{
		Base:    NewBase("avgpool2d"),
		kernelH: kernelH,
		kernelW: kernelW,
		strideH: strideH,
		strideW: strideW,
		padH:    padH,
		padW:    padW,
	}, nil
}

// Init initializes the layer, creating internal computation tensors.
func (a *AvgPool2D) Init(inputShape []int) error {
	if a == nil {
		return fmt.Errorf("AvgPool2D.Init: nil layer")
	}

	// Validate input shape: [batch, channels, height, width]
	if len(inputShape) != 4 {
		return fmt.Errorf("AvgPool2D.Init: input must be 4D [batch, channels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	channels := inputShape[1]
	inHeight := inputShape[2]
	inWidth := inputShape[3]

	// Compute output shape: [batch, channels, outHeight, outWidth]
	outHeight := (inHeight+2*a.padH-a.kernelH)/a.strideH + 1
	outWidth := (inWidth+2*a.padW-a.kernelW)/a.strideW + 1

	if outHeight <= 0 {
		return fmt.Errorf("AvgPool2D.Init: invalid output height %d (input height %d, kernel %d, pad %d, stride %d)",
			outHeight, inHeight, a.kernelH, a.padH, a.strideH)
	}
	if outWidth <= 0 {
		return fmt.Errorf("AvgPool2D.Init: invalid output width %d (input width %d, kernel %d, pad %d, stride %d)",
			outWidth, inWidth, a.kernelW, a.padW, a.strideW)
	}

	outputShape := []int{batchSize, channels, outHeight, outWidth}
	outputSize := batchSize * channels * outHeight * outWidth
	a.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass using tensor.AvgPool2D.
func (a *AvgPool2D) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if a == nil {
		return tensor.Tensor{}, fmt.Errorf("AvgPool2D.Forward: nil layer")
	}

	if input.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("AvgPool2D.Forward: empty input")
	}

	// Store input
	a.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := a.Base.Output()
	if output.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("AvgPool2D.Forward: output not allocated, must call Init first")
	}

	// Compute average pooling using tensor.AvgPool2D
	result := input.AvgPool2D([]int{a.kernelH, a.kernelW}, []int{a.strideH, a.strideW}, []int{a.padH, a.padW})

	// Copy result to pre-allocated output
	if len(result.Data()) != len(output.Data()) {
		return tensor.Tensor{}, fmt.Errorf("AvgPool2D.Forward: result size %d doesn't match output size %d",
			len(result.Data()), len(output.Data()))
	}
	copy(output.Data(), result.Data())

	// Store output
	a.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
func (a *AvgPool2D) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if a == nil {
		return tensor.Tensor{}, fmt.Errorf("AvgPool2D.Backward: nil layer")
	}

	if gradOutput.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("AvgPool2D.Backward: empty gradOutput")
	}

	input := a.Base.Input()
	if input.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("AvgPool2D.Backward: input not stored, must call Forward first")
	}

	output := a.Base.Output()
	if output.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("AvgPool2D.Backward: output not stored, must call Forward first")
	}

	// For now, AvgPool2D backward is not implemented
	if a.Base.CanLearn() {
		return tensor.Tensor{}, fmt.Errorf("AvgPool2D.Backward: backward pass not yet implemented")
	}

	// For inference-only, we don't compute gradients
	gradInput := tensor.New(tensor.DTFP32, gradOutput.Shape())

	a.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (a *AvgPool2D) OutputShape(inputShape []int) ([]int, error) {
	if a == nil {
		return nil, fmt.Errorf("AvgPool2D.OutputShape: nil layer")
	}

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("AvgPool2D.OutputShape: input must be 4D [batch, channels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	channels := inputShape[1]
	inHeight := inputShape[2]
	inWidth := inputShape[3]

	outHeight := (inHeight+2*a.padH-a.kernelH)/a.strideH + 1
	outWidth := (inWidth+2*a.padW-a.kernelW)/a.strideW + 1

	return []int{batchSize, channels, outHeight, outWidth}, nil
}

// GlobalAvgPool2D represents a global average pooling layer.
type GlobalAvgPool2D struct {
	Base
}

// NewGlobalAvgPool2D creates a new GlobalAvgPool2D layer.
func NewGlobalAvgPool2D() *GlobalAvgPool2D {
	return &GlobalAvgPool2D{
		Base: NewBase("globalavgpool2d"),
	}
}

// Init initializes the layer, creating internal computation tensors.
func (g *GlobalAvgPool2D) Init(inputShape []int) error {
	if g == nil {
		return fmt.Errorf("GlobalAvgPool2D.Init: nil layer")
	}

	// Validate input shape: [batch, channels, height, width]
	if len(inputShape) != 4 {
		return fmt.Errorf("GlobalAvgPool2D.Init: input must be 4D [batch, channels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	channels := inputShape[1]

	// Output shape: [batch, channels]
	outputShape := []int{batchSize, channels}
	outputSize := batchSize * channels
	g.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass using tensor.GlobalAvgPool2D.
func (g *GlobalAvgPool2D) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if g == nil {
		return tensor.Tensor{}, fmt.Errorf("GlobalAvgPool2D.Forward: nil layer")
	}

	if input.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("GlobalAvgPool2D.Forward: empty input")
	}

	// Store input
	g.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := g.Base.Output()
	if output.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("GlobalAvgPool2D.Forward: output not allocated, must call Init first")
	}

	// Compute global average pooling using tensor.GlobalAvgPool2D
	result := input.GlobalAvgPool2D()

	// Copy result to pre-allocated output
	if len(result.Data()) != len(output.Data()) {
		return tensor.Tensor{}, fmt.Errorf("GlobalAvgPool2D.Forward: result size %d doesn't match output size %d",
			len(result.Data()), len(output.Data()))
	}
	copy(output.Data(), result.Data())

	// Store output
	g.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
func (g *GlobalAvgPool2D) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if g == nil {
		return tensor.Tensor{}, fmt.Errorf("GlobalAvgPool2D.Backward: nil layer")
	}

	if gradOutput.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("GlobalAvgPool2D.Backward: empty gradOutput")
	}

	input := g.Base.Input()
	if input.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("GlobalAvgPool2D.Backward: input not stored, must call Forward first")
	}

	output := g.Base.Output()
	if output.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("GlobalAvgPool2D.Backward: output not stored, must call Forward first")
	}

	// For now, GlobalAvgPool2D backward is not implemented
	if g.Base.CanLearn() {
		return tensor.Tensor{}, fmt.Errorf("GlobalAvgPool2D.Backward: backward pass not yet implemented")
	}

	// For inference-only, we don't compute gradients
	gradInput := tensor.New(tensor.DTFP32, gradOutput.Shape())

	g.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (g *GlobalAvgPool2D) OutputShape(inputShape []int) ([]int, error) {
	if g == nil {
		return nil, fmt.Errorf("GlobalAvgPool2D.OutputShape: nil layer")
	}

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("GlobalAvgPool2D.OutputShape: input must be 4D [batch, channels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	channels := inputShape[1]

	return []int{batchSize, channels}, nil
}

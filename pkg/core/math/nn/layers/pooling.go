package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
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
	indices types.Tensor // Stored indices from forward pass for efficient backward pass
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
func (m *MaxPool2D) Init(inputShape tensor.Shape) error {
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

	outputShape := tensor.NewShape(batchSize, channels, outHeight, outWidth)
	outputSize := batchSize * channels * outHeight * outWidth
	m.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass using tensor.MaxPool2DWithIndices.
func (m *MaxPool2D) Forward(input types.Tensor) (types.Tensor, error) {
	if m == nil {
		return nil, fmt.Errorf("MaxPool2D.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("MaxPool2D.Forward: empty input")
	}

	// Store input
	m.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := m.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("MaxPool2D.Forward: output not allocated, must call Init first")
	}

	// Compute max pooling with indices using tensor.MaxPool2DWithIndices, writing directly to output
	_, indices := input.MaxPool2DWithIndices(output, nil, []int{m.kernelH, m.kernelW}, []int{m.strideH, m.strideW}, []int{m.padH, m.padW})

	// Store indices for backward pass
	m.indices = indices

	// Store output
	m.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input using stored indices.
// Uses MaxPool2DBackward for efficient gradient computation.
func (m *MaxPool2D) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if m == nil {
		return nil, fmt.Errorf("MaxPool2D.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("MaxPool2D.Backward: empty gradOutput")
	}

	input := m.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("MaxPool2D.Backward: input not stored, must call Forward first")
	}

	if tensor.IsNil(m.indices) {
		return nil, fmt.Errorf("MaxPool2D.Backward: indices not stored, must call Forward first")
	}

	// Get pre-allocated grad input tensor if available, otherwise create new
	gradInput := m.Base.Grad()
	if tensor.IsNil(gradInput) {
		// No pre-allocated grad input, create new one
		gradInput = input.MaxPool2DBackward(nil, gradOutput, m.indices, []int{m.kernelH, m.kernelW}, []int{m.strideH, m.strideW}, []int{m.padH, m.padW})
	} else {
		// Use pre-allocated grad input
		gradInput = input.MaxPool2DBackward(gradInput, gradOutput, m.indices, []int{m.kernelH, m.kernelW}, []int{m.strideH, m.strideW}, []int{m.padH, m.padW})
	}

	m.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (m *MaxPool2D) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
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

	return tensor.NewShape(batchSize, channels, outHeight, outWidth), nil
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
func (a *AvgPool2D) Init(inputShape tensor.Shape) error {
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

	outputShape := tensor.NewShape(batchSize, channels, outHeight, outWidth)
	outputSize := batchSize * channels * outHeight * outWidth
	a.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass using tensor.AvgPool2D.
func (a *AvgPool2D) Forward(input types.Tensor) (types.Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("AvgPool2D.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("AvgPool2D.Forward: empty input")
	}

	// Store input
	a.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := a.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("AvgPool2D.Forward: output not allocated, must call Init first")
	}

	// Compute average pooling using tensor.AvgPool2D, writing directly to output
	_ = input.AvgPool2D(output, []int{a.kernelH, a.kernelW}, []int{a.strideH, a.strideW}, []int{a.padH, a.padW})

	// Store output
	a.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input using AvgPool2DBackward.
func (a *AvgPool2D) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if a == nil {
		return nil, fmt.Errorf("AvgPool2D.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("AvgPool2D.Backward: empty gradOutput")
	}

	input := a.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("AvgPool2D.Backward: input not stored, must call Forward first")
	}

	// Get pre-allocated grad input tensor if available, otherwise create new
	gradInput := a.Base.Grad()
	if tensor.IsNil(gradInput) {
		// No pre-allocated grad input, create new one
		gradInput = input.AvgPool2DBackward(nil, gradOutput, []int{a.kernelH, a.kernelW}, []int{a.strideH, a.strideW}, []int{a.padH, a.padW})
	} else {
		// Use pre-allocated grad input
		gradInput = input.AvgPool2DBackward(gradInput, gradOutput, []int{a.kernelH, a.kernelW}, []int{a.strideH, a.strideW}, []int{a.padH, a.padW})
	}

	a.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (a *AvgPool2D) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
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

	return tensor.NewShape(batchSize, channels, outHeight, outWidth), nil
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
func (g *GlobalAvgPool2D) Init(inputShape tensor.Shape) error {
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
	outputShape := tensor.NewShape(batchSize, channels)
	outputSize := batchSize * channels
	g.Base.AllocOutput(outputShape, outputSize)

	return nil
}

// Forward computes the forward pass using tensor.GlobalAvgPool2D.
func (g *GlobalAvgPool2D) Forward(input types.Tensor) (types.Tensor, error) {
	if g == nil {
		return nil, fmt.Errorf("GlobalAvgPool2D.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("GlobalAvgPool2D.Forward: empty input")
	}

	// Store input
	g.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := g.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("GlobalAvgPool2D.Forward: output not allocated, must call Init first")
	}

	// Compute global average pooling using tensor.GlobalAvgPool2D, writing directly to output
	_ = input.GlobalAvgPool2D(output)

	// Store output
	g.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
func (g *GlobalAvgPool2D) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if g == nil {
		return nil, fmt.Errorf("GlobalAvgPool2D.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("GlobalAvgPool2D.Backward: empty gradOutput")
	}

	input := g.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("GlobalAvgPool2D.Backward: input not stored, must call Forward first")
	}

	output := g.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("GlobalAvgPool2D.Backward: output not stored, must call Forward first")
	}

	// For now, GlobalAvgPool2D backward is not implemented
	if g.Base.CanLearn() {
		return nil, fmt.Errorf("GlobalAvgPool2D.Backward: backward pass not yet implemented")
	}

	// For inference-only, we don't compute gradients
	// Use gradOutput's data type for input gradient (for correctness in backward pass)
	gradInput := tensor.New(gradOutput.DataType(), gradOutput.Shape())

	g.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (g *GlobalAvgPool2D) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if g == nil {
		return nil, fmt.Errorf("GlobalAvgPool2D.OutputShape: nil layer")
	}

	if len(inputShape) != 4 {
		return nil, fmt.Errorf("GlobalAvgPool2D.OutputShape: input must be 4D [batch, channels, height, width], got %dD: %v", len(inputShape), inputShape)
	}

	batchSize := inputShape[0]
	channels := inputShape[1]

	return tensor.NewShape(batchSize, channels), nil
}

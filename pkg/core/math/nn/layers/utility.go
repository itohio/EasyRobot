package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Flatten represents a layer that flattens multi-dimensional input to 2D.
type Flatten struct {
	*Base
	startDim int // Start dimension to flatten from
	endDim   int // End dimension to flatten to
}

// NewFlatten creates a new Flatten layer.
func NewFlatten(startDim, endDim int) *Flatten {
	return &Flatten{
		Base:     NewBase(""),
		startDim: startDim,
		endDim:   endDim,
	}
}

// Init initializes the layer.
func (f *Flatten) Init(inputShape []int) error {
	if f == nil {
		return fmt.Errorf("Flatten.Init: nil layer")
	}

	if len(inputShape) == 0 {
		return fmt.Errorf("Flatten.Init: empty input shape")
	}

	// Compute output shape by flattening dimensions [startDim:endDim]
	if f.startDim < 0 || f.endDim > len(inputShape) || f.startDim > f.endDim {
		return fmt.Errorf("Flatten.Init: invalid dim range [%d:%d] for input shape %v", f.startDim, f.endDim, inputShape)
	}

	// Build output shape
	outputShape := make([]int, 0, len(inputShape)-(f.endDim-f.startDim-1))
	outputShape = append(outputShape, inputShape[:f.startDim]...)

	// Flatten dimensions from startDim to endDim
	flattenedSize := 1
	for i := f.startDim; i < f.endDim && i < len(inputShape); i++ {
		flattenedSize *= inputShape[i]
	}
	outputShape = append(outputShape, flattenedSize)

	// Add remaining dimensions
	if f.endDim < len(inputShape) {
		outputShape = append(outputShape, inputShape[f.endDim:]...)
	}

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}

	f.Base.AllocOutput(outputShape, outputSize)
	return nil
}

// Forward computes the forward pass.
func (f *Flatten) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if f == nil {
		return tensor.Tensor{}, fmt.Errorf("Flatten.Forward: nil layer")
	}

	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Flatten.Forward: empty input")
	}

	// Store input
	f.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := f.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Flatten.Forward: output not allocated, must call Init first")
	}

	// Flatten is just a reshape - copy data (same memory for contiguous tensors)
	if len(input.Data) != len(output.Data) {
		return tensor.Tensor{}, fmt.Errorf("Flatten.Forward: input size %d doesn't match output size %d",
			len(input.Data), len(output.Data))
	}
	copy(output.Data, input.Data)

	// Store output
	f.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
func (f *Flatten) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if f == nil {
		return tensor.Tensor{}, fmt.Errorf("Flatten.Backward: nil layer")
	}

	if len(gradOutput.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Flatten.Backward: empty gradOutput")
	}

	input := f.Base.Input()
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Flatten.Backward: input not stored, must call Forward first")
	}

	// Gradient is just reshaping back
	gradSize := input.Size()
	gradInput := tensor.Tensor{
		Dim:  make([]int, len(input.Dim)),
		Data: make([]float32, gradSize),
	}
	copy(gradInput.Dim, input.Dim)

	if len(gradOutput.Data) != len(gradInput.Data) {
		return tensor.Tensor{}, fmt.Errorf("Flatten.Backward: gradOutput size %d doesn't match input size %d",
			len(gradOutput.Data), len(gradInput.Data))
	}
	copy(gradInput.Data, gradOutput.Data)

	f.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (f *Flatten) OutputShape(inputShape []int) ([]int, error) {
	if f == nil {
		return nil, fmt.Errorf("Flatten.OutputShape: nil layer")
	}

	if len(inputShape) == 0 {
		return nil, fmt.Errorf("Flatten.OutputShape: empty input shape")
	}

	if f.startDim < 0 || f.endDim > len(inputShape) || f.startDim > f.endDim {
		return nil, fmt.Errorf("Flatten.OutputShape: invalid dim range [%d:%d] for input shape %v", f.startDim, f.endDim, inputShape)
	}

	// Build output shape
	outputShape := make([]int, 0, len(inputShape)-(f.endDim-f.startDim-1))
	outputShape = append(outputShape, inputShape[:f.startDim]...)

	// Flatten dimensions from startDim to endDim
	flattenedSize := 1
	for i := f.startDim; i < f.endDim && i < len(inputShape); i++ {
		flattenedSize *= inputShape[i]
	}
	outputShape = append(outputShape, flattenedSize)

	// Add remaining dimensions
	if f.endDim < len(inputShape) {
		outputShape = append(outputShape, inputShape[f.endDim:]...)
	}

	return outputShape, nil
}

// Reshape represents a layer that reshapes the input tensor.
type Reshape struct {
	*Base
	targetShape []int
}

// NewReshape creates a new Reshape layer with the given target shape.
func NewReshape(targetShape []int) *Reshape {
	return &Reshape{
		Base:        NewBase(""),
		targetShape: append([]int(nil), targetShape...), // Copy slice
	}
}

// Init initializes the layer.
func (r *Reshape) Init(inputShape []int) error {
	if r == nil {
		return fmt.Errorf("Reshape.Init: nil layer")
	}

	if len(inputShape) == 0 {
		return fmt.Errorf("Reshape.Init: empty input shape")
	}

	// Compute total size of input
	inputSize := 1
	for _, dim := range inputShape {
		if dim <= 0 {
			return fmt.Errorf("Reshape.Init: invalid input dimension %d in %v", dim, inputShape)
		}
		inputSize *= dim
	}

	// Check if target shape is compatible
	targetSize := 1
	for _, dim := range r.targetShape {
		if dim <= 0 {
			return fmt.Errorf("Reshape.Init: invalid target dimension %d in %v", dim, r.targetShape)
		}
		targetSize *= dim
	}

	if inputSize != targetSize {
		return fmt.Errorf("Reshape.Init: incompatible sizes: input %v (size %d) vs target %v (size %d)",
			inputShape, inputSize, r.targetShape, targetSize)
	}

	r.Base.AllocOutput(r.targetShape, targetSize)
	return nil
}

// Forward computes the forward pass.
func (r *Reshape) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if r == nil {
		return tensor.Tensor{}, fmt.Errorf("Reshape.Forward: nil layer")
	}

	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Reshape.Forward: empty input")
	}

	// Store input
	r.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := r.Base.Output()
	if len(output.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Reshape.Forward: output not allocated, must call Init first")
	}

	// Reshape is just copying data with different dimensions
	if len(input.Data) != len(output.Data) {
		return tensor.Tensor{}, fmt.Errorf("Reshape.Forward: input size %d doesn't match output size %d",
			len(input.Data), len(output.Data))
	}
	copy(output.Data, input.Data)

	// Store output
	r.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
func (r *Reshape) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if r == nil {
		return tensor.Tensor{}, fmt.Errorf("Reshape.Backward: nil layer")
	}

	if len(gradOutput.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Reshape.Backward: empty gradOutput")
	}

	input := r.Base.Input()
	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("Reshape.Backward: input not stored, must call Forward first")
	}

	// Gradient is just reshaping back to input shape
	gradSize := input.Size()
	gradInput := tensor.Tensor{
		Dim:  make([]int, len(input.Dim)),
		Data: make([]float32, gradSize),
	}
	copy(gradInput.Dim, input.Dim)

	if len(gradOutput.Data) != len(gradInput.Data) {
		return tensor.Tensor{}, fmt.Errorf("Reshape.Backward: gradOutput size %d doesn't match input size %d",
			len(gradOutput.Data), len(gradInput.Data))
	}
	copy(gradInput.Data, gradOutput.Data)

	r.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (r *Reshape) OutputShape(inputShape []int) ([]int, error) {
	if r == nil {
		return nil, fmt.Errorf("Reshape.OutputShape: nil layer")
	}

	if len(inputShape) == 0 {
		return nil, fmt.Errorf("Reshape.OutputShape: empty input shape")
	}

	// Validate that sizes match
	inputSize := 1
	for _, dim := range inputShape {
		if dim <= 0 {
			return nil, fmt.Errorf("Reshape.OutputShape: invalid input dimension %d in %v", dim, inputShape)
		}
		inputSize *= dim
	}

	targetSize := 1
	for _, dim := range r.targetShape {
		if dim <= 0 {
			return nil, fmt.Errorf("Reshape.OutputShape: invalid target dimension %d in %v", dim, r.targetShape)
		}
		targetSize *= dim
	}

	if inputSize != targetSize {
		return nil, fmt.Errorf("Reshape.OutputShape: incompatible sizes: input %v (size %d) vs target %v (size %d)",
			inputShape, inputSize, r.targetShape, targetSize)
	}

	outputShape := make([]int, len(r.targetShape))
	copy(outputShape, r.targetShape)
	return outputShape, nil
}

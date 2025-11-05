package layers

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Flatten represents a layer that flattens multi-dimensional input to 2D.
type Flatten struct {
	Base
	startDim int // Start dimension to flatten from
	endDim   int // End dimension to flatten to
}

// NewFlatten creates a new Flatten layer.
func NewFlatten(startDim, endDim int) *Flatten {
	return &Flatten{
		Base:     NewBase("flatten"),
		startDim: startDim,
		endDim:   endDim,
	}
}

// Init initializes the layer.
func (f *Flatten) Init(inputShape tensor.Shape) error {
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
func (f *Flatten) Forward(input types.Tensor) (types.Tensor, error) {
	if f == nil {
		return nil, fmt.Errorf("Flatten.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Flatten.Forward: empty input")
	}

	// Store input
	f.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := f.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Flatten.Forward: output not allocated, must call Init first")
	}

	// Flatten is just a reshape - copy data (same memory for contiguous tensors)
	if input.Size() != output.Size() {
		return nil, fmt.Errorf("Flatten.Forward: input size %d doesn't match output size %d",
			input.Size(), output.Size())
	}
	// Reshape input to match output shape, then copy
	inputReshaped := input.Reshape(output.Shape())
	output.Copy(inputReshaped)

	// Store output
	f.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
func (f *Flatten) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if f == nil {
		return nil, fmt.Errorf("Flatten.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Flatten.Backward: empty gradOutput")
	}

	input := f.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Flatten.Backward: input not stored, must call Forward first")
	}

	// Gradient is just reshaping back
	// Use input's data type for input gradient (for correctness in backward pass)
	gradInput := tensor.New(input.DataType(), input.Shape())

	if gradOutput.Size() != gradInput.Size() {
		return nil, fmt.Errorf("Flatten.Backward: gradOutput size %d doesn't match input size %d",
			gradOutput.Size(), gradInput.Size())
	}
	// Reshape gradOutput to match input shape, then copy
	gradOutputReshaped := gradOutput.Reshape(input.Shape())
	gradInput.Copy(gradOutputReshaped)

	f.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (f *Flatten) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
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
	Base
	targetShape []int
}

// NewReshape creates a new Reshape layer with the given target shape.
func NewReshape(targetShape []int) *Reshape {
	return &Reshape{
		Base:        NewBase("reshape"),
		targetShape: append([]int(nil), targetShape...), // Copy slice
	}
}

// Init initializes the layer.
func (r *Reshape) Init(inputShape tensor.Shape) error {
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
func (r *Reshape) Forward(input types.Tensor) (types.Tensor, error) {
	if r == nil {
		return nil, fmt.Errorf("Reshape.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Reshape.Forward: empty input")
	}

	// Store input
	r.Base.StoreInput(input)

	// Get pre-allocated output tensor
	output := r.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Reshape.Forward: output not allocated, must call Init first")
	}

	// Reshape is just copying data with different dimensions
	if input.Size() != output.Size() {
		return nil, fmt.Errorf("Reshape.Forward: input size %d doesn't match output size %d",
			input.Size(), output.Size())
	}
	// Reshape input to match output shape, then copy
	inputReshaped := input.Reshape(output.Shape())
	output.Copy(inputReshaped)

	// Store output
	r.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
func (r *Reshape) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if r == nil {
		return nil, fmt.Errorf("Reshape.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Reshape.Backward: empty gradOutput")
	}

	input := r.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Reshape.Backward: input not stored, must call Forward first")
	}

	// Gradient is just reshaping back to input shape
	// Use input's data type for input gradient (for correctness in backward pass)
	gradInput := tensor.New(input.DataType(), input.Shape())

	if gradOutput.Size() != gradInput.Size() {
		return nil, fmt.Errorf("Reshape.Backward: gradOutput size %d doesn't match input size %d",
			gradOutput.Size(), gradInput.Size())
	}
	// Reshape gradOutput to match input shape, then copy
	gradOutputReshaped := gradOutput.Reshape(input.Shape())
	gradInput.Copy(gradOutputReshaped)

	r.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (r *Reshape) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
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

// Unsqueeze represents a layer that adds a dimension at the specified position.
type Unsqueeze struct {
	Base
	dim int // Dimension index where to add the new dimension
}

// NewUnsqueeze creates a new Unsqueeze layer.
// dim: dimension index where to insert (0-based, can be negative for counting from end)
func NewUnsqueeze(dim int) *Unsqueeze {
	return &Unsqueeze{
		Base: NewBase("unsqueeze"),
		dim:  dim,
	}
}

// Init initializes the layer.
func (u *Unsqueeze) Init(inputShape tensor.Shape) error {
	if u == nil {
		return fmt.Errorf("Unsqueeze.Init: nil layer")
	}

	if len(inputShape) == 0 {
		return fmt.Errorf("Unsqueeze.Init: empty input shape")
	}

	outputShape := u.computeOutputShape(inputShape)
	if outputShape == nil {
		return fmt.Errorf("Unsqueeze.Init: invalid dimension %d for input shape %v", u.dim, inputShape)
	}

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}

	u.Base.AllocOutput(outputShape, outputSize)
	return nil
}

// computeOutputShape computes the output shape after adding dimension.
func (u *Unsqueeze) computeOutputShape(inputShape tensor.Shape) tensor.Shape {
	dim := u.dim
	if dim < 0 {
		dim = len(inputShape) + 1 + dim
	}

	if dim < 0 || dim > len(inputShape) {
		return nil
	}

	outputShape := make([]int, 0, len(inputShape)+1)
	outputShape = append(outputShape, inputShape[:dim]...)
	outputShape = append(outputShape, 1)
	outputShape = append(outputShape, inputShape[dim:]...)
	return outputShape
}

// Forward computes the forward pass.
func (u *Unsqueeze) Forward(input types.Tensor) (types.Tensor, error) {
	if u == nil {
		return nil, fmt.Errorf("Unsqueeze.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Unsqueeze.Forward: empty input")
	}

	u.Base.StoreInput(input)

	output := u.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Unsqueeze.Forward: output not allocated, must call Init first")
	}

	// Unsqueeze is just a reshape - copy data
	if input.Size() != output.Size() {
		return nil, fmt.Errorf("Unsqueeze.Forward: input size %d doesn't match output size %d",
			input.Size(), output.Size())
	}
	// Copy data using optimized Tensor.Copy method
	inputReshaped := input.Reshape(output.Shape())
	output.Copy(inputReshaped)

	u.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
func (u *Unsqueeze) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if u == nil {
		return nil, fmt.Errorf("Unsqueeze.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Unsqueeze.Backward: empty gradOutput")
	}

	input := u.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Unsqueeze.Backward: input not stored, must call Forward first")
	}

	// Use input's data type for input gradient (for correctness in backward pass)
	gradInput := tensor.New(input.DataType(), input.Shape())

	if gradOutput.Size() != gradInput.Size() {
		return nil, fmt.Errorf("Unsqueeze.Backward: gradOutput size %d doesn't match input size %d",
			gradOutput.Size(), gradInput.Size())
	}
	// Copy data using optimized Tensor.Copy method
	gradInput.Copy(gradOutput)

	u.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (u *Unsqueeze) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if u == nil {
		return nil, fmt.Errorf("Unsqueeze.OutputShape: nil layer")
	}

	if len(inputShape) == 0 {
		return nil, fmt.Errorf("Unsqueeze.OutputShape: empty input shape")
	}

	outputShape := u.computeOutputShape(inputShape)
	if outputShape == nil {
		return nil, fmt.Errorf("Unsqueeze.OutputShape: invalid dimension %d for input shape %v", u.dim, inputShape)
	}

	return outputShape, nil
}

// Squeeze represents a layer that removes dimensions of size 1.
type Squeeze struct {
	Base
	dims []int // Specific dimensions to squeeze (nil means squeeze all size-1 dims)
}

// NewSqueeze creates a new Squeeze layer that removes all dimensions of size 1.
func NewSqueeze() *Squeeze {
	return &Squeeze{
		Base: NewBase("squeeze"),
		dims: nil,
	}
}

// NewSqueezeDims creates a new Squeeze layer that removes specific dimensions (must be size 1).
func NewSqueezeDims(dims ...int) *Squeeze {
	return &Squeeze{
		Base: NewBase("squeeze"),
		dims: append([]int(nil), dims...),
	}
}

// Init initializes the layer.
func (s *Squeeze) Init(inputShape tensor.Shape) error {
	if s == nil {
		return fmt.Errorf("Squeeze.Init: nil layer")
	}

	if len(inputShape) == 0 {
		return fmt.Errorf("Squeeze.Init: empty input shape")
	}

	outputShape := s.computeOutputShape(inputShape)
	if outputShape == nil {
		return fmt.Errorf("Squeeze.Init: cannot squeeze input shape %v", inputShape)
	}

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}

	s.Base.AllocOutput(outputShape, outputSize)
	return nil
}

// computeOutputShape computes the output shape after removing size-1 dimensions.
func (s *Squeeze) computeOutputShape(inputShape tensor.Shape) tensor.Shape {
	if s.dims == nil {
		// Remove all dimensions of size 1
		outputShape := make([]int, 0, len(inputShape))
		for _, dim := range inputShape {
			if dim != 1 {
				outputShape = append(outputShape, dim)
			}
		}
		return outputShape
	}

	// Remove specific dimensions (must be size 1)
	outputShape := make([]int, 0, len(inputShape))
	for i, dim := range inputShape {
		shouldRemove := false
		for _, removeDim := range s.dims {
			actualDim := removeDim
			if actualDim < 0 {
				actualDim = len(inputShape) + actualDim
			}
			if actualDim == i {
				if dim != 1 {
					return nil // Cannot squeeze dimension that is not size 1
				}
				shouldRemove = true
				break
			}
		}
		if !shouldRemove {
			outputShape = append(outputShape, dim)
		}
	}
	return outputShape
}

// Forward computes the forward pass.
func (s *Squeeze) Forward(input types.Tensor) (types.Tensor, error) {
	if s == nil {
		return nil, fmt.Errorf("Squeeze.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Squeeze.Forward: empty input")
	}

	s.Base.StoreInput(input)

	output := s.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Squeeze.Forward: output not allocated, must call Init first")
	}

	// Squeeze is just a reshape - copy data
	if input.Size() != output.Size() {
		return nil, fmt.Errorf("Squeeze.Forward: input size %d doesn't match output size %d",
			input.Size(), output.Size())
	}
	// Copy data using optimized Tensor.Copy method
	inputReshaped := input.Reshape(output.Shape())
	output.Copy(inputReshaped)

	s.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
func (s *Squeeze) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if s == nil {
		return nil, fmt.Errorf("Squeeze.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Squeeze.Backward: empty gradOutput")
	}

	input := s.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Squeeze.Backward: input not stored, must call Forward first")
	}

	// Use input's data type for input gradient (for correctness in backward pass)
	gradInput := tensor.New(input.DataType(), input.Shape())

	if gradOutput.Size() != gradInput.Size() {
		return nil, fmt.Errorf("Squeeze.Backward: gradOutput size %d doesn't match input size %d",
			gradOutput.Size(), gradInput.Size())
	}
	// Copy data using optimized Tensor.Copy method
	gradInput.Copy(gradOutput)

	s.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (s *Squeeze) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if s == nil {
		return nil, fmt.Errorf("Squeeze.OutputShape: nil layer")
	}

	if len(inputShape) == 0 {
		return nil, fmt.Errorf("Squeeze.OutputShape: empty input shape")
	}

	outputShape := s.computeOutputShape(inputShape)
	if outputShape == nil {
		return nil, fmt.Errorf("Squeeze.OutputShape: cannot squeeze input shape %v", inputShape)
	}

	return outputShape, nil
}

// Transpose represents a layer that transposes dimensions.
// Currently supports 2D transpose only.
type Transpose struct {
	Base
	dims []int // Dimension permutation (empty means default 2D transpose: [1, 0])
}

// NewTranspose creates a new Transpose layer for 2D tensors (default transpose).
func NewTranspose() *Transpose {
	return &Transpose{
		Base: NewBase("transpose"),
		dims: nil, // nil means default 2D transpose
	}
}

// NewTransposeDims creates a new Transpose layer with specific dimension permutation.
func NewTransposeDims(dims ...int) *Transpose {
	return &Transpose{
		Base: NewBase("transpose"),
		dims: append([]int(nil), dims...),
	}
}

// Init initializes the layer.
func (t *Transpose) Init(inputShape tensor.Shape) error {
	if t == nil {
		return fmt.Errorf("Transpose.Init: nil layer")
	}

	if len(inputShape) == 0 {
		return fmt.Errorf("Transpose.Init: empty input shape")
	}

	if len(inputShape) != 2 {
		return fmt.Errorf("Transpose.Init: only 2D tensors supported, got shape %v", inputShape)
	}

	// Output shape for 2D transpose: [N, M] -> [M, N]
	outputShape := tensor.NewShape(inputShape[1], inputShape[0])
	outputSize := outputShape[0] * outputShape[1]

	t.Base.AllocOutput(outputShape, outputSize)
	return nil
}

// Forward computes the forward pass.
func (t *Transpose) Forward(input types.Tensor) (types.Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("Transpose.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Transpose.Forward: empty input")
	}

	if input.Rank() != 2 {
		return nil, fmt.Errorf("Transpose.Forward: only 2D tensors supported, got shape %v", input.Shape().ToSlice())
	}

	t.Base.StoreInput(input)

	output := t.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Transpose.Forward: output not allocated, must call Init first")
	}

	// Use tensor.Transpose
	transposed := input.Transpose()
	if transposed == nil {
		return nil, fmt.Errorf("Transpose.Forward: transpose failed")
	}

	// Copy transposed data to output using optimized Tensor.Copy method
	output.Copy(transposed)

	t.Base.StoreOutput(output)
	return output, nil
}

// Backward computes gradients w.r.t. input.
func (t *Transpose) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if t == nil {
		return nil, fmt.Errorf("Transpose.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Transpose.Backward: empty gradOutput")
	}

	input := t.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Transpose.Backward: input not stored, must call Forward first")
	}

	// Gradient of transpose is transpose of gradient
	transposed := gradOutput.Transpose()
	if transposed == nil {
		return nil, fmt.Errorf("Transpose.Backward: transpose failed")
	}

	// Use input's data type for input gradient (for correctness in backward pass)
	gradInput := tensor.New(input.DataType(), input.Shape())
	// Copy transposed data using optimized Tensor.Copy method
	gradInput.Copy(transposed)

	t.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// OutputShape returns the output shape for given input shape.
func (t *Transpose) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if t == nil {
		return nil, fmt.Errorf("Transpose.OutputShape: nil layer")
	}

	if len(inputShape) == 0 {
		return nil, fmt.Errorf("Transpose.OutputShape: empty input shape")
	}

	if len(inputShape) != 2 {
		return nil, fmt.Errorf("Transpose.OutputShape: only 2D tensors supported, got shape %v", inputShape)
	}

	// Output shape for 2D transpose: [N, M] -> [M, N]
	return tensor.NewShape(inputShape[1], inputShape[0]), nil
}

// PaddingMode represents the type of padding.
type PaddingMode int

const (
	// PaddingConstant pads with a constant value (default: 0)
	PaddingConstant PaddingMode = iota
	// PaddingReflect reflects values at boundaries
	PaddingReflect
	// PaddingReplicate replicates edge values
	PaddingReplicate
)

// Pad represents a layer that pads tensor with specified values.
type Pad struct {
	Base
	padding []int       // Padding for each dimension: [padBefore0, padAfter0, padBefore1, padAfter1, ...]
	value   float32     // Constant padding value (for PaddingConstant mode)
	mode    PaddingMode // Padding mode
}

// NewPad creates a new Pad layer with constant padding.
// padding: per-dimension padding [padBefore0, padAfter0, padBefore1, padAfter1, ...]
// value: constant padding value (default: 0)
func NewPad(padding []int, value float32) *Pad {
	if len(padding)%2 != 0 {
		// Invalid padding format
		return nil
	}
	return &Pad{
		Base:    NewBase("pad"),
		padding: append([]int(nil), padding...),
		value:   value,
		mode:    PaddingConstant,
	}
}

// NewPadReflect creates a new Pad layer with reflect padding.
func NewPadReflect(padding []int) *Pad {
	if len(padding)%2 != 0 {
		return nil
	}
	return &Pad{
		Base:    NewBase("pad"),
		padding: append([]int(nil), padding...),
		value:   0,
		mode:    PaddingReflect,
	}
}

// Init initializes the layer.
func (p *Pad) Init(inputShape tensor.Shape) error {
	if p == nil {
		return fmt.Errorf("Pad.Init: nil layer")
	}

	if len(inputShape) == 0 {
		return fmt.Errorf("Pad.Init: empty input shape")
	}

	if len(p.padding) != 2*len(inputShape) {
		return fmt.Errorf("Pad.Init: padding length %d doesn't match input dimensions %d", len(p.padding), len(inputShape))
	}

	outputShape := make([]int, len(inputShape))
	for i := range inputShape {
		outputShape[i] = inputShape[i] + p.padding[2*i] + p.padding[2*i+1]
		if outputShape[i] < 0 {
			return fmt.Errorf("Pad.Init: invalid output dimension %d for input shape %v", outputShape[i], inputShape)
		}
	}

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}

	p.Base.AllocOutput(outputShape, outputSize)
	return nil
}

// Forward computes the forward pass.
func (p *Pad) Forward(input types.Tensor) (types.Tensor, error) {
	if p == nil {
		return nil, fmt.Errorf("Pad.Forward: nil layer")
	}

	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Pad.Forward: empty input")
	}

	p.Base.StoreInput(input)

	output := p.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Pad.Forward: output not allocated, must call Init first")
	}

	// Initialize output with padding value using Elements iterator
	for elem := range output.Elements() {
		// Convert float32 to float64 for Set interface
		elem.Set(float64(p.value))
	}

	// Copy input data to the appropriate position in output
	p.applyPadding(input, output)

	p.Base.StoreOutput(output)
	return output, nil
}

// applyPadding copies input data to output with padding.
func (p *Pad) applyPadding(input, output types.Tensor) {
	// Compute strides for both tensors
	inputStrides := computeStrides(input.Shape().ToSlice())
	outputStrides := computeStrides(output.Shape().ToSlice())

	// Recursively copy data with padding offset using tensor methods
	p.copyWithPadding(input.Shape().ToSlice(), input, inputStrides, 0, output.Shape().ToSlice(), output, outputStrides, make([]int, output.Rank()), 0)
}

// computeStrides computes row-major strides from shape.
func computeStrides(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// copyWithPadding recursively copies input data to output with padding.
func (p *Pad) copyWithPadding(inputShape []int, inputTensor types.Tensor, inputStrides []int, inputOffset int,
	outputShape []int, outputTensor types.Tensor, outputStrides []int, outputIdx []int, dim int) {
	if dim == len(inputShape) {
		// Base case: copy single element using tensor methods
		inputIndices := computeIndices(inputShape, inputStrides, inputOffset)
		outputIndices := make([]int, len(outputIdx))
		copy(outputIndices, outputIdx)
		outputTensor.SetAt(inputTensor.At(inputIndices...), outputIndices...)
		return
	}

	padBefore := p.padding[2*dim]
	// Iterate over input range
	for i := 0; i < inputShape[dim]; i++ {
		outputIdx[dim] = padBefore + i
		newInputOffset := inputOffset + i*inputStrides[dim]
		p.copyWithPadding(inputShape, inputTensor, inputStrides, newInputOffset,
			outputShape, outputTensor, outputStrides, outputIdx, dim+1)
	}
}

// Backward computes gradients w.r.t. input (unpad the gradient).
func (p *Pad) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if p == nil {
		return nil, fmt.Errorf("Pad.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Pad.Backward: empty gradOutput")
	}

	input := p.Base.Input()
	if tensor.IsNil(input) {
		return nil, fmt.Errorf("Pad.Backward: input not stored, must call Forward first")
	}

	// Use input's data type for input gradient (for correctness in backward pass)
	gradInput := tensor.New(input.DataType(), input.Shape())

	// Extract gradient from padded region using tensor methods
	inputStrides := computeStrides(input.Shape().ToSlice())
	outputStrides := computeStrides(gradOutput.Shape().ToSlice())
	p.extractGradient(input.Shape().ToSlice(), gradInput, inputStrides, 0,
		gradOutput.Shape().ToSlice(), gradOutput, outputStrides, make([]int, gradOutput.Rank()), 0)

	p.Base.StoreGrad(gradInput)
	return gradInput, nil
}

// extractGradient extracts gradient from padded output.
func (p *Pad) extractGradient(inputShape []int, inputTensor types.Tensor, inputStrides []int, inputOffset int,
	outputShape []int, outputTensor types.Tensor, outputStrides []int, outputIdx []int, dim int) {
	if dim == len(inputShape) {
		// Base case: copy single element using tensor methods
		inputIndices := computeIndices(inputShape, inputStrides, inputOffset)
		outputIndices := make([]int, len(outputIdx))
		copy(outputIndices, outputIdx)
		inputTensor.SetAt(outputTensor.At(outputIndices...), inputIndices...)
		return
	}

	padBefore := p.padding[2*dim]
	// Iterate over input range
	for i := 0; i < inputShape[dim]; i++ {
		outputIdx[dim] = padBefore + i
		newInputOffset := inputOffset + i*inputStrides[dim]
		p.extractGradient(inputShape, inputTensor, inputStrides, newInputOffset,
			outputShape, outputTensor, outputStrides, outputIdx, dim+1)
	}
}

// computeIndices converts a linear offset with strides to multi-dimensional indices.
func computeIndices(shape []int, strides []int, offset int) []int {
	indices := make([]int, len(shape))
	remaining := offset
	for i := 0; i < len(shape); i++ {
		indices[i] = remaining / strides[i]
		remaining = remaining % strides[i]
	}
	return indices
}

// OutputShape returns the output shape for given input shape.
func (p *Pad) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if p == nil {
		return nil, fmt.Errorf("Pad.OutputShape: nil layer")
	}

	if len(inputShape) == 0 {
		return nil, fmt.Errorf("Pad.OutputShape: empty input shape")
	}

	if len(p.padding) != 2*len(inputShape) {
		return nil, fmt.Errorf("Pad.OutputShape: padding length %d doesn't match input dimensions %d", len(p.padding), len(inputShape))
	}

	outputShape := make([]int, len(inputShape))
	for i := range inputShape {
		outputShape[i] = inputShape[i] + p.padding[2*i] + p.padding[2*i+1]
		if outputShape[i] < 0 {
			return nil, fmt.Errorf("Pad.OutputShape: invalid output dimension %d", outputShape[i])
		}
	}

	return outputShape, nil
}

// Concatenate represents a layer that concatenates multiple tensors along a dimension.
// Note: This layer breaks the standard Layer interface pattern due to multiple inputs.
type Concatenate struct {
	Base
	dim    int            // Dimension along which to concatenate
	inputs []types.Tensor // Stored inputs for concatenation
	shapes [][]int        // Shapes of inputs (for Init)
}

// NewConcatenate creates a new Concatenate layer.
// dim: dimension along which to concatenate (0-based)
func NewConcatenate(dim int) *Concatenate {
	return &Concatenate{
		Base: NewBase("concatenate"),
		dim:  dim,
	}
}

// SetInputs sets the input tensors for concatenation.
func (c *Concatenate) SetInputs(inputs []types.Tensor) {
	c.inputs = make([]types.Tensor, len(inputs))
	c.shapes = make([][]int, len(inputs))
	for i, input := range inputs {
		c.inputs[i] = input
		c.shapes[i] = input.Shape().ToSlice()
	}
}

// Init initializes the layer with a single input shape.
// For multiple inputs, call InitMulti instead or set inputs via SetInputs before Forward.
// This method is for Layer interface compatibility - actual initialization happens in ForwardMulti.
func (c *Concatenate) Init(inputShape tensor.Shape) error {
	if c == nil {
		return fmt.Errorf("Concatenate.Init: nil layer")
	}

	if len(inputShape) == 0 {
		return fmt.Errorf("Concatenate.Init: empty input shape")
	}

	// Store shape for reference, but output shape will be computed from actual inputs
	c.shapes = [][]int{inputShape}
	return nil
}

// InitMulti initializes the layer with multiple input shapes.
func (c *Concatenate) InitMulti(inputShapes [][]int) error {
	if c == nil {
		return fmt.Errorf("Concatenate.InitMulti: nil layer")
	}

	if len(inputShapes) == 0 {
		return fmt.Errorf("Concatenate.InitMulti: no input shapes provided")
	}

	if len(inputShapes[0]) == 0 {
		return fmt.Errorf("Concatenate.InitMulti: empty input shape")
	}

	// Validate all shapes are compatible
	ndims := len(inputShapes[0])
	for i, shape := range inputShapes {
		if len(shape) != ndims {
			return fmt.Errorf("Concatenate.InitMulti: incompatible shapes: shape[0] has %d dims, shape[%d] has %d dims",
				ndims, i, len(shape))
		}
		for j, dim := range shape {
			if j != c.dim && dim != inputShapes[0][j] {
				return fmt.Errorf("Concatenate.InitMulti: incompatible shapes: dimension %d differs", j)
			}
		}
	}

	// Compute output shape
	outputShape := make([]int, ndims)
	copy(outputShape, inputShapes[0])
	for i := 1; i < len(inputShapes); i++ {
		outputShape[c.dim] += inputShapes[i][c.dim]
	}

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}

	c.shapes = inputShapes
	c.Base.AllocOutput(outputShape, outputSize)
	return nil
}

// Forward computes the forward pass with stored inputs.
// If inputs are not set via SetInputs, initializes from the provided input and stored shapes.
func (c *Concatenate) Forward(input types.Tensor) (types.Tensor, error) {
	if c == nil {
		return nil, fmt.Errorf("Concatenate.Forward: nil layer")
	}

	if len(c.inputs) == 0 {
		return nil, fmt.Errorf("Concatenate.Forward: no inputs set, use SetInputs or ForwardMulti")
	}

	return c.ForwardMulti(c.inputs)
}

// ForwardMulti concatenates multiple input tensors.
func (c *Concatenate) ForwardMulti(inputs []types.Tensor) (types.Tensor, error) {
	if c == nil {
		return nil, fmt.Errorf("Concatenate.ForwardMulti: nil layer")
	}

	if len(inputs) == 0 {
		return nil, fmt.Errorf("Concatenate.ForwardMulti: no inputs provided")
	}

	// Validate inputs
	for i, input := range inputs {
		if tensor.IsNil(input) {
			return nil, fmt.Errorf("Concatenate.ForwardMulti: empty input[%d]", i)
		}
	}

	// Store inputs
	c.inputs = make([]types.Tensor, len(inputs))
	copy(c.inputs, inputs)

	output := c.Base.Output()
	if tensor.IsNil(output) {
		return nil, fmt.Errorf("Concatenate.ForwardMulti: output not allocated, must call Init first")
	}

	// Concatenate along the specified dimension
	c.concatenateTensors(inputs, output)

	c.Base.StoreOutput(output)
	return output, nil
}

// concatenateTensors concatenates multiple tensors along the specified dimension.
func (c *Concatenate) concatenateTensors(inputs []types.Tensor, output types.Tensor) {
	// Compute output offset for each input
	outputOffset := 0
	inputStrides := make([][]int, len(inputs))
	outputStrides := computeStrides(output.Shape().ToSlice())

	for i, input := range inputs {
		inputStrides[i] = computeStrides(input.Shape().ToSlice())
	}

	// For each input, copy its data to the appropriate position in output
	for inputIdx, input := range inputs {
		c.copyTensorToOutput(input, inputStrides[inputIdx], output, outputStrides, outputOffset, 0, make([]int, input.Rank()))
		// Update offset for next input along concatenation dimension
		if inputIdx < len(inputs)-1 {
			// Compute size of this input along concatenation dimension
			inputSize := 1
			inputShape := input.Shape().ToSlice()
			for i := 0; i < len(inputShape); i++ {
				if i == c.dim {
					inputSize *= inputShape[i]
				}
			}
			outputOffset += inputSize * outputStrides[c.dim]
		}
	}
}

// copyTensorToOutput copies tensor data to output at given offset.
func (c *Concatenate) copyTensorToOutput(input types.Tensor, inputStrides []int,
	output types.Tensor, outputStrides []int, outputOffset int, dim int, indices []int) {
	inputShape := input.Shape().ToSlice()
	outputShape := output.Shape().ToSlice()
	if dim == len(inputShape) {
		// Base case: copy single element using tensor methods
		// Compute input indices from strides
		inputIndices := make([]int, len(indices))
		copy(inputIndices, indices)

		// Compute output indices, accounting for concatenation dimension offset
		outputIndices := make([]int, len(indices))
		copy(outputIndices, indices)
		for i := range indices {
			if i == c.dim {
				// Use computeIndices to get the correct position along concatenation dimension
				offsetIndices := computeIndices(outputShape, outputStrides, outputOffset)
				outputIndices[i] = offsetIndices[i] + indices[i]
			} else {
				outputIndices[i] = indices[i]
			}
		}
		output.SetAt(input.At(inputIndices...), outputIndices...)
		return
	}

	// Recursive case: iterate over current dimension
	for i := 0; i < inputShape[dim]; i++ {
		if len(indices) <= dim {
			indices = append(indices, i)
		} else {
			indices[dim] = i
		}
		c.copyTensorToOutput(input, inputStrides, output, outputStrides, outputOffset, dim+1, indices)
	}
}

// Backward splits the gradient and returns gradients for each input.
// Returns a single gradient tensor (this is a limitation - ideally should return multiple).
// For proper multi-input backward, use BackwardMulti.
func (c *Concatenate) Backward(gradOutput types.Tensor) (types.Tensor, error) {
	if c == nil {
		return nil, fmt.Errorf("Concatenate.Backward: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Concatenate.Backward: empty gradOutput")
	}

	if len(c.inputs) == 0 {
		return nil, fmt.Errorf("Concatenate.Backward: no inputs stored, must call Forward first")
	}

	// For single gradient return, return gradient for first input
	grads, err := c.BackwardMulti(gradOutput)
	if err != nil || len(grads) == 0 {
		return nil, err
	}

	return grads[0], nil
}

// BackwardMulti splits the gradient and returns gradients for each input.
func (c *Concatenate) BackwardMulti(gradOutput types.Tensor) ([]types.Tensor, error) {
	if c == nil {
		return nil, fmt.Errorf("Concatenate.BackwardMulti: nil layer")
	}

	if tensor.IsNil(gradOutput) {
		return nil, fmt.Errorf("Concatenate.BackwardMulti: empty gradOutput")
	}

	if len(c.inputs) == 0 {
		return nil, fmt.Errorf("Concatenate.BackwardMulti: no inputs stored, must call Forward first")
	}

	grads := make([]types.Tensor, len(c.inputs))
	outputStrides := computeStrides(gradOutput.Shape().ToSlice())

	// Split gradient for each input
	outputOffset := 0
	for i, input := range c.inputs {
		// Use input's data type for input gradient (for correctness in backward pass)
		grads[i] = tensor.New(input.DataType(), input.Shape())

		inputStrides := computeStrides(input.Shape().ToSlice())
		// Extract gradient slice for this input
		c.extractGradientSlice(gradOutput, outputStrides, outputOffset,
			grads[i], inputStrides, 0, make([]int, input.Rank()))

		// Update offset for next input
		if i < len(c.inputs)-1 {
			inputSize := 1
			inputShape := input.Shape().ToSlice()
			for j := 0; j < len(inputShape); j++ {
				if j == c.dim {
					inputSize *= inputShape[j]
				}
			}
			outputOffset += inputSize * outputStrides[c.dim]
		}
	}

	// Store first gradient (for Layer interface compatibility)
	if len(grads) > 0 {
		c.Base.StoreGrad(grads[0])
	}

	return grads, nil
}

// extractGradientSlice extracts a slice of gradient for one input.
func (c *Concatenate) extractGradientSlice(gradOutput types.Tensor, outputStrides []int, outputOffset int,
	gradInput types.Tensor, inputStrides []int, dim int, indices []int) {
	gradInputShape := gradInput.Shape().ToSlice()
	gradOutputShape := gradOutput.Shape().ToSlice()
	if dim == len(gradInputShape) {
		// Base case: copy single element using tensor methods
		// Compute input indices from strides
		inputIndices := make([]int, len(indices))
		copy(inputIndices, indices)

		// Compute output indices, accounting for concatenation dimension offset
		outputIndices := make([]int, len(indices))
		copy(outputIndices, indices)
		for i := range indices {
			if i == c.dim {
				// Use computeIndices to get the correct position along concatenation dimension
				offsetIndices := computeIndices(gradOutputShape, outputStrides, outputOffset)
				outputIndices[i] = offsetIndices[i] + indices[i]
			} else {
				outputIndices[i] = indices[i]
			}
		}
		gradInput.SetAt(gradOutput.At(outputIndices...), inputIndices...)
		return
	}

	// Recursive case
	for i := 0; i < gradInputShape[dim]; i++ {
		if len(indices) <= dim {
			indices = append(indices, i)
		} else {
			indices[dim] = i
		}
		c.extractGradientSlice(gradOutput, outputStrides, outputOffset, gradInput, inputStrides, dim+1, indices)
	}
}

// OutputShape returns the output shape for given input shapes.
func (c *Concatenate) OutputShape(inputShapes [][]int) ([]int, error) {
	if c == nil {
		return nil, fmt.Errorf("Concatenate.OutputShape: nil layer")
	}

	if len(inputShapes) == 0 {
		return nil, fmt.Errorf("Concatenate.OutputShape: no input shapes provided")
	}

	if len(inputShapes[0]) == 0 {
		return nil, fmt.Errorf("Concatenate.OutputShape: empty input shape")
	}

	ndims := len(inputShapes[0])
	for _, shape := range inputShapes {
		if len(shape) != ndims {
			return nil, fmt.Errorf("Concatenate.OutputShape: incompatible shapes")
		}
		for j, dim := range shape {
			if j != c.dim && dim != inputShapes[0][j] {
				return nil, fmt.Errorf("Concatenate.OutputShape: incompatible shapes")
			}
		}
	}

	outputShape := make([]int, ndims)
	copy(outputShape, inputShapes[0])
	for i := 1; i < len(inputShapes); i++ {
		outputShape[c.dim] += inputShapes[i][c.dim]
	}

	return outputShape, nil
}

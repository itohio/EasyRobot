package tensor

import "fmt"

// Tensor represents a multi-dimensional array of float32 values.
type Tensor struct {
	dtype DataType
	shape Shape
	data  []float32
}

// New creates a new tensor with the provided data type and shape.
// The underlying buffer is zero-initialized.
func New(dtype DataType, shape Shape) Tensor {
	size := shape.Size()
	buf := make([]float32, size)
	return Tensor{dtype: dtype, shape: shape, data: buf}
}

// NewAs creates a new tensor with the same data type and shape as the given tensor.
func NewAs(t Tensor) Tensor {
	// Make new tensor using given tensor data (clone)
	clone := Tensor{
		dtype: t.dtype,
		shape: NewShape(t.shape...), // Clone shape
		data:  nil,
	}
	if t.data != nil {
		clone.data = make([]float32, len(t.data))
		copy(clone.data, t.data)
	}
	return clone
}

// Empty creates an empty tensor of given data type.
func Empty(dt DataType) Tensor {
	return Tensor{dtype: dt}
}

// Empty creates an empty tensor of given data type.
func EmptyAs(t Tensor) Tensor {
	return Tensor{dtype: t.DataType()}
}

// FromFloat32 constructs an FP32 tensor from an existing backing slice.
// If data is nil, a new buffer is allocated. The slice is used directly (no copy).
func FromFloat32(shape Shape, data []float32) Tensor {
	size := shape.Size()
	var buf []float32
	if data == nil {
		buf = make([]float32, size)
	} else {
		if len(data) != size {
			panic(fmt.Sprintf("tensor.FromFloat32: data length %d does not match shape size %d", len(data), size))
		}
		buf = data
	}
	return Tensor{dtype: DTFP32, shape: shape, data: buf}
}

func (t Tensor) Empty() bool {
	return t.shape == nil && t.data == nil
}

// DataType returns the tensor's data type.
func (t Tensor) DataType() DataType {
	if t.shape == nil && t.data == nil {
		return DTFP32
	}
	return t.dtype
}

// Shape returns a copy of the tensor's shape.
func (t Tensor) Shape() Shape {
	if t.shape == nil {
		return nil
	}
	return NewShape(t.shape...)
}

// Rank returns the number of dimensions.
func (t Tensor) Rank() int {
	if t.shape == nil {
		return 0
	}
	return t.shape.Rank()
}

// Size returns the total number of elements in the tensor.
func (t Tensor) Size() int {
	if t.shape == nil {
		if len(t.data) == 0 {
			return 0
		}
		return len(t.data)
	}
	return t.shape.Size()
}

// Data returns the underlying data slice. Mutating the returned slice mutates the tensor.
//
// Deprecated: Use Elements() for element access instead. Direct access to underlying data
// is discouraged for neural network operations as it bypasses tensor abstractions.
// Data() will be removed in a future version.
func (t Tensor) Data() []float32 {
	return t.data
}

// Flat returns the underlying data slice (zero-copy).
//
// Deprecated: Use Elements() for element access instead. Direct access to underlying data
// is discouraged for neural network operations as it bypasses tensor abstractions.
// Flat() will be removed in a future version.
func (t Tensor) Flat() []float32 {
	return t.data
}

// Clone creates a deep copy of the tensor.
func (t Tensor) Clone() *Tensor {
	if t.shape == nil && t.data == nil {
		return nil
	}
	clonedShape := t.Shape()
	clonedData := make([]float32, len(t.data))
	copy(clonedData, t.data)
	return &Tensor{dtype: t.dtype, shape: clonedShape, data: clonedData}
}

// isContiguous checks if the tensor data is contiguous (no gaps).
func (t Tensor) isContiguous() bool {
	if t.shape == nil {
		return true
	}
	if t.shape.Rank() == 0 {
		return true
	}
	strides := t.shape.Strides()
	expectedSize := strides[0] * t.shape[0]
	return len(t.data) == expectedSize
}

// elementIndex computes the linear index for given indices using strides.
func (t Tensor) elementIndex(indices []int, strides []int) int {
	idx := 0
	for i := range indices {
		idx += indices[i] * strides[i]
	}
	return idx
}

// At returns the element at the given indices.
// Indices must match the tensor's dimensions.
func (t Tensor) At(indices ...int) float32 {
	if t.shape == nil || (t.shape.Rank() == 0 && len(indices) == 0) {
		if len(t.data) == 0 {
			panic("tensor.At: empty tensor")
		}
		return t.data[0]
	}

	if len(indices) != t.shape.Rank() {
		panic("tensor.At: number of indices must match tensor dimensions")
	}

	for i, idx := range indices {
		if idx < 0 || idx >= t.shape[i] {
			panic("tensor.At: index out of bounds")
		}
	}

	strides := t.shape.Strides()
	linearIdx := t.elementIndex(indices, strides)
	if linearIdx >= len(t.data) {
		panic("tensor.At: computed index out of bounds")
	}
	return t.data[linearIdx]
}

// SetAt sets the element at the given indices.
// Indices must match the tensor's dimensions.
func (t *Tensor) SetAt(indices []int, value float32) {
	if t.shape == nil || (t.shape.Rank() == 0 && len(indices) == 0) {
		if len(t.data) == 0 {
			panic("tensor.SetAt: cannot set element of empty tensor")
		}
		t.data[0] = value
		return
	}

	if len(indices) != t.shape.Rank() {
		panic("tensor.SetAt: number of indices must match tensor dimensions")
	}

	for i, idx := range indices {
		if idx < 0 || idx >= t.shape[i] {
			panic("tensor.SetAt: index out of bounds")
		}
	}

	strides := t.shape.Strides()
	linearIdx := t.elementIndex(indices, strides)
	if linearIdx >= len(t.data) {
		panic("tensor.SetAt: computed index out of bounds")
	}
	t.data[linearIdx] = value
}

// Reshape returns a new tensor with the same data but different shape (zero-copy when possible).
// The total number of elements must remain the same.
func (t Tensor) Reshape(newShape []int) *Tensor {
	if t.shape == nil && t.data == nil {
		return nil
	}
	s := NewShape(newShape...)
	if s.Size() != len(t.data) {
		panic("tensor.Reshape: cannot reshape tensor with different total size")
	}
	return &Tensor{dtype: t.dtype, shape: s, data: t.data}
}

// reset replaces the tensor contents with the provided dtype, shape, and optional backing slice (no copy).
func (t *Tensor) reset(dtype DataType, shape []int, data []float32) {
	s := NewShape(shape...)
	size := s.Size()
	var buf []float32
	if data == nil {
		buf = make([]float32, size)
	} else {
		if len(data) != size {
			panic(fmt.Sprintf("tensor.reset: data length %d does not match shape size %d", len(data), size))
		}
		buf = data
	}
	t.dtype = dtype
	t.shape = s
	t.data = buf
}

// Element represents a single tensor element with Get and Set methods.
type Element struct {
	tensor *Tensor
	index  int
}

// Get returns the float32 value at this element's position.
func (e Element) Get() float32 {
	return e.tensor.data[e.index]
}

// Set sets the float32 value at this element's position.
func (e Element) Set(value float32) {
	e.tensor.data[e.index] = value
}

// Elements creates an iterator that fixes specified dimensions and iterates over the remaining ones.
// Returns a function that can be used in Go 1.22+ range loops.
// fixedAxisValuePairs are pairs of axis index and fixed value: axis1, value1, axis2, value2, ...
// The iterator yields Element objects with Get() and Set() methods for accessing tensor values.
//
// Usage:
//
//	for elem := range tensor.Elements() {
//	    value := elem.Get()
//	    elem.Set(value * 2)
//	}
//
//	for elem := range tensor.Elements(0, 1) {
//	    // Fixes dimension 0 at index 1, iterates over remaining dimensions
//	    elem.Set(0.0)
//	}
func (t *Tensor) Elements(fixedAxisValuePairs ...int) func(func(Element) bool) {
	if t.shape == nil || len(t.data) == 0 {
		return func(yield func(Element) bool) {
			// Empty tensor - yield once with invalid element
			yield(Element{tensor: t, index: 0})
		}
	}

	// Use shape iterator to get index combinations
	shapeIter := t.shape.Iterator(fixedAxisValuePairs...)
	strides := t.shape.Strides()

	return func(yield func(Element) bool) {
		for indices := range shapeIter {
			// Compute linear index from multi-dimensional indices
			linearIdx := t.elementIndex(indices, strides)
			if linearIdx >= len(t.data) {
				continue // Skip invalid indices
			}

			elem := Element{tensor: t, index: linearIdx}
			if !yield(elem) {
				return
			}
		}
	}
}

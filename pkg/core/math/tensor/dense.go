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
func New(dtype DataType, dims ...int) *Tensor {
	shape := NewShape(dims...)
	size := shape.Size()
	buf := make([]float32, size)
	return &Tensor{dtype: dtype, shape: shape, data: buf}
}

// FromFloat32 constructs an FP32 tensor from an existing backing slice.
// If data is nil, a new buffer is allocated. The slice is used directly (no copy).
func FromFloat32(shape []int, data []float32) *Tensor {
	s := NewShape(shape...)
	size := s.Size()
	var buf []float32
	if data == nil {
		buf = make([]float32, size)
	} else {
		if len(data) != size {
			panic(fmt.Sprintf("tensor.FromFloat32: data length %d does not match shape size %d", len(data), size))
		}
		buf = data
	}
	return &Tensor{dtype: DTFP32, shape: s, data: buf}
}

// DataType returns the tensor's data type.
func (t *Tensor) DataType() DataType {
	if t == nil {
		return DTFP32
	}
	return t.dtype
}

// Shape returns a copy of the tensor's shape.
func (t *Tensor) Shape() Shape {
	if t == nil || t.shape == nil {
		return nil
	}
	return NewShape(t.shape...)
}

// Rank returns the number of dimensions.
func (t *Tensor) Rank() int {
	if t == nil {
		return 0
	}
	return len(t.shape)
}

// Size returns the total number of elements in the tensor.
func (t *Tensor) Size() int {
	if t == nil {
		return 0
	}
	if t.shape == nil {
		if len(t.data) == 0 {
			return 0
		}
		return len(t.data)
	}
	return t.shape.Size()
}

// Data returns the underlying data slice. Mutating the returned slice mutates the tensor.
func (t *Tensor) Data() []float32 {
	if t == nil {
		return nil
	}
	return t.data
}

// Flat returns the underlying data slice (zero-copy).
func (t *Tensor) Flat() []float32 {
	return t.Data()
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	if t == nil {
		return nil
	}
	clonedShape := t.Shape()
	clonedData := make([]float32, len(t.data))
	copy(clonedData, t.data)
	return &Tensor{dtype: t.dtype, shape: clonedShape, data: clonedData}
}

// isContiguous checks if the tensor data is contiguous (no gaps).
func (t *Tensor) isContiguous() bool {
	if t == nil {
		return true
	}
	if len(t.shape) == 0 {
		return true
	}
	strides := t.shape.Strides()
	expectedSize := strides[0] * t.shape[0]
	return len(t.data) == expectedSize
}

// elementIndex computes the linear index for given indices using strides.
func (t *Tensor) elementIndex(indices []int, strides []int) int {
	idx := 0
	for i := range indices {
		idx += indices[i] * strides[i]
	}
	return idx
}

// At returns the element at the given indices.
// Indices must match the tensor's dimensions.
func (t *Tensor) At(indices ...int) float32 {
	if t == nil || (len(t.shape) == 0 && len(indices) == 0) {
		if len(t.data) == 0 {
			panic("tensor.At: empty tensor")
		}
		return t.data[0]
	}

	if len(indices) != len(t.shape) {
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
	if t == nil || (len(t.shape) == 0 && len(indices) == 0) {
		if t == nil || len(t.data) == 0 {
			panic("tensor.SetAt: cannot set element of nil or empty tensor")
		}
		t.data[0] = value
		return
	}

	if len(indices) != len(t.shape) {
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
func (t *Tensor) Reshape(newShape []int) *Tensor {
	if t == nil {
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
	if t == nil {
		return
	}
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

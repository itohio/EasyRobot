package tensor

// Tensor represents a multi-dimensional array of float32 values.
type Tensor struct {
	Dim  []int
	Data []float32
}

// Shape returns a copy of the tensor's dimensions.
func (t *Tensor) Shape() Shape {
	if t == nil || t.Dim == nil {
		return nil
	}
	return NewShape(t.Dim...)
}

// Size returns the total number of elements in the tensor.
func (t *Tensor) Size() int {
	if t == nil || len(t.Dim) == 0 {
		return 0
	}
	return Shape(t.Dim).Size()
}

// isContiguous checks if the tensor data is contiguous (no gaps).
func (t *Tensor) isContiguous() bool {
	if t == nil || len(t.Dim) == 0 {
		return true
	}
	strides := Shape(t.Dim).Strides()
	expectedSize := strides[0] * t.Dim[0]
	return len(t.Data) == expectedSize
}

// elementIndex computes the linear index for given indices using strides.
func (t *Tensor) elementIndex(indices []int, strides []int) int {
	idx := 0
	for i := range indices {
		idx += indices[i] * strides[i]
	}
	return idx
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	if t == nil {
		return nil
	}

	clone := &Tensor{
		Dim:  make([]int, len(t.Dim)),
		Data: make([]float32, len(t.Data)),
	}

	copy(clone.Dim, t.Dim)
	copy(clone.Data, t.Data)

	return clone
}

// Flat returns the underlying data slice (zero-copy).
// This returns the same slice as t.Data, allowing direct access to the data.
func (t *Tensor) Flat() []float32 {
	if t == nil {
		return nil
	}
	return t.Data
}

// At returns the element at the given indices.
// Indices must match the tensor's dimensions.
func (t *Tensor) At(indices ...int) float32 {
	if t == nil || len(t.Dim) == 0 {
		panic("tensor.At: cannot access element of nil or empty tensor")
	}

	if len(indices) != len(t.Dim) {
		panic("tensor.At: number of indices must match tensor dimensions")
	}

	// Validate indices
	for i, idx := range indices {
		if idx < 0 || idx >= t.Dim[i] {
			panic("tensor.At: index out of bounds")
		}
	}

	// Compute linear index using strides
	strides := Shape(t.Dim).Strides()
	idx := t.elementIndex(indices, strides)

	if idx >= len(t.Data) {
		panic("tensor.At: computed index out of bounds")
	}

	return t.Data[idx]
}

// SetAt sets the element at the given indices.
// Indices must match the tensor's dimensions.
func (t *Tensor) SetAt(indices []int, value float32) {
	if t == nil || len(t.Dim) == 0 {
		panic("tensor.SetAt: cannot set element of nil or empty tensor")
	}

	if len(indices) != len(t.Dim) {
		panic("tensor.SetAt: number of indices must match tensor dimensions")
	}

	// Validate indices
	for i, idx := range indices {
		if idx < 0 || idx >= t.Dim[i] {
			panic("tensor.SetAt: index out of bounds")
		}
	}

	// Compute linear index using strides
	strides := Shape(t.Dim).Strides()
	idx := t.elementIndex(indices, strides)

	if idx >= len(t.Data) {
		panic("tensor.SetAt: computed index out of bounds")
	}

	t.Data[idx] = value
}

// Reshape returns a new tensor with the same data but different shape (zero-copy when possible).
// The total number of elements must remain the same.
// If the tensor is contiguous, this creates a view without copying data.
func (t *Tensor) Reshape(newShape []int) *Tensor {
	if t == nil {
		return nil
	}

	// Calculate size of new shape
	newSize := 1
	for _, d := range newShape {
		if d <= 0 {
			panic("tensor.Reshape: all dimensions must be positive")
		}
		newSize *= d
	}

	// Validate that sizes match
	oldSize := t.Size()
	if newSize != oldSize {
		panic("tensor.Reshape: cannot reshape tensor with different total size")
	}

	// Create new tensor with same data (zero-copy view)
	// This is safe because we're just changing how we interpret the data
	return &Tensor{
		Dim:  newShape,
		Data: t.Data, // Share the same underlying slice
	}
}

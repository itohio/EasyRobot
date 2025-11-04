package eager_tensor

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Tensor represents a multi-dimensional array of values of type specified by types.DataType.
// Data is stored in a contiguous slice of values.
// This Tensor has eager execution semantics.
type Tensor struct {
	shape types.Shape
	data  any
}

// New creates a new tensor with the provided data type and shape.
// The underlying buffer is zero-initialized.
func New(dtype types.DataType, shape types.Shape) Tensor {
	size := shape.Size()
	buf := types.MakeTensorData(dtype, size)
	if buf == nil {
		panic(fmt.Sprintf("unsupported dtype: %v", dtype))
	}
	return Tensor{shape: shape, data: buf}
}

// NewAs creates a new tensor with the same data type and shape as the given tensor.
func NewAs(t types.Tensor) types.Tensor {
	return FromFloat32(t.Shape(), t.Data().([]float32))
}

// Empty creates an empty tensor of given data type.
func Empty(dt types.DataType) Tensor {
	return Tensor{}
}

// EmptyAs creates an empty tensor with the same data type as the given tensor.
func EmptyAs(t types.Tensor) Tensor {
	return Tensor{data: types.MakeTensorData(t.DataType(), 0)}
}

func FromArray[T types.DataElementType](shape types.Shape, data []T) Tensor {
	size := shape.Size()
	if len(data) != size {
		panic(fmt.Sprintf("tensor.FromArray: data length %d does not match shape size %d", len(data), size))
	}
	return Tensor{shape: shape, data: data}
}

// FromFloat32 constructs an FP32 tensor from an existing backing slice.
// If data is nil, a new buffer is allocated. The slice is used directly (no copy).
func FromFloat32(shape types.Shape, data []float32) Tensor {
	return FromArray(shape, data)
}

func (t Tensor) Empty() bool {
	return t.shape == nil && t.data == nil
}

// types.DataType returns the tensor's data type.
func (t Tensor) DataType() types.DataType {
	return types.TypeFromData(t.data)
}

func (t Tensor) Data() any {
	if t.data == nil {
		// Return typed nil slice to allow safe type assertion
		// Default to float32 for empty tensors (most common case)
		// This allows Data().([]float32) to work without panicking
		return []float32(nil)
	}
	return t.data
}

// Shape returns a copy of the tensor's shape.
func (t Tensor) Shape() types.Shape {
	if t.shape == nil {
		return nil
	}
	return t.shape
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
		tData := types.GetTensorData[[]float32](&t)
		if tData == nil {
			return 0
		}
		return len(tData)
	}
	return t.shape.Size()
}

// Clone creates a deep copy of the tensor.
func (t Tensor) Clone() types.Tensor {
	if t.shape == nil && t.data == nil {
		return nil
	}

	clonedData := types.CloneTensorDataTo(t.DataType(), t.Data())
	clonedShape := t.Shape().Clone()
	return &Tensor{shape: clonedShape, data: clonedData}
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
	tData := types.GetTensorData[[]float32](&t)
	if tData == nil {
		return false
	}
	return len(tData) == expectedSize
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
	tData := types.GetTensorData[[]float32](&t)
	if t.shape == nil || (t.shape.Rank() == 0 && len(indices) == 0) {
		if tData == nil || len(tData) == 0 {
			panic("tensor.At: empty tensor")
		}
		return tData[0]
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
	if tData == nil || linearIdx >= len(tData) {
		panic("tensor.At: computed index out of bounds")
	}
	return tData[linearIdx]
}

// SetAt sets the element at the given indices.
// Indices must match the tensor's dimensions.
func (t Tensor) SetAt(indices []int, value float32) {
	tData := types.GetTensorData[[]float32](&t)
	if t.shape == nil || (t.shape.Rank() == 0 && len(indices) == 0) {
		if tData == nil || len(tData) == 0 {
			panic("tensor.SetAt: cannot set element of empty tensor")
		}
		tData[0] = value
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
	if tData == nil || linearIdx >= len(tData) {
		panic("tensor.SetAt: computed index out of bounds")
	}
	tData[linearIdx] = value
}

// Reshape returns a new tensor with the same data but different shape (zero-copy when possible).
// The total number of elements must remain the same.
func (t Tensor) Reshape(newShape types.Shape) types.Tensor {
	if t.shape == nil && t.data == nil {
		return nil
	}
	shape := types.NewShape(newShape...)
	tData := types.GetTensorData[[]float32](&t)
	if tData == nil {
		panic("tensor.Reshape: cannot reshape tensor with nil data")
	}
	if shape.Size() != len(tData) {
		panic("tensor.Reshape: cannot reshape tensor with different total size")
	}
	return Tensor{shape: shape, data: t.data}
}

// reset replaces the tensor contents with the provided dtype, shape, and optional backing slice (no copy).
func (t Tensor) reset(dtype types.DataType, shape []int, data []float32) {
	s := types.NewShape(shape...)
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
	t.shape = s
	t.data = buf
}

// Element represents a single tensor element with Get and Set methods.
type Element struct {
	tensor Tensor
	index  int
}

// Get returns the float32 value at this element's position.
func (e Element) Get() float32 {
	tData := types.GetTensorData[[]float32](&e.tensor)
	return tData[e.index]
}

// Set sets the float32 value at this element's position.
func (e Element) Set(value float32) {
	tData := types.GetTensorData[[]float32](&e.tensor)
	tData[e.index] = value
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
func (t Tensor) Elements(fixedAxisValuePairs ...int) func(func(types.Element) bool) {
	tData := types.GetTensorData[[]float32](&t)
	if t.shape == nil || tData == nil || len(tData) == 0 {
		return func(yield func(types.Element) bool) {
			// Empty tensor - yield once with invalid element
			yield(Element{tensor: t, index: 0})
		}
	}

	// Use shape iterator to get index combinations
	shapeIter := t.shape.Iterator(fixedAxisValuePairs...)
	strides := t.shape.Strides()

	return func(yield func(types.Element) bool) {
		for indices := range shapeIter {
			// Compute linear index from multi-dimensional indices
			linearIdx := t.elementIndex(indices, strides)
			if linearIdx >= len(tData) {
				continue // Skip invalid indices
			}

			elem := Element{tensor: t, index: linearIdx}
			if !yield(elem) {
				return
			}
		}
	}
}

package eager_tensor

import (
	"fmt"
	"unsafe"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
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

// EmptyLike creates an empty tensor with the same data type as the given tensor.
func EmptyLike(t types.Tensor) Tensor {
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

func (t Tensor) ID() uintptr {
	if t.Empty() {
		return 0
	}
	switch t := t.data.(type) {
	case []float32:
		return uintptr(unsafe.Pointer(&t[0]))
	case []float64:
		return uintptr(unsafe.Pointer(&t[0]))
	case []int16:
		return uintptr(unsafe.Pointer(&t[0]))
	case []int8:
		return uintptr(unsafe.Pointer(&t[0]))
	default:
		return 0
	}
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
		tData := types.GetTensorData[[]float32](t)
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
	return Tensor{shape: clonedShape, data: clonedData}
}

// Copy copies data from src tensor into this tensor.
// Both tensors must have the same shape.
// Supports data type conversion between different tensor data types.
// Uses optimized primitive copy functions for efficient copying.
// Returns self for method chaining. Panics if shapes don't match.
func (t Tensor) Copy(src types.Tensor) types.Tensor {
	if src == nil {
		return t
	}
	if t.shape == nil && t.data == nil {
		return t
	}

	// Validate shapes match
	if t.Shape() == nil || src.Shape() == nil {
		panic("tensor.Copy: cannot copy to/from tensor with nil shape")
	}

	if !t.Shape().Equal(src.Shape()) {
		panic(fmt.Sprintf("tensor.Copy: shape mismatch: dst %v vs src %v", t.Shape(), src.Shape()))
	}

	// Use the existing optimized copyTensorData function
	// Pass pointer to t so copyTensorData can modify it
	copyTensorData(src, t)

	return t
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
	tData := types.GetTensorData[[]float32](t)
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

// getElementAtIndex returns the element at the given linear index, converting from the actual data type to float64.
func (t Tensor) getElementAtIndex(index int) float64 {
	if t.data == nil {
		panic("tensor.getElementAtIndex: tensor has nil data")
	}

	switch data := t.data.(type) {
	case []float32:
		if index >= len(data) {
			panic("tensor.getElementAtIndex: index out of bounds")
		}
		return primitive.ConvertValue[float32, float64](data[index])
	case []float64:
		if index >= len(data) {
			panic("tensor.getElementAtIndex: index out of bounds")
		}
		return data[index]
	case []int16:
		if index >= len(data) {
			panic("tensor.getElementAtIndex: index out of bounds")
		}
		return primitive.ConvertValue[int16, float64](data[index])
	case []int8:
		if index >= len(data) {
			panic("tensor.getElementAtIndex: index out of bounds")
		}
		return primitive.ConvertValue[int8, float64](data[index])
	default:
		panic(fmt.Sprintf("tensor.getElementAtIndex: unsupported data type"))
	}
}

// setElementAtIndex sets the element at the given linear index, converting from float64 to the actual data type.
func (t Tensor) setElementAtIndex(index int, value float64) {
	if t.data == nil {
		panic("tensor.setElementAtIndex: tensor has nil data")
	}

	switch data := t.data.(type) {
	case []float32:
		if index >= len(data) {
			panic("tensor.setElementAtIndex: index out of bounds")
		}
		data[index] = primitive.ConvertValue[float64, float32](value)
	case []float64:
		if index >= len(data) {
			panic("tensor.setElementAtIndex: index out of bounds")
		}
		data[index] = value
	case []int16:
		if index >= len(data) {
			panic("tensor.setElementAtIndex: index out of bounds")
		}
		data[index] = primitive.ConvertValue[float64, int16](value)
	case []int8:
		if index >= len(data) {
			panic("tensor.setElementAtIndex: index out of bounds")
		}
		data[index] = primitive.ConvertValue[float64, int8](value)
	default:
		panic(fmt.Sprintf("tensor.setElementAtIndex: unsupported data type"))
	}
}

// At returns the element at the given indices.
// When only one index is provided and tensor rank > 1, uses linear indexing (direct data access).
// Otherwise, indices must match the tensor's dimensions for multi-dimensional access.
func (t Tensor) At(indices ...int) float64 {
	if t.shape == nil || (t.shape.Rank() == 0 && len(indices) == 0) {
		if t.data == nil {
			panic("tensor.At: empty tensor")
		}
		// Handle scalar case - convert from actual type to float64
		return t.getElementAtIndex(0)
	}

	// Special case: single index with rank > 1 uses linear indexing
	if len(indices) == 1 && t.shape.Rank() > 1 {
		idx := indices[0]
		size := t.Size()
		if idx < 0 || idx >= size {
			panic("tensor.At: linear index out of bounds")
		}
		return t.getElementAtIndex(idx)
	}

	// Normal multi-dimensional indexing
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
	return t.getElementAtIndex(linearIdx)
}

// SetAt sets the element at the given indices.
// When only one index is provided and tensor rank > 1, uses linear indexing (direct data access).
// Otherwise, indices must match the tensor's dimensions for multi-dimensional access.
func (t Tensor) SetAt(value float64, indices ...int) {
	if t.shape == nil || (t.shape.Rank() == 0 && len(indices) == 0) {
		if t.data == nil {
			panic("tensor.SetAt: cannot set element of empty tensor")
		}
		// Handle scalar case - convert from float64 to actual type
		t.setElementAtIndex(0, value)
		return
	}

	// Special case: single index with rank > 1 uses linear indexing
	if len(indices) == 1 && t.shape.Rank() > 1 {
		idx := indices[0]
		size := t.Size()
		if idx < 0 || idx >= size {
			panic("tensor.SetAt: linear index out of bounds")
		}
		t.setElementAtIndex(idx, value)
		return
	}

	// Normal multi-dimensional indexing
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
	t.setElementAtIndex(linearIdx, value)
}

// Reshape returns a new tensor with the same data but different shape (zero-copy when possible).
// The total number of elements must remain the same.
func (t Tensor) Reshape(newShape types.Shape) types.Tensor {
	if t.shape == nil && t.data == nil {
		return nil
	}
	shape := types.NewShape(newShape...)
	tData := types.GetTensorData[[]float32](t)
	if tData == nil {
		panic("tensor.Reshape: cannot reshape tensor with nil data")
	}
	if shape.Size() != len(tData) {
		panic("tensor.Reshape: cannot reshape tensor with different total size")
	}
	return Tensor{shape: shape, data: t.data}
}

// Element represents a single tensor element with Get and Set methods.
type Element struct {
	tensor Tensor
	index  int
}

// Get returns the float64 value at this element's position.
func (e Element) Get() float64 {
	return e.tensor.getElementAtIndex(e.index)
}

// Set sets the float64 value at this element's position.
func (e Element) Set(value float64) {
	e.tensor.setElementAtIndex(e.index, value)
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
	if t.shape == nil || t.data == nil {
		return func(yield func(types.Element) bool) {
			// Empty tensor - yield once with invalid element
			yield(Element{tensor: t, index: 0})
		}
	}

	// Get size to validate indices
	size := t.Size()
	if size == 0 {
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
			if linearIdx >= size {
				continue // Skip invalid indices
			}

			elem := Element{tensor: t, index: linearIdx}
			if !yield(elem) {
				return
			}
		}
	}
}

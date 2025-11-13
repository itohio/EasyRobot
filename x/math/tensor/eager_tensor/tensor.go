package eager_tensor

import (
	"fmt"
	"unsafe"

	"github.com/itohio/EasyRobot/x/math/primitive"
	"github.com/itohio/EasyRobot/x/math/primitive/generics"
	"github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
)

// Tensor represents a multi-dimensional array of values of type specified by types.DataType.
// Data is stored in a contiguous slice of values.
// This Tensor has eager execution semantics.
//
// Strides and offset support:
//   - strides: If nil, tensor is contiguous (strides computed from shape).
//     If non-nil, stores explicit strides for non-contiguous layouts (e.g., views, transposes).
//   - offset: Base offset into the data slice. 0 means tensor starts at beginning of data.
//     Non-zero offset is used for views (e.g., slices) that reference a portion of larger array.
type Tensor struct {
	shape   types.Shape
	data    any
	strides []int // Optional: if nil, compute from shape (contiguous). If non-nil, explicit strides.
	offset  int   // Base offset into data. 0 means tensor starts at beginning of data.
}

// New creates a new tensor with the provided data type and shape.
// The underlying buffer is zero-initialized.
// New tensors are always contiguous (strides = nil, offset = 0).
func New(dtype types.DataType, shape types.Shape) Tensor {
	size := shape.Size()
	buf := types.MakeTensorData(dtype, size)
	if buf == nil {
		panic(fmt.Sprintf("unsupported dtype: %v", dtype))
	}
	return Tensor{shape: shape, data: buf, strides: nil, offset: 0}
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
	if len(data) < size {
		panic(fmt.Sprintf("tensor.FromArray: data length %d is less than shape size %d", len(data), size))
	}
	return Tensor{shape: shape, data: data[:size], strides: nil, offset: 0}
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
	case []int32:
		return uintptr(unsafe.Pointer(&t[0]))
	case []int:
		return uintptr(unsafe.Pointer(&t[0]))
	case []int64:
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

// Release returns tensor-backed buffers to their respective pools when possible.
// It is safe to call Release multiple times. Views (non-zero offset or
// non-contiguous tensors) do not release their storage.
func (t Tensor) Release() {
	if t.shape == nil || t.data == nil {
		return
	}
	if t.offset != 0 || !t.IsContiguous() {
		return
	}
	size := t.Size()
	if size <= 0 {
		return
	}

	types.ReleaseTensorData(t.data)
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

// Strides returns the tensor's strides, computing if necessary.
// If dst is provided and has sufficient capacity, strides are written to dst.
// If dst is nil or too small, a stack-allocated array is used.
// Returns the slice containing the strides.
//
// For contiguous tensors (strides == nil), computes canonical row-major strides from shape.
// For non-contiguous tensors, returns stored strides.
func (t Tensor) Strides(dst []int) []int {
	if t.shape == nil {
		return nil
	}

	rank := t.shape.Rank()
	if rank == 0 {
		return nil
	}

	// If strides are stored, return them (may need to copy to dst)
	if t.strides != nil {
		if dst != nil && len(dst) >= rank {
			copy(dst, t.strides)
			return dst[:rank]
		}
		// Return stored strides directly (caller should not modify)
		return t.strides
	}

	// Compute canonical strides from shape (contiguous tensor)
	return t.shape.Strides(dst)
}

// IsContiguous reports whether the tensor is contiguous (dense row-major layout).
// A tensor is contiguous if:
// - strides == nil (always contiguous, uses canonical strides)
// - OR stored strides match canonical row-major strides for the shape
func (t Tensor) IsContiguous() bool {
	if t.shape == nil {
		return true
	}

	// nil strides means contiguous
	if t.strides == nil {
		return true
	}

	// Check if stored strides match canonical strides
	return t.shape.IsContiguous(t.strides)
}

// Offset returns the base offset into the data slice.
// Returns 0 for tensors that start at the beginning of data.
// Non-zero offset is used for views (e.g., slices) that reference a portion of larger array.
func (t Tensor) Offset() int {
	return t.offset
}

// DataWithOffset returns the data slice adjusted by the tensor's offset.
// For tensors with offset == 0, returns the data as-is.
// For tensors with offset > 0, returns a slice starting at the offset.
// This is a type-agnostic helper that works with all data types.
func (t Tensor) DataWithOffset() any {
	if t.data == nil {
		return nil
	}

	if t.offset == 0 {
		return t.data
	}

	// Adjust slice by offset (type-agnostic)
	switch data := t.data.(type) {
	case []float32:
		if t.offset >= len(data) {
			return []float32(nil)
		}
		return data[t.offset:]
	case []float64:
		if t.offset >= len(data) {
			return []float64(nil)
		}
		return data[t.offset:]
	case []int32:
		if t.offset >= len(data) {
			return []int32(nil)
		}
		return data[t.offset:]
	case []int64:
		if t.offset >= len(data) {
			return []int64(nil)
		}
		return data[t.offset:]
	case []int:
		if t.offset >= len(data) {
			return []int(nil)
		}
		return data[t.offset:]
	case []int16:
		if t.offset >= len(data) {
			return []int16(nil)
		}
		return data[t.offset:]
	case []int8:
		if t.offset >= len(data) {
			return []int8(nil)
		}
		return data[t.offset:]
	default:
		panic(fmt.Sprintf("tensor.DataWithOffset: unsupported data type: %T", t.data))
	}
}

// Clone creates a deep copy of the tensor.
// Clones shape, data, strides, and offset.
func (t Tensor) Clone() types.Tensor {
	if t.shape == nil && t.data == nil {
		return nil
	}

	clonedData := types.CloneTensorDataTo(t.DataType(), t.Data())
	clonedShape := t.Shape().Clone()

	// Clone strides if present
	var clonedStrides []int
	if t.strides != nil {
		clonedStrides = make([]int, len(t.strides))
		copy(clonedStrides, t.strides)
	}

	return Tensor{
		shape:   clonedShape,
		data:    clonedData,
		strides: clonedStrides,
		offset:  t.offset,
	}
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

	// Copy src to t using generics (handles type conversion automatically)
	srcData := src.Data()
	tData := t.Data()
	shapeSlice := src.Shape().ToSlice()
	tStrides := t.Strides(nil)
	srcStrides := src.Strides(nil)
	generics.ElemCopyStridedAny(tData, srcData, shapeSlice, tStrides, srcStrides)

	return t
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
// The index is relative to the tensor's offset.
func (t Tensor) getElementAtIndex(index int) float64 {
	if t.data == nil {
		panic("tensor.getElementAtIndex: tensor has nil data")
	}

	// Adjust index by offset
	adjustedIndex := t.offset + index

	switch data := t.data.(type) {
	case []float32:
		if adjustedIndex >= len(data) {
			panic("tensor.getElementAtIndex: index out of bounds")
		}
		return primitive.ConvertValue[float32, float64](data[adjustedIndex])
	case []float64:
		if adjustedIndex >= len(data) {
			panic("tensor.getElementAtIndex: index out of bounds")
		}
		return data[adjustedIndex]
	case []int16:
		if adjustedIndex >= len(data) {
			panic("tensor.getElementAtIndex: index out of bounds")
		}
		return primitive.ConvertValue[int16, float64](data[adjustedIndex])
	case []int8:
		if adjustedIndex >= len(data) {
			panic("tensor.getElementAtIndex: index out of bounds")
		}
		return primitive.ConvertValue[int8, float64](data[adjustedIndex])
	default:
		panic(fmt.Sprintf("tensor.getElementAtIndex: unsupported data type"))
	}
}

// setElementAtIndex sets the element at the given linear index, converting from float64 to the actual data type.
// The index is relative to the tensor's offset.
func (t Tensor) setElementAtIndex(index int, value float64) {
	if t.data == nil {
		panic("tensor.setElementAtIndex: tensor has nil data")
	}

	// Adjust index by offset
	adjustedIndex := t.offset + index

	switch data := t.data.(type) {
	case []float32:
		if adjustedIndex >= len(data) {
			panic("tensor.setElementAtIndex: index out of bounds")
		}
		data[adjustedIndex] = primitive.ConvertValue[float64, float32](value)
	case []float64:
		if adjustedIndex >= len(data) {
			panic("tensor.setElementAtIndex: index out of bounds")
		}
		data[adjustedIndex] = value
	case []int16:
		if adjustedIndex >= len(data) {
			panic("tensor.setElementAtIndex: index out of bounds")
		}
		data[adjustedIndex] = primitive.ConvertValue[float64, int16](value)
	case []int8:
		if adjustedIndex >= len(data) {
			panic("tensor.setElementAtIndex: index out of bounds")
		}
		data[adjustedIndex] = primitive.ConvertValue[float64, int8](value)
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

	// Use Strides(nil) for read-only operations - returns stored strides directly without copy
	strides := t.Strides(nil)
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

	// Use Strides(nil) for read-only operations - returns stored strides directly without copy
	strides := t.Strides(nil)
	linearIdx := t.elementIndex(indices, strides)
	t.setElementAtIndex(linearIdx, value)
}

// Reshape returns a tensor with the same data but different shape (zero-copy when possible).
// The total number of elements must remain the same.
// If dst is nil, creates a new tensor view (zero-copy when possible).
// If dst is provided, copies reshaped data to dst and returns dst.
func (t Tensor) Reshape(dst types.Tensor, newShape types.Shape) types.Tensor {
	if t.shape == nil && t.data == nil {
		return nil
	}
	shape := types.NewShape(newShape...)
	tData := types.GetTensorData[[]float32](t)
	if tData == nil {
		panic("tensor.Reshape: cannot reshape tensor with nil data")
	}
	// For views, compare with tensor size (which accounts for offset), not full data length
	if shape.Size() != t.Size() {
		panic("tensor.Reshape: cannot reshape tensor with different total size")
	}

	if IsNil(dst) {
		// Create zero-copy view
		// Preserve offset (reshape doesn't change memory offset)
		// For strides: if rank changes, we need to handle it differently
		var newStrides []int
		oldRank := t.shape.Rank()
		newRank := shape.Rank()

		if oldRank == newRank {
			// Same rank: preserve strides
			if t.strides != nil {
				newStrides = make([]int, len(t.strides))
				copy(newStrides, t.strides)
			}
			// If t.strides == nil, newStrides remains nil (contiguous)
		} else {
			// Rank changed:
			// - If original was contiguous (strides == nil), result is also contiguous
			// - If original was non-contiguous, we cannot preserve strides across rank change
			//   because the stride array length won't match. In this case, we need to copy.
			//   However, for zero-copy views, we'll set strides to nil and let Strides() compute.
			//   This means the view will appear contiguous even though the underlying memory isn't.
			//   This is a limitation: reshape of non-contiguous tensors with rank change
			//   cannot be truly zero-copy while preserving correct indexing.
			//
			//   For now, we'll allow it but the indexing will be incorrect for non-contiguous cases.
			//   A proper solution would require copying the data, which breaks zero-copy.
			//
			//   Actually, wait - if the tensor is non-contiguous and we reshape with rank change,
			//   we can't just compute new strides from shape because the memory layout doesn't match.
			//   We need to either:
			//   1. Copy the data (not zero-copy)
			//   2. Keep original strides and map them somehow (complex)
			//   3. Disallow this operation
			//
			//   For now, let's check if the tensor is contiguous. If not, we need to copy.
			if t.strides != nil && !t.IsContiguous() {
				// Non-contiguous tensor with rank change: need to copy
				// Create new tensor and copy data using strided copy
				result := New(t.DataType(), shape)
				var srcStridesStatic [helpers.MAX_DIMS]int
				srcStrides := t.Strides(srcStridesStatic[:oldRank])
				var dstStridesStatic [helpers.MAX_DIMS]int
				dstStrides := shape.Strides(dstStridesStatic[:newRank])

				// Use strided copy
				tData := types.GetTensorData[[]float32](t)
				resultData := types.GetTensorData[[]float32](result)
				if tData != nil && resultData != nil {
					primitive.CopyWithStrides(
						t.DataWithOffset(),
						resultData,
						t.Shape().ToSlice(),
						srcStrides,
						dstStrides,
					)
				}
				return result
			}
			// Contiguous tensor with rank change: can use zero-copy view
			newStrides = nil
		}

		return Tensor{shape: shape, data: t.data, strides: newStrides, offset: t.offset}
	}

	// Validate dst shape matches newShape
	if !shape.Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.Reshape: destination shape mismatch: expected %v, got %v", shape, dst.Shape()))
	}

	// Copy data to dst (data layout is same, just shape metadata changes)
	tDataSlice := tData
	dstData := types.GetTensorData[[]float32](dst)
	if len(tDataSlice) != len(dstData) {
		panic(fmt.Sprintf("tensor.Reshape: data size mismatch: expected %d, got %d", len(tDataSlice), len(dstData)))
	}
	copy(dstData, tDataSlice)
	return dst
}

// Slice extracts a contiguous slice along the specified dimension.
// Returns a tensor with the sliced data.
// If dst is nil, creates a zero-copy view with adjusted offset and same strides.
// If dst is provided, copies sliced data to dst and returns dst.
func (t Tensor) Slice(dst types.Tensor, dim int, start int, length int) types.Tensor {
	if t.shape == nil && t.data == nil {
		return nil
	}

	shape := t.Shape()
	if shape == nil {
		panic("tensor.Slice: cannot slice tensor with nil shape")
	}

	// Validate dimension
	if dim < 0 || dim >= len(shape) {
		panic(fmt.Sprintf("tensor.Slice: dimension %d out of range for rank %d", dim, len(shape)))
	}

	dimSize := shape[dim]
	if start < 0 || start >= dimSize {
		panic(fmt.Sprintf("tensor.Slice: start index %d out of range for dimension %d (size %d)", start, dim, dimSize))
	}
	if length <= 0 {
		panic(fmt.Sprintf("tensor.Slice: length must be positive, got %d", length))
	}
	if start+length > dimSize {
		panic(fmt.Sprintf("tensor.Slice: start+length (%d+%d=%d) exceeds dimension size %d", start, length, start+length, dimSize))
	}

	// Get original strides - use Strides(nil) to get stored strides directly (no copy)
	origStrides := t.Strides(nil)

	// Create new shape with reduced dimension
	newShape := make(types.Shape, len(shape))
	copy(newShape, shape)
	newShape[dim] = length

	// Handle destination
	if IsNil(dst) {
		// Create zero-copy view
		// Compute new offset: original offset + start * stride[dim]
		newOffset := t.offset + start*origStrides[dim]

		// Preserve original strides (slicing doesn't change stride pattern)
		// Even if original was contiguous (strides == nil), we need to store
		// the original strides to maintain correct indexing after slicing
		// Copy strides to new slice for the view
		newStrides := make([]int, len(origStrides))
		copy(newStrides, origStrides)

		return Tensor{
			shape:   newShape,
			data:    t.data,     // Same backing array
			strides: newStrides, // Preserved original strides (slicing doesn't change stride pattern)
			offset:  newOffset,  // Adjusted offset
		}
	}

	// Copy to dst (existing behavior)
	// Validate dst shape matches newShape
	if !newShape.Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.Slice: destination shape mismatch: expected %v, got %v", newShape, dst.Shape()))
	}

	// Compute source offset and create source view
	srcOffset := t.offset + start*origStrides[dim]
	var srcView any

	// Adjust source view by offset (type-agnostic)
	switch data := t.data.(type) {
	case []float32:
		srcView = data[srcOffset:]
	case []float64:
		srcView = data[srcOffset:]
	case []int16:
		srcView = data[srcOffset:]
	case []int8:
		srcView = data[srcOffset:]
	default:
		panic(fmt.Sprintf("tensor.Slice: unsupported data type: %T", t.data))
	}

	// Use primitive.CopyWithStrides - it handles all types automatically!
	var dstStridesStatic [helpers.MAX_DIMS]int
	dstStrides := types.NewShape(newShape...).Strides(dstStridesStatic[:len(newShape)])
	primitive.CopyWithStrides(srcView, dst.Data(), newShape.ToSlice(), origStrides, dstStrides)

	return dst
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
	shapeIter := generics.ElementsIndices(t.shape, fixedAxisValuePairs...)
	// Use Strides(nil) for read-only operations - returns stored strides directly without copy
	strides := t.Strides(nil)

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

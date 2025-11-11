package gorgonia

import (
	"math"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/eager_tensor"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"gorgonia.org/tensor"
)

// Tensor is a lightweight wrapper around gorgonia's tensor.Dense
// that implements the types.Tensor interface using value receivers.
type Tensor struct {
	dense tensor.Dense
}

// New creates a new tensor with the specified data type and shape.
// All elements are initialized to zero.
//
// Example:
//
//	t := gorgonia.New(types.FP32, 2, 3, 4) // Creates a [2, 3, 4] FP32 tensor
func New(dtype types.DataType, shape ...int) Tensor {
	var dt tensor.Dtype
	switch dtype {
	case types.FP32:
		dt = tensor.Float32
	case types.FP64:
		dt = tensor.Float64
	case types.INT:
		dt = tensor.Int
	case types.INT32:
		dt = tensor.Int32
	case types.INT64:
		dt = tensor.Int64
	case types.INT16:
		dt = tensor.Int16
	case types.INT8:
		dt = tensor.Int8
	default:
		dt = tensor.Float32
	}

	return Tensor{
		dense: *tensor.New(
			tensor.WithShape(shape...),
			tensor.Of(dt),
		),
	}
}

// NewFromDense creates a tensor wrapper from an existing gorgonia Dense tensor.
// The tensor is used by value (embedded copy).
func NewFromDense(d tensor.Dense) Tensor {
	return Tensor{dense: d}
}

// ToEagerTensor converts this gorgonia tensor to an eager_tensor.
// This creates a copy of the data.
func (t Tensor) ToEagerTensor() types.Tensor {
	dt := t.DataType()
	shape := t.Shape()
	data := t.Data()

	// Create eager tensor with appropriate type using FromArray
	switch dt {
	case types.FP32:
		srcData := data.([]float32)
		dstData := make([]float32, len(srcData))
		copy(dstData, srcData)
		return eager_tensor.FromArray(shape, dstData)
	case types.FP64:
		srcData := data.([]float64)
		dstData := make([]float64, len(srcData))
		copy(dstData, srcData)
		return eager_tensor.FromArray(shape, dstData)
	case types.INT:
		srcData := data.([]int)
		dstData := make([]int, len(srcData))
		copy(dstData, srcData)
		return eager_tensor.FromArray(shape, dstData)
	case types.INT32:
		srcData := data.([]int32)
		dstData := make([]int32, len(srcData))
		copy(dstData, srcData)
		return eager_tensor.FromArray(shape, dstData)
	case types.INT64:
		srcData := data.([]int64)
		dstData := make([]int64, len(srcData))
		copy(dstData, srcData)
		return eager_tensor.FromArray(shape, dstData)
	case types.INT16:
		srcData := data.([]int16)
		dstData := make([]int16, len(srcData))
		copy(dstData, srcData)
		return eager_tensor.FromArray(shape, dstData)
	case types.INT8:
		srcData := data.([]int8)
		dstData := make([]int8, len(srcData))
		copy(dstData, srcData)
		return eager_tensor.FromArray(shape, dstData)
	default:
		panic("gorgonia.ToEagerTensor: unsupported data type")
	}
}

// FromEagerTensor creates a gorgonia tensor from an eager tensor.
// This creates a copy of the data.
func FromEagerTensor(t types.Tensor) Tensor {
	dt := t.DataType()
	shape := t.Shape()
	data := t.Data()

	var gdt tensor.Dtype
	switch dt {
	case types.FP32:
		gdt = tensor.Float32
	case types.FP64:
		gdt = tensor.Float64
	case types.INT:
		gdt = tensor.Int
	case types.INT32:
		gdt = tensor.Int32
	case types.INT64:
		gdt = tensor.Int64
	case types.INT16:
		gdt = tensor.Int16
	case types.INT8:
		gdt = tensor.Int8
	default:
		gdt = tensor.Float32
	}

	// Create new gorgonia tensor
	result := tensor.New(
		tensor.WithShape(shape...),
		tensor.Of(gdt),
		tensor.WithBacking(data),
	)

	return Tensor{dense: *result}
}

// convertToEager is a helper to convert any tensor to eager format.
// If already eager, returns as-is. If gorgonia, converts via ToEagerTensor.
func convertToEager(t types.Tensor) types.Tensor {
	if t == nil {
		return nil
	}

	// If it's already a gorgonia tensor, convert it
	if gTensor, ok := t.(Tensor); ok {
		return gTensor.ToEagerTensor()
	}

	// Otherwise assume it's already eager or compatible
	return t
}

// Dense returns a copy of the underlying gorgonia Dense tensor.
func (t Tensor) Dense() tensor.Dense {
	return t.dense
}

// Verify that Tensor implements the types.Tensor interface at compile time.
var _ types.Tensor = Tensor{}

//
// Core Interface Implementation
//

// ID returns a unique identifier for the tensor (memory address of underlying data).
func (t Tensor) ID() uintptr {
	if t.dense.IsManuallyManaged() {
		return uintptr(t.dense.Uintptr())
	}
	return 0
}

// DataType returns the tensor's data type.
func (t Tensor) DataType() types.DataType {
	switch t.dense.Dtype() {
	case tensor.Float32:
		return types.FP32
	case tensor.Float64:
		return types.FP64
	case tensor.Int:
		return types.INT
	case tensor.Int32:
		return types.INT32
	case tensor.Int64:
		return types.INT64
	case tensor.Int16:
		return types.INT16
	case tensor.Int8:
		return types.INT8
	default:
		return types.DT_UNKNOWN
	}
}

// Data returns the underlying data storage.
func (t Tensor) Data() any {
	return t.dense.Data()
}

// Shape returns a copy of the tensor's shape.
func (t Tensor) Shape() types.Shape {
	gShape := t.dense.Shape()
	if gShape == nil {
		return nil
	}
	shape := make(types.Shape, len(gShape))
	copy(shape, gShape)
	return shape
}

// Rank returns the number of dimensions.
func (t Tensor) Rank() int {
	return t.dense.Dims()
}

// Size returns the total number of elements.
func (t Tensor) Size() int {
	return t.dense.Size()
}

// Empty returns true if the tensor is empty.
func (t Tensor) Empty() bool {
	return t.dense.Size() == 0
}

// Strides returns the tensor's strides.
func (t Tensor) Strides(dst []int) []int {
	gStrides := t.dense.Strides()
	if dst == nil || cap(dst) < len(gStrides) {
		dst = make([]int, len(gStrides))
	} else {
		dst = dst[:len(gStrides)]
	}
	copy(dst, gStrides)
	return dst
}

// IsContiguous reports whether the tensor is contiguous.
func (t Tensor) IsContiguous() bool {
	// gorgonia doesn't expose IsContiguous, assume true for now
	// TODO: Implement proper contiguous check
	return true
}

// Offset returns the base offset into the data slice.
func (t Tensor) Offset() int {
	return 0 // gorgonia manages this internally
}

// DataWithOffset returns the data slice adjusted by the tensor's offset.
func (t Tensor) DataWithOffset() any {
	return t.dense.Data()
}

// At returns the element at the given indices.
// When only one index is provided and tensor rank > 1, uses linear indexing.
func (t Tensor) At(indices ...int) float64 {
	// For scalars (rank 0), ignore indices and access directly
	if t.Rank() == 0 {
		val, err := t.dense.At()
		if err != nil {
			panic(err)
		}
		switch v := val.(type) {
		case float32:
			return float64(v)
		case float64:
			return v
		case int:
			return float64(v)
		case int32:
			return float64(v)
		case int64:
			return float64(v)
		case int16:
			return float64(v)
		case int8:
			return float64(v)
		default:
			panic("unsupported data type")
		}
	}

	// Handle linear indexing: single index for multi-dimensional tensor
	if len(indices) == 1 && t.Rank() > 1 {
		linearIdx := indices[0]

		// Convert linear index to multi-dimensional indices
		shape := t.dense.Shape()
		multiIdx := make([]int, len(shape))
		remaining := linearIdx
		for i := len(shape) - 1; i >= 0; i-- {
			multiIdx[i] = remaining % shape[i]
			remaining /= shape[i]
		}
		indices = multiIdx
	}

	val, err := t.dense.At(indices...)
	if err != nil {
		panic(err)
	}

	switch v := val.(type) {
	case float32:
		return float64(v)
	case float64:
		return v
	case int:
		return float64(v)
	case int32:
		return float64(v)
	case int64:
		return float64(v)
	case int16:
		return float64(v)
	case int8:
		return float64(v)
	default:
		panic("unsupported data type")
	}
}

// SetAt sets the element at the given indices.
// When only one index is provided and tensor rank > 1, uses linear indexing.
func (t Tensor) SetAt(value float64, indices ...int) {
	// Handle scalar tensors (rank 0) - no indices needed
	if t.Rank() == 0 {
		indices = nil
	} else if len(indices) == 1 && t.Rank() > 1 {
		// Handle linear indexing: single index for multi-dimensional tensor
		linearIdx := indices[0]

		// Convert linear index to multi-dimensional indices
		shape := t.dense.Shape()
		multiIdx := make([]int, len(shape))
		remaining := linearIdx
		for i := len(shape) - 1; i >= 0; i-- {
			multiIdx[i] = remaining % shape[i]
			remaining /= shape[i]
		}
		indices = multiIdx
	}

	// Convert value to appropriate type for the tensor
	var val any
	switch t.DataType() {
	case types.FP32:
		val = float32(value)
	case types.FP64:
		val = value
	case types.INT:
		val = int(value)
	case types.INT32:
		val = int32(value)
	case types.INT64:
		val = int64(value)
	case types.INT16:
		val = int16(value)
	case types.INT8:
		val = int8(value)
	default:
		val = value
	}

	err := t.dense.SetAt(val, indices...)
	if err != nil {
		panic(err)
	}
}

// Elements creates an iterator over tensor elements.
func (t Tensor) Elements(fixedAxisValuePairs ...int) func(func(types.Element) bool) {
	return func(yield func(types.Element) bool) {
		// For now, implement simple iteration over all elements
		// TODO: Handle fixedAxisValuePairs
		it := t.dense.Iterator()
		for _, err := it.Next(); err == nil; _, err = it.Next() {
			idx := it.Coord()
			elem := &element{t: t, indices: idx}
			if !yield(elem) {
				return
			}
		}
	}
}

// Release allows tensors backed by pooled storage to return their buffers.
func (t Tensor) Release() {
	// gorgonia manages memory internally, nothing to do here
}

// element implements types.Element interface for iteration.
type element struct {
	t       Tensor
	indices []int
}

func (e *element) Get() float64 {
	return e.t.At(e.indices...)
}

func (e *element) Set(value float64) {
	e.t.SetAt(value, e.indices...)
}

//
// Manipulation Interface Implementation
//

// Clone creates a deep copy of the tensor.
func (t Tensor) Clone() types.Tensor {
	cloned := t.dense.Clone()
	return Tensor{dense: *cloned.(*tensor.Dense)}
}

// Copy copies data from src tensor into this tensor.
// Automatically handles conversion from eager tensors to gorgonia tensors.
func (t Tensor) Copy(src types.Tensor) types.Tensor {
	if !t.Shape().Equal(src.Shape()) {
		panic("tensor: shapes don't match for Copy")
	}

	if gSrc, ok := src.(Tensor); ok {
		// Source is also a gorgonia tensor - use native copy
		err := tensor.Copy(&t.dense, &gSrc.dense)
		if err != nil {
			panic(err)
		}
	} else {
		// Source is not a gorgonia tensor (e.g., eager_tensor)
		// Copy data directly using Data() for efficiency
		srcData := src.Data()
		dstData := t.Data()

		// Try direct type-matched copy first for better performance
		switch t.DataType() {
		case types.FP32:
			dstSlice := dstData.([]float32)
			if srcSlice, ok := srcData.([]float32); ok {
				copy(dstSlice, srcSlice)
			} else {
				// Type mismatch, fall back to element-by-element with conversion
				for i := 0; i < t.Size(); i++ {
					t.SetAt(src.At(i), i)
				}
			}
		case types.FP64:
			dstSlice := dstData.([]float64)
			if srcSlice, ok := srcData.([]float64); ok {
				copy(dstSlice, srcSlice)
			} else {
				for i := 0; i < t.Size(); i++ {
					t.SetAt(src.At(i), i)
				}
			}
		case types.INT:
			dstSlice := dstData.([]int)
			if srcSlice, ok := srcData.([]int); ok {
				copy(dstSlice, srcSlice)
			} else {
				for i := 0; i < t.Size(); i++ {
					t.SetAt(src.At(i), i)
				}
			}
		case types.INT32:
			dstSlice := dstData.([]int32)
			if srcSlice, ok := srcData.([]int32); ok {
				copy(dstSlice, srcSlice)
			} else {
				for i := 0; i < t.Size(); i++ {
					t.SetAt(src.At(i), i)
				}
			}
		case types.INT64:
			dstSlice := dstData.([]int64)
			if srcSlice, ok := srcData.([]int64); ok {
				copy(dstSlice, srcSlice)
			} else {
				for i := 0; i < t.Size(); i++ {
					t.SetAt(src.At(i), i)
				}
			}
		case types.INT16:
			dstSlice := dstData.([]int16)
			if srcSlice, ok := srcData.([]int16); ok {
				copy(dstSlice, srcSlice)
			} else {
				for i := 0; i < t.Size(); i++ {
					t.SetAt(src.At(i), i)
				}
			}
		case types.INT8:
			dstSlice := dstData.([]int8)
			if srcSlice, ok := srcData.([]int8); ok {
				copy(dstSlice, srcSlice)
			} else {
				for i := 0; i < t.Size(); i++ {
					t.SetAt(src.At(i), i)
				}
			}
		default:
			// Fallback: copy element by element
			for i := 0; i < t.Size(); i++ {
				t.SetAt(src.At(i), i)
			}
		}
	}
	return t
}

// Reshape returns a tensor with the same data but different shape.
func (t Tensor) Reshape(dst types.Tensor, newShape types.Shape) types.Tensor {
	// Clone first since Reshape is in-place in gorgonia
	cloned := t.dense.Clone().(*tensor.Dense)
	err := cloned.Reshape(newShape...)
	if err != nil {
		panic(err)
	}

	result := Tensor{dense: *cloned}
	if dst != nil {
		return dst.(Tensor).Copy(result)
	}
	return result
}

// Slice extracts a contiguous slice along the specified dimension.
func (t Tensor) Slice(dst types.Tensor, dim int, start int, length int) types.Tensor {
	end := start + length
	sliced, err := t.dense.Slice(tensor.S(start, end))
	if err != nil {
		panic(err)
	}

	result := Tensor{dense: *sliced.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(result)
	}
	return result
}

// Transpose transposes dimensions.
func (t Tensor) Transpose(dst types.Tensor, dims []int) types.Tensor {
	// Clone first since T/Transpose are in-place in gorgonia
	cloned := t.dense.Clone().(*tensor.Dense)
	var err error

	if len(dims) == 0 {
		// Default: transpose last two dimensions (for 2D) or just mark as transposed
		err = cloned.T()
	} else {
		// T with axes for multi-dimensional transpose
		err = cloned.T(dims...)
	}

	if err != nil {
		panic(err)
	}

	result := Tensor{dense: *cloned}
	if dst != nil {
		return dst.(Tensor).Copy(result)
	}
	return result
}

// Permute permutes dimensions according to the provided permutation.
func (t Tensor) Permute(dst types.Tensor, dims []int) types.Tensor {
	return t.Transpose(dst, dims)
}

// BroadcastTo broadcasts the tensor to the target shape.
func (t Tensor) BroadcastTo(dst types.Tensor, shape types.Shape) types.Tensor {
	// gorgonia doesn't have explicit broadcast, need to implement manually
	// For now, panic as not implemented
	panic("gorgonia.Tensor.BroadcastTo: not implemented")
}

// Fill fills the tensor with a constant value.
func (t Tensor) Fill(dst types.Tensor, value float64) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor)
	}

	// Convert value to appropriate type and fill
	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		for i := range data {
			data[i] = float32(value)
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			data[i] = value
		}
	case types.INT:
		data := target.dense.Data().([]int)
		for i := range data {
			data[i] = int(value)
		}
	case types.INT32:
		data := target.dense.Data().([]int32)
		for i := range data {
			data[i] = int32(value)
		}
	case types.INT64:
		data := target.dense.Data().([]int64)
		for i := range data {
			data[i] = int64(value)
		}
	case types.INT16:
		data := target.dense.Data().([]int16)
		for i := range data {
			data[i] = int16(value)
		}
	case types.INT8:
		data := target.dense.Data().([]int8)
		for i := range data {
			data[i] = int8(value)
		}
	default:
		panic("gorgonia.Tensor.Fill: unsupported data type")
	}

	return target
}

// FillFunc fills the tensor with values calculated by callback function.
func (t Tensor) FillFunc(dst types.Tensor, f func() float64) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor)
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		for i := range data {
			data[i] = float32(f())
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			data[i] = f()
		}
	default:
		panic("gorgonia.Tensor.FillFunc: unsupported data type")
	}

	return target
}

// Pad adds padding to tensor with constant value.
func (t Tensor) Pad(dst types.Tensor, padding []int, value float64) types.Tensor {
	panic("gorgonia.Tensor.Pad: not implemented")
}

// Unpad removes padding from tensor.
func (t Tensor) Unpad(dst types.Tensor, padding []int) types.Tensor {
	panic("gorgonia.Tensor.Unpad: not implemented")
}

//
// ElementWise Interface Implementation
//

// Add performs element-wise addition.
func (t Tensor) Add(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.Add(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Subtract performs element-wise subtraction.
func (t Tensor) Subtract(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.Sub(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Multiply performs element-wise multiplication.
func (t Tensor) Multiply(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.Mul(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Divide performs element-wise division.
func (t Tensor) Divide(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.Div(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// ScalarMul multiplies the tensor by a scalar.
func (t Tensor) ScalarMul(dst types.Tensor, scalar float64) types.Tensor {
	return t.MulScalar(dst, scalar)
}

// AddScalar adds a scalar value to all elements.
func (t Tensor) AddScalar(dst types.Tensor, scalar float64) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		// Clone to avoid modifying original
		target = t.Clone().(Tensor)
	}

	// Handle scalar tensors (rank 0) separately
	if target.Rank() == 0 {
		val := target.At(0) + scalar
		target.SetAt(val, 0)
		return target
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		s := float32(scalar)
		for i := range data {
			data[i] += s
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			data[i] += scalar
		}
	default:
		panic("gorgonia.Tensor.AddScalar: unsupported data type")
	}

	return target
}

// SubScalar subtracts a scalar value from all elements.
func (t Tensor) SubScalar(dst types.Tensor, scalar float64) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		// Clone to avoid modifying original
		target = t.Clone().(Tensor)
	}

	// Handle scalar tensors (rank 0) separately
	if target.Rank() == 0 {
		val := target.At(0) - scalar
		target.SetAt(val, 0)
		return target
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		s := float32(scalar)
		for i := range data {
			data[i] -= s
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			data[i] -= scalar
		}
	default:
		panic("gorgonia.Tensor.SubScalar: unsupported data type")
	}

	return target
}

// MulScalar multiplies all elements by a scalar.
func (t Tensor) MulScalar(dst types.Tensor, scalar float64) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		// Clone to avoid modifying original
		target = t.Clone().(Tensor)
	}

	// Handle scalar tensors (rank 0) separately
	if target.Rank() == 0 {
		val := target.At(0) * scalar
		target.SetAt(val, 0)
		return target
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		s := float32(scalar)
		for i := range data {
			data[i] *= s
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			data[i] *= scalar
		}
	default:
		panic("gorgonia.Tensor.MulScalar: unsupported data type")
	}

	return target
}

// DivScalar divides all elements by a scalar.
func (t Tensor) DivScalar(dst types.Tensor, scalar float64) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		// Clone to avoid modifying original
		target = t.Clone().(Tensor)
	}

	// Handle scalar tensors (rank 0) separately
	if target.Rank() == 0 {
		val := target.At(0) / scalar
		target.SetAt(val, 0)
		return target
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		s := float32(scalar)
		for i := range data {
			data[i] /= s
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			data[i] /= scalar
		}
	default:
		panic("gorgonia.Tensor.DivScalar: unsupported data type")
	}

	return target
}

// Square computes element-wise square.
func (t Tensor) Square(dst types.Tensor) types.Tensor {
	return t.Pow(dst, 2.0)
}

// Sqrt computes element-wise square root.
func (t Tensor) Sqrt(dst types.Tensor) types.Tensor {
	result, err := tensor.Sqrt(&t.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Exp computes element-wise exponential.
func (t Tensor) Exp(dst types.Tensor) types.Tensor {
	result, err := tensor.Exp(&t.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Log computes element-wise natural logarithm.
func (t Tensor) Log(dst types.Tensor) types.Tensor {
	result, err := tensor.Log(&t.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Pow computes element-wise power.
func (t Tensor) Pow(dst types.Tensor, power float64) types.Tensor {
	result, err := tensor.Pow(&t.dense, power)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Abs computes element-wise absolute value.
func (t Tensor) Abs(dst types.Tensor) types.Tensor {
	result, err := tensor.Abs(&t.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Sign computes element-wise sign.
func (t Tensor) Sign(dst types.Tensor) types.Tensor {
	result, err := tensor.Sign(&t.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Cos computes element-wise cosine.
func (t Tensor) Cos(dst types.Tensor) types.Tensor {
	// Implement using direct data access since gorgonia might not have Cos
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		for i := range data {
			data[i] = float32(math.Cos(float64(data[i])))
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			data[i] = math.Cos(data[i])
		}
	default:
		panic("gorgonia.Tensor.Cos: unsupported data type")
	}

	return target
}

// Sin computes element-wise sine.
func (t Tensor) Sin(dst types.Tensor) types.Tensor {
	// Implement using direct data access since gorgonia might not have Sin
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		for i := range data {
			data[i] = float32(math.Sin(float64(data[i])))
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			data[i] = math.Sin(data[i])
		}
	default:
		panic("gorgonia.Tensor.Sin: unsupported data type")
	}

	return target
}

// Negative computes element-wise negation.
func (t Tensor) Negative(dst types.Tensor) types.Tensor {
	result, err := tensor.Neg(&t.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Equal returns a tensor with 1.0 where t == other, 0.0 otherwise.
func (t Tensor) Equal(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.ElEq(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Greater returns a tensor with 1.0 where t > other, 0.0 otherwise.
func (t Tensor) Greater(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.Gt(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Less returns a tensor with 1.0 where t < other, 0.0 otherwise.
func (t Tensor) Less(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.Lt(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// NotEqual returns a tensor with 1.0 where t != other, 0.0 otherwise.
func (t Tensor) NotEqual(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.ElNe(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// GreaterEqual returns a tensor with 1.0 where t >= other, 0.0 otherwise.
func (t Tensor) GreaterEqual(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.Gte(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// LessEqual returns a tensor with 1.0 where t <= other, 0.0 otherwise.
func (t Tensor) LessEqual(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.Lte(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// EqualScalar returns a tensor with 1.0 where t == scalar, 0.0 otherwise.
func (t Tensor) EqualScalar(dst types.Tensor, scalar float64) types.Tensor {
	panic("gorgonia.Tensor.EqualScalar: not implemented")
}

// NotEqualScalar returns a tensor with 1.0 where t != scalar, 0.0 otherwise.
func (t Tensor) NotEqualScalar(dst types.Tensor, scalar float64) types.Tensor {
	panic("gorgonia.Tensor.NotEqualScalar: not implemented")
}

// GreaterScalar returns a tensor with 1.0 where t > scalar, 0.0 otherwise.
func (t Tensor) GreaterScalar(dst types.Tensor, scalar float64) types.Tensor {
	panic("gorgonia.Tensor.GreaterScalar: not implemented")
}

// LessScalar returns a tensor with 1.0 where t < scalar, 0.0 otherwise.
func (t Tensor) LessScalar(dst types.Tensor, scalar float64) types.Tensor {
	panic("gorgonia.Tensor.LessScalar: not implemented")
}

// GreaterEqualScalar returns a tensor with 1.0 where t >= scalar, 0.0 otherwise.
func (t Tensor) GreaterEqualScalar(dst types.Tensor, scalar float64) types.Tensor {
	panic("gorgonia.Tensor.GreaterEqualScalar: not implemented")
}

// LessEqualScalar returns a tensor with 1.0 where t <= scalar, 0.0 otherwise.
func (t Tensor) LessEqualScalar(dst types.Tensor, scalar float64) types.Tensor {
	panic("gorgonia.Tensor.LessEqualScalar: not implemented")
}

// Where performs element-wise selection.
func (t Tensor) Where(dst types.Tensor, condition, a, b types.Tensor) types.Tensor {
	panic("gorgonia.Tensor.Where: not implemented")
}

//
// Math Interface Implementation
//

// Sum sums along specified dimensions.
func (t Tensor) Sum(dst types.Tensor, dims []int) types.Tensor {
	var result tensor.Tensor
	var err error

	if len(dims) == 0 {
		// Sum all elements
		result, err = tensor.Sum(&t.dense)
	} else {
		// Sum along specific dimensions
		result = &t.dense
		for _, dim := range dims {
			result, err = tensor.Sum(result, dim)
			if err != nil {
				panic(err)
			}
		}
	}

	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// ReduceSum is an alias for Sum.
func (t Tensor) ReduceSum(dst types.Tensor, dims []int) types.Tensor {
	return t.Sum(dst, dims)
}

// Mean computes mean along specified dimensions.
func (t Tensor) Mean(dst types.Tensor, dims []int) types.Tensor {
	// Gorgonia doesn't have a Mean function, implement it using Sum / size
	sum := t.Sum(nil, dims).(Tensor)

	// Calculate the size of the reduced dimensions
	totalSize := float64(t.Size())
	if len(dims) > 0 {
		reducedSize := 1
		shape := t.Shape()
		for _, dim := range dims {
			reducedSize *= shape[dim]
		}
		totalSize = float64(reducedSize)
	}

	// Divide sum by size to get mean
	result := sum.DivScalar(nil, totalSize).(Tensor)

	if dst != nil {
		return dst.(Tensor).Copy(result)
	}
	return result
}

// ReduceMean is an alias for Mean.
func (t Tensor) ReduceMean(dst types.Tensor, dims []int) types.Tensor {
	return t.Mean(dst, dims)
}

// Max computes maximum along specified dimensions.
func (t Tensor) Max(dst types.Tensor, dims []int) types.Tensor {
	// gorgonia doesn't have a Max reduction function in the tensor package
	// This would need to be implemented manually
	panic("gorgonia.Tensor.Max: not implemented")
}

// ReduceMax is an alias for Max.
func (t Tensor) ReduceMax(dst types.Tensor, dims []int) types.Tensor {
	return t.Max(dst, dims)
}

// Min computes minimum along specified dimensions.
func (t Tensor) Min(dst types.Tensor, dims []int) types.Tensor {
	// gorgonia doesn't have a Min reduction function in the tensor package
	// This would need to be implemented manually
	panic("gorgonia.Tensor.Min: not implemented")
}

// ReduceMin is an alias for Min.
func (t Tensor) ReduceMin(dst types.Tensor, dims []int) types.Tensor {
	return t.Min(dst, dims)
}

// ArgMax returns the index of the maximum element along the specified dimension.
func (t Tensor) ArgMax(dst types.Tensor, dim int) types.Tensor {
	result, err := tensor.Argmax(&t.dense, dim)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// ArgMin returns the index of the minimum element along the specified dimension.
func (t Tensor) ArgMin(dst types.Tensor, dim int) types.Tensor {
	result, err := tensor.Argmin(&t.dense, dim)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// MatMul performs matrix multiplication.
func (t Tensor) MatMul(dst types.Tensor, other types.Tensor) types.Tensor {
	gOther := other.(Tensor)
	result, err := tensor.MatMul(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// MatMulTransposed performs matrix multiplication with optional transposition.
func (t Tensor) MatMulTransposed(dst types.Tensor, other types.Tensor, transposeA, transposeB bool) types.Tensor {
	panic("gorgonia.Tensor.MatMulTransposed: not implemented")
}

// MatVecMulTransposed performs matrix-vector multiplication with scaling.
func (t Tensor) MatVecMulTransposed(dst types.Tensor, matrix, vector types.Tensor, alpha, beta float64) types.Tensor {
	panic("gorgonia.Tensor.MatVecMulTransposed: not implemented")
}

// Dot computes dot product or Frobenius inner product.
func (t Tensor) Dot(other types.Tensor) float64 {
	gOther := other.(Tensor)
	result, err := tensor.Inner(&t.dense, &gOther.dense)
	if err != nil {
		panic(err)
	}

	// Extract scalar value
	val, err := result.(*tensor.Dense).At(0)
	if err != nil {
		panic(err)
	}

	switch v := val.(type) {
	case float32:
		return float64(v)
	case float64:
		return v
	default:
		panic("gorgonia.Tensor.Dot: unsupported result type")
	}
}

// Tensordot is an alias for Dot.
func (t Tensor) Tensordot(other types.Tensor) float64 {
	return t.Dot(other)
}

// Norm computes vector/matrix norm.
func (t Tensor) Norm(ord int) float64 {
	panic("gorgonia.Tensor.Norm: not implemented")
}

// L2Normalize performs L2 normalization along the specified dimension.
func (t Tensor) L2Normalize(dst types.Tensor, dim int) types.Tensor {
	panic("gorgonia.Tensor.L2Normalize: not implemented")
}

// Normalize is an alias for L2Normalize.
func (t Tensor) Normalize(dst types.Tensor, dim int) types.Tensor {
	return t.L2Normalize(dst, dim)
}

// AddScaled computes dst = t + alpha * other.
func (t Tensor) AddScaled(dst types.Tensor, other types.Tensor, alpha float64) types.Tensor {
	panic("gorgonia.Tensor.AddScaled: not implemented")
}

// ScatterAdd adds values to destination tensor at positions specified by indices.
func (t Tensor) ScatterAdd(dst, index, value types.Tensor) types.Tensor {
	panic("gorgonia.Tensor.ScatterAdd: not implemented")
}

//
// Normalizations Interface Implementation
//

// BatchNormForward performs batch normalization.
func (t Tensor) BatchNormForward(dst types.Tensor, gamma, beta types.Tensor, eps float64) types.Tensor {
	panic("gorgonia.Tensor.BatchNormForward: not implemented")
}

// BatchNormGrad computes gradients for batch normalization.
func (t Tensor) BatchNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	panic("gorgonia.Tensor.BatchNormGrad: not implemented")
}

// LayerNormForward performs layer normalization.
func (t Tensor) LayerNormForward(dst types.Tensor, gamma, beta types.Tensor, eps float64) types.Tensor {
	panic("gorgonia.Tensor.LayerNormForward: not implemented")
}

// LayerNormGrad computes gradients for layer normalization.
func (t Tensor) LayerNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	panic("gorgonia.Tensor.LayerNormGrad: not implemented")
}

// RMSNormForward performs RMS normalization.
func (t Tensor) RMSNormForward(dst types.Tensor, gamma types.Tensor, eps float64) types.Tensor {
	panic("gorgonia.Tensor.RMSNormForward: not implemented")
}

// RMSNormGrad computes gradients for RMS normalization.
func (t Tensor) RMSNormGrad(gradInputDst, gradGammaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor) {
	panic("gorgonia.Tensor.RMSNormGrad: not implemented")
}

// InstanceNorm2D performs instance normalization for 2D feature maps.
func (t Tensor) InstanceNorm2D(dst types.Tensor, gamma, beta types.Tensor, eps float64) types.Tensor {
	panic("gorgonia.Tensor.InstanceNorm2D: not implemented")
}

// InstanceNorm2DGrad computes gradients for 2D instance normalization.
func (t Tensor) InstanceNorm2DGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	panic("gorgonia.Tensor.InstanceNorm2DGrad: not implemented")
}

// GroupNormForward performs group normalization.
func (t Tensor) GroupNormForward(dst types.Tensor, gamma, beta types.Tensor, numGroups int, eps float64) types.Tensor {
	panic("gorgonia.Tensor.GroupNormForward: not implemented")
}

// GroupNormGrad computes gradients for group normalization.
func (t Tensor) GroupNormGrad(gradInputDst, gradGammaDst, gradBetaDst, gradOutput, input, gamma types.Tensor, numGroups int, eps float64) (types.Tensor, types.Tensor, types.Tensor) {
	panic("gorgonia.Tensor.GroupNormGrad: not implemented")
}

//
// Activations Interface Implementation
//

// ReLU applies Rectified Linear Unit activation.
func (t Tensor) ReLU(dst types.Tensor) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		for i := range data {
			if data[i] < 0 {
				data[i] = 0
			}
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			if data[i] < 0 {
				data[i] = 0
			}
		}
	default:
		panic("gorgonia.Tensor.ReLU: unsupported data type")
	}

	return target
}

// Sigmoid applies sigmoid activation.
func (t Tensor) Sigmoid(dst types.Tensor) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		for i := range data {
			data[i] = float32(1.0 / (1.0 + math.Exp(-float64(data[i]))))
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			data[i] = 1.0 / (1.0 + math.Exp(-data[i]))
		}
	default:
		panic("gorgonia.Tensor.Sigmoid: unsupported data type")
	}

	return target
}

// Tanh applies hyperbolic tangent activation.
func (t Tensor) Tanh(dst types.Tensor) types.Tensor {
	result, err := tensor.Tanh(&t.dense)
	if err != nil {
		panic(err)
	}

	resultTensor := Tensor{dense: *result.(*tensor.Dense)}
	if dst != nil {
		return dst.(Tensor).Copy(resultTensor)
	}
	return resultTensor
}

// Softmax applies softmax activation along the specified dimension.
func (t Tensor) Softmax(dim int, dst types.Tensor) types.Tensor {
	// For now, implement simple softmax over last dimension
	// Full multi-dim softmax is complex - this handles common case
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	shape := target.Shape()
	if dim < 0 {
		dim = len(shape) + dim
	}

	// TODO: Implement proper multi-dimensional softmax
	// For now, just implement simple last-dimension softmax
	size := target.Size()

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		// Find max for numerical stability
		maxVal := data[0]
		for i := 1; i < size; i++ {
			if data[i] > maxVal {
				maxVal = data[i]
			}
		}
		// Compute exp and sum
		var sum float32
		for i := 0; i < size; i++ {
			data[i] = float32(math.Exp(float64(data[i] - maxVal)))
			sum += data[i]
		}
		// Normalize
		for i := 0; i < size; i++ {
			data[i] /= sum
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		// Find max for numerical stability
		maxVal := data[0]
		for i := 1; i < size; i++ {
			if data[i] > maxVal {
				maxVal = data[i]
			}
		}
		// Compute exp and sum
		var sum float64
		for i := 0; i < size; i++ {
			data[i] = math.Exp(data[i] - maxVal)
			sum += data[i]
		}
		// Normalize
		for i := 0; i < size; i++ {
			data[i] /= sum
		}
	default:
		panic("gorgonia.Tensor.Softmax: unsupported data type")
	}

	return target
}

// ReLU6 applies ReLU6 activation (clamps between 0 and 6).
func (t Tensor) ReLU6(dst types.Tensor) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		for i := range data {
			if data[i] < 0 {
				data[i] = 0
			} else if data[i] > 6 {
				data[i] = 6
			}
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			if data[i] < 0 {
				data[i] = 0
			} else if data[i] > 6 {
				data[i] = 6
			}
		}
	default:
		panic("gorgonia.Tensor.ReLU6: unsupported data type")
	}

	return target
}

// LeakyReLU applies Leaky ReLU activation.
func (t Tensor) LeakyReLU(dst types.Tensor, alpha float64) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		a := float32(alpha)
		for i := range data {
			if data[i] < 0 {
				data[i] = a * data[i]
			}
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			if data[i] < 0 {
				data[i] = alpha * data[i]
			}
		}
	default:
		panic("gorgonia.Tensor.LeakyReLU: unsupported data type")
	}

	return target
}

// ELU applies ELU activation.
func (t Tensor) ELU(dst types.Tensor, alpha float64) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		a := float32(alpha)
		for i := range data {
			if data[i] < 0 {
				data[i] = a * (float32(math.Exp(float64(data[i]))) - 1)
			}
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			if data[i] < 0 {
				data[i] = alpha * (math.Exp(data[i]) - 1)
			}
		}
	default:
		panic("gorgonia.Tensor.ELU: unsupported data type")
	}

	return target
}

// Softplus applies softplus activation: log(1 + exp(x)).
func (t Tensor) Softplus(dst types.Tensor) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		for i := range data {
			data[i] = float32(math.Log(1.0 + math.Exp(float64(data[i]))))
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			data[i] = math.Log(1.0 + math.Exp(data[i]))
		}
	default:
		panic("gorgonia.Tensor.Softplus: unsupported data type")
	}

	return target
}

// Swish applies Swish activation: x * sigmoid(x).
func (t Tensor) Swish(dst types.Tensor) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		for i := range data {
			sigmoid := float32(1.0 / (1.0 + math.Exp(-float64(data[i]))))
			data[i] = data[i] * sigmoid
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			sigmoid := 1.0 / (1.0 + math.Exp(-data[i]))
			data[i] = data[i] * sigmoid
		}
	default:
		panic("gorgonia.Tensor.Swish: unsupported data type")
	}

	return target
}

// GELU applies GELU activation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
func (t Tensor) GELU(dst types.Tensor) types.Tensor {
	target := t
	if dst != nil {
		target = dst.(Tensor).Copy(t).(Tensor)
	} else {
		target = t.Clone().(Tensor)
	}

	const sqrt2OverPi = 0.7978845608028654 // sqrt(2/pi)
	const coeff = 0.044715

	switch target.DataType() {
	case types.FP32:
		data := target.dense.Data().([]float32)
		for i := range data {
			x := float64(data[i])
			inner := sqrt2OverPi * (x + coeff*x*x*x)
			data[i] = float32(0.5 * x * (1.0 + math.Tanh(inner)))
		}
	case types.FP64:
		data := target.dense.Data().([]float64)
		for i := range data {
			x := data[i]
			inner := sqrt2OverPi * (x + coeff*x*x*x)
			data[i] = 0.5 * x * (1.0 + math.Tanh(inner))
		}
	default:
		panic("gorgonia.Tensor.GELU: unsupported data type")
	}

	return target
}

// ReLUGrad computes the ReLU gradient.
func (t Tensor) ReLUGrad(dst types.Tensor, gradOutput types.Tensor) types.Tensor {
	panic("gorgonia.Tensor.ReLUGrad: not implemented")
}

// SigmoidGrad computes the sigmoid gradient.
func (t Tensor) SigmoidGrad(dst types.Tensor, gradOutput types.Tensor) types.Tensor {
	panic("gorgonia.Tensor.SigmoidGrad: not implemented")
}

// TanhGrad computes the tanh gradient.
func (t Tensor) TanhGrad(dst types.Tensor, gradOutput types.Tensor) types.Tensor {
	panic("gorgonia.Tensor.TanhGrad: not implemented")
}

// SoftmaxGrad computes the softmax gradient.
func (t Tensor) SoftmaxGrad(dst types.Tensor, gradOutput types.Tensor, dim int) types.Tensor {
	panic("gorgonia.Tensor.SoftmaxGrad: not implemented")
}

//
// Convolutions Interface Implementation
//

// Conv1D performs 1D convolution.
// Falls back to eager_tensor implementation for now.
func (t Tensor) Conv1D(dst types.Tensor, kernel, bias types.Tensor, stride, padding int) types.Tensor {
	// Convert to eager tensor, perform operation, convert back
	eagerInput := t.ToEagerTensor()
	eagerKernel := convertToEager(kernel)
	var eagerBias types.Tensor
	if bias != nil && bias.Size() > 0 {
		eagerBias = convertToEager(bias)
	}

	eagerResult := eagerInput.Conv1D(nil, eagerKernel, eagerBias, stride, padding)
	return FromEagerTensor(eagerResult)
}

// Conv2D performs 2D convolution.
// Falls back to eager_tensor implementation for now.
func (t Tensor) Conv2D(dst types.Tensor, kernel, bias types.Tensor, stride, padding []int) types.Tensor {
	// Convert to eager tensor, perform operation, convert back
	eagerInput := t.ToEagerTensor()
	eagerKernel := convertToEager(kernel)
	var eagerBias types.Tensor
	if bias != nil && bias.Size() > 0 {
		eagerBias = convertToEager(bias)
	}

	eagerResult := eagerInput.Conv2D(nil, eagerKernel, eagerBias, stride, padding)
	return FromEagerTensor(eagerResult)
}

// Conv2DTransposed performs transposed 2D convolution.
// Falls back to eager_tensor implementation.
func (t Tensor) Conv2DTransposed(dst types.Tensor, kernel, bias types.Tensor, stride, padding []int) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerKernel := convertToEager(kernel)
	var eagerBias types.Tensor
	if bias != nil && bias.Size() > 0 {
		eagerBias = convertToEager(bias)
	}

	eagerResult := eagerInput.Conv2DTransposed(nil, eagerKernel, eagerBias, stride, padding)
	return FromEagerTensor(eagerResult)
}

// Conv2DKernelGrad computes the gradient of the convolution kernel.
func (t Tensor) Conv2DKernelGrad(dst types.Tensor, outputGrad, kernel types.Tensor, stride, padding []int) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerOutputGrad := convertToEager(outputGrad)
	eagerKernel := convertToEager(kernel)

	eagerResult := eagerInput.Conv2DKernelGrad(nil, eagerOutputGrad, eagerKernel, stride, padding)
	return FromEagerTensor(eagerResult)
}

// Conv1DKernelGrad computes the gradient of the 1D convolution kernel.
func (t Tensor) Conv1DKernelGrad(dst types.Tensor, outputGrad, kernel types.Tensor, stride, padding int) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerOutputGrad := convertToEager(outputGrad)
	eagerKernel := convertToEager(kernel)

	eagerResult := eagerInput.Conv1DKernelGrad(nil, eagerOutputGrad, eagerKernel, stride, padding)
	return FromEagerTensor(eagerResult)
}

// Im2Col converts image patches to columns.
func (t Tensor) Im2Col(dst types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerResult := eagerInput.Im2Col(nil, kernelSize, stride, padding)
	return FromEagerTensor(eagerResult)
}

// Col2Im converts columns back to image.
func (t Tensor) Col2Im(dst types.Tensor, outputShape, kernelSize, stride, padding []int) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerResult := eagerInput.Col2Im(nil, outputShape, kernelSize, stride, padding)
	return FromEagerTensor(eagerResult)
}

//
// Pooling Interface Implementation
//

// MaxPool2D performs max pooling operation.
func (t Tensor) MaxPool2D(dst types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerResult := eagerInput.MaxPool2D(nil, kernelSize, stride, padding)
	return FromEagerTensor(eagerResult)
}

// MaxPool2DWithIndices performs max pooling and returns both output and indices.
func (t Tensor) MaxPool2DWithIndices(dst types.Tensor, indicesDst types.Tensor, kernelSize, stride, padding []int) (types.Tensor, types.Tensor) {
	eagerInput := t.ToEagerTensor()
	eagerOut, eagerIndices := eagerInput.MaxPool2DWithIndices(nil, nil, kernelSize, stride, padding)
	return FromEagerTensor(eagerOut), FromEagerTensor(eagerIndices)
}

// MaxPool2DBackward performs backward pass for max pooling.
func (t Tensor) MaxPool2DBackward(dst types.Tensor, gradOutput, indices types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerGradOutput := convertToEager(gradOutput)
	eagerIndices := convertToEager(indices)
	eagerResult := eagerInput.MaxPool2DBackward(nil, eagerGradOutput, eagerIndices, kernelSize, stride, padding)
	return FromEagerTensor(eagerResult)
}

// AvgPool2D performs average pooling operation.
func (t Tensor) AvgPool2D(dst types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerResult := eagerInput.AvgPool2D(nil, kernelSize, stride, padding)
	return FromEagerTensor(eagerResult)
}

// AvgPool2DBackward performs backward pass for average pooling.
func (t Tensor) AvgPool2DBackward(dst types.Tensor, gradOutput types.Tensor, kernelSize, stride, padding []int) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerGradOutput := convertToEager(gradOutput)
	eagerResult := eagerInput.AvgPool2DBackward(nil, eagerGradOutput, kernelSize, stride, padding)
	return FromEagerTensor(eagerResult)
}

// GlobalAvgPool2D performs global average pooling.
func (t Tensor) GlobalAvgPool2D(dst types.Tensor) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerResult := eagerInput.GlobalAvgPool2D(nil)
	return FromEagerTensor(eagerResult)
}

// AdaptiveAvgPool2D performs adaptive average pooling to fixed output size.
func (t Tensor) AdaptiveAvgPool2D(dst types.Tensor, outputSize []int) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerResult := eagerInput.AdaptiveAvgPool2D(nil, outputSize)
	return FromEagerTensor(eagerResult)
}

//
// Dropout Interface Implementation
//

// DropoutForward applies dropout mask during forward pass.
func (t Tensor) DropoutForward(dst types.Tensor, mask types.Tensor) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerMask := convertToEager(mask)
	eagerResult := eagerInput.DropoutForward(nil, eagerMask)
	return FromEagerTensor(eagerResult)
}

// DropoutMask creates a dropout mask with given probability and scale.
func (t Tensor) DropoutMask(p, scale float64, rng types.RNG) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerResult := eagerInput.DropoutMask(p, scale, rng)
	return FromEagerTensor(eagerResult)
}

// DropoutBackward computes dropout backward pass.
func (t Tensor) DropoutBackward(dst types.Tensor, gradOutput, mask types.Tensor) types.Tensor {
	eagerInput := t.ToEagerTensor()
	eagerGradOutput := convertToEager(gradOutput)
	eagerMask := convertToEager(mask)
	eagerResult := eagerInput.DropoutBackward(nil, eagerGradOutput, eagerMask)
	return FromEagerTensor(eagerResult)
}

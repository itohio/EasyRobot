package eager_tensor

import (
	"errors"
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

func isTensorContiguous(t types.Tensor) bool {
	otherTensor, ok := t.(Tensor)
	if !ok {
		// For non-eager tensors, assume non-contiguous (safer)
		return false
	}
	return otherTensor.isContiguous()
}

// copyTensorData copies data from src to dst using primitive.CopyWithConversion for type conversion.
// Both tensors must have the same shape.
// Supports data type conversion between different tensor data types.
func copyTensorData(src, dst types.Tensor) {
	if src == nil || dst == nil {
		return
	}

	srcShape := src.Shape()
	if srcShape == nil {
		return
	}

	if dst.Shape() == nil {
		return
	}

	if srcShape.Rank() != dst.Shape().Rank() {
		panic(fmt.Sprintf("copyTensorData: shape rank mismatch"))
	}

	size := src.Size()
	if size == 0 {
		return
	}

	srcDtype := src.DataType()
	dstDtype := dst.DataType()
	srcData := src.Data()
	dstData := dst.Data()

	if srcData == nil || dstData == nil {
		return
	}

	srcStrides := srcShape.Strides()
	dstStrides := dst.Shape().Strides()
	shapeSlice := srcShape.ToSlice()

	// Fast path: same type
	if srcDtype == dstDtype {
		srcContiguous := isTensorContiguous(src)
		dstContiguous := isTensorContiguous(dst)

		if srcContiguous && dstContiguous {
			// Use fast contiguous copy for float32
			if srcDtype == types.FP32 {
				srcSlice, ok1 := srcData.([]float32)
				dstSlice, ok2 := dstData.([]float32)
				if ok1 && ok2 {
					fp32.Copy(dstSlice, srcSlice, 1, 1, size)
					return
				}
			}
			// For other types, use primitive.CopyWithConversion (works on contiguous slices)
			primitive.CopyWithConversion(dstData, srcData)
			return
		}

		// For other types with strides, fall through to element-by-element copy
	}

	// Different types or non-float32: use primitive.CopyWithStrides
	primitive.CopyWithStrides(srcData, dstData, shapeSlice, srcStrides, dstStrides)
}

// Add adds another tensor element-wise in-place.
// Uses fp32 primitive.Axpy for efficient computation.
// Returns the tensor itself for method chaining.
func (t Tensor) Add(other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return t
	}

	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.Add: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	otherData := types.GetTensorData[[]float32](other)
	if t.isContiguous() && isTensorContiguous(other) {
		size := t.Size()
		fp32.Axpy(types.GetTensorData[[]float32](t), otherData, 1, 1, size, 1.0)
		return t
	}

	stridesT := t.shape.Strides()
	stridesOther := other.Shape().Strides()
	fp32.ElemAdd(types.GetTensorData[[]float32](t), types.GetTensorData[[]float32](t), otherData, []int(t.shape), stridesT, stridesT, stridesOther)
	return t
}

// Sub subtracts another tensor element-wise in-place.
// Uses fp32 primitive.Axpy with alpha=-1.
func (t Tensor) Sub(other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return t
	}

	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.Sub: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	otherData := types.GetTensorData[[]float32](other)
	if t.isContiguous() && isTensorContiguous(other) {
		size := t.Size()
		fp32.Axpy(types.GetTensorData[[]float32](t), otherData, 1, 1, size, -1.0)
		return t
	}

	stridesT := t.shape.Strides()
	stridesOther := other.Shape().Strides()
	fp32.ElemSub(types.GetTensorData[[]float32](t), types.GetTensorData[[]float32](t), otherData, []int(t.shape), stridesT, stridesT, stridesOther)
	return t
}

// Mul multiplies another tensor element-wise in-place.
func (t Tensor) Mul(other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return t
	}

	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.Mul: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	otherData := types.GetTensorData[[]float32](other)
	stridesT := t.shape.Strides()
	stridesOther := other.Shape().Strides()
	fp32.ElemMul(types.GetTensorData[[]float32](t), types.GetTensorData[[]float32](t), otherData, []int(t.shape), stridesT, stridesT, stridesOther)
	return t
}

// Multiply is an alias for Mul (matches TensorFlow naming: tf.multiply).
func (t Tensor) Multiply(other types.Tensor) types.Tensor {
	return t.Mul(other)
}

// Subtract is an alias for Sub (matches TensorFlow naming: tf.subtract).
func (t Tensor) Subtract(other types.Tensor) types.Tensor {
	return t.Sub(other)
}

// Divide is an alias for Div (matches TensorFlow naming: tf.divide).
func (t Tensor) Divide(other types.Tensor) types.Tensor {
	return t.Div(other)
}

// Div divides by another tensor element-wise in-place.
func (t Tensor) Div(other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return t
	}

	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.Div: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	otherData := types.GetTensorData[[]float32](other)
	stridesT := t.shape.Strides()
	stridesOther := other.Shape().Strides()
	fp32.ElemDiv(types.GetTensorData[[]float32](t), types.GetTensorData[[]float32](t), otherData, []int(t.shape), stridesT, stridesT, stridesOther)
	return t
}

// Scale multiplies the tensor by a scalar in-place.
// Uses fp32 primitive.Scal for efficient computation.
// Converts float64 scalar to float32 for internal computation.
func (t Tensor) Scale(scalar float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	scalar32 := float32(scalar)
	if t.isContiguous() {
		size := t.Size()
		fp32.Scal(types.GetTensorData[[]float32](t), 1, size, scalar32)
		return t
	}

	fp32.ElemScaleInPlace(types.GetTensorData[[]float32](t), scalar32, []int(t.shape), strides)
	return t
}

// ScalarMul is an alias for Scale (matches TensorFlow naming: tf.scalar_mul).
func (t Tensor) ScalarMul(scalar float64) types.Tensor {
	return t.Scale(scalar)
}

// AddScalar adds a scalar value to all elements in-place.
func (t Tensor) AddScalar(scalar float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	scalar32 := float32(scalar)
	strides := t.shape.Strides()
	tData := types.GetTensorData[[]float32](t)

	fp32.ElemAddScalar(tData, tData, scalar32, []int(t.shape), strides, strides)
	return t
}

// SubScalar subtracts a scalar value from all elements in-place.
func (t Tensor) SubScalar(scalar float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	scalar32 := float32(scalar)
	strides := t.shape.Strides()
	tData := types.GetTensorData[[]float32](t)

	fp32.ElemSubScalar(tData, tData, scalar32, []int(t.shape), strides, strides)
	return t
}

// MulScalar multiplies all elements by a scalar in-place.
func (t Tensor) MulScalar(scalar float64) types.Tensor {
	return t.Scale(scalar)
}

// DivScalar divides all elements by a scalar in-place.
func (t Tensor) DivScalar(scalar float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	scalar32 := float32(scalar)
	strides := t.shape.Strides()
	tData := types.GetTensorData[[]float32](t)

	fp32.ElemDivScalar(tData, tData, scalar32, []int(t.shape), strides, strides)
	return t
}

// ScaleTo multiplies the tensor by a scalar and stores result in dst.
func (t Tensor) ScaleTo(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	scalar32 := float32(scalar)
	if dst == nil {
		result := t.Clone()
		result.Scale(scalar)
		return result
	}

	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.ScaleTo: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	tData := types.GetTensorData[[]float32](t)
	dstData := types.GetTensorData[[]float32](dst)
	tStrides := t.shape.Strides()
	dstStrides := dst.Shape().Strides()

	fp32.ElemScale(dstData, tData, scalar32, []int(t.shape), dstStrides, tStrides)
	return dst
}

// AddScalarTo adds a scalar value to all elements and stores result in dst.
func (t Tensor) AddScalarTo(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	scalar32 := float32(scalar)
	if dst == nil {
		result := t.Clone()
		result.AddScalar(scalar)
		return result
	}

	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.AddScalarTo: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	tData := types.GetTensorData[[]float32](t)
	dstData := types.GetTensorData[[]float32](dst)
	tStrides := t.shape.Strides()
	dstStrides := dst.Shape().Strides()

	fp32.ElemAddScalar(dstData, tData, scalar32, []int(t.shape), dstStrides, tStrides)
	return dst
}

// SubScalarTo subtracts a scalar value from all elements and stores result in dst.
func (t Tensor) SubScalarTo(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	scalar32 := float32(scalar)
	if dst == nil {
		result := t.Clone()
		result.SubScalar(scalar)
		return result
	}

	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.SubScalarTo: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	tData := types.GetTensorData[[]float32](t)
	dstData := types.GetTensorData[[]float32](dst)
	tStrides := t.shape.Strides()
	dstStrides := dst.Shape().Strides()

	fp32.ElemSubScalar(dstData, tData, scalar32, []int(t.shape), dstStrides, tStrides)
	return dst
}

// MulScalarTo multiplies all elements by a scalar and stores result in dst.
func (t Tensor) MulScalarTo(dst types.Tensor, scalar float64) types.Tensor {
	return t.ScaleTo(dst, scalar)
}

// DivScalarTo divides all elements by a scalar and stores result in dst.
func (t Tensor) DivScalarTo(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	scalar32 := float32(scalar)
	if dst == nil {
		result := t.Clone()
		result.DivScalar(scalar)
		return result
	}

	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.DivScalarTo: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	tData := types.GetTensorData[[]float32](t)
	dstData := types.GetTensorData[[]float32](dst)
	tStrides := t.shape.Strides()
	dstStrides := dst.Shape().Strides()

	fp32.ElemDivScalar(dstData, tData, scalar32, []int(t.shape), dstStrides, tStrides)
	return dst
}

// AddTo adds two tensors and stores result in dst (or creates new tensor if dst is nil).
// Returns the destination tensor.
func (t Tensor) AddTo(other types.Tensor, dst types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.AddTo: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	if dst == nil {
		result := t.Clone()
		result.Add(other)
		return result
	}

	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.AddTo: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	// Copy t to dst, then add other
	copyTensorData(t, dst)
	dst.Add(other)
	return dst
}

// MulTo multiplies two tensors element-wise and stores result in dst (or creates new tensor if dst is nil).
func (t Tensor) MulTo(other types.Tensor, dst types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.MulTo: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	if dst == nil {
		result := t.Clone()
		result.Mul(other)
		return result
	}

	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.MulTo: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	// Copy t to dst, then multiply by other
	copyTensorData(t, dst)
	dst.Mul(other)
	return dst
}

// SubTo subtracts two tensors element-wise and stores result in dst (or creates new tensor if dst is nil).
func (t Tensor) SubTo(other types.Tensor, dst types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.SubTo: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	if dst == nil {
		result := t.Clone()
		result.Sub(other)
		return result
	}

	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.SubTo: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	// Copy t to dst, then subtract other
	copyTensorData(t, dst)
	dst.Sub(other)
	return dst
}

// DivTo divides two tensors element-wise and stores result in dst (or creates new tensor if dst is nil).
func (t Tensor) DivTo(other types.Tensor, dst types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.DivTo: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	if dst == nil {
		result := t.Clone()
		result.Div(other)
		return result
	}

	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.DivTo: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	// Copy t to dst, then divide by other
	copyTensorData(t, dst)
	dst.Div(other)
	return dst
}

// BroadcastTo broadcasts the tensor to a new shape.
// Currently creates a view-like operation (future: implement efficient broadcasting).
func (t Tensor) BroadcastTo(shape types.Shape) (types.Tensor, error) {
	if t.shape == nil {
		return nil, errors.New("tensor.BroadcastTo: nil tensor")
	}

	targetShape := types.NewShape(shape...)
	if len(shape) < t.shape.Rank() {
		return nil, fmt.Errorf("tensor.BroadcastTo: target shape %v has fewer dimensions than %v", shape, t.shape)
	}

	if shape.Equal(t.Shape()) {
		return t.Clone(), nil
	}

	if _, err := fp32.BroadcastStrides(t.shape.ToSlice(), t.shape.Strides(), shape); err != nil {
		return nil, fmt.Errorf("tensor.BroadcastTo: %w", err)
	}

	result := New(t.DataType(), targetShape)
	resultPtr := &result
	resultData := types.GetTensorData[[]float32](resultPtr)
	tData := types.GetTensorData[[]float32](t)
	if err := fp32.ExpandTo(
		resultData,
		tData,
		resultPtr.shape.ToSlice(),
		t.shape.ToSlice(),
		resultPtr.shape.Strides(),
		t.shape.Strides(),
	); err != nil {
		return nil, fmt.Errorf("tensor.BroadcastTo: %w", err)
	}

	return resultPtr, nil
}

// Sum computes sum along specified dimensions.
// If no dimensions specified, sums all elements.
func (t Tensor) Sum(dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	// If no dimensions specified, sum all elements
	if len(dims) == 0 {
		if t.isContiguous() {
			size := t.Size()
			sum := fp32.Asum(types.GetTensorData[[]float32](t), 1, size)
			result := FromFloat32(types.NewShape(1), []float32{sum})
			return &result
		}
		return t.reduceTensor(nil, true, fp32.ReduceSum)
	}

	return t.reduceTensor(dims, true, fp32.ReduceSum)
}

// ReduceSum is an alias for Sum (matches TensorFlow naming: tf.reduce_sum).
func (t Tensor) ReduceSum(dims ...int) types.Tensor {
	return t.Sum(dims...)
}

// SumTo sums along specified dimensions and stores result in dst.
func (t Tensor) SumTo(dst types.Tensor, dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	result := t.Sum(dims...)
	if result == nil {
		return nil
	}

	if dst == nil {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.SumTo: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	copyTensorData(result, dst)
	return dst
}

// Mean computes mean along specified dimensions.
func (t Tensor) Mean(dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMean)
}

// ReduceMean is an alias for Mean (matches TensorFlow naming: tf.reduce_mean).
func (t Tensor) ReduceMean(dims ...int) types.Tensor {
	return t.Mean(dims...)
}

// MeanTo computes mean along specified dimensions and stores result in dst.
func (t Tensor) MeanTo(dst types.Tensor, dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	result := t.Mean(dims...)
	if result == nil {
		return nil
	}

	if dst == nil {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.MeanTo: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	copyTensorData(result, dst)
	return dst
}

// Max computes maximum along specified dimensions.
func (t Tensor) Max(dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMax)
}

// ReduceMax is an alias for Max (matches TensorFlow naming: tf.reduce_max).
func (t Tensor) ReduceMax(dims ...int) types.Tensor {
	return t.Max(dims...)
}

// MaxTo computes maximum along specified dimensions and stores result in dst.
func (t Tensor) MaxTo(dst types.Tensor, dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	result := t.Max(dims...)
	if result == nil {
		return nil
	}

	if dst == nil {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.MaxTo: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	copyTensorData(result, dst)
	return dst
}

// Min computes minimum along specified dimensions.
func (t Tensor) Min(dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMin)
}

// ReduceMin is an alias for Min (matches TensorFlow naming: tf.reduce_min).
func (t Tensor) ReduceMin(dims ...int) types.Tensor {
	return t.Min(dims...)
}

// MinTo computes minimum along specified dimensions and stores result in dst.
func (t Tensor) MinTo(dst types.Tensor, dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	result := t.Min(dims...)
	if result == nil {
		return nil
	}

	if dst == nil {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.MinTo: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	copyTensorData(result, dst)
	return dst
}

// ArgMax returns indices of maximum elements along specified dimension.
// Uses fp32 primitive.Iamax for vector case.
func (t Tensor) ArgMax(dim int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	if dim < 0 || dim >= t.shape.Rank() {
		panic(fmt.Sprintf("tensor.ArgMax: dimension %d out of range for shape %v", dim, t.shape))
	}

	if t.shape.Rank() == 1 && t.isContiguous() {
		idx := fp32.Iamax(types.GetTensorData[[]float32](t), 1, t.Size())
		result := FromFloat32(types.NewShape(1), []float32{float32(idx)})
		return &result
	}

	resultShape, axis := t.prepareArgmax(dim)
	result := New(t.DataType(), types.NewShape(resultShape...))
	resultPtr := &result
	resultData := types.GetTensorData[[]float32](resultPtr)
	tData := types.GetTensorData[[]float32](t)
	fp32.Argmax(
		resultData,
		resultPtr.shape.ToSlice(),
		resultPtr.shape.Strides(),
		tData,
		t.shape.ToSlice(),
		t.shape.Strides(),
		axis,
	)
	return resultPtr
}

// ArgMin returns indices of minimum elements along specified dimension.
func (t Tensor) ArgMin(dim int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	if dim < 0 || dim >= t.shape.Rank() {
		panic(fmt.Sprintf("tensor.ArgMin: dimension %d out of range for shape %v", dim, t.shape))
	}

	resultShape, axis := t.prepareArgmax(dim)
	result := New(t.DataType(), types.NewShape(resultShape...))
	resultPtr := &result
	tData := types.GetTensorData[[]float32](t)

	// Argmin returns int32, but we need to store as float32 to match ArgMax pattern
	// Create a temporary int32 slice for the result
	resultDataInt32 := make([]int32, resultPtr.Size())
	fp32.Argmin(
		resultDataInt32,
		resultPtr.shape.ToSlice(),
		resultPtr.shape.Strides(),
		tData,
		t.shape.ToSlice(),
		t.shape.Strides(),
		axis,
	)

	// Convert int32 to float32
	resultData := types.GetTensorData[[]float32](resultPtr)
	for i := range resultDataInt32 {
		resultData[i] = float32(resultDataInt32[i])
	}

	return resultPtr
}

// Helper functions

type reduceFunc func(dst []float32, dstShape []int, dstStrides []int, src []float32, srcShape []int, srcStrides []int, axes []int)

func (t Tensor) reduceTensor(dims []int, scalarWhenEmpty bool, reducer reduceFunc) types.Tensor {
	if t.shape == nil {
		return nil
	}

	axes := t.normalizeAxes(dims)
	dimSet := make(map[int]struct{}, len(axes))
	for _, axis := range axes {
		dimSet[axis] = struct{}{}
	}

	resultShape := make([]int, 0, t.shape.Rank())
	for i, d := range t.shape {
		if _, ok := dimSet[i]; !ok {
			resultShape = append(resultShape, d)
		}
	}
	if len(resultShape) == 0 && scalarWhenEmpty {
		resultShape = []int{1}
	}

	res := New(t.DataType(), types.NewShape(resultShape...))
	resPtr := &res
	resData := types.GetTensorData[[]float32](resPtr)
	tData := types.GetTensorData[[]float32](t)

	reducer(
		resData,
		resPtr.shape.ToSlice(),
		resPtr.shape.Strides(),
		tData,
		t.shape.ToSlice(),
		t.shape.Strides(),
		axes,
	)

	return resPtr
}

func (t Tensor) normalizeAxes(dims []int) []int {
	if t.shape == nil || t.shape.Rank() == 0 {
		panic("tensor: reduction on empty tensor")
	}

	if len(dims) == 0 {
		axes := make([]int, t.shape.Rank())
		for i := range axes {
			axes[i] = i
		}
		return axes
	}

	axes := append([]int(nil), dims...)
	if err := types.Shape(t.shape).ValidateAxes(axes); err != nil {
		panic(err)
	}
	return axes
}

func (t Tensor) prepareArgmax(dim int) ([]int, int) {
	if t.shape == nil || t.shape.Rank() == 0 {
		panic("tensor.ArgMax: empty tensor")
	}
	if dim < 0 || dim >= t.shape.Rank() {
		panic(fmt.Sprintf("tensor.ArgMax: dimension %d out of range for shape %v", dim, t.shape))
	}

	shape := make([]int, 0, t.shape.Rank()-1)
	for i, d := range t.shape {
		if i != dim {
			shape = append(shape, d)
		}
	}
	if len(shape) == 0 {
		shape = []int{1}
	}

	return shape, dim
}

func (t Tensor) copyTo(dst Tensor) {
	if dst.Shape() == nil {
		return
	}

	if t.shape == nil {
		return
	}

	if dst.shape.Rank() != t.shape.Rank() {
		panic(fmt.Sprintf("tensor.copyTo: destination shape mismatch: %v vs %v", dst.shape, t.shape))
	}

	size := t.Size()
	if size == 0 {
		return
	}

	if t.isContiguous() && dst.isContiguous() {
		fp32.Copy(types.GetTensorData[[]float32](&dst), types.GetTensorData[[]float32](t), 1, 1, size)
		return
	}

	stridesSrc := t.shape.Strides()
	stridesDst := dst.shape.Strides()
	fp32.ElemCopy(types.GetTensorData[[]float32](&dst), types.GetTensorData[[]float32](t), []int(t.shape), stridesDst, stridesSrc)
}

// Where creates a new tensor by selecting elements from a where condition is true, otherwise from b.
// condition, a, b must have compatible shapes.
func (t Tensor) Where(condition, a, b types.Tensor) types.Tensor {
	if t.shape == nil || condition == nil || condition.Shape() == nil || a == nil || a.Shape() == nil || b == nil || b.Shape() == nil {
		return nil
	}

	result := t.Clone()
	if result == nil {
		return nil
	}

	if !t.Shape().Equal(condition.Shape()) || !t.Shape().Equal(a.Shape()) || !t.Shape().Equal(b.Shape()) {
		panic("tensor.Where: shape mismatch")
	}

	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)
	conditionData := types.GetTensorData[[]float32](condition)
	aData := types.GetTensorData[[]float32](a)
	bData := types.GetTensorData[[]float32](b)
	resultData := types.GetTensorData[[]float32](result)

	fp32.ElemWhere(
		resultData, conditionData, aData, bData,
		shape, strides, strides, strides, strides,
	)
	return result
}

// WhereTo performs element-wise selection and stores result in dst.
func (t Tensor) WhereTo(dst types.Tensor, condition, a, b types.Tensor) types.Tensor {
	if t.shape == nil || condition == nil || condition.Shape() == nil || a == nil || a.Shape() == nil || b == nil || b.Shape() == nil {
		return nil
	}

	if dst == nil {
		return t.Where(condition, a, b)
	}

	if !t.Shape().Equal(condition.Shape()) || !t.Shape().Equal(a.Shape()) || !t.Shape().Equal(b.Shape()) {
		panic("tensor.WhereTo: shape mismatch")
	}

	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.WhereTo: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)
	conditionData := types.GetTensorData[[]float32](condition)
	aData := types.GetTensorData[[]float32](a)
	bData := types.GetTensorData[[]float32](b)
	dstData := types.GetTensorData[[]float32](dst)

	fp32.ElemWhere(
		dstData, conditionData, aData, bData,
		shape, strides, strides, strides, strides,
	)
	return dst
}

// GreaterThan creates a tensor with 1.0 where t > other, 0.0 otherwise.
// t and other must have compatible shapes.
func (t Tensor) GreaterThan(other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.GreaterThan: shape mismatch")
	}

	result := New(t.DataType(), t.shape)
	resultPtr := &result
	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)
	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	fp32.ElemGreaterThan(
		resultData, tData, otherData,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// ZerosLike creates a new tensor with the same shape as t, filled with zeros.
func ZerosLike(t types.Tensor) types.Tensor {
	if t == nil || t.Shape() == nil {
		return nil
	}
	result := New(t.DataType(), t.Shape())
	return &result
}

// OnesLike creates a new tensor with the same shape as t, filled with ones.
func OnesLike(t types.Tensor) types.Tensor {
	if t == nil || t.Shape() == nil {
		return nil
	}
	result := New(t.DataType(), t.Shape())
	resultPtr := &result
	// Fill with ones by scaling a zero tensor by 0 and then adding 1
	// Actually, let's use the data directly for efficiency
	resultData := types.GetTensorData[[]float32](resultPtr)
	for i := range resultData {
		resultData[i] = 1.0
	}
	return resultPtr
}

// IsNil checks whether the tensor is nil or empty.
// Returns true if t is nil, or if t.Shape() is nil, or if t.Empty() is true.
func IsNil(t types.Tensor) bool {
	if t == nil {
		return true
	}
	if t.Shape() == nil {
		return true
	}
	return t.Empty()
}

// FullLike creates a new tensor with the same shape as t, filled with the given value.
func FullLike(t types.Tensor, value float32) types.Tensor {
	if t == nil || t.Shape() == nil {
		return nil
	}
	result := New(t.DataType(), t.Shape())
	resultPtr := &result
	resultData := types.GetTensorData[[]float32](resultPtr)
	for i := range resultData {
		resultData[i] = value
	}
	return resultPtr
}

// Square computes element-wise square in-place: t[i] = t[i]^2
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t Tensor) Square(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Square: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	strides := t.shape.Strides()
	dstStrides := strides
	if dst != nil && !dst.Empty() {
		dstStrides = dst.Shape().Strides()
	}

	fp32.ElemSquare(dstData, tData, []int(t.shape), dstStrides, strides)
	if dst == nil || dst.Empty() {
		return t
	}
	return dst
}

// Sqrt computes element-wise square root in-place: t[i] = sqrt(t[i])
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t Tensor) Sqrt(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Sqrt: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	strides := t.shape.Strides()
	dstStrides := strides
	if dst != nil && !dst.Empty() {
		dstStrides = dst.Shape().Strides()
	}

	fp32.ElemSqrt(dstData, tData, []int(t.shape), dstStrides, strides)
	if dst == nil || dst.Empty() {
		return t
	}
	return dst
}

// Exp computes element-wise exponential in-place: t[i] = exp(t[i])
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t Tensor) Exp(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Exp: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	strides := t.shape.Strides()
	dstStrides := strides
	if dst != nil && !dst.Empty() {
		dstStrides = dst.Shape().Strides()
	}

	fp32.ElemExp(dstData, tData, []int(t.shape), dstStrides, strides)
	if dst == nil || dst.Empty() {
		return t
	}
	return dst
}

// Log computes element-wise natural logarithm in-place: t[i] = log(t[i])
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t Tensor) Log(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Log: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	strides := t.shape.Strides()
	dstStrides := strides
	if dst != nil && !dst.Empty() {
		dstStrides = dst.Shape().Strides()
	}

	fp32.ElemLog(dstData, tData, []int(t.shape), dstStrides, strides)
	if dst == nil || dst.Empty() {
		return t
	}
	return dst
}

// Pow computes element-wise power in-place: t[i] = t[i]^power
// If dst is nil, applies in-place on t. Otherwise writes to dst.
// Converts float64 power to float32 for internal computation.
func (t Tensor) Pow(dst types.Tensor, power float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Pow: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	strides := t.shape.Strides()
	dstStrides := strides
	if dst != nil && !dst.Empty() {
		dstStrides = dst.Shape().Strides()
	}

	power32 := float32(power)
	fp32.ElemPow(dstData, tData, power32, []int(t.shape), dstStrides, strides)
	if dst == nil || dst.Empty() {
		return t
	}
	return dst
}

// Equal creates a tensor with 1.0 where t == other, 0.0 otherwise.
// t and other must have compatible shapes.
func (t Tensor) Equal(other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.Equal: shape mismatch")
	}

	result := New(t.DataType(), t.shape)
	resultPtr := &result
	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)
	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	fp32.ElemEqual(
		resultData, tData, otherData,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// Greater creates a tensor with 1.0 where t > other, 0.0 otherwise.
// t and other must have compatible shapes.
// Note: This is an alias for GreaterThan to match TensorFlow naming.
func (t Tensor) Greater(other types.Tensor) types.Tensor {
	return t.GreaterThan(other)
}

// Less creates a tensor with 1.0 where t < other, 0.0 otherwise.
// t and other must have compatible shapes.
func (t Tensor) Less(other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.Less: shape mismatch")
	}

	result := New(t.DataType(), t.shape)
	resultPtr := &result
	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)
	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	fp32.ElemLess(
		resultData, tData, otherData,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// NotEqual creates a tensor with 1.0 where t != other, 0.0 otherwise (matches tf.not_equal).
func (t Tensor) NotEqual(other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.NotEqual: shape mismatch")
	}

	result := New(t.DataType(), t.shape)
	resultPtr := &result
	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)
	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	fp32.ElemNotEqual(
		resultData, tData, otherData,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// GreaterEqual creates a tensor with 1.0 where t >= other, 0.0 otherwise (matches tf.greater_equal).
func (t Tensor) GreaterEqual(other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.GreaterEqual: shape mismatch")
	}

	result := New(t.DataType(), t.shape)
	resultPtr := &result
	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)
	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	fp32.ElemGreaterEqual(
		resultData, tData, otherData,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// LessEqual creates a tensor with 1.0 where t <= other, 0.0 otherwise (matches tf.less_equal).
func (t Tensor) LessEqual(other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.LessEqual: shape mismatch")
	}

	result := New(t.DataType(), t.shape)
	resultPtr := &result
	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)
	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	fp32.ElemLessEqual(
		resultData, tData, otherData,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// Abs computes element-wise absolute value in-place: t[i] = abs(t[i])
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t Tensor) Abs(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Abs: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	strides := t.shape.Strides()
	dstStrides := strides
	if dst != nil && !dst.Empty() {
		dstStrides = dst.Shape().Strides()
	}

	fp32.ElemAbs(dstData, tData, []int(t.shape), dstStrides, strides)
	if dst == nil || dst.Empty() {
		return t
	}
	return dst
}

// Sign computes element-wise sign in-place: t[i] = sign(t[i]) (-1, 0, or 1)
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t Tensor) Sign(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Sign: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	strides := t.shape.Strides()
	dstStrides := strides
	if dst != nil && !dst.Empty() {
		dstStrides = dst.Shape().Strides()
	}

	fp32.ElemSign(dstData, tData, []int(t.shape), dstStrides, strides)
	if dst == nil || dst.Empty() {
		return t
	}
	return dst
}

// Cos computes element-wise cosine in-place: t[i] = cos(t[i])
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t Tensor) Cos(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Cos: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	strides := t.shape.Strides()
	dstStrides := strides
	if dst != nil && !dst.Empty() {
		dstStrides = dst.Shape().Strides()
	}

	fp32.ElemCos(dstData, tData, []int(t.shape), dstStrides, strides)
	if dst == nil || dst.Empty() {
		return t
	}
	return dst
}

// Sin computes element-wise sine in-place: t[i] = sin(t[i])
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t Tensor) Sin(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Sin: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	strides := t.shape.Strides()
	dstStrides := strides
	if dst != nil && !dst.Empty() {
		dstStrides = dst.Shape().Strides()
	}

	fp32.ElemSin(dstData, tData, []int(t.shape), dstStrides, strides)
	if dst == nil || dst.Empty() {
		return t
	}
	return dst
}

// Negative computes element-wise negation in-place: t[i] = -t[i]
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t Tensor) Negative(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Negative: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	strides := t.shape.Strides()
	dstStrides := strides
	if dst != nil && !dst.Empty() {
		dstStrides = dst.Shape().Strides()
	}

	fp32.ElemNegative(dstData, tData, []int(t.shape), dstStrides, strides)
	if dst == nil || dst.Empty() {
		return t
	}
	return dst
}

// Explicit To variants for unary operations (for consistency with *To pattern)
// SquareTo is an alias for Square(dst)
func (t Tensor) SquareTo(dst types.Tensor) types.Tensor {
	return t.Square(dst)
}

// SqrtTo is an alias for Sqrt(dst)
func (t Tensor) SqrtTo(dst types.Tensor) types.Tensor {
	return t.Sqrt(dst)
}

// ExpTo is an alias for Exp(dst)
func (t Tensor) ExpTo(dst types.Tensor) types.Tensor {
	return t.Exp(dst)
}

// LogTo is an alias for Log(dst)
func (t Tensor) LogTo(dst types.Tensor) types.Tensor {
	return t.Log(dst)
}

// PowTo is an alias for Pow(dst, power)
func (t Tensor) PowTo(dst types.Tensor, power float64) types.Tensor {
	return t.Pow(dst, power)
}

// AbsTo is an alias for Abs(dst)
func (t Tensor) AbsTo(dst types.Tensor) types.Tensor {
	return t.Abs(dst)
}

// SignTo is an alias for Sign(dst)
func (t Tensor) SignTo(dst types.Tensor) types.Tensor {
	return t.Sign(dst)
}

// CosTo is an alias for Cos(dst)
func (t Tensor) CosTo(dst types.Tensor) types.Tensor {
	return t.Cos(dst)
}

// SinTo is an alias for Sin(dst)
func (t Tensor) SinTo(dst types.Tensor) types.Tensor {
	return t.Sin(dst)
}

// NegativeTo is an alias for Negative(dst)
func (t Tensor) NegativeTo(dst types.Tensor) types.Tensor {
	return t.Negative(dst)
}

// Fill fills the tensor with a constant value (in-place)
// Uses fp32.Fill primitive for efficient computation
func (t Tensor) Fill(value float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	value32 := float32(value)
	strides := t.shape.Strides()
	size := t.Size()

	if t.isContiguous() {
		// Use fp32.Fill with stride=1 for contiguous tensors
		tData := types.GetTensorData[[]float32](&t)
		fp32.Fill(tData, value32, size, 1)
		return t
	}

	// Handle non-contiguous case
	fp32.ElemFill(types.GetTensorData[[]float32](&t), value32, []int(t.shape), strides)
	return t
}

// FillTo fills the tensor with a constant value and stores result in dst.
func (t Tensor) FillTo(dst types.Tensor, value float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	if dst == nil {
		result := t.Clone()
		result.Fill(value)
		return result
	}

	if !t.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.FillTo: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
	}

	// Copy t to dst, then fill
	copyTensorData(t, dst)
	dst.Fill(value)
	return dst
}

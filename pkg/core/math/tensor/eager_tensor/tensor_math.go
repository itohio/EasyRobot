package eager_tensor

import (
	"errors"
	"fmt"

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

// copyTensorData copies data from src to dst using interface methods.
// Both tensors must have the same shape.
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

	srcData := types.GetTensorData[[]float32](src)
	dstData := types.GetTensorData[[]float32](dst)

	if srcData == nil || dstData == nil {
		return
	}

	// Check if both are contiguous for fast copy
	srcStrides := srcShape.Strides()
	dstStrides := dst.Shape().Strides()

	srcContiguous := isTensorContiguous(src)
	dstContiguous := isTensorContiguous(dst)

	if srcContiguous && dstContiguous {
		fp32.Copy(dstData, srcData, 1, 1, size)
		return
	}

	// Use element-wise copy for strided tensors
	shapeSlice := srcShape.ToSlice()
	fp32.ElemCopy(dstData, srcData, shapeSlice, dstStrides, srcStrides)
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
		fp32.Axpy(types.GetTensorData[[]float32](&t), otherData, 1, 1, size, 1.0)
		return t
	}

	stridesT := t.shape.Strides()
	stridesOther := other.Shape().Strides()
	fp32.ElemAdd(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), otherData, []int(t.shape), stridesT, stridesT, stridesOther)
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
		fp32.Axpy(types.GetTensorData[[]float32](&t), otherData, 1, 1, size, -1.0)
		return t
	}

	stridesT := t.shape.Strides()
	stridesOther := other.Shape().Strides()
	fp32.ElemSub(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), otherData, []int(t.shape), stridesT, stridesT, stridesOther)
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
	fp32.ElemMul(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), otherData, []int(t.shape), stridesT, stridesT, stridesOther)
	return t
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
	fp32.ElemDiv(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), otherData, []int(t.shape), stridesT, stridesT, stridesOther)
	return t
}

// Scale multiplies the tensor by a scalar in-place.
// Uses fp32 primitive.Scal for efficient computation.
func (t Tensor) Scale(scalar float32) types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	if t.isContiguous() {
		size := t.Size()
		fp32.Scal(types.GetTensorData[[]float32](&t), 1, size, scalar)
		return t
	}

	fp32.ElemScale(types.GetTensorData[[]float32](&t), scalar, []int(t.shape), strides)
	return t
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
	tData := types.GetTensorData[[]float32](&t)
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
			sum := fp32.Asum(types.GetTensorData[[]float32](&t), 1, size)
			result := FromFloat32(types.NewShape(1), []float32{sum})
			return &result
		}
		return t.reduceTensor(nil, true, fp32.ReduceSum)
	}

	return t.reduceTensor(dims, true, fp32.ReduceSum)
}

// Mean computes mean along specified dimensions.
func (t Tensor) Mean(dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMean)
}

// Max computes maximum along specified dimensions.
func (t Tensor) Max(dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMax)
}

// Min computes minimum along specified dimensions.
func (t Tensor) Min(dims ...int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMin)
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
		idx := fp32.Iamax(types.GetTensorData[[]float32](&t), 1, t.Size())
		result := FromFloat32(types.NewShape(1), []float32{float32(idx)})
		return &result
	}

	resultShape, axis := t.prepareArgmax(dim)
	result := New(t.DataType(), types.NewShape(resultShape...))
	resultPtr := &result
	resultData := types.GetTensorData[[]float32](resultPtr)
	tData := types.GetTensorData[[]float32](&t)
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
	tData := types.GetTensorData[[]float32](&t)

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
		fp32.Copy(types.GetTensorData[[]float32](&dst), types.GetTensorData[[]float32](&t), 1, 1, size)
		return
	}

	stridesSrc := t.shape.Strides()
	stridesDst := dst.shape.Strides()
	fp32.ElemCopy(types.GetTensorData[[]float32](&dst), types.GetTensorData[[]float32](&t), []int(t.shape), stridesDst, stridesSrc)
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
	tData := types.GetTensorData[[]float32](&t)
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
func (t Tensor) Square() types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemSquare(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), []int(t.shape), strides, strides)
	return t
}

// Sqrt computes element-wise square root in-place: t[i] = sqrt(t[i])
func (t Tensor) Sqrt() types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemSqrt(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), []int(t.shape), strides, strides)
	return t
}

// Exp computes element-wise exponential in-place: t[i] = exp(t[i])
func (t Tensor) Exp() types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemExp(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), []int(t.shape), strides, strides)
	return t
}

// Log computes element-wise natural logarithm in-place: t[i] = log(t[i])
func (t Tensor) Log() types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemLog(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), []int(t.shape), strides, strides)
	return t
}

// Pow computes element-wise power in-place: t[i] = t[i]^power
func (t Tensor) Pow(power float32) types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemPow(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), power, []int(t.shape), strides, strides)
	return t
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
	tData := types.GetTensorData[[]float32](&t)
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
	tData := types.GetTensorData[[]float32](&t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	fp32.ElemLess(
		resultData, tData, otherData,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// Abs computes element-wise absolute value in-place: t[i] = abs(t[i])
func (t Tensor) Abs() types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemAbs(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), []int(t.shape), strides, strides)
	return t
}

// Sign computes element-wise sign in-place: t[i] = sign(t[i]) (-1, 0, or 1)
func (t Tensor) Sign() types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemSign(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), []int(t.shape), strides, strides)
	return t
}

// Cos computes element-wise cosine in-place: t[i] = cos(t[i])
func (t Tensor) Cos() types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemCos(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), []int(t.shape), strides, strides)
	return t
}

// Sin computes element-wise sine in-place: t[i] = sin(t[i])
func (t Tensor) Sin() types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemSin(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), []int(t.shape), strides, strides)
	return t
}

// Negative computes element-wise negation in-place: t[i] = -t[i]
func (t Tensor) Negative() types.Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemNegative(types.GetTensorData[[]float32](&t), types.GetTensorData[[]float32](&t), []int(t.shape), strides, strides)
	return t
}

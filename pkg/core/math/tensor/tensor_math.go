package tensor

import (
	"errors"
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

// Add adds another tensor element-wise in-place.
// Uses fp32 primitive.Axpy for efficient computation.
// Returns the tensor itself for method chaining.
func (t *Tensor) Add(other Tensor) *Tensor {
	if t.shape == nil || other.shape == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Add: shape mismatch: %v vs %v", t.shape, other.shape))
	}

	if t.isContiguous() && other.isContiguous() {
		size := t.Size()
		fp32.Axpy(t.data, other.data, 1, 1, size, 1.0)
		return t
	}

	stridesT := t.shape.Strides()
	stridesOther := other.shape.Strides()
	fp32.ElemAdd(t.data, t.data, other.data, []int(t.shape), stridesT, stridesT, stridesOther)
	return t
}

// Sub subtracts another tensor element-wise in-place.
// Uses fp32 primitive.Axpy with alpha=-1.
func (t *Tensor) Sub(other Tensor) *Tensor {
	if t.shape == nil || other.shape == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Sub: shape mismatch: %v vs %v", t.shape, other.shape))
	}

	if t.isContiguous() && other.isContiguous() {
		size := t.Size()
		fp32.Axpy(t.data, other.data, 1, 1, size, -1.0)
		return t
	}

	stridesT := t.shape.Strides()
	stridesOther := other.shape.Strides()
	fp32.ElemSub(t.data, t.data, other.data, []int(t.shape), stridesT, stridesT, stridesOther)
	return t
}

// Mul multiplies another tensor element-wise in-place.
func (t *Tensor) Mul(other Tensor) *Tensor {
	if t.shape == nil || other.shape == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Mul: shape mismatch: %v vs %v", t.shape, other.shape))
	}

	stridesT := t.shape.Strides()
	stridesOther := other.shape.Strides()
	fp32.ElemMul(t.data, t.data, other.data, []int(t.shape), stridesT, stridesT, stridesOther)
	return t
}

// Div divides by another tensor element-wise in-place.
func (t *Tensor) Div(other Tensor) *Tensor {
	if t.shape == nil || other.shape == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Div: shape mismatch: %v vs %v", t.shape, other.shape))
	}

	stridesT := t.shape.Strides()
	stridesOther := other.shape.Strides()
	fp32.ElemDiv(t.data, t.data, other.data, []int(t.shape), stridesT, stridesT, stridesOther)
	return t
}

// Scale multiplies the tensor by a scalar in-place.
// Uses fp32 primitive.Scal for efficient computation.
func (t *Tensor) Scale(scalar float32) *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	if t.isContiguous() {
		size := t.Size()
		fp32.Scal(t.data, 1, size, scalar)
		return t
	}

	fp32.ElemScale(t.data, scalar, []int(t.shape), strides)
	return t
}

// AddTo adds two tensors and stores result in dst (or creates new tensor if dst is nil).
// Returns the destination tensor.
func (t *Tensor) AddTo(other Tensor, dst *Tensor) *Tensor {
	if t.shape == nil || other.shape == nil {
		return nil
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.AddTo: shape mismatch: %v vs %v", t.shape, other.shape))
	}

	if dst == nil {
		dst = t.Clone()
		dst.Add(other)
		return dst
	}

	if !dst.sameShape(*t) {
		panic(fmt.Sprintf("tensor.AddTo: destination shape mismatch: %v vs %v", dst.shape, t.shape))
	}

	// Copy t to dst, then add other
	t.copyTo(dst)
	dst.Add(other)
	return dst
}

// MulTo multiplies two tensors element-wise and stores result in dst (or creates new tensor if dst is nil).
func (t *Tensor) MulTo(other Tensor, dst *Tensor) *Tensor {
	if t.shape == nil || other.shape == nil {
		return nil
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.MulTo: shape mismatch: %v vs %v", t.shape, other.shape))
	}

	if dst == nil {
		dst = t.Clone()
		dst.Mul(other)
		return dst
	}

	if !dst.sameShape(*t) {
		panic(fmt.Sprintf("tensor.MulTo: destination shape mismatch: %v vs %v", dst.shape, t.shape))
	}

	// Copy t to dst, then multiply by other
	t.copyTo(dst)
	dst.Mul(other)
	return dst
}

// BroadcastTo broadcasts the tensor to a new shape.
// Currently creates a view-like operation (future: implement efficient broadcasting).
func (t Tensor) BroadcastTo(shape []int) (*Tensor, error) {
	if t.shape == nil {
		return nil, errors.New("tensor.BroadcastTo: nil tensor")
	}

	if len(shape) < t.shape.Rank() {
		return nil, fmt.Errorf("tensor.BroadcastTo: target shape %v has fewer dimensions than %v", shape, t.shape)
	}

	if t.sameShapeInt(shape) {
		return t.Clone(), nil
	}

	if _, err := fp32.BroadcastStrides(t.shape.ToSlice(), t.shape.Strides(), shape); err != nil {
		return nil, fmt.Errorf("tensor.BroadcastTo: %w", err)
	}

	result := New(t.dtype, NewShape(shape...))
	resultPtr := &result
	if err := fp32.ExpandTo(
		resultPtr.data,
		t.data,
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
func (t Tensor) Sum(dims ...int) *Tensor {
	if t.shape == nil {
		return nil
	}

	// If no dimensions specified, sum all elements
	if len(dims) == 0 {
		if t.isContiguous() {
			size := t.Size()
			sum := fp32.Asum(t.data, 1, size)
			result := FromFloat32(NewShape(1), []float32{sum})
			return &result
		}
		return t.reduceTensor(nil, true, fp32.ReduceSum)
	}

	return t.reduceTensor(dims, true, fp32.ReduceSum)
}

// Mean computes mean along specified dimensions.
func (t Tensor) Mean(dims ...int) *Tensor {
	if t.shape == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMean)
}

// Max computes maximum along specified dimensions.
func (t Tensor) Max(dims ...int) *Tensor {
	if t.shape == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMax)
}

// Min computes minimum along specified dimensions.
func (t Tensor) Min(dims ...int) *Tensor {
	if t.shape == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMin)
}

// ArgMax returns indices of maximum elements along specified dimension.
// Uses fp32 primitive.Iamax for vector case.
func (t Tensor) ArgMax(dim int) *Tensor {
	if t.shape == nil {
		return nil
	}

	if dim < 0 || dim >= t.shape.Rank() {
		panic(fmt.Sprintf("tensor.ArgMax: dimension %d out of range for shape %v", dim, t.shape))
	}

	if t.shape.Rank() == 1 && t.isContiguous() {
		idx := fp32.Iamax(t.data, 1, t.Size())
		result := FromFloat32(NewShape(1), []float32{float32(idx)})
		return &result
	}

	resultShape, axis := t.prepareArgmax(dim)
	result := New(t.dtype, NewShape(resultShape...))
	resultPtr := &result
	fp32.Argmax(
		resultPtr.data,
		resultPtr.shape.ToSlice(),
		resultPtr.shape.Strides(),
		t.data,
		t.shape.ToSlice(),
		t.shape.Strides(),
		axis,
	)
	return resultPtr
}

// Helper functions

func (t Tensor) sameShape(other Tensor) bool {
	if t.shape == nil || other.shape == nil {
		return false
	}
	if t.shape.Rank() != other.shape.Rank() {
		return false
	}
	for i := range t.shape {
		if t.shape[i] != other.shape[i] {
			return false
		}
	}
	return true
}

func (t Tensor) sameShapeInt(shape []int) bool {
	if t.shape == nil {
		return false
	}
	if t.shape.Rank() != len(shape) {
		return false
	}
	for i := range t.shape {
		if t.shape[i] != shape[i] {
			return false
		}
	}
	return true
}

type reduceFunc func(dst []float32, dstShape []int, dstStrides []int, src []float32, srcShape []int, srcStrides []int, axes []int)

func (t Tensor) reduceTensor(dims []int, scalarWhenEmpty bool, reducer reduceFunc) *Tensor {
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

	res := New(t.dtype, NewShape(resultShape...))
	resPtr := &res

	reducer(
		resPtr.data,
		resPtr.shape.ToSlice(),
		resPtr.shape.Strides(),
		t.data,
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
	if err := Shape(t.shape).ValidateAxes(axes); err != nil {
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

func (t Tensor) copyTo(dst *Tensor) {
	if dst == nil {
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
		fp32.Copy(dst.data, t.data, 1, 1, size)
		return
	}

	stridesSrc := t.shape.Strides()
	stridesDst := dst.shape.Strides()
	fp32.ElemCopy(dst.data, t.data, []int(t.shape), stridesDst, stridesSrc)
}

// Where creates a new tensor by selecting elements from a where condition is true, otherwise from b.
// condition, a, b must have compatible shapes.
func (t Tensor) Where(condition, a, b Tensor) *Tensor {
	if t.shape == nil || condition.shape == nil || a.shape == nil || b.shape == nil {
		return nil
	}

	if !condition.sameShape(a) || !condition.sameShape(b) {
		panic("tensor.Where: shape mismatch")
	}

	result := t.Clone()
	if result == nil {
		return nil
	}
	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)

	fp32.ElemWhere(
		result.data, condition.data, a.data, b.data,
		shape, strides, strides, strides, strides,
	)
	return result
}

// GreaterThan creates a tensor with 1.0 where t > other, 0.0 otherwise.
// t and other must have compatible shapes.
func (t Tensor) GreaterThan(other Tensor) *Tensor {
	if t.shape == nil || other.shape == nil {
		return nil
	}

	if !t.sameShape(other) {
		panic("tensor.GreaterThan: shape mismatch")
	}

	result := New(t.dtype, t.shape)
	resultPtr := &result
	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)

	fp32.ElemGreaterThan(
		resultPtr.data, t.data, other.data,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// ZerosLike creates a new tensor with the same shape as t, filled with zeros.
func ZerosLike(t Tensor) *Tensor {
	if t.shape == nil {
		return nil
	}
	result := New(t.dtype, t.shape)
	return &result
}

// OnesLike creates a new tensor with the same shape as t, filled with ones.
func OnesLike(t Tensor) *Tensor {
	if t.shape == nil {
		return nil
	}
	result := New(t.dtype, t.shape)
	resultPtr := &result
	// Fill with ones by scaling a zero tensor by 0 and then adding 1
	// Actually, let's use the data directly for efficiency
	for i := range resultPtr.data {
		resultPtr.data[i] = 1.0
	}
	return resultPtr
}

// FullLike creates a new tensor with the same shape as t, filled with the given value.
func FullLike(t Tensor, value float32) *Tensor {
	if t.shape == nil {
		return nil
	}
	result := New(t.dtype, t.shape)
	resultPtr := &result
	for i := range resultPtr.data {
		resultPtr.data[i] = value
	}
	return resultPtr
}

// Square computes element-wise square in-place: t[i] = t[i]^2
func (t *Tensor) Square() *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemSquare(t.data, t.data, []int(t.shape), strides, strides)
	return t
}

// Sqrt computes element-wise square root in-place: t[i] = sqrt(t[i])
func (t *Tensor) Sqrt() *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemSqrt(t.data, t.data, []int(t.shape), strides, strides)
	return t
}

// Exp computes element-wise exponential in-place: t[i] = exp(t[i])
func (t *Tensor) Exp() *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemExp(t.data, t.data, []int(t.shape), strides, strides)
	return t
}

// Log computes element-wise natural logarithm in-place: t[i] = log(t[i])
func (t *Tensor) Log() *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemLog(t.data, t.data, []int(t.shape), strides, strides)
	return t
}

// Pow computes element-wise power in-place: t[i] = t[i]^power
func (t *Tensor) Pow(power float32) *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemPow(t.data, t.data, power, []int(t.shape), strides, strides)
	return t
}

// Equal creates a tensor with 1.0 where t == other, 0.0 otherwise.
// t and other must have compatible shapes.
func (t Tensor) Equal(other Tensor) *Tensor {
	if t.shape == nil || other.shape == nil {
		return nil
	}

	if !t.sameShape(other) {
		panic("tensor.Equal: shape mismatch")
	}

	result := New(t.dtype, t.shape)
	resultPtr := &result
	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)

	fp32.ElemEqual(
		resultPtr.data, t.data, other.data,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// Greater creates a tensor with 1.0 where t > other, 0.0 otherwise.
// t and other must have compatible shapes.
// Note: This is an alias for GreaterThan to match TensorFlow naming.
func (t Tensor) Greater(other Tensor) *Tensor {
	return t.GreaterThan(other)
}

// Less creates a tensor with 1.0 where t < other, 0.0 otherwise.
// t and other must have compatible shapes.
func (t Tensor) Less(other Tensor) *Tensor {
	if t.shape == nil || other.shape == nil {
		return nil
	}

	if !t.sameShape(other) {
		panic("tensor.Less: shape mismatch")
	}

	result := New(t.dtype, t.shape)
	resultPtr := &result
	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)

	fp32.ElemLess(
		resultPtr.data, t.data, other.data,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// Abs computes element-wise absolute value in-place: t[i] = abs(t[i])
func (t *Tensor) Abs() *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemAbs(t.data, t.data, []int(t.shape), strides, strides)
	return t
}

// Sign computes element-wise sign in-place: t[i] = sign(t[i]) (-1, 0, or 1)
func (t *Tensor) Sign() *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemSign(t.data, t.data, []int(t.shape), strides, strides)
	return t
}

// Cos computes element-wise cosine in-place: t[i] = cos(t[i])
func (t *Tensor) Cos() *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemCos(t.data, t.data, []int(t.shape), strides, strides)
	return t
}

// Sin computes element-wise sine in-place: t[i] = sin(t[i])
func (t *Tensor) Sin() *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemSin(t.data, t.data, []int(t.shape), strides, strides)
	return t
}

// Negative computes element-wise negation in-place: t[i] = -t[i]
func (t *Tensor) Negative() *Tensor {
	if t.shape == nil {
		return t
	}

	strides := t.shape.Strides()
	fp32.ElemNegative(t.data, t.data, []int(t.shape), strides, strides)
	return t
}

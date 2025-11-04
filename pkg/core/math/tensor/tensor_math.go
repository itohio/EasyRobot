package tensor

import (
	"errors"
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

// Add adds another tensor element-wise in-place.
// Uses fp32 primitive.Axpy for efficient computation.
// Returns the tensor itself for method chaining.
func (t *Tensor) Add(other *Tensor) *Tensor {
	if t == nil || other == nil {
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
func (t *Tensor) Sub(other *Tensor) *Tensor {
	if t == nil || other == nil {
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
func (t *Tensor) Mul(other *Tensor) *Tensor {
	if t == nil || other == nil {
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
func (t *Tensor) Div(other *Tensor) *Tensor {
	if t == nil || other == nil {
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
	if t == nil {
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
func (t *Tensor) AddTo(other *Tensor, dst *Tensor) *Tensor {
	if t == nil || other == nil {
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

	if !dst.sameShape(t) {
		panic(fmt.Sprintf("tensor.AddTo: destination shape mismatch: %v vs %v", dst.shape, t.shape))
	}

	// Copy t to dst, then add other
	t.copyTo(dst)
	dst.Add(other)
	return dst
}

// MulTo multiplies two tensors element-wise and stores result in dst (or creates new tensor if dst is nil).
func (t *Tensor) MulTo(other *Tensor, dst *Tensor) *Tensor {
	if t == nil || other == nil {
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

	if !dst.sameShape(t) {
		panic(fmt.Sprintf("tensor.MulTo: destination shape mismatch: %v vs %v", dst.shape, t.shape))
	}

	// Copy t to dst, then multiply by other
	t.copyTo(dst)
	dst.Mul(other)
	return dst
}

// BroadcastTo broadcasts the tensor to a new shape.
// Currently creates a view-like operation (future: implement efficient broadcasting).
func (t *Tensor) BroadcastTo(shape []int) (*Tensor, error) {
	if t == nil {
		return nil, errors.New("tensor.BroadcastTo: nil tensor")
	}

	if len(shape) < len(t.shape) {
		return nil, fmt.Errorf("tensor.BroadcastTo: target shape %v has fewer dimensions than %v", shape, t.shape)
	}

	if t.sameShapeInt(shape) {
		return t.Clone(), nil
	}

	if _, err := fp32.BroadcastStrides(t.shape.ToSlice(), t.shape.Strides(), shape); err != nil {
		return nil, fmt.Errorf("tensor.BroadcastTo: %w", err)
	}

	result := New(t.dtype, NewShape(shape...))
	if err := fp32.ExpandTo(
		result.data,
		t.data,
		result.shape.ToSlice(),
		t.shape.ToSlice(),
		result.shape.Strides(),
		t.shape.Strides(),
	); err != nil {
		return nil, fmt.Errorf("tensor.BroadcastTo: %w", err)
	}

	return result, nil
}

// Sum computes sum along specified dimensions.
// If no dimensions specified, sums all elements.
func (t *Tensor) Sum(dims ...int) *Tensor {
	if t == nil {
		return nil
	}

	// If no dimensions specified, sum all elements
	if len(dims) == 0 {
		if t.isContiguous() {
			size := t.Size()
			sum := fp32.Asum(t.data, 1, size)
			return FromFloat32(NewShape(1), []float32{sum})
		}
		return t.reduceTensor(nil, true, fp32.ReduceSum)
	}

	return t.reduceTensor(dims, true, fp32.ReduceSum)
}

// Mean computes mean along specified dimensions.
func (t *Tensor) Mean(dims ...int) *Tensor {
	if t == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMean)
}

// Max computes maximum along specified dimensions.
func (t *Tensor) Max(dims ...int) *Tensor {
	if t == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMax)
}

// Min computes minimum along specified dimensions.
func (t *Tensor) Min(dims ...int) *Tensor {
	if t == nil {
		return nil
	}

	return t.reduceTensor(dims, true, fp32.ReduceMin)
}

// ArgMax returns indices of maximum elements along specified dimension.
// Uses fp32 primitive.Iamax for vector case.
func (t *Tensor) ArgMax(dim int) *Tensor {
	if t == nil {
		return nil
	}

	if dim < 0 || dim >= len(t.shape) {
		panic(fmt.Sprintf("tensor.ArgMax: dimension %d out of range for shape %v", dim, t.shape))
	}

	if len(t.shape) == 1 && t.isContiguous() {
		idx := fp32.Iamax(t.data, 1, t.Size())
		return FromFloat32(NewShape(1), []float32{float32(idx)})
	}

	resultShape, axis := t.prepareArgmax(dim)
	result := New(t.dtype, NewShape(resultShape...))
	fp32.Argmax(
		result.data,
		result.shape.ToSlice(),
		result.shape.Strides(),
		t.data,
		t.shape.ToSlice(),
		t.shape.Strides(),
		axis,
	)
	return result
}

// Helper functions

func (t *Tensor) sameShape(other *Tensor) bool {
	if t == nil || other == nil {
		return false
	}
	if len(t.shape) != len(other.shape) {
		return false
	}
	for i := range t.shape {
		if t.shape[i] != other.shape[i] {
			return false
		}
	}
	return true
}

func (t *Tensor) sameShapeInt(shape []int) bool {
	if len(t.shape) != len(shape) {
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

func (t *Tensor) reduceTensor(dims []int, scalarWhenEmpty bool, reducer reduceFunc) *Tensor {
	if t == nil {
		return nil
	}

	axes := t.normalizeAxes(dims)
	dimSet := make(map[int]struct{}, len(axes))
	for _, axis := range axes {
		dimSet[axis] = struct{}{}
	}

	resultShape := make([]int, 0, len(t.shape))
	for i, d := range t.shape {
		if _, ok := dimSet[i]; !ok {
			resultShape = append(resultShape, d)
		}
	}
	if len(resultShape) == 0 && scalarWhenEmpty {
		resultShape = []int{1}
	}

	res := New(t.dtype, NewShape(resultShape...))

	reducer(
		res.data,
		res.shape.ToSlice(),
		res.shape.Strides(),
		t.data,
		t.shape.ToSlice(),
		t.shape.Strides(),
		axes,
	)

	return res
}

func (t *Tensor) normalizeAxes(dims []int) []int {
	if len(t.shape) == 0 {
		panic("tensor: reduction on empty tensor")
	}

	if len(dims) == 0 {
		axes := make([]int, len(t.shape))
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

func (t *Tensor) prepareArgmax(dim int) ([]int, int) {
	if len(t.shape) == 0 {
		panic("tensor.ArgMax: empty tensor")
	}
	if dim < 0 || dim >= len(t.shape) {
		panic(fmt.Sprintf("tensor.ArgMax: dimension %d out of range for shape %v", dim, t.shape))
	}

	shape := make([]int, 0, len(t.shape)-1)
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

func (t *Tensor) copyTo(dst *Tensor) {
	if dst == nil {
		return
	}

	if len(dst.shape) != len(t.shape) {
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

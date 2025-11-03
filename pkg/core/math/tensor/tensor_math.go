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
		panic(fmt.Sprintf("tensor.Add: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	if t.isContiguous() && other.isContiguous() {
		size := t.Size()
		fp32.Axpy(t.Data, other.Data, 1, 1, size, 1.0)
		return t
	}

	shape := Shape(t.Dim)
	otherShape := Shape(other.Dim)
	stridesT := shape.Strides()
	stridesOther := otherShape.Strides()
	fp32.ElemAdd(t.Data, t.Data, other.Data, []int(shape), stridesT, stridesT, stridesOther)
	return t
}

// Sub subtracts another tensor element-wise in-place.
// Uses fp32 primitive.Axpy with alpha=-1.
func (t *Tensor) Sub(other *Tensor) *Tensor {
	if t == nil || other == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Sub: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	if t.isContiguous() && other.isContiguous() {
		size := t.Size()
		fp32.Axpy(t.Data, other.Data, 1, 1, size, -1.0)
		return t
	}

	shape := Shape(t.Dim)
	otherShape := Shape(other.Dim)
	stridesT := shape.Strides()
	stridesOther := otherShape.Strides()
	fp32.ElemSub(t.Data, t.Data, other.Data, []int(shape), stridesT, stridesT, stridesOther)
	return t
}

// Mul multiplies another tensor element-wise in-place.
func (t *Tensor) Mul(other *Tensor) *Tensor {
	if t == nil || other == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Mul: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	shape := Shape(t.Dim)
	otherShape := Shape(other.Dim)
	stridesT := shape.Strides()
	stridesOther := otherShape.Strides()
	fp32.ElemMul(t.Data, t.Data, other.Data, []int(shape), stridesT, stridesT, stridesOther)
	return t
}

// Div divides by another tensor element-wise in-place.
func (t *Tensor) Div(other *Tensor) *Tensor {
	if t == nil || other == nil {
		return t
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.Div: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	shape := Shape(t.Dim)
	otherShape := Shape(other.Dim)
	stridesT := shape.Strides()
	stridesOther := otherShape.Strides()
	fp32.ElemDiv(t.Data, t.Data, other.Data, []int(shape), stridesT, stridesT, stridesOther)
	return t
}

// Scale multiplies the tensor by a scalar in-place.
// Uses fp32 primitive.Scal for efficient computation.
func (t *Tensor) Scale(scalar float32) *Tensor {
	if t == nil {
		return t
	}

	shape := Shape(t.Dim)
	strides := shape.Strides()
	if t.isContiguous() {
		size := t.Size()
		fp32.Scal(t.Data, 1, size, scalar)
		return t
	}

	fp32.ElemScale(t.Data, scalar, []int(shape), strides)
	return t
}

// AddTo adds two tensors and stores result in dst (or creates new tensor if dst is nil).
// Returns the destination tensor.
func (t *Tensor) AddTo(other *Tensor, dst *Tensor) *Tensor {
	if t == nil || other == nil {
		return nil
	}

	if !t.sameShape(other) {
		panic(fmt.Sprintf("tensor.AddTo: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	if dst == nil {
		dst = t.Clone()
		dst.Add(other)
		return dst
	}

	if !dst.sameShape(t) {
		panic(fmt.Sprintf("tensor.AddTo: destination shape mismatch: %v vs %v", dst.Dim, t.Dim))
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
		panic(fmt.Sprintf("tensor.MulTo: shape mismatch: %v vs %v", t.Dim, other.Dim))
	}

	if dst == nil {
		dst = t.Clone()
		dst.Mul(other)
		return dst
	}

	if !dst.sameShape(t) {
		panic(fmt.Sprintf("tensor.MulTo: destination shape mismatch: %v vs %v", dst.Dim, t.Dim))
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

	if len(shape) < len(t.Dim) {
		return nil, fmt.Errorf("tensor.BroadcastTo: target shape %v has fewer dimensions than %v", shape, t.Dim)
	}

	// Simple clone when shapes already match
	if t.sameShapeInt(shape) {
		return t.Clone(), nil
	}

	// Validate broadcasting via primitive helper
	if _, err := fp32.BroadcastStrides(t.Dim, Shape(t.Dim).Strides(), shape); err != nil {
		return nil, fmt.Errorf("tensor.BroadcastTo: %w", err)
	}

	size := sizeFromShape(shape)
	result := &Tensor{
		Dim:  make([]int, len(shape)),
		Data: make([]float32, size),
	}
	copy(result.Dim, shape)

	if err := fp32.ExpandTo(
		result.Data,
		t.Data,
		result.Dim,
		t.Dim,
		Shape(result.Dim).Strides(),
		Shape(t.Dim).Strides(),
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
			sum := fp32.Asum(t.Data, 1, size)
			return &Tensor{Dim: []int{1}, Data: []float32{sum}}
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

	if dim < 0 || dim >= len(t.Dim) {
		panic(fmt.Sprintf("tensor.ArgMax: dimension %d out of range for shape %v", dim, t.Dim))
	}

	// For 1D tensor, prefer primitive.Iamax fast path
	if len(t.Dim) == 1 && t.isContiguous() {
		idx := fp32.Iamax(t.Data, 1, t.Size())
		return &Tensor{Dim: []int{1}, Data: []float32{float32(idx)}}
	}

	resultShape, axis := t.prepareArgmax(dim)
	result := &Tensor{
		Dim:  resultShape,
		Data: make([]float32, sizeFromShape(resultShape)),
	}
	fp32.Argmax(result.Data, result.Dim, Shape(result.Dim).Strides(), t.Data, t.Dim, Shape(t.Dim).Strides(), axis)
	return result
}

// Helper functions

func (t *Tensor) sameShape(other *Tensor) bool {
	if len(t.Dim) != len(other.Dim) {
		return false
	}
	for i := range t.Dim {
		if t.Dim[i] != other.Dim[i] {
			return false
		}
	}
	return true
}

func (t *Tensor) sameShapeInt(shape []int) bool {
	if len(t.Dim) != len(shape) {
		return false
	}
	for i := range t.Dim {
		if t.Dim[i] != shape[i] {
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

	resultShape := make([]int, 0, len(t.Dim))
	for i, d := range t.Dim {
		if _, ok := dimSet[i]; !ok {
			resultShape = append(resultShape, d)
		}
	}
	if len(resultShape) == 0 && scalarWhenEmpty {
		resultShape = []int{1}
	}

	res := &Tensor{
		Dim:  resultShape,
		Data: make([]float32, sizeFromShape(resultShape)),
	}

	reducer(
		res.Data,
		res.Dim,
		Shape(res.Dim).Strides(),
		t.Data,
		t.Dim,
		Shape(t.Dim).Strides(),
		axes,
	)

	return res
}

func (t *Tensor) normalizeAxes(dims []int) []int {
	if len(t.Dim) == 0 {
		panic("tensor: reduction on empty tensor")
	}

	if len(dims) == 0 {
		axes := make([]int, len(t.Dim))
		for i := range axes {
			axes[i] = i
		}
		return axes
	}

	axes := append([]int(nil), dims...)
	if err := Shape(t.Dim).ValidateAxes(axes); err != nil {
		panic(err)
	}
	return axes
}

func (t *Tensor) prepareArgmax(dim int) ([]int, int) {
	if len(t.Dim) == 0 {
		panic("tensor.ArgMax: empty tensor")
	}
	if dim < 0 || dim >= len(t.Dim) {
		panic(fmt.Sprintf("tensor.ArgMax: dimension %d out of range for shape %v", dim, t.Dim))
	}

	shape := make([]int, 0, len(t.Dim)-1)
	for i, d := range t.Dim {
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

	if len(dst.Dim) != len(t.Dim) {
		panic(fmt.Sprintf("tensor.copyTo: destination shape mismatch: %v vs %v", dst.Dim, t.Dim))
	}

	shape := Shape(t.Dim)
	size := t.Size()
	if size == 0 {
		return
	}

	if t.isContiguous() && dst.isContiguous() {
		fp32.Copy(dst.Data, t.Data, 1, 1, size)
		return
	}

	stridesSrc := shape.Strides()
	stridesDst := Shape(dst.Dim).Strides()
	fp32.ElemCopy(dst.Data, t.Data, []int(shape), stridesDst, stridesSrc)
}

func sizeFromShape(shape []int) int {
	return Shape(shape).Size()
}

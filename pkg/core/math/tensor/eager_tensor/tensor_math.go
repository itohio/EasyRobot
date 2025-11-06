package eager_tensor

import (
	"errors"
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/generics"
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// Add performs element-wise addition: dst = t + other (matches tf.add).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Add(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}
	if IsNil(other) {
		return t
	}
	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.Add: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Add: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	otherData := types.GetTensorData[[]float32](other)
	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	otherStrides := other.Shape().Strides()

	// Fast path for contiguous tensors
	shapeSlice := []int(t.shape)
	if t.isContiguous() && IsContiguous(otherStrides, shapeSlice) && IsContiguous(dstStrides, shapeSlice) {
		if !IsNil(dst) {
			// Copy t to dst using generics
			generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
		}
		size := t.Size()
		fp32.Axpy(dstData, otherData, 1, 1, size, 1.0)
		return result
	}

	// General path
	fp32.ElemAdd(dstData, tData, otherData, []int(t.shape), dstStrides, tStrides, otherStrides)
	return result
}

// Subtract performs element-wise subtraction: dst = t - other (matches tf.subtract).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Subtract(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}
	if IsNil(other) {
		return t
	}
	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.Subtract: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Subtract: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	otherData := types.GetTensorData[[]float32](other)
	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	otherStrides := other.Shape().Strides()

	// Fast path for contiguous tensors
	shapeSlice := []int(t.shape)
	if t.isContiguous() && IsContiguous(otherStrides, shapeSlice) && IsContiguous(dstStrides, shapeSlice) {
		if !IsNil(dst) {
			// Copy t to dst using generics
			generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
		}
		size := t.Size()
		fp32.Axpy(dstData, otherData, 1, 1, size, -1.0)
		return result
	}

	// General path
	fp32.ElemSub(dstData, tData, otherData, []int(t.shape), dstStrides, tStrides, otherStrides)
	return result
}

// Multiply performs element-wise multiplication: dst = t * other (matches tf.multiply).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Multiply(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}
	if IsNil(other) {
		return t
	}
	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.Multiply: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Multiply: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	otherData := types.GetTensorData[[]float32](other)
	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	otherStrides := other.Shape().Strides()

	fp32.ElemMul(dstData, tData, otherData, []int(t.shape), dstStrides, tStrides, otherStrides)
	return result
}

// Divide performs element-wise division: dst = t / other (matches tf.divide).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Divide(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}
	if IsNil(other) {
		return t
	}
	if !t.Shape().Equal(other.Shape()) {
		panic(fmt.Sprintf("tensor.Divide: shape mismatch: %v vs %v", t.shape, other.Shape()))
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Divide: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	otherData := types.GetTensorData[[]float32](other)
	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	otherStrides := other.Shape().Strides()

	fp32.ElemDiv(dstData, tData, otherData, []int(t.shape), dstStrides, tStrides, otherStrides)
	return result
}

// ScalarMul multiplies the tensor by a scalar: dst = scalar * t (matches tf.scalar_mul).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) ScalarMul(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	scalar32 := float32(scalar)

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.ScalarMul: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	// Fast path for contiguous tensors (in-place only)
	if IsNil(dst) && t.isContiguous() {
		size := t.Size()
		fp32.Scal(dstData, 1, size, scalar32)
		return result
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	if IsNil(dst) {
		fp32.ElemScaleInPlace(dstData, scalar32, []int(t.shape), tStrides)
	} else {
		fp32.ElemScale(dstData, tData, scalar32, []int(t.shape), dstStrides, tStrides)
	}
	return result
}

// AddScalar adds a scalar value to all elements: dst[i] = t[i] + scalar.
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) AddScalar(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	scalar32 := float32(scalar)

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.AddScalar: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemAddScalar(dstData, tData, scalar32, []int(t.shape), dstStrides, tStrides)
	return result
}

// SubScalar subtracts a scalar value from all elements: dst[i] = t[i] - scalar.
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) SubScalar(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	scalar32 := float32(scalar)

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.SubScalar: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemSubScalar(dstData, tData, scalar32, []int(t.shape), dstStrides, tStrides)
	return result
}

// MulScalar multiplies all elements by a scalar: dst[i] = t[i] * scalar.
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) MulScalar(dst types.Tensor, scalar float64) types.Tensor {
	return t.ScalarMul(dst, scalar)
}

// DivScalar divides all elements by a scalar: dst[i] = t[i] / scalar.
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) DivScalar(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	scalar32 := float32(scalar)

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.DivScalar: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemDivScalar(dstData, tData, scalar32, []int(t.shape), dstStrides, tStrides)
	return result
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
// Sum sums along specified dimensions. If no dimensions specified, sums all elements (matches tf.reduce_sum).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Sum(dst types.Tensor, dims []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	result := t.reduceTensor(dims, true, fp32.ReduceSum)
	if result == nil {
		return nil
	}

	if IsNil(dst) {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.Sum: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	// Copy result to dst using generics
	resultData := types.GetTensorData[[]float32](result)
	dstData := types.GetTensorData[[]float32](dst)
	shapeSlice := result.Shape().ToSlice()
	generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dst.Shape().Strides(), result.Shape().Strides())
	return dst
}

// ReduceSum is an alias for Sum (matches TensorFlow naming: tf.reduce_sum).
func (t Tensor) ReduceSum(dst types.Tensor, dims []int) types.Tensor {
	return t.Sum(dst, dims)
}

// Mean computes mean along specified dimensions. If no dimensions specified, means all elements (matches tf.reduce_mean).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Mean(dst types.Tensor, dims []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	result := t.reduceTensor(dims, true, fp32.ReduceMean)
	if result == nil {
		return nil
	}

	if IsNil(dst) {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.Mean: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	// Copy result to dst using generics
	resultData := types.GetTensorData[[]float32](result)
	dstData := types.GetTensorData[[]float32](dst)
	shapeSlice := result.Shape().ToSlice()
	generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dst.Shape().Strides(), result.Shape().Strides())
	return dst
}

// ReduceMean is an alias for Mean (matches TensorFlow naming: tf.reduce_mean).
func (t Tensor) ReduceMean(dst types.Tensor, dims []int) types.Tensor {
	return t.Mean(dst, dims)
}

// Max computes maximum along specified dimensions. If no dimensions specified, finds global maximum (matches tf.reduce_max).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Max(dst types.Tensor, dims []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	result := t.reduceTensor(dims, true, fp32.ReduceMax)
	if result == nil {
		return nil
	}

	if IsNil(dst) {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.Max: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	// Copy result to dst using generics
	resultData := types.GetTensorData[[]float32](result)
	dstData := types.GetTensorData[[]float32](dst)
	shapeSlice := result.Shape().ToSlice()
	generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dst.Shape().Strides(), result.Shape().Strides())
	return dst
}

// ReduceMax is an alias for Max (matches TensorFlow naming: tf.reduce_max).
func (t Tensor) ReduceMax(dst types.Tensor, dims []int) types.Tensor {
	return t.Max(dst, dims)
}

// Min computes minimum along specified dimensions. If no dimensions specified, finds global minimum (matches tf.reduce_min).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Min(dst types.Tensor, dims []int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	result := t.reduceTensor(dims, true, fp32.ReduceMin)
	if result == nil {
		return nil
	}

	if IsNil(dst) {
		return result
	}

	if !result.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.Min: destination shape mismatch: expected %v, got %v", result.Shape(), dst.Shape()))
	}

	// Copy result to dst using generics
	resultData := types.GetTensorData[[]float32](result)
	dstData := types.GetTensorData[[]float32](dst)
	shapeSlice := result.Shape().ToSlice()
	generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dst.Shape().Strides(), result.Shape().Strides())
	return dst
}

// ReduceMin is an alias for Min (matches TensorFlow naming: tf.reduce_min).
func (t Tensor) ReduceMin(dst types.Tensor, dims []int) types.Tensor {
	return t.Min(dst, dims)
}

// ArgMax returns the index of the maximum element along the specified dimension (matches tf.argmax).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) ArgMax(dst types.Tensor, dim int) types.Tensor {
	if t.shape == nil {
		return nil
	}

	if dim < 0 || dim >= t.shape.Rank() {
		panic(fmt.Sprintf("tensor.ArgMax: dimension %d out of range for shape %v", dim, t.shape))
	}

	var resultPtr types.Tensor
	if t.shape.Rank() == 1 && t.isContiguous() {
		idx := fp32.Iamax(types.GetTensorData[[]float32](t), 1, t.Size())
		result := FromFloat32(types.NewShape(1), []float32{float32(idx)})
		resultPtr = &result
	} else {
		resultShape, axis := t.prepareArgmax(dim)
		result := New(t.DataType(), types.NewShape(resultShape...))
		resultPtr = &result
		resultData := types.GetTensorData[[]float32](resultPtr)
		tData := types.GetTensorData[[]float32](t)
		fp32.Argmax(
			resultData,
			resultPtr.Shape().ToSlice(),
			resultPtr.Shape().Strides(),
			tData,
			t.shape.ToSlice(),
			t.shape.Strides(),
			axis,
		)
	}

	if IsNil(dst) {
		return resultPtr
	}

	if !resultPtr.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.ArgMax: destination shape mismatch: expected %v, got %v", resultPtr.Shape(), dst.Shape()))
	}

	// Copy result to dst using generics
	resultData := types.GetTensorData[[]float32](resultPtr)
	dstData := types.GetTensorData[[]float32](dst)
	shapeSlice := resultPtr.Shape().ToSlice()
	generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dst.Shape().Strides(), resultPtr.Shape().Strides())
	return dst
}

// ArgMin returns the index of the minimum element along the specified dimension (matches tf.argmin).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) ArgMin(dst types.Tensor, dim int) types.Tensor {
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

	if IsNil(dst) {
		return resultPtr
	}

	if !resultPtr.Shape().Equal(dst.Shape()) {
		panic(fmt.Sprintf("tensor.ArgMin: destination shape mismatch: expected %v, got %v", resultPtr.Shape(), dst.Shape()))
	}

	// Copy result to dst using generics
	dstData := types.GetTensorData[[]float32](dst)
	shapeSlice := resultPtr.Shape().ToSlice()
	generics.ElemCopyStrided[float32](dstData, resultData, shapeSlice, dst.Shape().Strides(), resultPtr.Shape().Strides())
	return dst
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

	dstData := types.GetTensorData[[]float32](dst)
	tData := types.GetTensorData[[]float32](t)
	shapeSlice := []int(t.shape)
	stridesSrc := t.shape.Strides()
	stridesDst := dst.shape.Strides()

	// Use generics.ElemCopyStrided which handles both contiguous and strided cases
	generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, stridesDst, stridesSrc)
}

// Where creates a new tensor by selecting elements from a where condition is true, otherwise from b.
// condition, a, b must have compatible shapes.
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Where(dst types.Tensor, condition, a, b types.Tensor) types.Tensor {
	if t.shape == nil || condition == nil || condition.Shape() == nil || a == nil || a.Shape() == nil || b == nil || b.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(condition.Shape()) || !t.Shape().Equal(a.Shape()) || !t.Shape().Equal(b.Shape()) {
		panic("tensor.Where: shape mismatch")
	}

	var result types.Tensor
	if IsNil(dst) {
		// Create new tensor
		result = t.Clone()
		if result == nil {
			return nil
		}
	} else {
		// Validate dst shape
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Where: destination shape mismatch: expected %v, got %v", t.Shape(), dst.Shape()))
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	strides := fp32.ComputeStrides(shape)
	conditionData := types.GetTensorData[[]float32](condition)
	aData := types.GetTensorData[[]float32](a)
	bData := types.GetTensorData[[]float32](b)
	resultData := types.GetTensorData[[]float32](result)

	generics.ElemWhere[float32](
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
	tData := types.GetTensorData[[]float32](t)
	otherData := types.GetTensorData[[]float32](other)
	resultData := types.GetTensorData[[]float32](resultPtr)

	generics.ElemGreaterThanStrided[float32](
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

// Square computes element-wise square: dst[i] = t[i]^2 (matches tf.square).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Square(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Square: destination shape mismatch")
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemSquare(dstData, tData, []int(t.shape), dstStrides, tStrides)
	return result
}

// Sqrt computes element-wise square root: dst[i] = sqrt(t[i]) (matches tf.sqrt).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Sqrt(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Sqrt: destination shape mismatch")
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemSqrt(dstData, tData, []int(t.shape), dstStrides, tStrides)
	return result
}

// Exp computes element-wise exponential: dst[i] = exp(t[i]) (matches tf.exp).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Exp(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Exp: destination shape mismatch")
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemExp(dstData, tData, []int(t.shape), dstStrides, tStrides)
	return result
}

// Log computes element-wise natural logarithm: dst[i] = log(t[i]) (matches tf.log).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Log(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Log: destination shape mismatch")
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemLog(dstData, tData, []int(t.shape), dstStrides, tStrides)
	return result
}

// Pow computes element-wise power: dst[i] = t[i]^power (matches tf.pow).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Pow(dst types.Tensor, power float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	power32 := float32(power)

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Pow: destination shape mismatch")
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemPow(dstData, tData, power32, []int(t.shape), dstStrides, tStrides)
	return result
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

	generics.ElemEqualStrided[float32](
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

	generics.ElemLessStrided[float32](
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

	generics.ElemNotEqualStrided[float32](
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

	generics.ElemGreaterEqualStrided[float32](
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

	generics.ElemLessEqualStrided[float32](
		resultData, tData, otherData,
		shape, strides, strides, strides,
	)
	return resultPtr
}

// Abs computes element-wise absolute value: dst[i] = |t[i]| (matches tf.abs).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Abs(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Abs: destination shape mismatch")
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemAbs(dstData, tData, []int(t.shape), dstStrides, tStrides)
	return result
}

// Sign computes element-wise sign: dst[i] = sign(t[i]) (-1, 0, or 1) (matches tf.sign).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Sign(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Sign: destination shape mismatch")
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	generics.ElemSignStrided[float32](dstData, tData, []int(t.shape), dstStrides, tStrides)
	return result
}

// Cos computes element-wise cosine: dst[i] = cos(t[i]) (matches tf.cos).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Cos(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Cos: destination shape mismatch")
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemCos(dstData, tData, []int(t.shape), dstStrides, tStrides)
	return result
}

// Sin computes element-wise sine: dst[i] = sin(t[i]) (matches tf.sin).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Sin(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Sin: destination shape mismatch")
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	fp32.ElemSin(dstData, tData, []int(t.shape), dstStrides, tStrides)
	return result
}

// Negative computes element-wise negation: dst[i] = -t[i] (matches tf.negative).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Negative(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var tData []float32
	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Negative: destination shape mismatch")
		}
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	tStrides := t.shape.Strides()
	dstStrides := result.Shape().Strides()
	generics.ElemNegativeStrided[float32](dstData, tData, []int(t.shape), dstStrides, tStrides)
	return result
}

// Fill fills the tensor with a constant value.
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Fill(dst types.Tensor, value float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	value32 := float32(value)

	var dstData []float32
	var result types.Tensor
	if IsNil(dst) {
		dstData = types.GetTensorData[[]float32](t)
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Fill: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		dstData = types.GetTensorData[[]float32](dst)
		result = dst
	}

	shapeSlice := []int(t.shape)
	strides := result.Shape().Strides()
	if IsContiguous(strides, shapeSlice) {
		// Fast path: contiguous - use ElemFill
		size := t.Size()
		generics.ElemFill[float32](dstData, value32, size)
	} else {
		// Strided path: use ElemFillStrided
		generics.ElemFillStrided[float32](dstData, value32, shapeSlice, strides)
	}
	return result
}

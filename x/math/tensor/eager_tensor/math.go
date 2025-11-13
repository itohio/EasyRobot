package eager_tensor

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/primitive/fp32"
	"github.com/itohio/EasyRobot/x/math/primitive/generics"
	. "github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
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

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Add: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)
	otherStrides := other.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		otherData := types.GetTensorData[[]float32](other)
		dstData := types.GetTensorData[[]float32](result)

		// Fast path for contiguous tensors
		if t.IsContiguous() && other.IsContiguous() && result.IsContiguous() {
			if !IsNil(dst) {
				// Copy t to dst using generics
				generics.ElemCopyStrided[float32](dstData, tData, shape, dstStrides, tStrides)
			}
			size := t.Size()
			fp32.Axpy(dstData, otherData, 1, 1, size, 1.0)
			return result
		}

		// General path
		fp32.ElemAdd(dstData, tData, otherData, shape, dstStrides, tStrides, otherStrides)
	default:
		panic(fmt.Sprintf("tensor.Add: unsupported data type: %T", tData))
	}
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

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Subtract: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)
	otherStrides := other.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		otherData := types.GetTensorData[[]float32](other)
		dstData := types.GetTensorData[[]float32](result)

		// Fast path for contiguous tensors
		if t.IsContiguous() && other.IsContiguous() && result.IsContiguous() {
			if !IsNil(dst) {
				// Copy t to dst using generics
				generics.ElemCopyStrided[float32](dstData, tData, shape, dstStrides, tStrides)
			}
			size := t.Size()
			fp32.Axpy(dstData, otherData, 1, 1, size, -1.0)
			return result
		}

		// General path
		fp32.ElemSub(dstData, tData, otherData, shape, dstStrides, tStrides, otherStrides)
	default:
		panic(fmt.Sprintf("tensor.Subtract: unsupported data type: %T", tData))
	}
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

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Multiply: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)
	otherStrides := other.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		otherData := types.GetTensorData[[]float32](other)
		dstData := types.GetTensorData[[]float32](result)
		fp32.ElemMul(dstData, tData, otherData, shape, dstStrides, tStrides, otherStrides)
	default:
		panic(fmt.Sprintf("tensor.Multiply: unsupported data type: %T", tData))
	}
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

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Divide: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)
	otherStrides := other.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		otherData := types.GetTensorData[[]float32](other)
		dstData := types.GetTensorData[[]float32](result)
		fp32.ElemDiv(dstData, tData, otherData, shape, dstStrides, tStrides, otherStrides)
	default:
		panic(fmt.Sprintf("tensor.Divide: unsupported data type: %T", tData))
	}
	return result
}

// ScalarMul multiplies the tensor by a scalar: dst = scalar * t (matches tf.scalar_mul).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) ScalarMul(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.ScalarMul: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		scalar32 := float32(scalar)

		// Fast path for contiguous tensors (in-place only)
		if IsNil(dst) && t.IsContiguous() {
			size := t.Size()
			fp32.Scal(dstData, 1, size, scalar32)
			return result
		}

		if IsNil(dst) {
			fp32.ElemScaleInPlace(dstData, scalar32, shape, tStrides)
		} else {
			fp32.ElemScale(dstData, tData, scalar32, shape, dstStrides, tStrides)
		}
	default:
		panic(fmt.Sprintf("tensor.ScalarMul: unsupported data type: %T", tData))
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

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.AddScalar: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		scalar32 := float32(scalar)
		fp32.ElemAddScalar(dstData, tData, scalar32, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.AddScalar: unsupported data type: %T", tData))
	}
	return result
}

// SubScalar subtracts a scalar value from all elements: dst[i] = t[i] - scalar.
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) SubScalar(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.SubScalar: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		scalar32 := float32(scalar)
		fp32.ElemSubScalar(dstData, tData, scalar32, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.SubScalar: unsupported data type: %T", tData))
	}
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

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.DivScalar: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		scalar32 := float32(scalar)
		fp32.ElemDivScalar(dstData, tData, scalar32, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.DivScalar: unsupported data type: %T", tData))
	}
	return result
}

// BroadcastTo broadcasts the tensor to a new shape.
// Currently creates a view-like operation (future: implement efficient broadcasting).
func (t Tensor) BroadcastTo(dst types.Tensor, shape types.Shape) types.Tensor {
	if t.shape == nil {
		panic("tensor.BroadcastTo: nil tensor")
	}

	targetShape := types.NewShape(shape...)
	if len(shape) < t.shape.Rank() {
		panic(fmt.Sprintf("tensor.BroadcastTo: target shape %v has fewer dimensions than %v", shape, t.shape))
	}

	// Handle destination
	var result types.Tensor
	if IsNil(dst) {
		// If shapes match exactly, we can create a copy or use Clone semantics
		if shape.Equal(t.Shape()) {
			result = t.Clone()
			return result
		}

		// Validate broadcasting is possible
		tStrides := t.Strides(nil)
		if _, err := fp32.BroadcastStrides(t.shape.ToSlice(), tStrides, shape); err != nil {
			panic(fmt.Sprintf("tensor.BroadcastTo: %v", err))
		}

		result = New(t.DataType(), targetShape)
	} else {
		// Validate dst shape matches target shape
		if !targetShape.Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.BroadcastTo: destination shape mismatch: expected %v, got %v", targetShape, dst.Shape()))
		}

		// If shapes match exactly, copy t to dst
		if shape.Equal(t.Shape()) {
			dst.Copy(t)
			return dst
		}

		// Validate broadcasting is possible
		tStrides := t.Strides(nil)
		if _, err := fp32.BroadcastStrides(t.shape.ToSlice(), tStrides, shape); err != nil {
			panic(fmt.Sprintf("tensor.BroadcastTo: %v", err))
		}

		result = dst
	}

	switch tData := t.Data().(type) {
	case []float32:
		resultData := types.GetTensorData[[]float32](result)
		tStrides := t.Strides(nil)
		resultStrides := result.Strides(nil)
		if err := fp32.ExpandTo(
			resultData,
			tData,
			result.Shape().ToSlice(),
			t.shape.ToSlice(),
			resultStrides,
			tStrides,
		); err != nil {
			panic(fmt.Sprintf("tensor.BroadcastTo: %v", err))
		}
	default:
		panic(fmt.Sprintf("tensor.BroadcastTo: unsupported data type: %T", tData))
	}

	return result
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

	result, scratch := t.reduceTensor(dst, dims, true, fp32.ReduceSum)
	if result == nil {
		return nil
	}

	if scratch {
		if IsNil(dst) {
			panic("tensor.Sum: internal scratch buffer without destination")
		}
		dstData := types.GetTensorData[[]float32](dst)
		srcData := types.GetTensorData[[]float32](result)
		shapeSlice := dst.Shape().ToSlice()
		dstStrides := dst.Strides(nil)
		srcStrides := result.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, srcData, shapeSlice, dstStrides, srcStrides)
		result.Release()
		return dst
	}

	if IsNil(dst) {
		return result
	}

	return result
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

	result, scratch := t.reduceTensor(dst, dims, true, fp32.ReduceMean)
	if result == nil {
		return nil
	}

	if scratch {
		if IsNil(dst) {
			panic("tensor.Mean: internal scratch buffer without destination")
		}
		dstData := types.GetTensorData[[]float32](dst)
		srcData := types.GetTensorData[[]float32](result)
		shapeSlice := dst.Shape().ToSlice()
		dstStrides := dst.Strides(nil)
		srcStrides := result.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, srcData, shapeSlice, dstStrides, srcStrides)
		result.Release()
		return dst
	}

	if IsNil(dst) {
		return result
	}

	return result
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

	result, scratch := t.reduceTensor(dst, dims, true, fp32.ReduceMax)
	if result == nil {
		return nil
	}

	if scratch {
		if IsNil(dst) {
			panic("tensor.Max: internal scratch buffer without destination")
		}
		dstData := types.GetTensorData[[]float32](dst)
		srcData := types.GetTensorData[[]float32](result)
		shapeSlice := dst.Shape().ToSlice()
		dstStrides := dst.Strides(nil)
		srcStrides := result.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, srcData, shapeSlice, dstStrides, srcStrides)
		result.Release()
		return dst
	}

	if IsNil(dst) {
		return result
	}

	return result
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

	result, scratch := t.reduceTensor(dst, dims, true, fp32.ReduceMin)
	if result == nil {
		return nil
	}

	if scratch {
		if IsNil(dst) {
			panic("tensor.Min: internal scratch buffer without destination")
		}
		dstData := types.GetTensorData[[]float32](dst)
		srcData := types.GetTensorData[[]float32](result)
		shapeSlice := dst.Shape().ToSlice()
		dstStrides := dst.Strides(nil)
		srcStrides := result.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, srcData, shapeSlice, dstStrides, srcStrides)
		result.Release()
		return dst
	}

	if IsNil(dst) {
		return result
	}

	return result
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

	resultShape, axis := t.prepareArgmax(dim)
	targetShape := types.NewShape(resultShape...)
	dtype := t.DataType()

	var target types.Tensor
	useScratch := false

	if IsNil(dst) {
		target = New(dtype, targetShape)
	} else {
		if dst.DataType() != dtype {
			panic(fmt.Sprintf("tensor.ArgMax: destination dtype mismatch: expected %v, got %v", dtype, dst.DataType()))
		}
		if !targetShape.Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.ArgMax: destination shape mismatch: expected %v, got %v", targetShape, dst.Shape()))
		}
		if dst.IsContiguous() && dst.Offset() == 0 {
			target = dst
		} else {
			tmp := New(dtype, targetShape)
			target = tmp
			useScratch = true
		}
	}

	tData := types.GetTensorData[[]float32](t)
	targetData := types.GetTensorData[[]float32](target)
	if tData == nil || targetData == nil {
		panic("tensor.ArgMax: unsupported data type")
	}

	resultStrides := target.Strides(nil)
	tStrides := t.Strides(nil)
	fp32.Argmax(
		targetData,
		target.Shape().ToSlice(),
		resultStrides,
		tData,
		t.shape.ToSlice(),
		tStrides,
		axis,
	)

	if useScratch {
		dstData := types.GetTensorData[[]float32](dst)
		shapeSlice := dst.Shape().ToSlice()
		dstStrides := dst.Strides(nil)
		srcStrides := target.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, targetData, shapeSlice, dstStrides, srcStrides)
		target.Release()
		return dst
	}

	if IsNil(dst) {
		return target
	}

	return target
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
	targetShape := types.NewShape(resultShape...)
	dtype := t.DataType()

	var target types.Tensor
	useScratch := false

	if IsNil(dst) {
		target = New(dtype, targetShape)
	} else {
		if dst.DataType() != dtype {
			panic(fmt.Sprintf("tensor.ArgMin: destination dtype mismatch: expected %v, got %v", dtype, dst.DataType()))
		}
		if !targetShape.Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.ArgMin: destination shape mismatch: expected %v, got %v", targetShape, dst.Shape()))
		}
		if dst.IsContiguous() && dst.Offset() == 0 {
			target = dst
		} else {
			tmp := New(dtype, targetShape)
			target = tmp
			useScratch = true
		}
	}

	tData := types.GetTensorData[[]float32](t)
	targetData := types.GetTensorData[[]float32](target)
	if tData == nil || targetData == nil {
		panic("tensor.ArgMin: unsupported data type")
	}

	targetStrides := target.Strides(nil)
	tStrides := t.Strides(nil)
	indices := make([]int32, target.Size())
	fp32.Argmin(
		indices,
		target.Shape().ToSlice(),
		targetStrides,
		tData,
		t.shape.ToSlice(),
		tStrides,
		axis,
	)

	for i := range indices {
		targetData[i] = float32(indices[i])
	}

	if useScratch {
		dstData := types.GetTensorData[[]float32](dst)
		shapeSlice := dst.Shape().ToSlice()
		dstStrides := dst.Strides(nil)
		srcStrides := target.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, targetData, shapeSlice, dstStrides, srcStrides)
		target.Release()
		return dst
	}

	if IsNil(dst) {
		return target
	}

	return target
}

// Helper functions

type reduceFunc func(dst []float32, dstShape []int, dstStrides []int, src []float32, srcShape []int, srcStrides []int, axes []int)

func (t Tensor) reduceTensor(dst types.Tensor, dims []int, scalarWhenEmpty bool, reducer reduceFunc) (types.Tensor, bool) {
	if t.shape == nil {
		return nil, false
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

	targetShape := types.NewShape(resultShape...)
	dtype := t.DataType()

	var target types.Tensor
	useScratch := false

	if IsNil(dst) {
		target = New(dtype, targetShape)
	} else {
		if dst.DataType() != dtype {
			panic(fmt.Sprintf("tensor.reduceTensor: destination dtype mismatch: expected %v, got %v", dtype, dst.DataType()))
		}
		if !targetShape.Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.reduceTensor: destination shape mismatch: expected %v, got %v", targetShape, dst.Shape()))
		}
		if dst.IsContiguous() && dst.Offset() == 0 {
			target = dst
		} else {
			tmp := New(dtype, targetShape)
			target = tmp
			useScratch = true
		}
	}

	tData := types.GetTensorData[[]float32](t)
	targetData := types.GetTensorData[[]float32](target)
	if tData == nil || targetData == nil {
		panic("tensor.reduceTensor: unsupported data type")
	}

	var targetStridesStatic [MAX_DIMS]int
	targetStrides := target.Strides(targetStridesStatic[:target.Shape().Rank()])
	var tStridesStatic [MAX_DIMS]int
	tStrides := t.Strides(tStridesStatic[:t.shape.Rank()])

	reducer(
		targetData,
		target.Shape().ToSlice(),
		targetStrides,
		tData,
		t.shape.ToSlice(),
		tStrides,
		axes,
	)

	return target, useScratch
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
	dstStrides := result.Strides(nil)
	conditionStrides := condition.Strides(nil)
	aStrides := a.Strides(nil)
	bStrides := b.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		conditionData := types.GetTensorData[[]float32](condition)
		aData := types.GetTensorData[[]float32](a)
		bData := types.GetTensorData[[]float32](b)
		resultData := types.GetTensorData[[]float32](result)
		generics.ElemWhere[float32](
			resultData, conditionData, aData, bData,
			shape, dstStrides, conditionStrides, aStrides, bStrides,
		)
	case []float64:
		conditionData := types.GetTensorData[[]float64](condition)
		aData := types.GetTensorData[[]float64](a)
		bData := types.GetTensorData[[]float64](b)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemWhere[float64](
			resultData, conditionData, aData, bData,
			shape, dstStrides, conditionStrides, aStrides, bStrides,
		)
	case []int32:
		conditionData := types.GetTensorData[[]int32](condition)
		aData := types.GetTensorData[[]int32](a)
		bData := types.GetTensorData[[]int32](b)
		resultData := types.GetTensorData[[]int32](result)
		generics.ElemWhere[int32](
			resultData, conditionData, aData, bData,
			shape, dstStrides, conditionStrides, aStrides, bStrides,
		)
	case []int64:
		conditionData := types.GetTensorData[[]int64](condition)
		aData := types.GetTensorData[[]int64](a)
		bData := types.GetTensorData[[]int64](b)
		resultData := types.GetTensorData[[]int64](result)
		generics.ElemWhere[int64](
			resultData, conditionData, aData, bData,
			shape, dstStrides, conditionStrides, aStrides, bStrides,
		)
	case []int:
		conditionData := types.GetTensorData[[]int](condition)
		aData := types.GetTensorData[[]int](a)
		bData := types.GetTensorData[[]int](b)
		resultData := types.GetTensorData[[]int](result)
		generics.ElemWhere[int](
			resultData, conditionData, aData, bData,
			shape, dstStrides, conditionStrides, aStrides, bStrides,
		)
	case []int16:
		conditionData := types.GetTensorData[[]int16](condition)
		aData := types.GetTensorData[[]int16](a)
		bData := types.GetTensorData[[]int16](b)
		resultData := types.GetTensorData[[]int16](result)
		generics.ElemWhere[int16](
			resultData, conditionData, aData, bData,
			shape, dstStrides, conditionStrides, aStrides, bStrides,
		)
	case []int8:
		conditionData := types.GetTensorData[[]int8](condition)
		aData := types.GetTensorData[[]int8](a)
		bData := types.GetTensorData[[]int8](b)
		resultData := types.GetTensorData[[]int8](result)
		generics.ElemWhere[int8](
			resultData, conditionData, aData, bData,
			shape, dstStrides, conditionStrides, aStrides, bStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.Where: unsupported data type: %T", tData))
	}
	return result
}

// Greater creates a tensor with 1.0 where t > other, 0.0 otherwise (matches tf.greater).
// t and other must have compatible shapes.
func (t Tensor) Greater(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.Greater: shape mismatch")
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	otherStrides := other.Strides(nil)

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Greater: destination shape mismatch")
		}
		result = dst
	}
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		otherData := types.GetTensorData[[]float32](other)
		resultData := types.GetTensorData[[]float32](result)
		generics.ElemGreaterThanStrided[float32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []float64:
		otherData := types.GetTensorData[[]float64](other)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemGreaterThanStrided[float64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int32:
		otherData := types.GetTensorData[[]int32](other)
		resultData := types.GetTensorData[[]int32](result)
		generics.ElemGreaterThanStrided[int32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int64:
		otherData := types.GetTensorData[[]int64](other)
		resultData := types.GetTensorData[[]int64](result)
		generics.ElemGreaterThanStrided[int64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int:
		otherData := types.GetTensorData[[]int](other)
		resultData := types.GetTensorData[[]int](result)
		generics.ElemGreaterThanStrided[int](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int16:
		otherData := types.GetTensorData[[]int16](other)
		resultData := types.GetTensorData[[]int16](result)
		generics.ElemGreaterThanStrided[int16](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int8:
		otherData := types.GetTensorData[[]int8](other)
		resultData := types.GetTensorData[[]int8](result)
		generics.ElemGreaterThanStrided[int8](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.Greater: unsupported data type: %T", tData))
	}
	return result
}

// ZerosLike creates a new tensor with the same shape as t, filled with zeros.
func ZerosLike(t types.Tensor) types.Tensor {
	if t == nil || t.Shape() == nil {
		return nil
	}
	result := New(t.DataType(), t.Shape())
	return result
}

// OnesLike creates a new tensor with the same shape as t, filled with ones.
func OnesLike(t types.Tensor) types.Tensor {
	if t == nil || t.Shape() == nil {
		return nil
	}
	result := New(t.DataType(), t.Shape())
	// Fill with ones by scaling a zero tensor by 0 and then adding 1
	// Actually, let's use the data directly for efficiency
	resultData := types.GetTensorData[[]float32](result)
	for i := range resultData {
		resultData[i] = 1.0
	}
	return result
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
	resultData := types.GetTensorData[[]float32](result)
	for i := range resultData {
		resultData[i] = value
	}
	return result
}

// Square computes element-wise square: dst[i] = t[i]^2 (matches tf.square).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Square(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Square: destination shape mismatch")
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		fp32.ElemSquare(dstData, tData, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.Square: unsupported data type: %T", tData))
	}
	return result
}

// Sqrt computes element-wise square root: dst[i] = sqrt(t[i]) (matches tf.sqrt).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Sqrt(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Sqrt: destination shape mismatch")
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		fp32.ElemSqrt(dstData, tData, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.Sqrt: unsupported data type: %T", tData))
	}
	return result
}

// Exp computes element-wise exponential: dst[i] = exp(t[i]) (matches tf.exp).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Exp(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Exp: destination shape mismatch")
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		fp32.ElemExp(dstData, tData, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.Exp: unsupported data type: %T", tData))
	}
	return result
}

// Log computes element-wise natural logarithm: dst[i] = log(t[i]) (matches tf.log).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Log(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Log: destination shape mismatch")
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		fp32.ElemLog(dstData, tData, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.Log: unsupported data type: %T", tData))
	}
	return result
}

// Pow computes element-wise power: dst[i] = t[i]^power (matches tf.pow).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Pow(dst types.Tensor, power float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Pow: destination shape mismatch")
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		power32 := float32(power)
		fp32.ElemPow(dstData, tData, power32, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.Pow: unsupported data type: %T", tData))
	}
	return result
}

// Equal creates a tensor with 1.0 where t == other, 0.0 otherwise.
// t and other must have compatible shapes.
func (t Tensor) Equal(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.Equal: shape mismatch")
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	otherStrides := other.Strides(nil)

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Equal: destination shape mismatch")
		}
		result = dst
	}
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		otherData := types.GetTensorData[[]float32](other)
		resultData := types.GetTensorData[[]float32](result)
		generics.ElemEqualStrided[float32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []float64:
		otherData := types.GetTensorData[[]float64](other)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemEqualStrided[float64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int32:
		otherData := types.GetTensorData[[]int32](other)
		resultData := types.GetTensorData[[]int32](result)
		generics.ElemEqualStrided[int32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int64:
		otherData := types.GetTensorData[[]int64](other)
		resultData := types.GetTensorData[[]int64](result)
		generics.ElemEqualStrided[int64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int:
		otherData := types.GetTensorData[[]int](other)
		resultData := types.GetTensorData[[]int](result)
		generics.ElemEqualStrided[int](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int16:
		otherData := types.GetTensorData[[]int16](other)
		resultData := types.GetTensorData[[]int16](result)
		generics.ElemEqualStrided[int16](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int8:
		otherData := types.GetTensorData[[]int8](other)
		resultData := types.GetTensorData[[]int8](result)
		generics.ElemEqualStrided[int8](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.Equal: unsupported data type: %T", tData))
	}
	return result
}

// EqualScalar returns a tensor with 1.0 where t == scalar, 0.0 otherwise (matches tf.equal with scalar).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) EqualScalar(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape().ToSlice()
	var tStridesStatic [MAX_DIMS]int
	tStrides := t.Strides(tStridesStatic[:t.shape.Rank()])

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.EqualScalar: destination shape mismatch: expected %v, got %v", t.Shape(), dst.Shape()))
		}
		result = dst
	}
	var dstStridesStatic [MAX_DIMS]int
	dstStrides := result.Strides(dstStridesStatic[:result.Shape().Rank()])

	// Type switch to support all data types
	switch t.DataType() {
	case types.FP32:
		tData := types.GetTensorData[[]float32](t)
		resultData := types.GetTensorData[[]float32](result)
		scalar32 := float32(scalar)
		generics.ElemEqualScalarStrided[float32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.FP64:
		tData := types.GetTensorData[[]float64](t)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemEqualScalarStrided[float64](
			resultData, tData, scalar,
			shape, dstStrides, tStrides,
		)
	case types.INT32:
		tData := types.GetTensorData[[]int32](t)
		resultData := types.GetTensorData[[]int32](result)
		scalar32 := int32(scalar)
		generics.ElemEqualScalarStrided[int32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.INT64:
		tData := types.GetTensorData[[]int64](t)
		resultData := types.GetTensorData[[]int64](result)
		scalar64 := int64(scalar)
		generics.ElemEqualScalarStrided[int64](
			resultData, tData, scalar64,
			shape, dstStrides, tStrides,
		)
	case types.INT:
		tData := types.GetTensorData[[]int](t)
		resultData := types.GetTensorData[[]int](result)
		scalarInt := int(scalar)
		generics.ElemEqualScalarStrided[int](
			resultData, tData, scalarInt,
			shape, dstStrides, tStrides,
		)
	case types.INT16:
		tData := types.GetTensorData[[]int16](t)
		resultData := types.GetTensorData[[]int16](result)
		scalar16 := int16(scalar)
		generics.ElemEqualScalarStrided[int16](
			resultData, tData, scalar16,
			shape, dstStrides, tStrides,
		)
	case types.INT8:
		tData := types.GetTensorData[[]int8](t)
		resultData := types.GetTensorData[[]int8](result)
		scalar8 := int8(scalar)
		generics.ElemEqualScalarStrided[int8](
			resultData, tData, scalar8,
			shape, dstStrides, tStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.EqualScalar: unsupported data type: %v", t.DataType()))
	}

	return result
}

// NotEqualScalar returns a tensor with 1.0 where t != scalar, 0.0 otherwise (matches tf.not_equal with scalar).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) NotEqualScalar(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape().ToSlice()
	var tStridesStatic [MAX_DIMS]int
	tStrides := t.Strides(tStridesStatic[:t.shape.Rank()])

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.NotEqualScalar: destination shape mismatch: expected %v, got %v", t.Shape(), dst.Shape()))
		}
		result = dst
	}
	var dstStridesStatic [MAX_DIMS]int
	dstStrides := result.Strides(dstStridesStatic[:result.Shape().Rank()])

	// Type switch to support all data types
	switch t.DataType() {
	case types.FP32:
		tData := types.GetTensorData[[]float32](t)
		resultData := types.GetTensorData[[]float32](result)
		scalar32 := float32(scalar)
		generics.ElemNotEqualScalarStrided[float32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.FP64:
		tData := types.GetTensorData[[]float64](t)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemNotEqualScalarStrided[float64](
			resultData, tData, scalar,
			shape, dstStrides, tStrides,
		)
	case types.INT32:
		tData := types.GetTensorData[[]int32](t)
		resultData := types.GetTensorData[[]int32](result)
		scalar32 := int32(scalar)
		generics.ElemNotEqualScalarStrided[int32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.INT64:
		tData := types.GetTensorData[[]int64](t)
		resultData := types.GetTensorData[[]int64](result)
		scalar64 := int64(scalar)
		generics.ElemNotEqualScalarStrided[int64](
			resultData, tData, scalar64,
			shape, dstStrides, tStrides,
		)
	case types.INT:
		tData := types.GetTensorData[[]int](t)
		resultData := types.GetTensorData[[]int](result)
		scalarInt := int(scalar)
		generics.ElemNotEqualScalarStrided[int](
			resultData, tData, scalarInt,
			shape, dstStrides, tStrides,
		)
	case types.INT16:
		tData := types.GetTensorData[[]int16](t)
		resultData := types.GetTensorData[[]int16](result)
		scalar16 := int16(scalar)
		generics.ElemNotEqualScalarStrided[int16](
			resultData, tData, scalar16,
			shape, dstStrides, tStrides,
		)
	case types.INT8:
		tData := types.GetTensorData[[]int8](t)
		resultData := types.GetTensorData[[]int8](result)
		scalar8 := int8(scalar)
		generics.ElemNotEqualScalarStrided[int8](
			resultData, tData, scalar8,
			shape, dstStrides, tStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.NotEqualScalar: unsupported data type: %v", t.DataType()))
	}

	return result
}

// GreaterScalar returns a tensor with 1.0 where t > scalar, 0.0 otherwise (matches tf.greater with scalar).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) GreaterScalar(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape().ToSlice()
	var tStridesStatic [MAX_DIMS]int
	tStrides := t.Strides(tStridesStatic[:t.shape.Rank()])

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.GreaterScalar: destination shape mismatch: expected %v, got %v", t.Shape(), dst.Shape()))
		}
		result = dst
	}
	var dstStridesStatic [MAX_DIMS]int
	dstStrides := result.Strides(dstStridesStatic[:result.Shape().Rank()])

	// Type switch to support all data types
	switch t.DataType() {
	case types.FP32:
		tData := types.GetTensorData[[]float32](t)
		resultData := types.GetTensorData[[]float32](result)
		scalar32 := float32(scalar)
		generics.ElemGreaterScalarStrided[float32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.FP64:
		tData := types.GetTensorData[[]float64](t)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemGreaterScalarStrided[float64](
			resultData, tData, scalar,
			shape, dstStrides, tStrides,
		)
	case types.INT32:
		tData := types.GetTensorData[[]int32](t)
		resultData := types.GetTensorData[[]int32](result)
		scalar32 := int32(scalar)
		generics.ElemGreaterScalarStrided[int32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.INT64:
		tData := types.GetTensorData[[]int64](t)
		resultData := types.GetTensorData[[]int64](result)
		scalar64 := int64(scalar)
		generics.ElemGreaterScalarStrided[int64](
			resultData, tData, scalar64,
			shape, dstStrides, tStrides,
		)
	case types.INT:
		tData := types.GetTensorData[[]int](t)
		resultData := types.GetTensorData[[]int](result)
		scalarInt := int(scalar)
		generics.ElemGreaterScalarStrided[int](
			resultData, tData, scalarInt,
			shape, dstStrides, tStrides,
		)
	case types.INT16:
		tData := types.GetTensorData[[]int16](t)
		resultData := types.GetTensorData[[]int16](result)
		scalar16 := int16(scalar)
		generics.ElemGreaterScalarStrided[int16](
			resultData, tData, scalar16,
			shape, dstStrides, tStrides,
		)
	case types.INT8:
		tData := types.GetTensorData[[]int8](t)
		resultData := types.GetTensorData[[]int8](result)
		scalar8 := int8(scalar)
		generics.ElemGreaterScalarStrided[int8](
			resultData, tData, scalar8,
			shape, dstStrides, tStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.GreaterScalar: unsupported data type: %v", t.DataType()))
	}

	return result
}

// LessScalar returns a tensor with 1.0 where t < scalar, 0.0 otherwise (matches tf.less with scalar).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) LessScalar(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape().ToSlice()
	var tStridesStatic [MAX_DIMS]int
	tStrides := t.Strides(tStridesStatic[:t.shape.Rank()])

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.LessScalar: destination shape mismatch: expected %v, got %v", t.Shape(), dst.Shape()))
		}
		result = dst
	}
	var dstStridesStatic [MAX_DIMS]int
	dstStrides := result.Strides(dstStridesStatic[:result.Shape().Rank()])

	// Type switch to support all data types
	switch t.DataType() {
	case types.FP32:
		tData := types.GetTensorData[[]float32](t)
		resultData := types.GetTensorData[[]float32](result)
		scalar32 := float32(scalar)
		generics.ElemLessScalarStrided[float32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.FP64:
		tData := types.GetTensorData[[]float64](t)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemLessScalarStrided[float64](
			resultData, tData, scalar,
			shape, dstStrides, tStrides,
		)
	case types.INT32:
		tData := types.GetTensorData[[]int32](t)
		resultData := types.GetTensorData[[]int32](result)
		scalar32 := int32(scalar)
		generics.ElemLessScalarStrided[int32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.INT64:
		tData := types.GetTensorData[[]int64](t)
		resultData := types.GetTensorData[[]int64](result)
		scalar64 := int64(scalar)
		generics.ElemLessScalarStrided[int64](
			resultData, tData, scalar64,
			shape, dstStrides, tStrides,
		)
	case types.INT:
		tData := types.GetTensorData[[]int](t)
		resultData := types.GetTensorData[[]int](result)
		scalarInt := int(scalar)
		generics.ElemLessScalarStrided[int](
			resultData, tData, scalarInt,
			shape, dstStrides, tStrides,
		)
	case types.INT16:
		tData := types.GetTensorData[[]int16](t)
		resultData := types.GetTensorData[[]int16](result)
		scalar16 := int16(scalar)
		generics.ElemLessScalarStrided[int16](
			resultData, tData, scalar16,
			shape, dstStrides, tStrides,
		)
	case types.INT8:
		tData := types.GetTensorData[[]int8](t)
		resultData := types.GetTensorData[[]int8](result)
		scalar8 := int8(scalar)
		generics.ElemLessScalarStrided[int8](
			resultData, tData, scalar8,
			shape, dstStrides, tStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.LessScalar: unsupported data type: %v", t.DataType()))
	}

	return result
}

// GreaterEqualScalar returns a tensor with 1.0 where t >= scalar, 0.0 otherwise (matches tf.greater_equal with scalar).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) GreaterEqualScalar(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape().ToSlice()
	var tStridesStatic [MAX_DIMS]int
	tStrides := t.Strides(tStridesStatic[:t.shape.Rank()])

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.GreaterEqualScalar: destination shape mismatch: expected %v, got %v", t.Shape(), dst.Shape()))
		}
		result = dst
	}
	var dstStridesStatic [MAX_DIMS]int
	dstStrides := result.Strides(dstStridesStatic[:result.Shape().Rank()])

	// Type switch to support all data types
	switch t.DataType() {
	case types.FP32:
		tData := types.GetTensorData[[]float32](t)
		resultData := types.GetTensorData[[]float32](result)
		scalar32 := float32(scalar)
		generics.ElemGreaterEqualScalarStrided[float32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.FP64:
		tData := types.GetTensorData[[]float64](t)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemGreaterEqualScalarStrided[float64](
			resultData, tData, scalar,
			shape, dstStrides, tStrides,
		)
	case types.INT32:
		tData := types.GetTensorData[[]int32](t)
		resultData := types.GetTensorData[[]int32](result)
		scalar32 := int32(scalar)
		generics.ElemGreaterEqualScalarStrided[int32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.INT64:
		tData := types.GetTensorData[[]int64](t)
		resultData := types.GetTensorData[[]int64](result)
		scalar64 := int64(scalar)
		generics.ElemGreaterEqualScalarStrided[int64](
			resultData, tData, scalar64,
			shape, dstStrides, tStrides,
		)
	case types.INT:
		tData := types.GetTensorData[[]int](t)
		resultData := types.GetTensorData[[]int](result)
		scalarInt := int(scalar)
		generics.ElemGreaterEqualScalarStrided[int](
			resultData, tData, scalarInt,
			shape, dstStrides, tStrides,
		)
	case types.INT16:
		tData := types.GetTensorData[[]int16](t)
		resultData := types.GetTensorData[[]int16](result)
		scalar16 := int16(scalar)
		generics.ElemGreaterEqualScalarStrided[int16](
			resultData, tData, scalar16,
			shape, dstStrides, tStrides,
		)
	case types.INT8:
		tData := types.GetTensorData[[]int8](t)
		resultData := types.GetTensorData[[]int8](result)
		scalar8 := int8(scalar)
		generics.ElemGreaterEqualScalarStrided[int8](
			resultData, tData, scalar8,
			shape, dstStrides, tStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.GreaterEqualScalar: unsupported data type: %v", t.DataType()))
	}

	return result
}

// LessEqualScalar returns a tensor with 1.0 where t <= scalar, 0.0 otherwise (matches tf.less_equal with scalar).
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) LessEqualScalar(dst types.Tensor, scalar float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	shape := t.Shape().ToSlice()
	var tStridesStatic [MAX_DIMS]int
	tStrides := t.Strides(tStridesStatic[:t.shape.Rank()])

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.LessEqualScalar: destination shape mismatch: expected %v, got %v", t.Shape(), dst.Shape()))
		}
		result = dst
	}
	var dstStridesStatic [MAX_DIMS]int
	dstStrides := result.Strides(dstStridesStatic[:result.Shape().Rank()])

	// Type switch to support all data types
	switch t.DataType() {
	case types.FP32:
		tData := types.GetTensorData[[]float32](t)
		resultData := types.GetTensorData[[]float32](result)
		scalar32 := float32(scalar)
		generics.ElemLessEqualScalarStrided[float32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.FP64:
		tData := types.GetTensorData[[]float64](t)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemLessEqualScalarStrided[float64](
			resultData, tData, scalar,
			shape, dstStrides, tStrides,
		)
	case types.INT32:
		tData := types.GetTensorData[[]int32](t)
		resultData := types.GetTensorData[[]int32](result)
		scalar32 := int32(scalar)
		generics.ElemLessEqualScalarStrided[int32](
			resultData, tData, scalar32,
			shape, dstStrides, tStrides,
		)
	case types.INT64:
		tData := types.GetTensorData[[]int64](t)
		resultData := types.GetTensorData[[]int64](result)
		scalar64 := int64(scalar)
		generics.ElemLessEqualScalarStrided[int64](
			resultData, tData, scalar64,
			shape, dstStrides, tStrides,
		)
	case types.INT:
		tData := types.GetTensorData[[]int](t)
		resultData := types.GetTensorData[[]int](result)
		scalarInt := int(scalar)
		generics.ElemLessEqualScalarStrided[int](
			resultData, tData, scalarInt,
			shape, dstStrides, tStrides,
		)
	case types.INT16:
		tData := types.GetTensorData[[]int16](t)
		resultData := types.GetTensorData[[]int16](result)
		scalar16 := int16(scalar)
		generics.ElemLessEqualScalarStrided[int16](
			resultData, tData, scalar16,
			shape, dstStrides, tStrides,
		)
	case types.INT8:
		tData := types.GetTensorData[[]int8](t)
		resultData := types.GetTensorData[[]int8](result)
		scalar8 := int8(scalar)
		generics.ElemLessEqualScalarStrided[int8](
			resultData, tData, scalar8,
			shape, dstStrides, tStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.LessEqualScalar: unsupported data type: %v", t.DataType()))
	}

	return result
}

// Less creates a tensor with 1.0 where t < other, 0.0 otherwise.
// t and other must have compatible shapes.
func (t Tensor) Less(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.Less: shape mismatch")
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	otherStrides := other.Strides(nil)

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Less: destination shape mismatch")
		}
		result = dst
	}
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		otherData := types.GetTensorData[[]float32](other)
		resultData := types.GetTensorData[[]float32](result)
		generics.ElemLessStrided[float32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []float64:
		otherData := types.GetTensorData[[]float64](other)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemLessStrided[float64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int32:
		otherData := types.GetTensorData[[]int32](other)
		resultData := types.GetTensorData[[]int32](result)
		generics.ElemLessStrided[int32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int64:
		otherData := types.GetTensorData[[]int64](other)
		resultData := types.GetTensorData[[]int64](result)
		generics.ElemLessStrided[int64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int:
		otherData := types.GetTensorData[[]int](other)
		resultData := types.GetTensorData[[]int](result)
		generics.ElemLessStrided[int](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int16:
		otherData := types.GetTensorData[[]int16](other)
		resultData := types.GetTensorData[[]int16](result)
		generics.ElemLessStrided[int16](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int8:
		otherData := types.GetTensorData[[]int8](other)
		resultData := types.GetTensorData[[]int8](result)
		generics.ElemLessStrided[int8](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.Less: unsupported data type: %T", tData))
	}
	return result
}

// NotEqual creates a tensor with 1.0 where t != other, 0.0 otherwise (matches tf.not_equal).
func (t Tensor) NotEqual(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.NotEqual: shape mismatch")
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	otherStrides := other.Strides(nil)

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.NotEqual: destination shape mismatch")
		}
		result = dst
	}
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		otherData := types.GetTensorData[[]float32](other)
		resultData := types.GetTensorData[[]float32](result)
		generics.ElemNotEqualStrided[float32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []float64:
		otherData := types.GetTensorData[[]float64](other)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemNotEqualStrided[float64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int32:
		otherData := types.GetTensorData[[]int32](other)
		resultData := types.GetTensorData[[]int32](result)
		generics.ElemNotEqualStrided[int32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int64:
		otherData := types.GetTensorData[[]int64](other)
		resultData := types.GetTensorData[[]int64](result)
		generics.ElemNotEqualStrided[int64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int:
		otherData := types.GetTensorData[[]int](other)
		resultData := types.GetTensorData[[]int](result)
		generics.ElemNotEqualStrided[int](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int16:
		otherData := types.GetTensorData[[]int16](other)
		resultData := types.GetTensorData[[]int16](result)
		generics.ElemNotEqualStrided[int16](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int8:
		otherData := types.GetTensorData[[]int8](other)
		resultData := types.GetTensorData[[]int8](result)
		generics.ElemNotEqualStrided[int8](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.NotEqual: unsupported data type: %T", tData))
	}
	return result
}

// GreaterEqual creates a tensor with 1.0 where t >= other, 0.0 otherwise (matches tf.greater_equal).
func (t Tensor) GreaterEqual(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.GreaterEqual: shape mismatch")
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	otherStrides := other.Strides(nil)

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.GreaterEqual: destination shape mismatch")
		}
		result = dst
	}
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		otherData := types.GetTensorData[[]float32](other)
		resultData := types.GetTensorData[[]float32](result)
		generics.ElemGreaterEqualStrided[float32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []float64:
		otherData := types.GetTensorData[[]float64](other)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemGreaterEqualStrided[float64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int32:
		otherData := types.GetTensorData[[]int32](other)
		resultData := types.GetTensorData[[]int32](result)
		generics.ElemGreaterEqualStrided[int32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int64:
		otherData := types.GetTensorData[[]int64](other)
		resultData := types.GetTensorData[[]int64](result)
		generics.ElemGreaterEqualStrided[int64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int:
		otherData := types.GetTensorData[[]int](other)
		resultData := types.GetTensorData[[]int](result)
		generics.ElemGreaterEqualStrided[int](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int16:
		otherData := types.GetTensorData[[]int16](other)
		resultData := types.GetTensorData[[]int16](result)
		generics.ElemGreaterEqualStrided[int16](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int8:
		otherData := types.GetTensorData[[]int8](other)
		resultData := types.GetTensorData[[]int8](result)
		generics.ElemGreaterEqualStrided[int8](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.GreaterEqual: unsupported data type: %T", tData))
	}
	return result
}

// LessEqual creates a tensor with 1.0 where t <= other, 0.0 otherwise (matches tf.less_equal).
func (t Tensor) LessEqual(dst types.Tensor, other types.Tensor) types.Tensor {
	if t.shape == nil || other == nil || other.Shape() == nil {
		return nil
	}

	if !t.Shape().Equal(other.Shape()) {
		panic("tensor.LessEqual: shape mismatch")
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	otherStrides := other.Strides(nil)

	var result types.Tensor
	if IsNil(dst) {
		result = New(t.DataType(), t.shape)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.LessEqual: destination shape mismatch")
		}
		result = dst
	}
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		otherData := types.GetTensorData[[]float32](other)
		resultData := types.GetTensorData[[]float32](result)
		generics.ElemLessEqualStrided[float32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []float64:
		otherData := types.GetTensorData[[]float64](other)
		resultData := types.GetTensorData[[]float64](result)
		generics.ElemLessEqualStrided[float64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int32:
		otherData := types.GetTensorData[[]int32](other)
		resultData := types.GetTensorData[[]int32](result)
		generics.ElemLessEqualStrided[int32](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int64:
		otherData := types.GetTensorData[[]int64](other)
		resultData := types.GetTensorData[[]int64](result)
		generics.ElemLessEqualStrided[int64](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int:
		otherData := types.GetTensorData[[]int](other)
		resultData := types.GetTensorData[[]int](result)
		generics.ElemLessEqualStrided[int](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int16:
		otherData := types.GetTensorData[[]int16](other)
		resultData := types.GetTensorData[[]int16](result)
		generics.ElemLessEqualStrided[int16](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	case []int8:
		otherData := types.GetTensorData[[]int8](other)
		resultData := types.GetTensorData[[]int8](result)
		generics.ElemLessEqualStrided[int8](
			resultData, tData, otherData,
			shape, dstStrides, tStrides, otherStrides,
		)
	default:
		panic(fmt.Sprintf("tensor.LessEqual: unsupported data type: %T", tData))
	}
	return result
}

// Abs computes element-wise absolute value: dst[i] = |t[i]| (matches tf.abs).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Abs(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Abs: destination shape mismatch")
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		fp32.ElemAbs(dstData, tData, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.Abs: unsupported data type: %T", tData))
	}
	return result
}

// Sign computes element-wise sign: dst[i] = sign(t[i]) (-1, 0, or 1) (matches tf.sign).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Sign(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Sign: destination shape mismatch")
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		generics.ElemSignStrided[float32](dstData, tData, shape, dstStrides, tStrides)
	case []float64:
		dstData := types.GetTensorData[[]float64](result)
		generics.ElemSignStrided[float64](dstData, tData, shape, dstStrides, tStrides)
	case []int32:
		dstData := types.GetTensorData[[]int32](result)
		generics.ElemSignStrided[int32](dstData, tData, shape, dstStrides, tStrides)
	case []int64:
		dstData := types.GetTensorData[[]int64](result)
		generics.ElemSignStrided[int64](dstData, tData, shape, dstStrides, tStrides)
	case []int:
		dstData := types.GetTensorData[[]int](result)
		generics.ElemSignStrided[int](dstData, tData, shape, dstStrides, tStrides)
	case []int16:
		dstData := types.GetTensorData[[]int16](result)
		generics.ElemSignStrided[int16](dstData, tData, shape, dstStrides, tStrides)
	case []int8:
		dstData := types.GetTensorData[[]int8](result)
		generics.ElemSignStrided[int8](dstData, tData, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.Sign: unsupported data type: %T", tData))
	}
	return result
}

// Cos computes element-wise cosine: dst[i] = cos(t[i]) (matches tf.cos).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Cos(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Cos: destination shape mismatch")
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		fp32.ElemCos(dstData, tData, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.Cos: unsupported data type: %T", tData))
	}
	return result
}

// Sin computes element-wise sine: dst[i] = sin(t[i]) (matches tf.sin).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Sin(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Sin: destination shape mismatch")
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		fp32.ElemSin(dstData, tData, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.Sin: unsupported data type: %T", tData))
	}
	return result
}

// Negative computes element-wise negation: dst[i] = -t[i] (matches tf.negative).
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Negative(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Negative: destination shape mismatch")
		}
		result = dst
	}

	shape := t.Shape().ToSlice()
	tStrides := t.Strides(nil)
	dstStrides := result.Strides(nil)

	switch tData := t.Data().(type) {
	case []float32:
		dstData := types.GetTensorData[[]float32](result)
		generics.ElemNegativeStrided[float32](dstData, tData, shape, dstStrides, tStrides)
	case []float64:
		dstData := types.GetTensorData[[]float64](result)
		generics.ElemNegativeStrided[float64](dstData, tData, shape, dstStrides, tStrides)
	case []int32:
		dstData := types.GetTensorData[[]int32](result)
		generics.ElemNegativeStrided[int32](dstData, tData, shape, dstStrides, tStrides)
	case []int64:
		dstData := types.GetTensorData[[]int64](result)
		generics.ElemNegativeStrided[int64](dstData, tData, shape, dstStrides, tStrides)
	case []int:
		dstData := types.GetTensorData[[]int](result)
		generics.ElemNegativeStrided[int](dstData, tData, shape, dstStrides, tStrides)
	case []int16:
		dstData := types.GetTensorData[[]int16](result)
		generics.ElemNegativeStrided[int16](dstData, tData, shape, dstStrides, tStrides)
	case []int8:
		dstData := types.GetTensorData[[]int8](result)
		generics.ElemNegativeStrided[int8](dstData, tData, shape, dstStrides, tStrides)
	default:
		panic(fmt.Sprintf("tensor.Negative: unsupported data type: %T", tData))
	}
	return result
}

// Fill fills the tensor with a constant value.
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) Fill(dst types.Tensor, value float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		dst = t
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Fill: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		result = dst
	}

	switch dstData := dst.Data().(type) {
	case []float32:
		value32 := float32(value)
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemFill[float32](dstData, value32, size)
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemFillStrided[float32](dstData, value32, dst.Shape(), resultStrides)
		}
	case []float64:
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemFill[float64](dstData, value, size)
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemFillStrided[float64](dstData, value, dst.Shape(), resultStrides)
		}
	case []int32:
		value32 := int32(value)
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemFill[int32](dstData, value32, size)
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemFillStrided[int32](dstData, value32, dst.Shape(), resultStrides)
		}
	case []int64:
		value64 := int64(value)
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemFill[int64](dstData, value64, size)
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemFillStrided[int64](dstData, value64, dst.Shape(), resultStrides)
		}
	case []int:
		valueInt := int(value)
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemFill[int](dstData, valueInt, size)
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemFillStrided[int](dstData, valueInt, dst.Shape(), resultStrides)
		}
	case []int16:
		value16 := int16(value)
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemFill[int16](dstData, value16, size)
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemFillStrided[int16](dstData, value16, dst.Shape(), resultStrides)
		}
	case []int8:
		value8 := int8(value)
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemFill[int8](dstData, value8, size)
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemFillStrided[int8](dstData, value8, dst.Shape(), resultStrides)
		}
	default:
		panic(fmt.Sprintf("tensor.Fill: unsupported data type: %T", dstData))
	}
	return result
}

func (t Tensor) FillFunc(dst types.Tensor, f func() float64) types.Tensor {
	if t.shape == nil {
		return t
	}

	var result types.Tensor
	if IsNil(dst) {
		dst = t
		result = t
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic(fmt.Sprintf("tensor.Fill: destination shape mismatch: %v vs %v", dst.Shape(), t.shape))
		}
		result = dst
	}

	switch dstData := dst.Data().(type) {
	case []float32:
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemApplyUnary[float32](dstData, dstData, size, func(v float32) float32 {
				return float32(f())
			})
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemApplyUnaryStrided[float32](dstData, dstData, dst.Shape(), resultStrides, resultStrides, func(v float32) float32 {
				return float32(f())
			})
		}
	case []float64:
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemApplyUnary[float64](dstData, dstData, size, func(v float64) float64 {
				return float64(f())
			})
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemApplyUnaryStrided[float64](dstData, dstData, dst.Shape(), resultStrides, resultStrides, func(v float64) float64 {
				return float64(f())
			})
		}
	case []int32:
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemApplyUnary[int32](dstData, dstData, size, func(v int32) int32 {
				return int32(f())
			})
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemApplyUnaryStrided[int32](dstData, dstData, dst.Shape(), resultStrides, resultStrides, func(v int32) int32 {
				return int32(f())
			})
		}
	case []int64:
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemApplyUnary[int64](dstData, dstData, size, func(v int64) int64 {
				return int64(f())
			})
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemApplyUnaryStrided[int64](dstData, dstData, dst.Shape(), resultStrides, resultStrides, func(v int64) int64 {
				return int64(f())
			})
		}
	case []int:
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemApplyUnary[int](dstData, dstData, size, func(v int) int {
				return int(f())
			})
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemApplyUnaryStrided[int](dstData, dstData, dst.Shape(), resultStrides, resultStrides, func(v int) int {
				return int(f())
			})
		}
	case []int16:
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemApplyUnary[int16](dstData, dstData, size, func(v int16) int16 {
				return int16(f())
			})
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemApplyUnaryStrided[int16](dstData, dstData, dst.Shape(), resultStrides, resultStrides, func(v int16) int16 {
				return int16(f())
			})
		}
	case []int8:
		if result.IsContiguous() {
			size := t.Size()
			generics.ElemApplyUnary[int8](dstData, dstData, size, func(v int8) int8 {
				return int8(f())
			})
		} else {
			resultStrides := result.Strides(nil)
			generics.ElemApplyUnaryStrided[int8](dstData, dstData, dst.Shape(), resultStrides, resultStrides, func(v int8) int8 {
				return int8(f())
			})
		}
	default:
		panic(fmt.Sprintf("tensor.FillFunc: unsupported data type: %T", dstData))
	}

	return result
}

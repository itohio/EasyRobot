package eager_tensor

import (
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// ReLU applies the Rectified Linear Unit activation function: dst[i] = max(0, t[i])
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) ReLU(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.ReLU: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	size := t.Size()

	fp32.ReLU(dstData, tData, size)
	return dst
}

// Sigmoid applies the sigmoid activation function: dst[i] = 1 / (1 + exp(-t[i]))
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) Sigmoid(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Sigmoid: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	size := t.Size()

	fp32.Sigmoid(dstData, tData, size)
	return dst
}

// Tanh applies the hyperbolic tangent activation function: dst[i] = tanh(t[i])
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) Tanh(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Tanh: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	size := t.Size()

	fp32.Tanh(dstData, tData, size)
	return dst
}

// Softmax applies softmax along the specified dimension.
// Currently supports 1D tensors and 2D tensors with dim=0 (rows) or dim=1 (columns).
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) Softmax(dim int, dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	if dim < 0 || dim >= t.shape.Rank() {
		panic("tensor.Softmax: dimension out of range")
	}

	var tData []float32
	var dstData []float32
	if dst == nil || dst.Empty() {
		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](t)
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Softmax: destination shape mismatch")
		}

		tData = types.GetTensorData[[]float32](t)
		dstData = types.GetTensorData[[]float32](dst)
	}

	dstShape := t.shape
	if dst != nil && !dst.Empty() {
		dstShape = dst.Shape()
	}

	if t.shape.Rank() == 1 {
		// Softmax1D reads from src and writes to dst
		fp32.Softmax1D(tData, dstData, t.Size())
	} else if t.shape.Rank() == 2 {
		// Softmax2D functions operate in-place on dst, so copy first if needed
		if dst != nil && !dst.Empty() {
			copyTensorData(t, dst)
		}
		if dim == 0 {
			fp32.Softmax2DRows(dstData, dstShape[0], dstShape[1])
		} else if dim == 1 {
			fp32.Softmax2DCols(dstData, dstShape[0], dstShape[1])
		} else {
			panic("tensor.Softmax: invalid dimension for 2D tensor")
		}
	} else {
		panic("tensor.Softmax: only 1D and 2D tensors supported")
	}

	return dst
}

// DropoutForward applies dropout during training.
// mask[i] should contain 0.0 for dropped elements, scale (1/(1-p)) for kept elements.
// Modifies the tensor in-place: t[i] *= mask[i]
func (t Tensor) DropoutForward(mask types.Tensor) types.Tensor {
	if t.shape == nil || mask == nil || mask.Shape() == nil {
		return t
	}

	if !t.Shape().Equal(mask.Shape()) {
		panic("tensor.DropoutForward: mask shape mismatch")
	}

	maskData := types.GetTensorData[[]float32](mask)
	size := t.Size()
	tData := types.GetTensorData[[]float32](t)
	fp32.ElemMul(tData, tData, maskData, []int{size}, []int{1}, []int{1}, []int{1})
	return t
}

// DropoutMask generates a dropout mask tensor with the given dropout rate.
// mask[i] = 0.0 if dropped (with probability p), scale otherwise.
// scale = 1.0 / (1.0 - p) for inverted dropout.
// rng is typed as interface{} to avoid importing math/rand in interface; actual type is *rand.Rand.
func (t Tensor) DropoutMask(p float32, scale float32, rng interface{}) types.Tensor {
	if t.shape == nil {
		return nil
	}

	size := t.Size()
	if size == 0 {
		return t
	}

	// Type assert to *rand.Rand for random generation
	randRng, ok := rng.(interface {
		Float32() float32
	})
	if !ok {
		panic("tensor.DropoutMask: rng must implement Float32() method")
	}

	// For now, use direct data access since random generation is layer-specific
	// In the future, this could be moved to fp32 package with a random interface
	data := types.GetTensorData[[]float32](t)
	for i := range data {
		if randRng.Float32() < p {
			data[i] = 0.0
		} else {
			data[i] = scale
		}
	}

	return t
}

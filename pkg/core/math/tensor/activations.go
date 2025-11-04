package tensor

import (
	"math/rand"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

// ReLU applies the Rectified Linear Unit activation function in-place: t[i] = max(0, t[i])
func (t *Tensor) ReLU() *Tensor {
	if t == nil {
		return nil
	}

	size := t.Size()
	fp32.ReLU(t.data, t.data, size)
	return t
}

// ReLUGrad computes the ReLU gradient: dst[i] = gradOutput[i] * (input[i] > 0 ? 1 : 0)
// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
func (t *Tensor) ReLUGrad(gradOutput *Tensor, dst *Tensor) *Tensor {
	if t == nil || gradOutput == nil {
		return nil
	}

	if !t.sameShape(gradOutput) {
		panic("tensor.ReLUGrad: shape mismatch")
	}

	if dst == nil {
		dst = t.Clone()
	} else if !dst.sameShape(t) {
		panic("tensor.ReLUGrad: destination shape mismatch")
	}

	size := t.Size()
	fp32.ReLUGrad(dst.data, gradOutput.data, t.data, size)
	return dst
}

// Sigmoid applies the sigmoid activation function in-place: t[i] = 1 / (1 + exp(-t[i]))
func (t *Tensor) Sigmoid() *Tensor {
	if t == nil {
		return nil
	}

	size := t.Size()
	fp32.Sigmoid(t.data, t.data, size)
	return t
}

// SigmoidGrad computes the sigmoid gradient: dst[i] = gradOutput[i] * output[i] * (1 - output[i])
// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
func (t *Tensor) SigmoidGrad(gradOutput *Tensor, dst *Tensor) *Tensor {
	if t == nil || gradOutput == nil {
		return nil
	}

	if !t.sameShape(gradOutput) {
		panic("tensor.SigmoidGrad: shape mismatch")
	}

	if dst == nil {
		dst = t.Clone()
	} else if !dst.sameShape(t) {
		panic("tensor.SigmoidGrad: destination shape mismatch")
	}

	size := t.Size()
	fp32.SigmoidGrad(dst.data, gradOutput.data, t.data, size)
	return dst
}

// Tanh applies the hyperbolic tangent activation function in-place: t[i] = tanh(t[i])
func (t *Tensor) Tanh() *Tensor {
	if t == nil {
		return nil
	}

	size := t.Size()
	fp32.Tanh(t.data, t.data, size)
	return t
}

// TanhGrad computes the tanh gradient: dst[i] = gradOutput[i] * (1 - output[i]^2)
// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
func (t *Tensor) TanhGrad(gradOutput *Tensor, dst *Tensor) *Tensor {
	if t == nil || gradOutput == nil {
		return nil
	}

	if !t.sameShape(gradOutput) {
		panic("tensor.TanhGrad: shape mismatch")
	}

	if dst == nil {
		dst = t.Clone()
	} else if !dst.sameShape(t) {
		panic("tensor.TanhGrad: destination shape mismatch")
	}

	size := t.Size()
	fp32.TanhGrad(dst.data, gradOutput.data, t.data, size)
	return dst
}

// Softmax applies softmax along the specified dimension.
// Currently supports 1D tensors and 2D tensors with dim=0 (rows) or dim=1 (columns).
func (t *Tensor) Softmax(dim int) *Tensor {
	if t == nil {
		return nil
	}

	if dim < 0 || dim >= t.shape.Rank() {
		panic("tensor.Softmax: dimension out of range")
	}

	if t.shape.Rank() == 1 {
		fp32.Softmax1D(t.data, t.Size())
	} else if t.shape.Rank() == 2 {
		if dim == 0 {
			fp32.Softmax2DRows(t.data, t.shape[0], t.shape[1])
		} else if dim == 1 {
			fp32.Softmax2DCols(t.data, t.shape[0], t.shape[1])
		} else {
			panic("tensor.Softmax: invalid dimension for 2D tensor")
		}
	} else {
		panic("tensor.Softmax: only 1D and 2D tensors supported")
	}

	return t
}

// SoftmaxGrad computes softmax gradient along the specified dimension.
// dst[i] = output[i] * (gradOutput[i] - sum(gradOutput[j] * output[j]))
// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
func (t *Tensor) SoftmaxGrad(gradOutput *Tensor, dim int, dst *Tensor) *Tensor {
	if t == nil || gradOutput == nil {
		return nil
	}

	if !t.sameShape(gradOutput) {
		panic("tensor.SoftmaxGrad: shape mismatch")
	}

	if dim < 0 || dim >= t.shape.Rank() {
		panic("tensor.SoftmaxGrad: dimension out of range")
	}

	if dst == nil {
		dst = t.Clone()
	} else if !dst.sameShape(t) {
		panic("tensor.SoftmaxGrad: destination shape mismatch")
	}

	if t.shape.Rank() == 1 {
		fp32.Softmax1DGrad(dst.data, gradOutput.data, t.data, t.Size())
	} else if t.shape.Rank() == 2 {
		if dim == 0 {
			fp32.Softmax2DRowsGrad(dst.data, gradOutput.data, t.data, t.shape[0], t.shape[1])
		} else if dim == 1 {
			fp32.Softmax2DColsGrad(dst.data, gradOutput.data, t.data, t.shape[0], t.shape[1])
		} else {
			panic("tensor.SoftmaxGrad: invalid dimension for 2D tensor")
		}
	} else {
		panic("tensor.SoftmaxGrad: only 1D and 2D tensors supported")
	}

	return dst
}

// DropoutForward applies dropout during training.
// mask[i] should contain 0.0 for dropped elements, scale (1/(1-p)) for kept elements.
// Modifies the tensor in-place: t[i] *= mask[i]
func (t *Tensor) DropoutForward(mask *Tensor) *Tensor {
	if t == nil || mask == nil {
		return t
	}

	if !t.sameShape(mask) {
		panic("tensor.DropoutForward: mask shape mismatch")
	}

	size := t.Size()
	fp32.ElemMask(t.data, t.data, mask.data, size)
	return t
}

// DropoutBackward computes dropout gradient.
// Modifies dst in-place: dst[i] = gradOutput[i] * mask[i]
// If dst is nil, creates a new tensor. If dst is provided, uses it (must match shape).
func (t *Tensor) DropoutBackward(gradOutput, mask *Tensor, dst *Tensor) *Tensor {
	if gradOutput == nil || mask == nil {
		return nil
	}

	if !gradOutput.sameShape(mask) {
		panic("tensor.DropoutBackward: gradOutput and mask shape mismatch")
	}

	if dst == nil {
		dst = gradOutput.Clone()
	} else if !dst.sameShape(gradOutput) {
		panic("tensor.DropoutBackward: destination shape mismatch")
	}

	size := gradOutput.Size()
	fp32.ElemMask(dst.data, gradOutput.data, mask.data, size)
	return dst
}

// DropoutMask generates a dropout mask tensor with the given dropout rate.
// mask[i] = 0.0 if dropped (with probability p), scale otherwise.
// scale = 1.0 / (1.0 - p) for inverted dropout.
func (t *Tensor) DropoutMask(p float32, scale float32, rng *rand.Rand) *Tensor {
	if t == nil {
		return nil
	}

	size := t.Size()
	if size == 0 {
		return t
	}

	// For now, use direct data access since random generation is layer-specific
	// In the future, this could be moved to fp32 package with a random interface
	data := t.Data()
	for i := range data {
		if rng.Float32() < p {
			data[i] = 0.0
		} else {
			data[i] = scale
		}
	}

	return t
}

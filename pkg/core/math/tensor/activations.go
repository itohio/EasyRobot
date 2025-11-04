package tensor

import (
	"math/rand"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
)

// ReLU applies the Rectified Linear Unit activation function: dst[i] = max(0, t[i])
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t *Tensor) ReLU(dst *Tensor) *Tensor {
	if t.shape == nil {
		return nil
	}

	size := t.Size()
	if dst == nil {
		// Apply in-place
		fp32.ReLU(t.data, t.data, size)
		return t
	}

	if !dst.sameShape(*t) {
		panic("tensor.ReLU: destination shape mismatch")
	}

	fp32.ReLU(dst.data, t.data, size)
	return dst
}

// Sigmoid applies the sigmoid activation function: dst[i] = 1 / (1 + exp(-t[i]))
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t *Tensor) Sigmoid(dst *Tensor) *Tensor {
	if t.shape == nil {
		return nil
	}

	size := t.Size()
	if dst == nil {
		// Apply in-place
		fp32.Sigmoid(t.data, t.data, size)
		return t
	}

	if !dst.sameShape(*t) {
		panic("tensor.Sigmoid: destination shape mismatch")
	}

	fp32.Sigmoid(dst.data, t.data, size)
	return dst
}

// Tanh applies the hyperbolic tangent activation function: dst[i] = tanh(t[i])
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t *Tensor) Tanh(dst *Tensor) *Tensor {
	if t.shape == nil {
		return nil
	}

	size := t.Size()
	if dst == nil {
		// Apply in-place
		fp32.Tanh(t.data, t.data, size)
		return t
	}

	if !dst.sameShape(*t) {
		panic("tensor.Tanh: destination shape mismatch")
	}

	fp32.Tanh(dst.data, t.data, size)
	return dst
}

// Softmax applies softmax along the specified dimension.
// Currently supports 1D tensors and 2D tensors with dim=0 (rows) or dim=1 (columns).
// If dst is nil, applies in-place on t. Otherwise writes to dst.
func (t *Tensor) Softmax(dim int, dst *Tensor) *Tensor {
	if t.shape == nil {
		return nil
	}

	if dim < 0 || dim >= t.shape.Rank() {
		panic("tensor.Softmax: dimension out of range")
	}

	if dst == nil {
		// Apply in-place
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

	if !dst.sameShape(*t) {
		panic("tensor.Softmax: destination shape mismatch")
	}

	// Copy source to destination if different
	if dst != t {
		t.copyTo(dst)
	}

	if t.shape.Rank() == 1 {
		fp32.Softmax1D(dst.data, dst.Size())
	} else if t.shape.Rank() == 2 {
		if dim == 0 {
			fp32.Softmax2DRows(dst.data, dst.shape[0], dst.shape[1])
		} else if dim == 1 {
			fp32.Softmax2DCols(dst.data, dst.shape[0], dst.shape[1])
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
func (t *Tensor) DropoutForward(mask Tensor) *Tensor {
	if t.shape == nil || mask.shape == nil {
		return t
	}

	tVal := *t
	if !tVal.sameShape(mask) {
		panic("tensor.DropoutForward: mask shape mismatch")
	}

	size := t.Size()
	fp32.ElemMul(t.data, t.data, mask.data, []int{size}, []int{1}, []int{1}, []int{1})
	return t
}

// DropoutMask generates a dropout mask tensor with the given dropout rate.
// mask[i] = 0.0 if dropped (with probability p), scale otherwise.
// scale = 1.0 / (1.0 - p) for inverted dropout.
func (t *Tensor) DropoutMask(p float32, scale float32, rng *rand.Rand) *Tensor {
	if t.shape == nil {
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

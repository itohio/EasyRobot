package eager_tensor

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/generics"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
)

// ReLU applies the Rectified Linear Unit activation function: dst[i] = max(0, t[i])
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) ReLU(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	switch t.Data().(type) {
	case []float32:
		var tData []float32
		var dstData []float32
		var result types.Tensor
		if dst == nil || dst.Empty() {
			tData = types.GetTensorData[[]float32](t)
			dstData = types.GetTensorData[[]float32](t)
			result = t
		} else {
			if !t.Shape().Equal(dst.Shape()) {
				panic("tensor.ReLU: destination shape mismatch")
			}

			tData = types.GetTensorData[[]float32](t)
			dstData = types.GetTensorData[[]float32](dst)
			result = dst
		}

		size := t.Size()
		fp32.ReLU(dstData, tData, size)
		return result
	default:
		panic(fmt.Sprintf("tensor.ReLU: unsupported data type: %T", t.Data()))
	}
}

// Sigmoid applies the sigmoid activation function: dst[i] = 1 / (1 + exp(-t[i]))
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) Sigmoid(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	switch t.Data().(type) {
	case []float32:
		var tData []float32
		var dstData []float32
		var result types.Tensor
		if dst == nil || dst.Empty() {
			tData = types.GetTensorData[[]float32](t)
			dstData = types.GetTensorData[[]float32](t)
			result = t
		} else {
			if !t.Shape().Equal(dst.Shape()) {
				panic("tensor.Sigmoid: destination shape mismatch")
			}

			tData = types.GetTensorData[[]float32](t)
			dstData = types.GetTensorData[[]float32](dst)
			result = dst
		}

		size := t.Size()
		fp32.Sigmoid(dstData, tData, size)
		return result
	default:
		panic(fmt.Sprintf("tensor.Sigmoid: unsupported data type: %T", t.Data()))
	}
}

// Tanh applies the hyperbolic tangent activation function: dst[i] = tanh(t[i])
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) Tanh(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	switch t.Data().(type) {
	case []float32:
		var tData []float32
		var dstData []float32
		var result types.Tensor
		if dst == nil || dst.Empty() {
			tData = types.GetTensorData[[]float32](t)
			dstData = types.GetTensorData[[]float32](t)
			result = t
		} else {
			if !t.Shape().Equal(dst.Shape()) {
				panic("tensor.Tanh: destination shape mismatch")
			}

			tData = types.GetTensorData[[]float32](t)
			dstData = types.GetTensorData[[]float32](dst)
			result = dst
		}

		size := t.Size()
		fp32.Tanh(dstData, tData, size)
		return result
	default:
		panic(fmt.Sprintf("tensor.Tanh: unsupported data type: %T", t.Data()))
	}
}

// ReLUGrad computes the ReLU gradient: dst[i] = gradOutput[i] * (input[i] > 0 ? 1 : 0)
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (input Tensor) ReLUGrad(dst types.Tensor, gradOutput types.Tensor) types.Tensor {
	if input.shape == nil {
		return nil
	}
	if IsNil(gradOutput) {
		return nil
	}
	if !input.Shape().Equal(gradOutput.Shape()) {
		panic("tensor.ReLUGrad: input and gradOutput shape mismatch")
	}

	switch input.Data().(type) {
	case []float32:
		var dstData []float32
		var result types.Tensor
		if IsNil(dst) {
			result = New(gradOutput.DataType(), gradOutput.Shape())
			dstData = types.GetTensorData[[]float32](result)
		} else {
			if !gradOutput.Shape().Equal(dst.Shape()) {
				panic("tensor.ReLUGrad: destination shape mismatch")
			}
			result = dst
			dstData = types.GetTensorData[[]float32](dst)
		}

		inputData := types.GetTensorData[[]float32](input)
		gradOutputData := types.GetTensorData[[]float32](gradOutput)

		// Check if all tensors are contiguous for fast path
		shapeSlice := input.Shape().ToSlice()
		inputStrides := input.Strides(nil)
		dstStrides := result.Strides(nil)
		gradStrides := gradOutput.Strides(nil)
		if input.IsContiguous() && result.IsContiguous() && gradOutput.IsContiguous() {
			// Fast path for contiguous tensors
			size := input.Size()
			fp32.ReLUGrad(dstData, gradOutputData, inputData, size)
		} else {
			// General path with stride support
			fp32.ReLUGradStride(dstData, gradOutputData, inputData, shapeSlice, dstStrides, gradStrides, inputStrides)
		}

		return result
	default:
		panic(fmt.Sprintf("tensor.ReLUGrad: unsupported data type: %T", input.Data()))
	}
}

// SigmoidGrad computes the sigmoid gradient: dst[i] = gradOutput[i] * output[i] * (1 - output[i])
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (output Tensor) SigmoidGrad(dst types.Tensor, gradOutput types.Tensor) types.Tensor {
	if output.shape == nil {
		return nil
	}
	if IsNil(gradOutput) {
		return nil
	}
	if !output.Shape().Equal(gradOutput.Shape()) {
		panic("tensor.SigmoidGrad: output and gradOutput shape mismatch")
	}

	switch output.Data().(type) {
	case []float32:
		var dstData []float32
		var result types.Tensor
		if IsNil(dst) {
			result = New(gradOutput.DataType(), gradOutput.Shape())
			dstData = types.GetTensorData[[]float32](result)
		} else {
			if !gradOutput.Shape().Equal(dst.Shape()) {
				panic("tensor.SigmoidGrad: destination shape mismatch")
			}
			result = dst
			dstData = types.GetTensorData[[]float32](dst)
		}

		outputData := types.GetTensorData[[]float32](output)
		gradOutputData := types.GetTensorData[[]float32](gradOutput)

		// Check if all tensors are contiguous for fast path
		shapeSlice := output.Shape().ToSlice()
		outputStrides := output.Strides(nil)
		dstStrides := result.Strides(nil)
		gradStrides := gradOutput.Strides(nil)
		if output.IsContiguous() && result.IsContiguous() && gradOutput.IsContiguous() {
			// Fast path for contiguous tensors
			size := output.Size()
			fp32.SigmoidGrad(dstData, gradOutputData, outputData, size)
		} else {
			// General path with stride support
			fp32.SigmoidGradStride(dstData, gradOutputData, outputData, shapeSlice, dstStrides, gradStrides, outputStrides)
		}

		return result
	default:
		panic(fmt.Sprintf("tensor.SigmoidGrad: unsupported data type: %T", output.Data()))
	}
}

// TanhGrad computes the tanh gradient: dst[i] = gradOutput[i] * (1 - output[i]^2)
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (output Tensor) TanhGrad(dst types.Tensor, gradOutput types.Tensor) types.Tensor {
	if output.shape == nil {
		return nil
	}
	if IsNil(gradOutput) {
		return nil
	}
	if !output.Shape().Equal(gradOutput.Shape()) {
		panic("tensor.TanhGrad: output and gradOutput shape mismatch")
	}

	switch output.Data().(type) {
	case []float32:
		var dstData []float32
		var result types.Tensor
		if IsNil(dst) {
			result = New(gradOutput.DataType(), gradOutput.Shape())
			dstData = types.GetTensorData[[]float32](result)
		} else {
			if !gradOutput.Shape().Equal(dst.Shape()) {
				panic("tensor.TanhGrad: destination shape mismatch")
			}
			result = dst
			dstData = types.GetTensorData[[]float32](dst)
		}

		outputData := types.GetTensorData[[]float32](output)
		gradOutputData := types.GetTensorData[[]float32](gradOutput)

		// Check if all tensors are contiguous for fast path
		shapeSlice := output.Shape().ToSlice()
		outputStrides := output.Strides(nil)
		dstStrides := result.Strides(nil)
		gradStrides := gradOutput.Strides(nil)
		if output.IsContiguous() && result.IsContiguous() && gradOutput.IsContiguous() {
			// Fast path for contiguous tensors
			size := output.Size()
			fp32.TanhGrad(dstData, gradOutputData, outputData, size)
		} else {
			// General path with stride support
			fp32.TanhGradStride(dstData, gradOutputData, outputData, shapeSlice, dstStrides, gradStrides, outputStrides)
		}

		return result
	default:
		panic(fmt.Sprintf("tensor.TanhGrad: unsupported data type: %T", output.Data()))
	}
}

// SoftmaxGrad computes the softmax gradient along the specified dimension.
// Currently supports 1D tensors and 2D tensors with dim=0 (rows) or dim=1 (columns).
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (output Tensor) SoftmaxGrad(dst types.Tensor, gradOutput types.Tensor, dim int) types.Tensor {
	if output.shape == nil {
		return nil
	}
	if IsNil(gradOutput) {
		return nil
	}
	if !output.Shape().Equal(gradOutput.Shape()) {
		panic("tensor.SoftmaxGrad: output and gradOutput shape mismatch")
	}
	if dim < 0 || dim >= output.shape.Rank() {
		panic("tensor.SoftmaxGrad: dimension out of range")
	}

	switch output.Data().(type) {
	case []float32:
		var dstData []float32
		var result types.Tensor
		if IsNil(dst) {
			result = New(gradOutput.DataType(), gradOutput.Shape())
			dstData = types.GetTensorData[[]float32](result)
		} else {
			if !gradOutput.Shape().Equal(dst.Shape()) {
				panic("tensor.SoftmaxGrad: destination shape mismatch")
			}
			result = dst
			dstData = types.GetTensorData[[]float32](dst)
		}

		outputData := types.GetTensorData[[]float32](output)
		gradOutputData := types.GetTensorData[[]float32](gradOutput)

		if output.shape.Rank() == 1 {
			// Softmax1DGrad
			size := output.Size()
			fp32.Softmax1DGrad(dstData, gradOutputData, outputData, size)
		} else if output.shape.Rank() == 2 {
			rows := output.shape[0]
			cols := output.shape[1]
			if dim == 0 {
				// Softmax2DRowsGrad
				fp32.Softmax2DRowsGrad(dstData, gradOutputData, outputData, rows, cols)
			} else if dim == 1 {
				// Softmax2DColsGrad
				fp32.Softmax2DColsGrad(dstData, gradOutputData, outputData, rows, cols)
			} else {
				panic("tensor.SoftmaxGrad: invalid dimension for 2D tensor")
			}
		} else {
			panic("tensor.SoftmaxGrad: only 1D and 2D tensors supported")
		}

		return result
	default:
		panic(fmt.Sprintf("tensor.SoftmaxGrad: unsupported data type: %T", output.Data()))
	}
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

	switch t.Data().(type) {
	case []float32:
		var tData []float32
		var dstData []float32
		var result types.Tensor
		if dst == nil || dst.Empty() {
			tData = types.GetTensorData[[]float32](t)
			dstData = types.GetTensorData[[]float32](t)
			result = t // Return input tensor when dst is nil
		} else {
			if !t.Shape().Equal(dst.Shape()) {
				panic("tensor.Softmax: destination shape mismatch")
			}

			tData = types.GetTensorData[[]float32](t)
			dstData = types.GetTensorData[[]float32](dst)
			result = dst
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
				tData := types.GetTensorData[[]float32](t)
				dstData := types.GetTensorData[[]float32](dst)
				shapeSlice := t.Shape().ToSlice()
				dstStrides := dst.Strides(nil)
				tStrides := t.Strides(nil)
				generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
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

		return result
	default:
		panic(fmt.Sprintf("tensor.Softmax: unsupported data type: %T", t.Data()))
	}
}

// DropoutForward applies dropout mask during forward pass: dst = t * mask.
// If dst is nil, operation is in-place (modifies t) and returns t.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) DropoutForward(dst types.Tensor, mask types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}
	if IsNil(mask) {
		return nil
	}

	if !t.Shape().Equal(mask.Shape()) {
		panic("tensor.DropoutForward: mask shape mismatch")
	}

	switch t.Data().(type) {
	case []float32:
		var tData []float32
		var dstData []float32
		var result types.Tensor
		if IsNil(dst) {
			tData = types.GetTensorData[[]float32](t)
			dstData = types.GetTensorData[[]float32](t)
			result = t
		} else {
			if !t.Shape().Equal(dst.Shape()) {
				panic("tensor.DropoutForward: destination shape mismatch")
			}
			// Copy t to dst first
			tData = types.GetTensorData[[]float32](t)
			dstData = types.GetTensorData[[]float32](dst)
			shapeSlice := t.Shape().ToSlice()
			dstStrides := dst.Strides(nil)
			tStrides := t.Strides(nil)
			generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
			result = dst
		}

		maskData := types.GetTensorData[[]float32](mask)
		tStrides := t.Strides(nil)
		dstStrides := result.Strides(nil)
		maskStrides := mask.Strides(nil)
		fp32.ElemMul(dstData, tData, maskData, []int(t.shape), dstStrides, tStrides, maskStrides)
		return result
	default:
		panic(fmt.Sprintf("tensor.DropoutForward: unsupported data type: %T", t.Data()))
	}
}

// DropoutMask generates a dropout mask tensor with the given dropout rate.
// mask[i] = 0.0 if dropped (with probability p), scale otherwise.
// scale = 1.0 / (1.0 - p) for inverted dropout.
// rng: random number generator implementing types.RNG interface
func (t Tensor) DropoutMask(p float64, scale float64, rng types.RNG) types.Tensor {
	if t.shape == nil {
		return nil
	}

	size := t.Size()
	if size == 0 {
		return t
	}

	if rng == nil {
		panic("tensor.DropoutMask: rng must not be nil")
	}

	t.FillFunc(t, func() float64 {
		if rng.Float64() < p {
			return 0.0
		} else {
			return scale
		}
	})

	return t
}

// DropoutBackward computes dropout backward pass: dst = gradOutput * mask.
// gradOutput: gradient from next layer
// mask: dropout mask used in forward pass
// If dst is nil, creates a new tensor.
// If dst is provided, writes result to dst and returns dst.
func (t Tensor) DropoutBackward(dst types.Tensor, gradOutput, mask types.Tensor) types.Tensor {
	if IsNil(gradOutput) || IsNil(mask) {
		return nil
	}

	if !gradOutput.Shape().Equal(mask.Shape()) {
		panic("tensor.DropoutBackward: gradOutput and mask shape mismatch")
	}

	var result types.Tensor
	if IsNil(dst) {
		result = gradOutput.Clone()
	} else {
		if !gradOutput.Shape().Equal(dst.Shape()) {
			panic("tensor.DropoutBackward: destination shape mismatch")
		}
		gradData := types.GetTensorData[[]float32](gradOutput)
		dstData := types.GetTensorData[[]float32](dst)
		shapeSlice := gradOutput.Shape().ToSlice()
		// Use Strides(nil) for read-only operations - returns stored strides directly without copy
		dstStrides := dst.Strides(nil)
		gradStrides := gradOutput.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, gradData, shapeSlice, dstStrides, gradStrides)
		result = dst
	}

	// Dropout backward: result = gradOutput * mask
	result.Multiply(nil, mask)

	return result
}

// ReLU6 applies ReLU6 activation: result[i] = min(max(t[i], 0), 6) (matches tf.nn.relu6).
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) ReLU6(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	var result types.Tensor
	if dst == nil || dst.Empty() {
		result = t.Clone()
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.ReLU6: destination shape mismatch")
		}
		tData := types.GetTensorData[[]float32](t)
		dstData := types.GetTensorData[[]float32](dst)
		shapeSlice := t.Shape().ToSlice()
		// Use Strides(nil) for read-only operations - returns stored strides directly without copy
		dstStrides := dst.Strides(nil)
		tStrides := t.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
		result = dst
	}

	// ReLU6: min(max(x, 0), 6)
	// First apply ReLU (max(x, 0))
	result.ReLU(result)

	// Then apply min(_, 6) by using Where with condition
	// Create a tensor filled with 6.0 for comparison
	sixTensor := result.Clone().Fill(nil, 6.0)
	// Use Where: result[i] = result[i] < 6 ? result[i] : 6
	condition := result.Less(nil, sixTensor)
	result.Where(result, condition, result, sixTensor)

	if dst == nil || dst.Empty() {
		return result
	}
	return dst
}

// LeakyReLU applies Leaky ReLU activation: result[i] = max(t[i], alpha * t[i]) (matches tf.nn.leaky_relu).
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) LeakyReLU(dst types.Tensor, alpha float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	var result types.Tensor
	if dst == nil || dst.Empty() {
		result = t.Clone()
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.LeakyReLU: destination shape mismatch")
		}
		tData := types.GetTensorData[[]float32](t)
		dstData := types.GetTensorData[[]float32](dst)
		shapeSlice := t.Shape().ToSlice()
		// Use Strides(nil) for read-only operations - returns stored strides directly without copy
		dstStrides := dst.Strides(nil)
		tStrides := t.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
		result = dst
	}

	// LeakyReLU: max(x, alpha * x) = x > 0 ? x : alpha * x
	alphaTensor := result.Clone().ScalarMul(nil, alpha)
	zeroTensor := result.Clone().Fill(nil, 0.0)
	condition := result.Greater(nil, zeroTensor)
	result.Where(result, condition, result, alphaTensor)

	if dst == nil || dst.Empty() {
		return result
	}
	return dst
}

// ELU applies ELU activation: result[i] = t[i] > 0 ? t[i] : alpha * (exp(t[i]) - 1) (matches tf.nn.elu).
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) ELU(dst types.Tensor, alpha float64) types.Tensor {
	if t.shape == nil {
		return nil
	}

	var result types.Tensor
	if dst == nil || dst.Empty() {
		result = t.Clone()
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.ELU: destination shape mismatch")
		}
		tData := types.GetTensorData[[]float32](t)
		dstData := types.GetTensorData[[]float32](dst)
		shapeSlice := t.Shape().ToSlice()
		// Use Strides(nil) for read-only operations - returns stored strides directly without copy
		dstStrides := dst.Strides(nil)
		tStrides := t.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
		result = dst
	}

	// ELU: x > 0 ? x : alpha * (exp(x) - 1)
	zeroTensor := result.Clone().Fill(nil, 0.0)
	condition := result.Greater(nil, zeroTensor)

	// Compute exp(x) - 1 for negative values
	expResult := result.Clone()
	expResult.Exp(expResult)
	expResult.SubScalar(expResult, 1.0)
	expResult.ScalarMul(expResult, alpha)

	result.Where(result, condition, result, expResult)

	if dst == nil || dst.Empty() {
		return result
	}
	return dst
}

// Softplus applies softplus activation: result[i] = log(1 + exp(t[i])) (matches tf.nn.softplus).
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) Softplus(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	var result types.Tensor
	if dst == nil || dst.Empty() {
		result = t.Clone()
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Softplus: destination shape mismatch")
		}
		tData := types.GetTensorData[[]float32](t)
		dstData := types.GetTensorData[[]float32](dst)
		shapeSlice := t.Shape().ToSlice()
		// Use Strides(nil) for read-only operations - returns stored strides directly without copy
		dstStrides := dst.Strides(nil)
		tStrides := t.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
		result = dst
	}

	// Softplus: log(1 + exp(x))
	result.Exp(result)
	result.AddScalar(result, 1.0)
	result.Log(result)

	if dst == nil || dst.Empty() {
		return result
	}
	return dst
}

// Swish applies Swish activation: result[i] = t[i] * sigmoid(t[i]) (matches tf.nn.swish).
// If dst is nil, creates a new tensor. Otherwise writes to dst.
func (t Tensor) Swish(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	var result types.Tensor
	if dst == nil || dst.Empty() {
		result = t.Clone()
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.Swish: destination shape mismatch")
		}
		tData := types.GetTensorData[[]float32](t)
		dstData := types.GetTensorData[[]float32](dst)
		shapeSlice := t.Shape().ToSlice()
		// Use Strides(nil) for read-only operations - returns stored strides directly without copy
		dstStrides := dst.Strides(nil)
		tStrides := t.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
		result = dst
	}

	// Swish: x * sigmoid(x)
	sigmoidResult := result.Clone()
	sigmoidResult.Sigmoid(sigmoidResult)
	result.Multiply(result, sigmoidResult)

	if dst == nil || dst.Empty() {
		return result
	}
	return dst
}

// GELU applies GELU activation: result[i] = t[i] * 0.5 * (1 + erf(t[i]/sqrt(2))) (matches tf.nn.gelu).
// If dst is nil, creates a new tensor. Otherwise writes to dst.
// Uses approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
func (t Tensor) GELU(dst types.Tensor) types.Tensor {
	if t.shape == nil {
		return nil
	}

	var result types.Tensor
	if dst == nil || dst.Empty() {
		result = t.Clone()
	} else {
		if !t.Shape().Equal(dst.Shape()) {
			panic("tensor.GELU: destination shape mismatch")
		}
		tData := types.GetTensorData[[]float32](t)
		dstData := types.GetTensorData[[]float32](dst)
		shapeSlice := t.Shape().ToSlice()
		// Use Strides(nil) for read-only operations - returns stored strides directly without copy
		dstStrides := dst.Strides(nil)
		tStrides := t.Strides(nil)
		generics.ElemCopyStrided[float32](dstData, tData, shapeSlice, dstStrides, tStrides)
		result = dst
	}

	// GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
	// sqrt(2/π) ≈ 0.7978845608
	// 0.044715 * x^3
	x3 := result.Clone()
	x3.Pow(x3, 3.0)
	x3.ScalarMul(x3, 0.044715)

	// x + 0.044715 * x^3
	inner := result.Clone()
	inner.Add(inner, x3)

	// sqrt(2/π) * (x + 0.044715 * x^3)
	inner.ScalarMul(inner, 0.7978845608)

	// tanh(...)
	inner.Tanh(inner)

	// 1 + tanh(...)
	inner.AddScalar(inner, 1.0)

	// 0.5 * x * (1 + tanh(...))
	result.Multiply(result, inner)
	result.ScalarMul(result, 0.5)

	if dst == nil || dst.Empty() {
		return result
	}
	return dst
}

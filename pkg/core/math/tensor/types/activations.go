package types

// Activations defines activation functions for neural networks.
// This interface contains non-linear transformations applied element-wise.
type Activations interface {
	// ReLU applies Rectified Linear Unit activation: result[i] = max(0, t[i]).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	ReLU(dst Tensor) Tensor

	// Sigmoid applies sigmoid activation: result[i] = 1 / (1 + exp(-t[i])).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	Sigmoid(dst Tensor) Tensor

	// Tanh applies hyperbolic tangent activation: result[i] = tanh(t[i]).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	Tanh(dst Tensor) Tensor

	// Softmax applies softmax activation along the specified dimension.
	// result[i] = exp(t[i]) / sum(exp(t[j])) for all j along dimension dim.
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	// Panics if dimension is out of range.
	Softmax(dim int, dst Tensor) Tensor

	// ReLU6 applies ReLU6 activation: result[i] = min(max(t[i], 0), 6) (matches tf.nn.relu6).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	ReLU6(dst Tensor) Tensor

	// LeakyReLU applies Leaky ReLU activation: result[i] = max(t[i], alpha * t[i]) (matches tf.nn.leaky_relu).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	LeakyReLU(dst Tensor, alpha float64) Tensor

	// ELU applies ELU activation: result[i] = t[i] > 0 ? t[i] : alpha * (exp(t[i]) - 1) (matches tf.nn.elu).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	ELU(dst Tensor, alpha float64) Tensor

	// Softplus applies softplus activation: result[i] = log(1 + exp(t[i])) (matches tf.nn.softplus).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	Softplus(dst Tensor) Tensor

	// Swish applies Swish activation: result[i] = t[i] * sigmoid(t[i]) (matches tf.nn.swish).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	Swish(dst Tensor) Tensor

	// GELU applies GELU activation: result[i] = t[i] * 0.5 * (1 + erf(t[i]/sqrt(2))) (matches tf.nn.gelu).
	// If dst is nil, creates a new tensor. Returns the destination tensor.
	GELU(dst Tensor) Tensor

	// ReLUGrad computes the ReLU gradient: dst[i] = gradOutput[i] * (input[i] > 0 ? 1 : 0)
	// If dst is nil, creates a new tensor. Otherwise writes to dst.
	// This is an optimized gradient computation that directly uses fp32.ReLUGrad primitive.
	ReLUGrad(dst Tensor, gradOutput Tensor) Tensor

	// SigmoidGrad computes the sigmoid gradient: dst[i] = gradOutput[i] * output[i] * (1 - output[i])
	// If dst is nil, creates a new tensor. Otherwise writes to dst.
	// This is an optimized gradient computation that directly uses fp32.SigmoidGrad primitive.
	SigmoidGrad(dst Tensor, gradOutput Tensor) Tensor

	// TanhGrad computes the tanh gradient: dst[i] = gradOutput[i] * (1 - output[i]^2)
	// If dst is nil, creates a new tensor. Otherwise writes to dst.
	// This is an optimized gradient computation that directly uses fp32.TanhGrad primitive.
	TanhGrad(dst Tensor, gradOutput Tensor) Tensor

	// SoftmaxGrad computes the softmax gradient along the specified dimension.
	// Currently supports 1D tensors and 2D tensors with dim=0 (rows) or dim=1 (columns).
	// If dst is nil, creates a new tensor. Otherwise writes to dst.
	// This is an optimized gradient computation that directly uses fp32.Softmax*Grad primitives.
	SoftmaxGrad(dst Tensor, gradOutput Tensor, dim int) Tensor
}

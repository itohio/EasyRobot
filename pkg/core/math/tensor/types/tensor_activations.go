package types

// TensorActivations defines activation functions for neural networks.
// This interface contains non-linear transformations applied element-wise.
type TensorActivations interface {
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
}


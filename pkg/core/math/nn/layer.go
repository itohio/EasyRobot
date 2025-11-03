package nn

import (
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Layer represents a neural network layer with forward and backward operations.
type Layer interface {
	// Name returns the name of this layer.
	Name() string

	// Init initializes the layer, creating internal computation tensors.
	// Should be called after the layer is added to a model and input shape is known.
	Init(inputShape []int) error

	// Forward computes the forward pass: output = layer(input)
	// Validates input dimensions and computes directly into pre-allocated output tensor.
	// Input tensor is stored internally for backward pass.
	// Returns the output tensor (which should be pre-allocated and reused).
	Forward(input tensor.Tensor) (tensor.Tensor, error)

	// Backward computes gradients: gradInput = backward(gradOutput)
	// Returns gradient w.r.t. input and updates internal gradients w.r.t. weights.
	// Uses stored input/output from Forward pass.
	// Only computes gradients if CanLearn() returns true.
	Backward(gradOutput tensor.Tensor) (tensor.Tensor, error)

	// OutputShape returns the output shape for given input shape.
	// Used for dimension validation before Forward.
	OutputShape(inputShape []int) ([]int, error)

	// CanLearn returns whether this layer computes gradients during backward pass.
	// If false, backward pass only computes gradient w.r.t. input (propagation).
	// If true, backward pass also computes and stores gradients w.r.t. parameters.
	CanLearn() bool

	// SetCanLearn sets whether this layer computes gradients.
	SetCanLearn(canLearn bool)

	// Input returns the input tensor from the last Forward pass.
	Input() tensor.Tensor

	// Output returns the output tensor from the last Forward pass.
	Output() tensor.Tensor
}

// ParameterLayer extends Layer with trainable parameters.
type ParameterLayer interface {
	Layer

	// Parameters returns all trainable parameters (weights, biases, etc.)
	Parameters() []*Parameter

	// ZeroGrad zeros out all parameter gradients
	ZeroGrad()
}

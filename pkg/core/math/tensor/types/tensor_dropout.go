package types

// TensorDropout defines dropout operations for neural networks.
// This interface contains operations for regularization during training.
type TensorDropout interface {
	// DropoutForward applies dropout mask during forward pass: dst = t * mask.
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	DropoutForward(dst Tensor, mask Tensor) Tensor

	// DropoutMask creates a dropout mask with given probability and scale.
	// p: probability of keeping an element (0.0 to 1.0)
	// scale: scaling factor (typically 1.0 / (1.0 - p))
	// rng: random number generator implementing RNG interface (e.g., *rand.Rand)
	// Returns a new tensor with the mask.
	DropoutMask(p, scale float64, rng RNG) Tensor

	// DropoutBackward computes dropout backward pass: dst = gradOutput * mask.
	// gradOutput: gradient from next layer
	// mask: dropout mask used in forward pass
	// If dst is nil, creates a new tensor.
	// If dst is provided, writes result to dst and returns dst.
	DropoutBackward(dst Tensor, gradOutput, mask Tensor) Tensor
}

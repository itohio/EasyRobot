package types

// TensorDropout defines dropout operations for neural networks.
// This interface contains operations for regularization during training.
type TensorDropout interface {
	// DropoutForward applies dropout mask during forward pass: result = t * mask.
	// Returns a new tensor with dropout applied.
	DropoutForward(mask Tensor) Tensor

	// DropoutMask creates a dropout mask with given probability and scale.
	// p: probability of keeping an element (0.0 to 1.0)
	// scale: scaling factor (typically 1.0 / (1.0 - p))
	// rng: random number generator implementing RNG interface (e.g., *rand.Rand)
	// Returns a new tensor with the mask.
	DropoutMask(p, scale float64, rng RNG) Tensor
}


package types

// Optimizer interface for updating parameters during training.
type Optimizer interface {
	// Update updates parameter using gradient.
	Update(param Parameter) error
}

// Model represents a trainable neural network model or layer.
// Model embeds Layer, so any Model can be used as a Layer.
// This allows both Sequential models and individual learnable layers to be trained.
type Model interface {
	Layer // Model is also a Layer

	// ZeroGrad zeros all parameter gradients.
	ZeroGrad()
}

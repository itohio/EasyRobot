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

	// LayerCount returns the number of layers in the model.
	// For Sequential models, returns the number of layers.
	// For individual layers, returns 1.
	LayerCount() int

	// GetLayer returns the layer at the specified index.
	// For Sequential models, returns the layer at that index.
	// For individual layers, returns the layer itself if index is 0.
	GetLayer(index int) Layer
}

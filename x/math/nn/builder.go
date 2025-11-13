package nn

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/nn/layers"
	"github.com/itohio/EasyRobot/x/math/nn/models"
	"github.com/itohio/EasyRobot/x/math/nn/types"
	tensorTypes "github.com/itohio/EasyRobot/x/math/tensor/types"
)

// SequentialModelBuilder helps construct models by adding layers sequentially.
type SequentialModelBuilder struct {
	layers     []types.Layer
	inputShape tensorTypes.Shape
}

// NewSequentialModelBuilder creates a new sequential model builder with the given input shape.
func NewSequentialModelBuilder(inputShape tensorTypes.Shape) *SequentialModelBuilder {
	if inputShape == nil {
		return nil
	}
	// Validate input shape
	for _, dim := range inputShape {
		if dim <= 0 {
			panic(fmt.Sprintf("SequentialModelBuilder: input shape dimensions must be positive, got %v", inputShape))
		}
	}
	return &SequentialModelBuilder{
		layers:     []types.Layer{},
		inputShape: inputShape,
	}
}

// AddLayer adds a layer to the model.
func (b *SequentialModelBuilder) AddLayer(layer types.Layer) *SequentialModelBuilder {
	if b == nil {
		return nil
	}
	if layer == nil {
		panic("SequentialModelBuilder.AddLayer: cannot add nil layer")
	}
	b.layers = append(b.layers, layer)
	return b
}

// WithLayers adds multiple layers to the model as varargs.
func (b *SequentialModelBuilder) WithLayers(layers ...types.Layer) *SequentialModelBuilder {
	if b == nil {
		return nil
	}
	for _, layer := range layers {
		if layer == nil {
			panic("SequentialModelBuilder.WithLayers: cannot add nil layer")
		}
		b.layers = append(b.layers, layer)
	}
	return b
}

// Build creates the Model from the builder.
// Validates that all layer shapes are compatible.
func (b *SequentialModelBuilder) Build() (*models.Sequential, error) {
	if b == nil {
		return nil, fmt.Errorf("SequentialModelBuilder.Build: nil builder")
	}

	if len(b.layers) == 0 {
		return nil, fmt.Errorf("SequentialModelBuilder.Build: no layers added")
	}

	// Validate layer shapes
	currentShape := b.inputShape
	for i, layer := range b.layers {
		outputShape, err := layer.OutputShape(currentShape)
		if err != nil {
			return nil, fmt.Errorf("SequentialModelBuilder.Build: layer %d shape validation failed: %w", i, err)
		}
		currentShape = outputShape
	}

	// Build layer name map
	layerNames := make(map[string]int)
	for i, layer := range b.layers {
		if layer != nil {
			name := layer.Name()
			if name != "" {
				// Check for duplicate names
				if _, exists := layerNames[name]; exists {
					return nil, fmt.Errorf("SequentialModelBuilder.Build: duplicate layer name: %s", name)
				}
				layerNames[name] = i
			}
		}
	}

	// Initialize Base for the model
	base := layers.NewBase("model")

	// Use NewSequential constructor
	return models.NewSequential(base, b.layers, layerNames, b.inputShape), nil
}

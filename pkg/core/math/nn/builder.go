package nn

import (
	"fmt"
)

// ModelBuilder helps construct models by adding layers sequentially.
type ModelBuilder struct {
	layers     []Layer
	inputShape []int
}

// NewModelBuilder creates a new model builder with the given input shape.
func NewModelBuilder(inputShape []int) *ModelBuilder {
	if inputShape == nil {
		return nil
	}
	// Validate input shape
	for _, dim := range inputShape {
		if dim <= 0 {
			panic(fmt.Sprintf("ModelBuilder: input shape dimensions must be positive, got %v", inputShape))
		}
	}
	return &ModelBuilder{
		layers:     []Layer{},
		inputShape: inputShape,
	}
}

// AddLayer adds a layer to the model.
func (b *ModelBuilder) AddLayer(layer Layer) *ModelBuilder {
	if b == nil {
		return nil
	}
	if layer == nil {
		panic("ModelBuilder.AddLayer: cannot add nil layer")
	}
	b.layers = append(b.layers, layer)
	return b
}

// Build creates the Model from the builder.
// Validates that all layer shapes are compatible.
func (b *ModelBuilder) Build() (*Model, error) {
	if b == nil {
		return nil, fmt.Errorf("ModelBuilder.Build: nil builder")
	}

	if len(b.layers) == 0 {
		return nil, fmt.Errorf("ModelBuilder.Build: no layers added")
	}

	// Validate layer shapes
	currentShape := b.inputShape
	for i, layer := range b.layers {
		outputShape, err := layer.OutputShape(currentShape)
		if err != nil {
			return nil, fmt.Errorf("ModelBuilder.Build: layer %d shape validation failed: %w", i, err)
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
					return nil, fmt.Errorf("ModelBuilder.Build: duplicate layer name: %s", name)
				}
				layerNames[name] = i
			}
		}
	}

	return &Model{
		layers:     b.layers,
		layerNames: layerNames,
		inputShape: b.inputShape,
	}, nil
}

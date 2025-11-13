package models

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/nn/layers"
	"github.com/itohio/EasyRobot/x/math/nn/types"
	"github.com/itohio/EasyRobot/x/math/tensor"
)

// Sequential represents a neural network model composed of layers.
// Sequential embeds layers.Base and implements the Layer interface, allowing models to be used as layers.
type Sequential struct {
	layers.Base
	layers     []types.Layer
	layerNames map[string]int // Map from layer name to index
	inputShape tensor.Shape
}

// NewSequential creates a new Sequential model with the given layers, layer names, and input shape.
// This is used by the builder to construct models.
func NewSequential(base layers.Base, layers []types.Layer, layerNames map[string]int, inputShape tensor.Shape) *Sequential {
	return &Sequential{
		Base:       base,
		layers:     layers,
		layerNames: layerNames,
		inputShape: inputShape,
	}
}

// GetLayer returns the layer at the given index.
func (m *Sequential) GetLayer(index int) types.Layer {
	if m == nil || index < 0 || index >= len(m.layers) {
		return nil
	}
	return m.layers[index]
}

// GetLayerByName returns the layer with the given name, or nil if not found.
func (m *Sequential) GetLayerByName(name string) types.Layer {
	if m == nil || m.layerNames == nil {
		return nil
	}
	if index, ok := m.layerNames[name]; ok && index >= 0 && index < len(m.layers) {
		return m.layers[index]
	}
	return nil
}

// LayerCount returns the number of layers in the model.
func (m *Sequential) LayerCount() int {
	if m == nil {
		return 0
	}
	return len(m.layers)
}

// Init initializes all layers in the model.
// This should be called after building the model and before first Forward pass.
// Init initializes the model with the given input shape.
// This implements the types.Layer interface.
func (m *Sequential) Init(inputShape tensor.Shape) error {
	if m == nil {
		return fmt.Errorf("model.Init: nil model")
	}

	if len(inputShape) == 0 {
		return fmt.Errorf("model.Init: input shape is empty")
	}

	// Store the input shape
	m.inputShape = inputShape

	// Initialize each layer with its input shape
	currentShape := inputShape
	for i, layer := range m.layers {
		if layer == nil {
			return fmt.Errorf("model.Init: nil layer at index %d", i)
		}

		// Call Init on the layer
		if err := layer.Init(currentShape); err != nil {
			return fmt.Errorf("model.Init: layer %d (%s) init failed: %w", i, layer.Name(), err)
		}

		// Get output shape for next layer
		outputShape, err := layer.OutputShape(currentShape)
		if err != nil {
			return fmt.Errorf("model.Init: layer %d (%s) output shape failed: %w", i, layer.Name(), err)
		}
		currentShape = outputShape
	}

	return nil
}

// Forward computes forward pass through all layers.
// Sets m.Input to input and m.Output to final output.
// Layers store their own inputs/outputs internally.
func (m *Sequential) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if m == nil {
		return nil, fmt.Errorf("model.Forward: nil model")
	}

	if input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("model.Forward: empty input")
	}

	// Validate input shape
	inputShape := input.Shape()
	if !tensor.Shape(inputShape).Equal(tensor.Shape(m.inputShape)) {
		return nil, fmt.Errorf("model.Forward: input shape %v does not match expected shape %v", inputShape, m.inputShape)
	}

	// Store input in Base
	m.Base.StoreInput(input)

	// Forward pass through all layers
	// Reuse output tensors: each layer computes into its output tensor,
	// and the next layer uses that same tensor as its input
	current := input
	for i, layer := range m.layers {
		if layer == nil {
			return nil, fmt.Errorf("model.Forward: nil layer at index %d", i)
		}

		// Forward computes into layer's pre-allocated output tensor
		output, err := layer.Forward(current)
		if err != nil {
			return nil, fmt.Errorf("model.Forward: layer %d (%s) failed: %w", i, layer.Name(), err)
		}

		// Next layer uses this layer's output tensor directly
		current = output
	}

	// Store output in Base
	m.Base.StoreOutput(current)

	return current, nil
}

// Backward computes backward pass through all layers in reverse order.
// Layers use their stored inputs/outputs from Forward pass.
func (m *Sequential) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
	if m == nil {
		return nil, fmt.Errorf("model.Backward: nil model")
	}

	if gradOutput.Shape().Rank() == 0 {
		return nil, fmt.Errorf("model.Backward: empty gradOutput")
	}

	output := m.Base.Output()
	if output == nil || output.Shape().Rank() == 0 {
		return nil, fmt.Errorf("model.Backward: model.Output is empty, must call Forward first")
	}

	// Backward pass through layers in reverse order
	currentGrad := gradOutput
	for i := len(m.layers) - 1; i >= 0; i-- {
		layer := m.layers[i]
		if layer == nil {
			return nil, fmt.Errorf("model.Backward: nil layer at index %d", i)
		}

		gradInput, err := layer.Backward(currentGrad)
		if err != nil {
			return nil, fmt.Errorf("model.Backward: layer %d failed: %w", i, err)
		}

		currentGrad = gradInput
	}

	// Store gradient in Base
	m.Base.StoreGrad(currentGrad)

	return currentGrad, nil
}

// Parameters returns all trainable parameters from all layers.
// For Sequential models with multiple layers, this aggregates parameters from all layers.
// Since different layers may have the same ParamIndex, we use a composite key approach:
// we collect parameters from all layers and return them with their ParamIndex keys.
// Note: If multiple layers have the same ParamIndex, the last layer's parameter will overwrite previous ones.
func (m *Sequential) Parameters() map[types.ParamIndex]types.Parameter {
	if m == nil {
		return nil
	}

	result := make(map[types.ParamIndex]types.Parameter)
	for _, layer := range m.layers {
		// Check if layer has Parameters() method
		// Layers embedding Base will have this method promoted
		type paramsGetter interface {
			Parameters() map[types.ParamIndex]types.Parameter
		}

		if pg, ok := layer.(paramsGetter); ok {
			layerParams := pg.Parameters()
			// Merge parameters from this layer into result
			// Note: If multiple layers have the same ParamIndex, later layers overwrite earlier ones
			for paramIdx, param := range layerParams {
				result[paramIdx] = param
			}
		}
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

// ZeroGrad zeros all parameter gradients.
func (m *Sequential) ZeroGrad() {
	if m == nil {
		return
	}

	for _, layer := range m.layers {
		// Check if layer has ZeroGrad() method
		// Layers embedding Base will have this method promoted
		type zeroGradder interface {
			ZeroGrad()
		}

		if zg, ok := layer.(zeroGradder); ok {
			zg.ZeroGrad()
		}
	}
}

// Update updates all parameters using optimizer.
// Use TrainStep from learn package for full training loop.
// Delegates to each layer's Update method.
func (m *Sequential) Update(optimizer types.Optimizer) error {
	if m == nil {
		return fmt.Errorf("model.Update: nil model")
	}

	if optimizer == nil {
		return fmt.Errorf("model.Update: nil optimizer")
	}

	// Delegate to each layer's Update method
	// Each layer (which embeds Base) will handle its own parameter updates
	for i, layer := range m.layers {
		type updater interface {
			Update(optimizer types.Optimizer) error
		}

		if u, ok := layer.(updater); ok {
			if err := u.Update(optimizer); err != nil {
				return fmt.Errorf("model.Update: layer %d failed: %w", i, err)
			}
		}
	}

	return nil
}

// Name returns the name of the model (from Base).
func (m *Sequential) Name() string {
	if m == nil {
		return ""
	}
	return m.Base.Name()
}

// CanLearn returns whether the model can learn (from Base).
func (m *Sequential) CanLearn() bool {
	if m == nil {
		return false
	}
	return m.Base.CanLearn()
}

// SetCanLearn sets whether the model can learn (from Base).
func (m *Sequential) SetCanLearn(canLearn bool) {
	if m == nil {
		return
	}
	m.Base.SetCanLearn(canLearn)
}

// Input returns the input tensor from the last Forward pass (from Base).
func (m *Sequential) Input() tensor.Tensor {
	if m == nil {
		return nil
	}
	return m.Base.Input()
}

// Output returns the output tensor from the last Forward pass (from Base).
func (m *Sequential) Output() tensor.Tensor {
	if m == nil {
		return nil
	}
	return m.Base.Output()
}

// OutputShape returns the output shape for given input shape.
// Computes the output shape by propagating through all layers.
func (m *Sequential) OutputShape(inputShape tensor.Shape) (tensor.Shape, error) {
	if m == nil {
		return nil, fmt.Errorf("model.OutputShape: nil model")
	}

	if len(inputShape) == 0 {
		return nil, fmt.Errorf("model.OutputShape: empty input shape")
	}

	// Validate input shape matches model's expected input shape
	if !inputShape.Equal(m.inputShape) {
		return nil, fmt.Errorf("model.OutputShape: input shape %v does not match model input shape %v", inputShape, m.inputShape)
	}

	// Propagate shape through all layers
	currentShape := inputShape
	for i, layer := range m.layers {
		if layer == nil {
			return nil, fmt.Errorf("model.OutputShape: nil layer at index %d", i)
		}

		outputShape, err := layer.OutputShape(currentShape)
		if err != nil {
			return nil, fmt.Errorf("model.OutputShape: layer %d (%s) output shape failed: %w", i, layer.Name(), err)
		}
		currentShape = outputShape
	}

	return currentShape, nil
}

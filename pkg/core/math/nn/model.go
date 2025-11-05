package nn

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/layers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Optimizer interface for updating parameters during training.
type Optimizer interface {
	// Update updates parameter using gradient.
	Update(param *layers.Parameter) error
}

// Model represents a neural network model composed of layers.
// Model embeds layers.Base and implements the Layer interface, allowing models to be used as layers.
type Model struct {
	layers.Base
	layers     []Layer
	layerNames map[string]int // Map from layer name to index
	inputShape []int
}

// GetLayer returns the layer at the given index.
func (m *Model) GetLayer(index int) Layer {
	if m == nil || index < 0 || index >= len(m.layers) {
		return nil
	}
	return m.layers[index]
}

// GetLayerByName returns the layer with the given name, or nil if not found.
func (m *Model) GetLayerByName(name string) Layer {
	if m == nil || m.layerNames == nil {
		return nil
	}
	if index, ok := m.layerNames[name]; ok && index >= 0 && index < len(m.layers) {
		return m.layers[index]
	}
	return nil
}

// LayerCount returns the number of layers in the model.
func (m *Model) LayerCount() int {
	if m == nil {
		return 0
	}
	return len(m.layers)
}

// Init initializes all layers in the model.
// This should be called after building the model and before first Forward pass.
func (m *Model) Init() error {
	if m == nil {
		return fmt.Errorf("model.Init: nil model")
	}

	if len(m.inputShape) == 0 {
		return fmt.Errorf("model.Init: input shape not set")
	}

	// Initialize each layer with its input shape
	currentShape := m.inputShape
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
func (m *Model) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if m == nil {
		return nil, fmt.Errorf("model.Forward: nil model")
	}

	if input.Shape().Rank() == 0 {
		return nil, fmt.Errorf("model.Forward: empty input")
	}

	// Validate input shape
	inputShape := input.Shape()
	if !shapesEqual(inputShape, m.inputShape) {
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
func (m *Model) Backward(gradOutput tensor.Tensor) (tensor.Tensor, error) {
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
// Collects parameters from layers that have a Parameters() method returning map[layers.ParamIndex]layers.Parameter.
// Returns a combined map where keys are "layer_index:param_index" and values are layers.Parameter structs (not pointers).
func (m *Model) Parameters() map[string]layers.Parameter {
	if m == nil {
		return nil
	}

	result := make(map[string]layers.Parameter)
	for layerIdx, layer := range m.layers {
		// Check if layer has Parameters() method
		// Layers embedding Base will have this method promoted
		type paramsGetter interface {
			Parameters() map[layers.ParamIndex]layers.Parameter
		}

		if pg, ok := layer.(paramsGetter); ok {
			params := pg.Parameters()
			for paramIdx, param := range params {
				key := fmt.Sprintf("%d:%v", layerIdx, paramIdx)
				result[key] = param
			}
		}
	}

	if len(result) == 0 {
		return nil
	}
	return result
}

// ZeroGrad zeros all parameter gradients.
func (m *Model) ZeroGrad() {
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
func (m *Model) Update(optimizer Optimizer) error {
	if m == nil {
		return fmt.Errorf("model.Update: nil model")
	}

	if optimizer == nil {
		return fmt.Errorf("model.Update: nil optimizer")
	}

	// Note: Update() works with parameters as values.
	// For optimizers that need pointer access, we create temporary pointers.
	// After optimization, parameters are written back to layers via SetParam.
	for _, layer := range m.layers {
		type paramsGetter interface {
			Parameters() map[layers.ParamIndex]layers.Parameter
		}
		type paramSetter interface {
			SetParam(layers.ParamIndex, layers.Parameter)
		}

		if pg, ok := layer.(paramsGetter); ok {
			params := pg.Parameters()
			if len(params) > 0 {
				if ps, ok2 := layer.(paramSetter); ok2 {
					for paramIdx, param := range params {
						// Create a pointer to the parameter for optimizer
						paramPtr := &param

						// Optimizer.Update updates the parameter in-place
						if err := optimizer.Update(paramPtr); err != nil {
							return fmt.Errorf("model.Update: failed to update parameter: %w", err)
						}
						// Update the layer with the modified parameter
						ps.SetParam(paramIdx, *paramPtr)
					}
				}
			}
		}
	}

	return nil
}

// Name returns the name of the model (from Base).
func (m *Model) Name() string {
	if m == nil {
		return ""
	}
	return m.Base.Name()
}

// CanLearn returns whether the model can learn (from Base).
func (m *Model) CanLearn() bool {
	if m == nil {
		return false
	}
	return m.Base.CanLearn()
}

// SetCanLearn sets whether the model can learn (from Base).
func (m *Model) SetCanLearn(canLearn bool) {
	if m == nil {
		return
	}
	m.Base.SetCanLearn(canLearn)
}

// Input returns the input tensor from the last Forward pass (from Base).
func (m *Model) Input() tensor.Tensor {
	if m == nil {
		return nil
	}
	return m.Base.Input()
}

// Output returns the output tensor from the last Forward pass (from Base).
func (m *Model) Output() tensor.Tensor {
	if m == nil {
		return nil
	}
	return m.Base.Output()
}

// OutputShape returns the output shape for given input shape.
// Computes the output shape by propagating through all layers.
func (m *Model) OutputShape(inputShape []int) ([]int, error) {
	if m == nil {
		return nil, fmt.Errorf("model.OutputShape: nil model")
	}

	if len(inputShape) == 0 {
		return nil, fmt.Errorf("model.OutputShape: empty input shape")
	}

	// Validate input shape matches model's expected input shape
	if !shapesEqual(inputShape, m.inputShape) {
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

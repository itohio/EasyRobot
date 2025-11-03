package nn

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/layers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// Model represents a neural network model composed of layers.
type Model struct {
	layers     []Layer
	layerNames map[string]int // Map from layer name to index
	inputShape []int
	Input      tensor.Tensor // Input tensor (set by user)
	Output     tensor.Tensor // Output tensor (set after Forward)
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
		return tensor.Tensor{}, fmt.Errorf("model.Forward: nil model")
	}

	if len(input.Dim) == 0 {
		return tensor.Tensor{}, fmt.Errorf("model.Forward: empty input")
	}

	// Validate input shape
	inputShape := input.Shape()
	if !shapesEqual(inputShape, m.inputShape) {
		return tensor.Tensor{}, fmt.Errorf("model.Forward: input shape %v does not match expected shape %v", inputShape, m.inputShape)
	}

	// Set model input
	m.Input = input

	// Forward pass through all layers
	// Reuse output tensors: each layer computes into its output tensor,
	// and the next layer uses that same tensor as its input
	current := input
	for i, layer := range m.layers {
		if layer == nil {
			return tensor.Tensor{}, fmt.Errorf("model.Forward: nil layer at index %d", i)
		}

		// Forward computes into layer's pre-allocated output tensor
		output, err := layer.Forward(current)
		if err != nil {
			return tensor.Tensor{}, fmt.Errorf("model.Forward: layer %d (%s) failed: %w", i, layer.Name(), err)
		}

		// Next layer uses this layer's output tensor directly
		current = output
	}

	// Set model output
	m.Output = current

	return current, nil
}

// Backward computes backward pass through all layers in reverse order.
// Layers use their stored inputs/outputs from Forward pass.
func (m *Model) Backward(gradOutput tensor.Tensor) error {
	if m == nil {
		return fmt.Errorf("model.Backward: nil model")
	}

	if len(gradOutput.Dim) == 0 {
		return fmt.Errorf("model.Backward: empty gradOutput")
	}

	if len(m.Output.Dim) == 0 {
		return fmt.Errorf("model.Backward: model.Output is empty, must call Forward first")
	}

	// Backward pass through layers in reverse order
	currentGrad := gradOutput
	for i := len(m.layers) - 1; i >= 0; i-- {
		layer := m.layers[i]
		if layer == nil {
			return fmt.Errorf("model.Backward: nil layer at index %d", i)
		}

		gradInput, err := layer.Backward(currentGrad)
		if err != nil {
			return fmt.Errorf("model.Backward: layer %d failed: %w", i, err)
		}

		currentGrad = gradInput
	}

	return nil
}

// Parameters returns all trainable parameters from all layers.
// Collects parameters from layers that have a Parameters() method returning map[interface{}]layers.Parameter.
// Returns a combined map where keys are "layer_index:param_index" and values are layers.Parameter structs (not pointers).
// Note: This method works with layers embedding Base, which have Parameters() returning map[layers.ParamIndex]layers.Parameter.
// The layers.ParamIndex type is converted to interface{} to avoid import cycles.
func (m *Model) Parameters() map[string]layers.Parameter {
	if m == nil {
		return nil
	}

	result := make(map[string]layers.Parameter)
	for layerIdx, layer := range m.layers {
		// Check if layer has ParametersAsInterface() or Parameters() method
		// Layers embedding Base will have these methods promoted
		// We use interface{} to avoid import cycles with layers package
		type paramsGetterInterface interface {
			ParametersAsInterface() map[interface{}]layers.Parameter
		}
		type paramsGetter interface {
			Parameters() map[interface{}]layers.Parameter
		}

		// Try ParametersAsInterface first (from Base)
		if pgi, ok := layer.(paramsGetterInterface); ok {
			params := pgi.ParametersAsInterface()
			for paramIdx, param := range params {
				key := fmt.Sprintf("%d:%v", layerIdx, paramIdx)
				result[key] = param
			}
		} else if pg, ok := layer.(paramsGetter); ok {
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
// Optimizer is from math/learn package.
func (m *Model) Update(optimizer interface{}) error {
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
		type paramsGetterInterface interface {
			ParametersAsInterface() map[interface{}]layers.Parameter
		}
		type paramSetterInterface interface {
			SetParamInterface(interface{}, layers.Parameter)
		}

		var params map[interface{}]layers.Parameter
		var ok bool

		// Try ParametersAsInterface first (from Base)
		if pgi, ok2 := layer.(paramsGetterInterface); ok2 {
			params = pgi.ParametersAsInterface()
			ok = true
		}

		if ok && params != nil {
			if psi, ok2 := layer.(paramSetterInterface); ok2 {
				for paramIdx, param := range params {
					// Create a pointer to the parameter for optimizer
					paramPtr := &param

					// Use type assertion to get learn.Optimizer
					type learnOptimizer interface {
						Update(*layers.Parameter) error
					}
					if opt, ok3 := optimizer.(learnOptimizer); ok3 {
						if err := opt.Update(paramPtr); err != nil {
							return fmt.Errorf("model.Update: failed to update parameter: %w", err)
						}
						// Update the layer with the modified parameter
						psi.SetParamInterface(paramIdx, *paramPtr)
					} else {
						return fmt.Errorf("model.Update: optimizer does not implement learn.Optimizer interface")
					}
				}
			}
		}
	}

	return nil
}

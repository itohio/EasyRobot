package learn

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/nn"
	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// TrainStep performs a single training step: forward pass, loss computation, backward pass, and weight update.
// Optimizer must implement the types.Optimizer interface.
// Layer can be either a Sequential model or a single Layer that implements the Layer interface.
func TrainStep(layer types.Layer, optimizer types.Optimizer, lossFn nn.LossFunction, input, target tensor.Tensor) (float64, error) {
	if layer == nil {
		return 0, fmt.Errorf("TrainStep: nil layer")
	}
	if optimizer == nil {
		return 0, fmt.Errorf("TrainStep: nil optimizer")
	}
	if lossFn == nil {
		return 0, fmt.Errorf("TrainStep: nil loss function")
	}
	if input.Size() == 0 {
		return 0, fmt.Errorf("TrainStep: empty input")
	}
	if target.Size() == 0 {
		return 0, fmt.Errorf("TrainStep: empty target")
	}

	// Forward pass
	output, err := layer.Forward(input)
	if err != nil {
		return 0, fmt.Errorf("TrainStep: forward pass failed: %w", err)
	}

	// Compute loss
	loss, err := lossFn.Compute(output, target)
	if err != nil {
		return 0, fmt.Errorf("TrainStep: loss computation failed: %w", err)
	}

	// Compute loss gradient
	gradOutput, err := lossFn.Gradient(output, target)
	if err != nil {
		return 0, fmt.Errorf("TrainStep: loss gradient computation failed: %w", err)
	}

	// Backward pass (layers use their stored inputs/outputs)
	layer.ZeroGrad()
	_, err = layer.Backward(gradOutput)
	if err != nil {
		return 0, fmt.Errorf("TrainStep: backward pass failed: %w", err)
	}

	// Update weights
	err = layer.Update(optimizer)
	if err != nil {
		return 0, fmt.Errorf("TrainStep: weight update failed: %w", err)
	}

	return float64(loss), nil
}

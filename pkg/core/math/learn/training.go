package learn

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/nn"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// TrainStep performs a single training step: forward pass, loss computation, backward pass, and weight update.
// Optimizer must implement the nn.Optimizer interface.
func TrainStep(model *nn.Model, optimizer nn.Optimizer, lossFn nn.LossFunction, input, target tensor.Tensor) (float64, error) {
	if model == nil {
		return 0, fmt.Errorf("TrainStep: nil model")
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
	output, err := model.Forward(input)
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
	model.ZeroGrad()
	_, err = model.Backward(gradOutput)
	if err != nil {
		return 0, fmt.Errorf("TrainStep: backward pass failed: %w", err)
	}

	// Update weights
	err = model.Update(optimizer)
	if err != nil {
		return 0, fmt.Errorf("TrainStep: weight update failed: %w", err)
	}

	return float64(loss), nil
}

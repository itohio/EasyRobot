package nn

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// TrainStep performs a single training step: forward pass, loss computation, backward pass, and weight update.
// Optimizer should be from math/learn package.
func TrainStep(model *Model, optimizer interface{}, lossFn LossFunction, input, target tensor.Tensor) (float32, error) {
	if model == nil {
		return 0, fmt.Errorf("TrainStep: nil model")
	}
	if optimizer == nil {
		return 0, fmt.Errorf("TrainStep: nil optimizer")
	}
	if lossFn == nil {
		return 0, fmt.Errorf("TrainStep: nil loss function")
	}
	if len(input.Dim) == 0 {
		return 0, fmt.Errorf("TrainStep: empty input")
	}
	if len(target.Dim) == 0 {
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
	err = model.Backward(gradOutput)
	if err != nil {
		return 0, fmt.Errorf("TrainStep: backward pass failed: %w", err)
	}

	// Update weights
	err = model.Update(optimizer)
	if err != nil {
		return 0, fmt.Errorf("TrainStep: weight update failed: %w", err)
	}

	return loss, nil
}

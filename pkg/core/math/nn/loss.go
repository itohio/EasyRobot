package nn

import (
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// LossFunction represents a loss function.
type LossFunction interface {
	// Compute computes loss: loss = lossFn(pred, target)
	Compute(pred, target tensor.Tensor) (float32, error)

	// Gradient computes gradient w.r.t. predictions: gradPred = d(loss)/d(pred)
	Gradient(pred, target tensor.Tensor) (tensor.Tensor, error)
}

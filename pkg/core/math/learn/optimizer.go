package learn

import (
	"fmt"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/layers"
)

// Optimizer interface for updating parameters during training.
type Optimizer interface {
	// Update updates parameter using gradient.
	Update(param *layers.Parameter) error
}

// SGD implements Stochastic Gradient Descent optimizer.
type SGD struct {
	lr float32 // Learning rate
}

// NewSGD creates a new SGD optimizer with the given learning rate.
func NewSGD(lr float32) *SGD {
	if lr <= 0 {
		panic("SGD: learning rate must be positive")
	}
	return &SGD{lr: lr}
}

// Update applies SGD update: param.Data = param.Data - lr * param.Grad.
func (s *SGD) Update(param *layers.Parameter) error {
	if s == nil {
		return fmt.Errorf("SGD.Update: nil optimizer")
	}

	if param == nil {
		return fmt.Errorf("SGD.Update: nil parameter")
	}

	if !param.RequiresGrad {
		return nil // No gradient tracking
	}

	if len(param.Grad.Dim) == 0 {
		return nil // No gradient computed
	}

	if len(param.Data.Dim) == 0 {
		return fmt.Errorf("SGD.Update: empty parameter data")
	}

	// Validate shapes match
	if !shapesEqual(param.Data.Dim, param.Grad.Dim) {
		return fmt.Errorf("SGD.Update: parameter and gradient shapes mismatch: %v vs %v", param.Data.Dim, param.Grad.Dim)
	}

	// SGD update: data = data - lr * grad
	for i := range param.Data.Data {
		param.Data.Data[i] -= s.lr * param.Grad.Data[i]
	}

	return nil
}

// shapesEqual checks if two shapes are equal.
func shapesEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

package nn

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// MSELoss implements Mean Squared Error loss function.
type MSELoss struct{}

// NewMSE creates a new MSE loss function.
func NewMSE() *MSELoss {
	return &MSELoss{}
}

// Compute computes MSE: loss = mean((pred - target)^2).
func (m *MSELoss) Compute(pred, target tensor.Tensor) (float32, error) {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return 0, fmt.Errorf("MSE.Compute: empty input")
	}

	return MSE(pred, target), nil
}

// Gradient computes gradient w.r.t. predictions: gradPred = 2 * (pred - target) / size.
func (m *MSELoss) Gradient(pred, target tensor.Tensor) (tensor.Tensor, error) {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("MSE.Gradient: empty input")
	}

	predShape := pred.Shape()
	targetShape := target.Shape()

	if len(predShape) != len(targetShape) {
		return tensor.Tensor{}, fmt.Errorf("MSE.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
	}

	for i := range predShape {
		if predShape[i] != targetShape[i] {
			return tensor.Tensor{}, fmt.Errorf("MSE.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
		}
	}

	// gradPred = 2 * (pred - target) / size
	size := pred.Shape().Size()
	grad := pred.Clone()
	grad = grad.Sub(target)
	grad = grad.Scale(2.0 / float32(size))

	return *grad, nil
}

// CrossEntropyLoss implements cross-entropy loss function.
type CrossEntropyLoss struct{}

// NewCrossEntropy creates a new CrossEntropy loss function.
func NewCrossEntropy() *CrossEntropyLoss {
	return &CrossEntropyLoss{}
}

// Compute computes cross-entropy: loss = -sum(target * log(pred + epsilon)).
func (c *CrossEntropyLoss) Compute(pred, target tensor.Tensor) (float32, error) {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return 0, fmt.Errorf("CrossEntropy.Compute: empty input")
	}

	return CrossEntropy(pred, target), nil
}

// Gradient computes gradient w.r.t. predictions: gradPred = -target / (pred + epsilon).
func (c *CrossEntropyLoss) Gradient(pred, target tensor.Tensor) (tensor.Tensor, error) {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("CrossEntropy.Gradient: empty input")
	}

	predShape := pred.Shape()
	targetShape := target.Shape()

	if len(predShape) != len(targetShape) {
		return tensor.Tensor{}, fmt.Errorf("CrossEntropy.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
	}

	for i := range predShape {
		if predShape[i] != targetShape[i] {
			return tensor.Tensor{}, fmt.Errorf("CrossEntropy.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
		}
	}

	// gradPred = -target / (pred + epsilon)
	const epsilon = 1e-10
	grad := pred.Clone()
	predData := pred.Data()
	targetData := target.Data()
	gradData := grad.Data()

	for i := range gradData {
		if predData[i] > 0 {
			gradData[i] = -targetData[i] / (predData[i] + epsilon)
		} else {
			gradData[i] = 0
		}
	}

	return *grad, nil
}

// CategoricalCrossEntropy implements categorical cross-entropy loss with optional softmax.
type CategoricalCrossEntropy struct {
	fromLogits bool // If true, apply softmax internally
}

// NewCategoricalCrossEntropy creates a new CategoricalCrossEntropy loss function.
func NewCategoricalCrossEntropy(fromLogits bool) *CategoricalCrossEntropy {
	return &CategoricalCrossEntropy{fromLogits: fromLogits}
}

// Compute computes categorical cross-entropy.
// If fromLogits is true, applies softmax then cross-entropy.
// Otherwise, assumes pred is already probabilities.
func (c *CategoricalCrossEntropy) Compute(pred, target tensor.Tensor) (float32, error) {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return 0, fmt.Errorf("CategoricalCrossEntropy.Compute: empty input")
	}

	if c.fromLogits {
		// Apply softmax first (assumes last dimension for classification)
		predShape := pred.Shape()
		if len(predShape) == 0 {
			return 0, fmt.Errorf("CategoricalCrossEntropy.Compute: invalid pred shape")
		}
		dim := len(predShape) - 1 // Apply softmax along last dimension
		predProb := pred.Softmax(dim, nil)
		if predProb == nil {
			return 0, fmt.Errorf("CategoricalCrossEntropy.Compute: softmax returned nil")
		}
		return CrossEntropy(*predProb, target), nil
	}

	return CrossEntropy(pred, target), nil
}

// Gradient computes gradient w.r.t. predictions.
// If fromLogits is true and softmax was applied, returns: gradPred = pred - target.
// Otherwise, returns: gradPred = -target / (pred + epsilon).
func (c *CategoricalCrossEntropy) Gradient(pred, target tensor.Tensor) (tensor.Tensor, error) {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return tensor.Tensor{}, fmt.Errorf("CategoricalCrossEntropy.Gradient: empty input")
	}

	predShape := pred.Shape()
	targetShape := target.Shape()

	if len(predShape) != len(targetShape) {
		return tensor.Tensor{}, fmt.Errorf("CategoricalCrossEntropy.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
	}

	for i := range predShape {
		if predShape[i] != targetShape[i] {
			return tensor.Tensor{}, fmt.Errorf("CategoricalCrossEntropy.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
		}
	}

	if c.fromLogits {
		// Apply softmax first
		dim := len(predShape) - 1
		predProb := pred.Softmax(dim, nil)
		if predProb == nil {
			return tensor.Tensor{}, fmt.Errorf("CategoricalCrossEntropy.Gradient: softmax returned nil")
		}

		// Gradient after softmax: pred - target
		grad := predProb.Clone()
		grad = grad.Sub(target)
		return *grad, nil
	}

	// Standard cross-entropy gradient
	const epsilon = 1e-10
	grad := pred.Clone()
	predData := pred.Data()
	targetData := target.Data()
	gradData := grad.Data()

	for i := range gradData {
		if predData[i] > 0 {
			gradData[i] = -targetData[i] / (predData[i] + epsilon)
		} else {
			gradData[i] = 0
		}
	}

	return *grad, nil
}

// MSE computes Mean Squared Error between tensor and target
func MSE(pred, target tensor.Tensor) float32 {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return 0
	}

	squaredDiff := pred.Clone()
	squaredDiff = squaredDiff.Sub(target)
	squaredDiff = squaredDiff.Mul(*squaredDiff)

	size := pred.Shape().Size()
	sum := squaredDiff.Sum()
	if size > 0 {
		return sum.Data()[0] / float32(size)
	}

	return 0
}

// CrossEntropy computes cross-entropy loss between predictions and targets
func CrossEntropy(pred, target tensor.Tensor) float32 {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return 0
	}

	var loss float32
	predData := pred.Data()
	targetData := target.Data()
	for i := range predData {
		if targetData[i] != 0 && predData[i] > 0 {
			loss -= targetData[i] * math32.Log(predData[i]+1e-10)
		}
	}

	return loss
}

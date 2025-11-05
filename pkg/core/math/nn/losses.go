package nn

import (
	"fmt"

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
		return nil, fmt.Errorf("MSE.Gradient: empty input")
	}

	predShape := pred.Shape()
	targetShape := target.Shape()

	if len(predShape) != len(targetShape) {
		return nil, fmt.Errorf("MSE.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
	}

	for i := range predShape {
		if predShape[i] != targetShape[i] {
			return nil, fmt.Errorf("MSE.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
		}
	}

	// gradPred = 2 * (pred - target) / size
	size := pred.Shape().Size()
	var grad tensor.Tensor
	grad = pred.Clone()
	grad = grad.Sub(target)
	grad = grad.Scale(2.0 / float32(size))

	return grad, nil
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
		return nil, fmt.Errorf("CrossEntropy.Gradient: empty input")
	}

	predShape := pred.Shape()
	targetShape := target.Shape()

	if len(predShape) != len(targetShape) {
		return nil, fmt.Errorf("CrossEntropy.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
	}

	for i := range predShape {
		if predShape[i] != targetShape[i] {
			return nil, fmt.Errorf("CrossEntropy.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
		}
	}

	// gradPred = -target / (pred + epsilon) where pred > 0, else 0
	const epsilon = 1e-10
	zeros := tensor.ZerosLike(pred)
	mask := pred.GreaterThan(zeros) // 1.0 where pred > 0, 0.0 otherwise
	epsilonTensor := tensor.FullLike(pred, epsilon)
	predPlusEps := pred.Clone().Add(epsilonTensor)
	negTarget := target.Clone().Negative()
	gradComputed := negTarget.Div(predPlusEps)
	grad := pred.Where(mask, gradComputed, zeros)

	return grad, nil
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
		return CrossEntropy(predProb, target), nil
	}

	return CrossEntropy(pred, target), nil
}

// Gradient computes gradient w.r.t. predictions.
// If fromLogits is true and softmax was applied, returns: gradPred = pred - target.
// Otherwise, returns: gradPred = -target / (pred + epsilon).
func (c *CategoricalCrossEntropy) Gradient(pred, target tensor.Tensor) (tensor.Tensor, error) {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return nil, fmt.Errorf("CategoricalCrossEntropy.Gradient: empty input")
	}

	predShape := pred.Shape()
	targetShape := target.Shape()

	if len(predShape) != len(targetShape) {
		return nil, fmt.Errorf("CategoricalCrossEntropy.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
	}

	for i := range predShape {
		if predShape[i] != targetShape[i] {
			return nil, fmt.Errorf("CategoricalCrossEntropy.Gradient: shape mismatch: pred %v, target %v", predShape, targetShape)
		}
	}

	if c.fromLogits {
		// Apply softmax first
		dim := len(predShape) - 1
		predProb := pred.Softmax(dim, nil)
		if predProb == nil {
			return nil, fmt.Errorf("CategoricalCrossEntropy.Gradient: softmax returned nil")
		}

		// Gradient after softmax: pred - target
		grad := predProb.Clone()
		grad = grad.Sub(target)
		return grad, nil
	}

	// Standard cross-entropy gradient: -target / (pred + epsilon) where pred > 0, else 0
	const epsilon = 1e-10
	zeros := tensor.ZerosLike(pred)
	mask := pred.GreaterThan(zeros) // 1.0 where pred > 0, 0.0 otherwise
	epsilonTensor := tensor.FullLike(pred, epsilon)
	predPlusEps := pred.Clone().Add(epsilonTensor)
	negTarget := target.Clone().Negative()
	gradComputed := negTarget.Div(predPlusEps)
	grad := pred.Where(mask, gradComputed, zeros)

	return grad, nil
}

// MSE computes Mean Squared Error between tensor and target
func MSE(pred, target tensor.Tensor) float32 {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return 0
	}

	squaredDiff := pred.Clone()
	squaredDiff = squaredDiff.Sub(target)
	squaredDiff = squaredDiff.Mul(squaredDiff)

	size := pred.Shape().Size()
	sum := squaredDiff.Sum()
	if size > 0 {
		return sum.At(0) / float32(size)
	}

	return 0
}

// CrossEntropy computes cross-entropy loss between predictions and targets
func CrossEntropy(pred, target tensor.Tensor) float32 {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return 0
	}

	// Create masks: target != 0 && pred > 0
	zeros := tensor.ZerosLike(pred)
	targetAbs := target.Clone().Abs()
	targetNonZero := targetAbs.GreaterThan(zeros)   // mask for target != 0
	predPositive := pred.GreaterThan(zeros)         // mask for pred > 0
	combinedMask := targetNonZero.Mul(predPositive) // combined condition mask

	// Compute: -target * log(pred + epsilon) where condition is true
	const epsilon = 1e-10
	epsilonTensor := tensor.FullLike(pred, epsilon)
	predPlusEps := pred.Clone().Add(epsilonTensor)
	logPred := predPlusEps.Log()
	targetLogPred := target.Clone().Mul(logPred)
	maskedLoss := targetLogPred.Mul(combinedMask)

	// Sum and negate to get final loss
	loss := -maskedLoss.Sum().At(0)

	return loss
}

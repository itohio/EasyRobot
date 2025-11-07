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
	// Create gradient tensor and compute in-place to avoid Clone
	grad := tensor.New(pred.DataType(), pred.Shape())
	grad.Copy(pred)
	grad.Subtract(nil, target)
	// Convert float32 to float64 for Scale interface
	grad.MulScalar(nil, float64(2.0/float32(size)))

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
	// Pre-allocate intermediate tensors to avoid multiple allocations
	zeros := tensor.ZerosLike(pred)
	mask := pred.GreaterThan(nil, zeros) // 1.0 where pred > 0, 0.0 otherwise
	epsilonTensor := tensor.FullLike(pred, epsilon)
	// Use destination parameter for pred + epsilon
	predPlusEps := tensor.New(pred.DataType(), pred.Shape())
	pred.Add(predPlusEps, epsilonTensor)
	// Use destination parameter for negative target
	negTarget := tensor.New(target.DataType(), target.Shape())
	target.Negative(negTarget)
	// Use destination parameter for division
	gradComputed := tensor.New(pred.DataType(), pred.Shape())
	negTarget.Divide(gradComputed, predPlusEps)
	// Use destination parameter for Where
	grad := tensor.New(pred.DataType(), pred.Shape())
	pred.Where(grad, mask, gradComputed, zeros)

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
		// Pre-allocate tensor for softmax result
		predProb := tensor.New(pred.DataType(), pred.Shape())
		pred.Softmax(dim, predProb)
		if tensor.IsNil(predProb) {
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
		// Pre-allocate tensor for softmax result
		predProb := tensor.New(pred.DataType(), pred.Shape())
		pred.Softmax(dim, predProb)
		if tensor.IsNil(predProb) {
			return nil, fmt.Errorf("CategoricalCrossEntropy.Gradient: softmax returned nil")
		}

		// Gradient after softmax: pred - target
		// Use destination parameter to avoid Clone
		grad := tensor.New(predProb.DataType(), predProb.Shape())
		grad.Copy(predProb)
		grad.Subtract(nil, target)
		return grad, nil
	}

	// Standard cross-entropy gradient: -target / (pred + epsilon) where pred > 0, else 0
	const epsilon = 1e-10
	// Pre-allocate intermediate tensors to avoid multiple allocations
	zeros := tensor.ZerosLike(pred)
	mask := pred.GreaterThan(nil, zeros) // 1.0 where pred > 0, 0.0 otherwise
	epsilonTensor := tensor.FullLike(pred, epsilon)
	// Use destination parameter for pred + epsilon
	predPlusEps := tensor.New(pred.DataType(), pred.Shape())
	pred.Add(predPlusEps, epsilonTensor)
	// Use destination parameter for negative target
	negTarget := tensor.New(target.DataType(), target.Shape())
	target.Negative(negTarget)
	// Use destination parameter for division
	gradComputed := tensor.New(pred.DataType(), pred.Shape())
	negTarget.Divide(gradComputed, predPlusEps)
	// Use destination parameter for Where
	grad := tensor.New(pred.DataType(), pred.Shape())
	pred.Where(grad, mask, gradComputed, zeros)

	return grad, nil
}

// MSE computes Mean Squared Error between tensor and target
func MSE(pred, target tensor.Tensor) float32 {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return 0
	}

	// Use destination parameter to avoid Clone
	squaredDiff := tensor.New(pred.DataType(), pred.Shape())
	squaredDiff.Copy(pred)
	squaredDiff.Subtract(nil, target)
	squaredDiff.Multiply(nil, squaredDiff)

	size := pred.Shape().Size()
	// Use destination parameter for Sum (scalar result)
	sumResult := tensor.New(pred.DataType(), tensor.NewShape(1))
	sum := squaredDiff.Sum(sumResult, nil)
	if size > 0 {
		// At() returns float64, convert to float32 for division
		return float32(sum.At(0)) / float32(size)
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
	targetAbs := target.Clone().Abs(nil)
	targetNonZero := targetAbs.GreaterThan(nil, zeros)        // mask for target != 0
	predPositive := pred.GreaterThan(nil, zeros)              // mask for pred > 0
	combinedMask := targetNonZero.Multiply(nil, predPositive) // combined condition mask

	// Compute: -target * log(pred + epsilon) where condition is true
	const epsilon = 1e-10
	epsilonTensor := tensor.FullLike(pred, epsilon)
	predPlusEps := pred.Clone().Add(nil, epsilonTensor)
	logPred := predPlusEps.Log(nil)
	targetLogPred := target.Clone().Multiply(nil, logPred)
	maskedLoss := targetLogPred.Multiply(nil, combinedMask)

	// Sum and negate to get final loss
	// At() returns float64, convert to float32 for return
	loss := -float32(maskedLoss.Sum(nil, nil).At(0))

	return loss
}

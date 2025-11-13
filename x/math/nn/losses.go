package nn

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/tensor"
)

// MSELoss implements Mean Squared Error loss function.
type MSELoss struct {
	// Pre-allocated scratch tensors for gradient computation
	grad tensor.Tensor // Scratch tensor for gradient computation
}

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
	if err := validateShapes(pred, target); err != nil {
		return nil, fmt.Errorf("MSE.Gradient: %w", err)
	}

	// Reallocate grad tensor if shape changed
	predShape := pred.Shape()
	if tensor.IsNil(m.grad) || !m.grad.Shape().Equal(predShape) {
		m.grad = tensor.New(pred.DataType(), predShape)
	}

	// gradPred = 2 * (pred - target) / size
	size := pred.Shape().Size()
	pred.Subtract(m.grad, target)
	m.grad.MulScalar(nil, float64(2.0/float32(size)))

	return m.grad, nil
}

// CrossEntropyLoss implements cross-entropy loss function.
type CrossEntropyLoss struct {
	// Pre-allocated scratch tensors for gradient computation
	mask          tensor.Tensor // Mask for pred > 0
	zeros         tensor.Tensor // Zeros tensor for Where operation
	epsilonTensor tensor.Tensor // Epsilon tensor for numerical stability
	predPlusEps   tensor.Tensor // Scratch tensor for pred + epsilon
	negTarget     tensor.Tensor // Scratch tensor for negative target
	gradComputed  tensor.Tensor // Scratch tensor for computed gradient
	grad          tensor.Tensor // Scratch tensor for final gradient
}

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
	if err := validateShapes(pred, target); err != nil {
		return nil, fmt.Errorf("CrossEntropy.Gradient: %w", err)
	}

	// Reallocate scratch tensors if shape changed
	c.ensureScratchTensors(pred, target)

	// gradPred = -target / (pred + epsilon) where pred > 0, else 0
	const epsilon = 1e-10
	// Use pre-allocated mask
	pred.GreaterScalar(c.mask, 0) // 1.0 where pred > 0, 0.0 otherwise
	// Use pre-allocated epsilon tensor (constant value, reused if same shape)
	if tensor.IsNil(c.epsilonTensor) || !c.epsilonTensor.Shape().Equal(pred.Shape()) {
		c.epsilonTensor = tensor.FullLike(pred, epsilon)
	}
	// Use pre-allocated scratch tensors
	pred.Add(c.predPlusEps, c.epsilonTensor)
	target.Negative(c.negTarget)
	c.negTarget.Divide(c.gradComputed, c.predPlusEps)
	// Use pre-allocated mask and zeros
	pred.Where(c.grad, c.mask, c.gradComputed, c.zeros)

	return c.grad, nil
}

// ensureScratchTensors allocates or reallocates scratch tensors if shape changed
func (c *CrossEntropyLoss) ensureScratchTensors(pred, target tensor.Tensor) {
	predShape := pred.Shape()
	dtype := pred.DataType()

	// Allocate or reallocate mask if needed
	if tensor.IsNil(c.mask) || !c.mask.Shape().Equal(predShape) {
		c.mask = tensor.New(dtype, predShape)
	}

	// Allocate or reallocate zeros if needed
	if tensor.IsNil(c.zeros) || !c.zeros.Shape().Equal(predShape) {
		c.zeros = tensor.ZerosLike(pred)
	}

	// Epsilon tensor is handled separately in Gradient method since it needs epsilon constant

	// Allocate or reallocate predPlusEps if needed
	if tensor.IsNil(c.predPlusEps) || !c.predPlusEps.Shape().Equal(predShape) {
		c.predPlusEps = tensor.New(dtype, predShape)
	}

	// Allocate or reallocate negTarget if needed
	if tensor.IsNil(c.negTarget) || !c.negTarget.Shape().Equal(target.Shape()) {
		c.negTarget = tensor.New(dtype, target.Shape())
	}

	// Allocate or reallocate gradComputed if needed
	if tensor.IsNil(c.gradComputed) || !c.gradComputed.Shape().Equal(predShape) {
		c.gradComputed = tensor.New(dtype, predShape)
	}

	// Allocate or reallocate grad if needed
	if tensor.IsNil(c.grad) || !c.grad.Shape().Equal(predShape) {
		c.grad = tensor.New(dtype, predShape)
	}
}

// CategoricalCrossEntropy implements categorical cross-entropy loss with optional softmax.
type CategoricalCrossEntropy struct {
	fromLogits bool // If true, apply softmax internally
	// Pre-allocated scratch tensors for gradient computation
	predProb      tensor.Tensor // Tensor for softmax result (when fromLogits is true)
	grad          tensor.Tensor // Scratch tensor for final gradient
	mask          tensor.Tensor // Mask for pred > 0 (when fromLogits is false)
	zeros         tensor.Tensor // Zeros tensor for Where operation
	epsilonTensor tensor.Tensor // Epsilon tensor for numerical stability
	predPlusEps   tensor.Tensor // Scratch tensor for pred + epsilon
	negTarget     tensor.Tensor // Scratch tensor for negative target
	gradComputed  tensor.Tensor // Scratch tensor for computed gradient
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
		// Reuse pre-allocated tensor for softmax result if shape matches
		if tensor.IsNil(c.predProb) || !c.predProb.Shape().Equal(pred.Shape()) {
			c.predProb = tensor.New(pred.DataType(), pred.Shape())
		}
		pred.Softmax(dim, c.predProb)
		if tensor.IsNil(c.predProb) {
			return 0, fmt.Errorf("CategoricalCrossEntropy.Compute: softmax returned nil")
		}
		return CrossEntropy(c.predProb, target), nil
	}

	return CrossEntropy(pred, target), nil
}

// Gradient computes gradient w.r.t. predictions.
// If fromLogits is true and softmax was applied, returns: gradPred = pred - target.
// Otherwise, returns: gradPred = -target / (pred + epsilon).
func (c *CategoricalCrossEntropy) Gradient(pred, target tensor.Tensor) (tensor.Tensor, error) {
	if err := validateShapes(pred, target); err != nil {
		return nil, fmt.Errorf("CategoricalCrossEntropy.Gradient: %w", err)
	}

	predShape := pred.Shape()

	if c.fromLogits {
		// Apply softmax first
		dim := len(predShape) - 1
		// Reallocate predProb if shape changed
		if tensor.IsNil(c.predProb) || !c.predProb.Shape().Equal(pred.Shape()) {
			c.predProb = tensor.New(pred.DataType(), pred.Shape())
		}
		pred.Softmax(dim, c.predProb)
		if tensor.IsNil(c.predProb) {
			return nil, fmt.Errorf("CategoricalCrossEntropy.Gradient: softmax returned nil")
		}

		// Reallocate grad if shape changed
		if tensor.IsNil(c.grad) || !c.grad.Shape().Equal(c.predProb.Shape()) {
			c.grad = tensor.New(c.predProb.DataType(), c.predProb.Shape())
		}

		// Gradient after softmax: pred - target
		c.predProb.Subtract(c.grad, target)
		return c.grad, nil
	}

	// Standard cross-entropy gradient: -target / (pred + epsilon) where pred > 0, else 0
	const epsilon = 1e-10
	// Reallocate scratch tensors if shape changed
	c.ensureScratchTensors(pred, target)

	// Use pre-allocated mask
	pred.GreaterScalar(c.mask, 0) // 1.0 where pred > 0, 0.0 otherwise
	// Use pre-allocated epsilon tensor (constant value, reused if same shape)
	if tensor.IsNil(c.epsilonTensor) || !c.epsilonTensor.Shape().Equal(pred.Shape()) {
		c.epsilonTensor = tensor.FullLike(pred, epsilon)
	}
	// Use pre-allocated scratch tensors
	pred.Add(c.predPlusEps, c.epsilonTensor)
	target.Negative(c.negTarget)
	c.negTarget.Divide(c.gradComputed, c.predPlusEps)
	// Use pre-allocated mask and zeros
	pred.Where(c.grad, c.mask, c.gradComputed, c.zeros)

	return c.grad, nil
}

// ensureScratchTensors allocates or reallocates scratch tensors if shape changed
func (c *CategoricalCrossEntropy) ensureScratchTensors(pred, target tensor.Tensor) {
	predShape := pred.Shape()
	dtype := pred.DataType()

	// Allocate or reallocate mask if needed
	if tensor.IsNil(c.mask) || !c.mask.Shape().Equal(predShape) {
		c.mask = tensor.New(dtype, predShape)
	}

	// Allocate or reallocate zeros if needed
	if tensor.IsNil(c.zeros) || !c.zeros.Shape().Equal(predShape) {
		c.zeros = tensor.ZerosLike(pred)
	}

	// Epsilon tensor is handled separately in Gradient method since it needs epsilon constant

	// Allocate or reallocate predPlusEps if needed
	if tensor.IsNil(c.predPlusEps) || !c.predPlusEps.Shape().Equal(predShape) {
		c.predPlusEps = tensor.New(dtype, predShape)
	}

	// Allocate or reallocate negTarget if needed
	if tensor.IsNil(c.negTarget) || !c.negTarget.Shape().Equal(target.Shape()) {
		c.negTarget = tensor.New(dtype, target.Shape())
	}

	// Allocate or reallocate gradComputed if needed
	if tensor.IsNil(c.gradComputed) || !c.gradComputed.Shape().Equal(predShape) {
		c.gradComputed = tensor.New(dtype, predShape)
	}

	// Allocate or reallocate grad if needed
	if tensor.IsNil(c.grad) || !c.grad.Shape().Equal(predShape) {
		c.grad = tensor.New(dtype, predShape)
	}
}

// MSE computes Mean Squared Error between tensor and target
func MSE(pred, target tensor.Tensor) float32 {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return 0
	}

	// Use destination parameter to avoid Clone
	// Note: This creates temporary tensors, but MSE is typically called less frequently than Gradient
	squaredDiff := tensor.New(pred.DataType(), pred.Shape())
	pred.Subtract(squaredDiff, target)
	squaredDiff.Square(squaredDiff)

	size := pred.Shape().Size()
	if size == 0 {
		return 0
	}
	// Use destination parameter for Sum (scalar result)
	sumResult := tensor.New(pred.DataType(), tensor.NewShape(1))
	sum := squaredDiff.Sum(sumResult, nil)
	// At() returns float64, convert to float32 for division
	return float32(sum.At(0)) / float32(size)
}

// CrossEntropy computes cross-entropy loss between predictions and targets
func CrossEntropy(pred, target tensor.Tensor) float32 {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return 0
	}

	// Optimize: Create masks and compute loss with fewer intermediate allocations
	// Use destination-based operations to avoid unnecessary clones
	const epsilon = 1e-10

	// Compute: -target * log(pred + epsilon) where target != 0 && pred > 0
	// Pre-allocate intermediate tensors
	predPlusEps := tensor.New(pred.DataType(), pred.Shape())
	epsilonTensor := tensor.FullLike(pred, epsilon)
	pred.Add(predPlusEps, epsilonTensor)

	// Compute log(pred + epsilon)
	logPred := predPlusEps.Log(nil)

	// Compute target * log(pred + epsilon) using destination to avoid clone
	targetLogPred := tensor.New(target.DataType(), target.Shape())
	target.Multiply(targetLogPred, logPred)

	// Create combined mask: target != 0 && pred > 0
	// Use absolute value for target check to handle negative targets
	targetAbs := tensor.New(target.DataType(), target.Shape())
	target.Abs(targetAbs)
	targetNonZero := targetAbs.GreaterScalar(nil, 0)          // mask for target != 0
	predPositive := pred.GreaterScalar(nil, 0)                // mask for pred > 0
	combinedMask := targetNonZero.Multiply(nil, predPositive) // combined condition mask

	// Apply mask to loss
	maskedLoss := targetLogPred.Multiply(nil, combinedMask)

	// Sum and negate to get final loss
	sumResult := tensor.New(pred.DataType(), tensor.NewShape(1))
	sum := maskedLoss.Sum(sumResult, nil)
	// At() returns float64, convert to float32 for return
	return -float32(sum.At(0))
}

// validateShapes validates that two tensors have compatible shapes for loss computation
func validateShapes(pred, target tensor.Tensor) error {
	if pred.Shape().Rank() == 0 || target.Shape().Rank() == 0 {
		return fmt.Errorf("empty input")
	}

	predShape := pred.Shape()
	targetShape := target.Shape()

	if len(predShape) != len(targetShape) {
		return fmt.Errorf("shape rank mismatch: pred %v, target %v", predShape, targetShape)
	}

	for i := range predShape {
		if predShape[i] != targetShape[i] {
			return fmt.Errorf("shape mismatch: pred %v, target %v", predShape, targetShape)
		}
	}

	return nil
}

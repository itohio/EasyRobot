package learn

import (
	"fmt"
	"math"
	"sync"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/types"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// SGD implements Stochastic Gradient Descent optimizer.
// It implements nn.Optimizer interface.
type SGD struct {
	lr float64 // Learning rate
}

// NewSGD creates a new SGD optimizer with the given learning rate.
func NewSGD(lr float64) *SGD {
	if lr <= 0 {
		panic("SGD: learning rate must be positive")
	}
	return &SGD{lr: lr}
}

// Update applies SGD update: param.Data = param.Data - lr * param.Grad.
func (s *SGD) Update(param types.Parameter) error {
	if s == nil {
		return fmt.Errorf("SGD.Update: nil optimizer")
	}

	// Note: param is passed by value, but Data and Grad are tensor references
	// We modify the underlying tensor data in place
	if !param.RequiresGrad {
		return nil // No gradient tracking
	}

	if param.Grad == nil || tensor.IsNil(param.Grad) || len(param.Grad.Shape()) == 0 {
		return nil // No gradient computed
	}

	if param.Data == nil || tensor.IsNil(param.Data) || len(param.Data.Shape()) == 0 {
		return fmt.Errorf("SGD.Update: empty parameter data")
	}

	// Validate shapes match
	if !shapesEqual(param.Data.Shape(), param.Grad.Shape()) {
		return fmt.Errorf("SGD.Update: parameter and gradient shapes mismatch: %v vs %v", param.Data.Shape(), param.Grad.Shape())
	}

	// SGD update: data = data - lr * grad
	// Use AddScaled with negative learning rate for efficient combined operation
	// This eliminates the need for Clone and intermediate tensor
	param.Data.AddScaled(nil, param.Grad, -s.lr)

	return nil
}

// Adam implements Adaptive Moment Estimation optimizer.
// It implements nn.Optimizer interface.
// Adam combines advantages of AdaGrad and RMSProp by using both
// first and second moment estimates of gradients.
type Adam struct {
	lr      float64                // Learning rate
	beta1   float64                // Exponential decay rate for first moment estimates
	beta2   float64                // Exponential decay rate for second moment estimates
	epsilon float64                // Small constant for numerical stability
	mu      sync.Mutex             // Mutex for thread-safe state access
	state   map[uintptr]*adamState // Per-parameter state keyed by data pointer
}

// adamState holds per-parameter state for Adam optimizer.
type adamState struct {
	m    tensor.Tensor // First moment estimate
	v    tensor.Tensor // Second moment estimate
	step int           // Step counter for bias correction
	// Pre-allocated scratch tensors to avoid allocations during updates
	scaledGrad1   tensor.Tensor // For (1-beta1) * grad
	scaledGrad2   tensor.Tensor // For (1-beta2) * grad^2
	gradSquared   tensor.Tensor // For grad^2
	mHat          tensor.Tensor // Bias-corrected first moment
	vHat          tensor.Tensor // Bias-corrected second moment
	sqrtVHat      tensor.Tensor // sqrt(vHat)
	epsilonTensor tensor.Tensor // Epsilon tensor
	update        tensor.Tensor // Final update term
}

// NewAdam creates a new Adam optimizer with the given hyperparameters.
// Default values (if not specified): lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
func NewAdam(lr, beta1, beta2, epsilon float64) *Adam {
	if lr <= 0 {
		panic("Adam: learning rate must be positive")
	}
	if beta1 < 0 || beta1 >= 1 {
		panic("Adam: beta1 must be in [0, 1)")
	}
	if beta2 < 0 || beta2 >= 1 {
		panic("Adam: beta2 must be in [0, 1)")
	}
	if epsilon <= 0 {
		panic("Adam: epsilon must be positive")
	}
	return &Adam{
		lr:      lr,
		beta1:   beta1,
		beta2:   beta2,
		epsilon: epsilon,
		state:   make(map[uintptr]*adamState),
	}
}

// Update applies Adam update to the parameter.
// Algorithm:
//
//	m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
//	v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
//	m_hat_t = m_t / (1 - beta1^t)
//	v_hat_t = v_t / (1 - beta2^t)
//	param = param - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
func (a *Adam) Update(param types.Parameter) error {
	if a == nil {
		return fmt.Errorf("Adam.Update: nil optimizer")
	}

	// Note: param is passed by value, but Data and Grad are tensor references
	// We modify the underlying tensor data in place
	if !param.RequiresGrad {
		return nil // No gradient tracking
	}

	if param.Grad == nil || tensor.IsNil(param.Grad) || len(param.Grad.Shape()) == 0 {
		return nil // No gradient computed
	}

	if param.Data == nil || tensor.IsNil(param.Data) || len(param.Data.Shape()) == 0 {
		return fmt.Errorf("Adam.Update: empty parameter data")
	}

	// Validate shapes match
	if !shapesEqual(param.Data.Shape(), param.Grad.Shape()) {
		return fmt.Errorf("Adam.Update: parameter and gradient shapes mismatch: %v vs %v", param.Data.Shape(), param.Grad.Shape())
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	key := param.Data.ID()

	// Get or create state for this parameter
	state, exists := a.state[key]
	if !exists {
		// Initialize state with zero tensors and pre-allocated scratch tensors
		shape := param.Data.Shape()
		state = &adamState{
			m:             tensor.New(tensor.DTFP32, shape),
			v:             tensor.New(tensor.DTFP32, shape),
			step:          0,
			scaledGrad1:   tensor.New(tensor.DTFP32, shape),
			scaledGrad2:   tensor.New(tensor.DTFP32, shape),
			gradSquared:   tensor.New(tensor.DTFP32, shape),
			mHat:          tensor.New(tensor.DTFP32, shape),
			vHat:          tensor.New(tensor.DTFP32, shape),
			sqrtVHat:      tensor.New(tensor.DTFP32, shape),
			epsilonTensor: tensor.New(tensor.DTFP32, shape),
			update:        tensor.New(tensor.DTFP32, shape),
		}
		a.state[key] = state
	}

	// Increment step counter
	state.step++

	// Compute bias correction coefficients
	beta1Power := math.Pow(a.beta1, float64(state.step))
	beta2Power := math.Pow(a.beta2, float64(state.step))
	biasCorrection1 := 1 - beta1Power
	biasCorrection2 := 1 - beta2Power

	// Update first moment estimate: m = beta1 * m + (1-beta1) * g
	// Use in-place MulScalar followed by AddScaled for efficiency
	state.m.MulScalar(nil, a.beta1)
	state.m.AddScaled(nil, param.Grad, 1-a.beta1)

	// Update second moment estimate: v = beta2 * v + (1-beta2) * g^2
	// Compute grad^2 using pre-allocated scratch tensor
	param.Grad.Multiply(state.gradSquared, param.Grad)
	state.v.MulScalar(nil, a.beta2)
	state.v.AddScaled(nil, state.gradSquared, 1-a.beta2)

	// Compute bias-corrected estimates: mHat = m / (1 - beta1^t), vHat = v / (1 - beta2^t)
	// Use pre-allocated scratch tensors with destination parameters
	state.m.MulScalar(state.mHat, 1.0/biasCorrection1)
	state.v.MulScalar(state.vHat, 1.0/biasCorrection2)

	// Compute sqrt(vHat) + epsilon using pre-allocated scratch tensors
	state.vHat.Sqrt(state.sqrtVHat)
	// Use Fill instead of Elements() loop for efficiency
	state.epsilonTensor.Fill(nil, a.epsilon)
	state.sqrtVHat.Add(nil, state.epsilonTensor)

	// Compute update: param = param - lr * mHat / (sqrt(vHat) + epsilon)
	// Use pre-allocated scratch tensor with destination parameters
	state.mHat.Divide(state.update, state.sqrtVHat)
	state.update.MulScalar(nil, a.lr)
	// Use AddScaled with negative learning rate for final update
	param.Data.AddScaled(nil, state.update, -1.0)

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

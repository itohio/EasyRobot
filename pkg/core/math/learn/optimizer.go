package learn

import (
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/itohio/EasyRobot/pkg/core/math/nn/layers"
	"github.com/itohio/EasyRobot/pkg/core/math/tensor"
)

// SGD implements Stochastic Gradient Descent optimizer.
// It implements nn.Optimizer interface.
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

// Adam implements Adaptive Moment Estimation optimizer.
// It implements nn.Optimizer interface.
// Adam combines advantages of AdaGrad and RMSProp by using both
// first and second moment estimates of gradients.
type Adam struct {
	lr      float32                // Learning rate
	beta1   float32                // Exponential decay rate for first moment estimates
	beta2   float32                // Exponential decay rate for second moment estimates
	epsilon float32                // Small constant for numerical stability
	mu      sync.Mutex             // Mutex for thread-safe state access
	state   map[uintptr]*adamState // Per-parameter state keyed by data pointer
}

// adamState holds per-parameter state for Adam optimizer.
type adamState struct {
	m    tensor.Tensor // First moment estimate
	v    tensor.Tensor // Second moment estimate
	step int           // Step counter for bias correction
}

// NewAdam creates a new Adam optimizer with the given hyperparameters.
// Default values (if not specified): lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8
func NewAdam(lr, beta1, beta2, epsilon float32) *Adam {
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
func (a *Adam) Update(param *layers.Parameter) error {
	if a == nil {
		return fmt.Errorf("Adam.Update: nil optimizer")
	}

	if param == nil {
		return fmt.Errorf("Adam.Update: nil parameter")
	}

	if !param.RequiresGrad {
		return nil // No gradient tracking
	}

	if len(param.Grad.Dim) == 0 {
		return nil // No gradient computed
	}

	if len(param.Data.Dim) == 0 {
		return fmt.Errorf("Adam.Update: empty parameter data")
	}

	// Validate shapes match
	if !shapesEqual(param.Data.Dim, param.Grad.Dim) {
		return fmt.Errorf("Adam.Update: parameter and gradient shapes mismatch: %v vs %v", param.Data.Dim, param.Grad.Dim)
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Use the data pointer as a stable key for this parameter
	// Even when Parameter struct is copied, the underlying slice points to the same array
	if len(param.Data.Data) == 0 {
		return fmt.Errorf("Adam.Update: parameter data slice is empty")
	}
	key := uintptr(unsafe.Pointer(&param.Data.Data[0]))

	// Get or create state for this parameter
	state, exists := a.state[key]
	if !exists {
		// Initialize state with zero tensors
		state = &adamState{
			m: tensor.Tensor{
				Dim:  make([]int, len(param.Data.Dim)),
				Data: make([]float32, param.Data.Size()),
			},
			v: tensor.Tensor{
				Dim:  make([]int, len(param.Data.Dim)),
				Data: make([]float32, param.Data.Size()),
			},
			step: 0,
		}
		copy(state.m.Dim, param.Data.Dim)
		copy(state.v.Dim, param.Data.Dim)
		a.state[key] = state
	}

	// Increment step counter
	state.step++

	// Compute bias correction coefficients
	beta1Power := float32(math.Pow(float64(a.beta1), float64(state.step)))
	beta2Power := float32(math.Pow(float64(a.beta2), float64(state.step)))
	biasCorrection1 := 1 - beta1Power
	biasCorrection2 := 1 - beta2Power

	// Update first and second moment estimates and apply parameter update
	for i := range param.Data.Data {
		g := param.Grad.Data[i]

		// Update biased first moment estimate
		state.m.Data[i] = a.beta1*state.m.Data[i] + (1-a.beta1)*g

		// Update biased second moment estimate
		state.v.Data[i] = a.beta2*state.v.Data[i] + (1-a.beta2)*g*g

		// Compute bias-corrected estimates
		mHat := state.m.Data[i] / biasCorrection1
		vHat := state.v.Data[i] / biasCorrection2

		// Update parameter
		param.Data.Data[i] -= a.lr * mHat / (float32(math.Sqrt(float64(vHat))) + a.epsilon)
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

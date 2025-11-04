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

	if len(param.Grad.Shape()) == 0 {
		return nil // No gradient computed
	}

	if len(param.Data.Shape()) == 0 {
		return fmt.Errorf("SGD.Update: empty parameter data")
	}

	// Validate shapes match
	if !shapesEqual(param.Data.Shape(), param.Grad.Shape()) {
		return fmt.Errorf("SGD.Update: parameter and gradient shapes mismatch: %v vs %v", param.Data.Shape(), param.Grad.Shape())
	}

	// SGD update: data = data - lr * grad
	// Use tensor operations: scale gradient by learning rate, then subtract from data
	scaledGrad := param.Grad.Clone()
	scaledGrad.Scale(s.lr)
	(&param.Data).Sub(*scaledGrad)

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

	if len(param.Grad.Shape()) == 0 {
		return nil // No gradient computed
	}

	if len(param.Data.Shape()) == 0 {
		return fmt.Errorf("Adam.Update: empty parameter data")
	}

	// Validate shapes match
	if !shapesEqual(param.Data.Shape(), param.Grad.Shape()) {
		return fmt.Errorf("Adam.Update: parameter and gradient shapes mismatch: %v vs %v", param.Data.Shape(), param.Grad.Shape())
	}

	a.mu.Lock()
	defer a.mu.Unlock()

	// Use the data pointer as a stable key for this parameter
	// Even when Parameter struct is copied, the underlying slice points to the same array
	data := param.Data.Data()
	if len(data) == 0 {
		return fmt.Errorf("Adam.Update: parameter data slice is empty")
	}
	key := uintptr(unsafe.Pointer(&data[0]))

	// Get or create state for this parameter
	state, exists := a.state[key]
	if !exists {
		// Initialize state with zero tensors
		shape := param.Data.Shape()
		state = &adamState{
			m:    tensor.New(tensor.DTFP32, shape),
			v:    tensor.New(tensor.DTFP32, shape),
			step: 0,
		}
		a.state[key] = state
	}

	// Increment step counter
	state.step++

	// Compute bias correction coefficients
	beta1Power := float32(math.Pow(float64(a.beta1), float64(state.step)))
	beta2Power := float32(math.Pow(float64(a.beta2), float64(state.step)))
	biasCorrection1 := 1 - beta1Power
	biasCorrection2 := 1 - beta2Power

	// Update first moment estimate: m = beta1 * m + (1-beta1) * g
	(&state.m).Scale(a.beta1)
	scaledGrad1 := param.Grad.Clone()
	scaledGrad1.Scale(1 - a.beta1)
	(&state.m).Add(*scaledGrad1)

	// Update second moment estimate: v = beta2 * v + (1-beta2) * g^2
	gradSquared := param.Grad.Clone()
	gradSquared.Mul(param.Grad)
	(&state.v).Scale(a.beta2)
	scaledGrad2 := gradSquared.Clone()
	scaledGrad2.Scale(1 - a.beta2)
	(&state.v).Add(*scaledGrad2)

	// Compute bias-corrected estimates: mHat = m / (1 - beta1^t), vHat = v / (1 - beta2^t)
	mHat := state.m.Clone()
	mHat.Scale(1.0 / biasCorrection1)
	vHat := state.v.Clone()
	vHat.Scale(1.0 / biasCorrection2)

	// Compute sqrt(vHat) + epsilon
	sqrtVHat := sqrtTensor(*vHat)
	epsilonTensor := tensor.FullLike(*sqrtVHat, a.epsilon)
	sqrtVHat.Add(*epsilonTensor)

	// Compute update: param = param - lr * mHat / (sqrt(vHat) + epsilon)
	update := mHat.Clone()
	update.Div(*sqrtVHat)
	update.Scale(a.lr)
	(&param.Data).Sub(*update)

	return nil
}

// sqrtTensor computes element-wise square root of a tensor.
// Since there's no tensor sqrt operation, this helper computes sqrt element-wise.
func sqrtTensor(t tensor.Tensor) *tensor.Tensor {
	result := t.Clone()
	data := result.Data()
	for i := range data {
		data[i] = float32(math.Sqrt(float64(data[i])))
	}
	return result
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

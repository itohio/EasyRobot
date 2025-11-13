package ekalman

import (
	"github.com/itohio/EasyRobot/x/math/filter"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// StateTransitionFunc defines the nonlinear state transition function.
// Returns: x_next = f(x, u, dt)
// If control input is not used, u will be nil.
type StateTransitionFunc func(x, u vec.Vector, dt float32) vec.Vector

// MeasurementFunc defines the nonlinear measurement function.
// Returns: z = h(x)
type MeasurementFunc func(x vec.Vector) vec.Vector

// StateJacobianFunc computes the Jacobian of the state transition function.
// Returns: F = ∂f/∂x (n x n matrix)
// If control input is not used, u will be nil.
type StateJacobianFunc func(x, u vec.Vector, dt float32) mat.Matrix

// MeasurementJacobianFunc computes the Jacobian of the measurement function.
// Returns: H = ∂h/∂x (m x n matrix)
type MeasurementJacobianFunc func(x vec.Vector) mat.Matrix

// EKF implements an Extended Kalman filter for nonlinear state estimation.
// It uses mat.Matrix and vec.Vector for all computations.
type EKF struct {
	// State
	x vec.Vector // State vector (n dimensions)

	// Covariances
	P mat.Matrix // State covariance (n x n)
	Q mat.Matrix // Process noise covariance (n x n)
	R mat.Matrix // Measurement noise covariance (m x m)

	// Nonlinear functions
	fFunc StateTransitionFunc // State transition function
	hFunc MeasurementFunc     // Measurement function

	// Jacobian functions (if nil, will use numerical Jacobians)
	fJacobian StateJacobianFunc       // State transition Jacobian
	hJacobian MeasurementJacobianFunc // Measurement Jacobian

	// Temporary matrices for computations
	K      mat.Matrix // Kalman gain (n x m)
	S      mat.Matrix // Innovation covariance (m x m)
	F      mat.Matrix // State transition Jacobian (n x n)
	H      mat.Matrix // Measurement Jacobian (m x n)
	tempP  mat.Matrix // Temporary for P_pred
	tempM  mat.Matrix // Temporary for matrix operations
	tempN  mat.Matrix // Temporary for matrix operations
	tempN2 mat.Matrix // Temporary for matrix operations (n x n)
	tempHP mat.Matrix // Temporary for H * P (m x n)
	tempHT mat.Matrix // Temporary for Hᵀ (n x m)
	tempPH mat.Matrix // Temporary for P * Hᵀ (n x m)
	tempV  vec.Vector // Temporary for vector operations
	tempV2 vec.Vector // Temporary for vector operations
	tempV3 vec.Vector // Temporary for vector operations

	// Dimensions
	n int // State dimension
	m int // Measurement dimension
	k int // Control input dimension (0 if not used)

	// Numerical Jacobian settings
	JacobianEpsilon float32 // Step size for numerical differentiation

	// Filter interface
	Input  vec.Vector // Measurement input
	Output vec.Vector // Estimated state output
	Target vec.Vector // Target state (optional)
}

const (
	// DefaultJacobianEpsilon is the default step size for numerical Jacobian computation
	DefaultJacobianEpsilon = 1e-5
)

// New creates a new Extended Kalman Filter with user-provided Jacobian functions.
// n: state dimension
// m: measurement dimension
// fFunc: state transition function f(x, u, dt)
// hFunc: measurement function h(x)
// fJacobian: Jacobian of state transition function (can be nil to use numerical)
// hJacobian: Jacobian of measurement function (can be nil to use numerical)
// Q: process noise covariance (n x n)
// R: measurement noise covariance (m x m)
func New(
	n, m int,
	fFunc StateTransitionFunc,
	hFunc MeasurementFunc,
	fJacobian StateJacobianFunc,
	hJacobian MeasurementJacobianFunc,
	Q, R mat.Matrix,
) *EKF {
	if len(Q) != n || (len(Q) > 0 && len(Q[0]) != n) {
		panic("ekf: Q must be n x n")
	}
	if len(R) != m || (len(R) > 0 && len(R[0]) != m) {
		panic("ekf: R must be m x m")
	}

	ekf := &EKF{
		n:               n,
		m:               m,
		k:               0,
		fFunc:           fFunc,
		hFunc:           hFunc,
		fJacobian:       fJacobian,
		hJacobian:       hJacobian,
		Q:               Q,
		R:               R,
		x:               vec.New(n),
		P:               mat.New(n, n),
		F:               mat.New(n, n),
		H:               mat.New(m, n),
		K:               mat.New(n, m),
		S:               mat.New(m, m),
		tempP:           mat.New(n, n),
		tempM:           mat.New(n, n),
		tempN:           mat.New(m, m),
		tempN2:          mat.New(n, n),
		tempHP:          mat.New(m, n),
		tempHT:          mat.New(n, m),
		tempPH:          mat.New(n, m),
		tempV:           vec.New(n),
		tempV2:          vec.New(n),
		tempV3:          vec.New(n),
		JacobianEpsilon: DefaultJacobianEpsilon,
		Input:           vec.New(m),
		Output:          vec.New(n),
		Target:          vec.New(n),
	}

	// Initialize P as identity
	ekf.P.Eye()
	ekf.x.FillC(0)

	return ekf
}

// NewWithControl creates a new Extended Kalman Filter with control input.
// n: state dimension
// m: measurement dimension
// k: control input dimension
// fFunc: state transition function f(x, u, dt)
// hFunc: measurement function h(x)
// fJacobian: Jacobian of state transition function (can be nil to use numerical)
// hJacobian: Jacobian of measurement function (can be nil to use numerical)
// Q: process noise covariance (n x n)
// R: measurement noise covariance (m x m)
func NewWithControl(
	n, m, k int,
	fFunc StateTransitionFunc,
	hFunc MeasurementFunc,
	fJacobian StateJacobianFunc,
	hJacobian MeasurementJacobianFunc,
	Q, R mat.Matrix,
) *EKF {
	if len(Q) != n || (len(Q) > 0 && len(Q[0]) != n) {
		panic("ekf: Q must be n x n")
	}
	if len(R) != m || (len(R) > 0 && len(R[0]) != m) {
		panic("ekf: R must be m x m")
	}

	ekf := &EKF{
		n:               n,
		m:               m,
		k:               k,
		fFunc:           fFunc,
		hFunc:           hFunc,
		fJacobian:       fJacobian,
		hJacobian:       hJacobian,
		Q:               Q,
		R:               R,
		x:               vec.New(n),
		P:               mat.New(n, n),
		F:               mat.New(n, n),
		H:               mat.New(m, n),
		K:               mat.New(n, m),
		S:               mat.New(m, m),
		tempP:           mat.New(n, n),
		tempM:           mat.New(n, n),
		tempN:           mat.New(m, m),
		tempN2:          mat.New(n, n),
		tempHP:          mat.New(m, n),
		tempHT:          mat.New(n, m),
		tempPH:          mat.New(n, m),
		tempV:           vec.New(n),
		tempV2:          vec.New(n),
		tempV3:          vec.New(n),
		JacobianEpsilon: DefaultJacobianEpsilon,
		Input:           vec.New(m),
		Output:          vec.New(n),
		Target:          vec.New(n),
	}

	// Initialize P as identity
	ekf.P.Eye()
	ekf.x.FillC(0)

	return ekf
}

// SetState sets the initial state vector.
func (e *EKF) SetState(x vec.Vector) *EKF {
	if len(x) != e.n {
		panic("ekf: state vector dimension mismatch")
	}
	copy(e.x, x)
	copy(e.Output, x)
	return e
}

// SetCovariance sets the initial state covariance matrix.
func (e *EKF) SetCovariance(P mat.Matrix) *EKF {
	if len(P) != e.n || (len(P) > 0 && len(P[0]) != e.n) {
		panic("ekf: covariance matrix dimension mismatch")
	}
	for i := range P {
		copy(e.P[i], P[i])
	}
	return e
}

// SetJacobianEpsilon sets the step size for numerical Jacobian computation.
func (e *EKF) SetJacobianEpsilon(epsilon float32) *EKF {
	e.JacobianEpsilon = epsilon
	return e
}

// Reset resets the filter state to zero.
func (e *EKF) Reset() filter.Filter {
	e.x.FillC(0)
	e.P.Eye()
	copy(e.Input, e.x)
	copy(e.Output, e.x)
	copy(e.Target, e.x)
	return e
}

// computeStateJacobian computes the Jacobian of the state transition function.
// Uses analytical Jacobian if available, otherwise numerical.
func (e *EKF) computeStateJacobian(x, u vec.Vector, dt float32) {
	if e.fJacobian != nil {
		// Use analytical Jacobian
		F := e.fJacobian(x, u, dt)
		for i := range F {
			copy(e.F[i], F[i])
		}
	} else {
		// Compute numerical Jacobian
		e.computeNumericalStateJacobian(x, u, dt)
	}
}

// computeNumericalStateJacobian computes the state transition Jacobian numerically.
func (e *EKF) computeNumericalStateJacobian(x, u vec.Vector, dt float32) {
	epsilon := e.JacobianEpsilon

	// Compute Jacobian using central differences
	for j := 0; j < e.n; j++ {
		// Perturb x[j]
		xPlus := vec.New(e.n)
		copy(xPlus, x)
		xPlus[j] += epsilon

		xMinus := vec.New(e.n)
		copy(xMinus, x)
		xMinus[j] -= epsilon

		fPlus := e.fFunc(xPlus, u, dt)
		fMinus := e.fFunc(xMinus, u, dt)

		// Central difference: ∂f/∂x[j] = (f(x+ε) - f(x-ε)) / (2ε)
		for i := 0; i < e.n; i++ {
			e.F[i][j] = (fPlus[i] - fMinus[i]) / (2.0 * epsilon)
		}
	}
}

// computeMeasurementJacobian computes the Jacobian of the measurement function.
// Uses analytical Jacobian if available, otherwise numerical.
func (e *EKF) computeMeasurementJacobian(x vec.Vector) {
	if e.hJacobian != nil {
		// Use analytical Jacobian
		H := e.hJacobian(x)
		for i := range H {
			copy(e.H[i], H[i])
		}
	} else {
		// Compute numerical Jacobian
		e.computeNumericalMeasurementJacobian(x)
	}
}

// computeNumericalMeasurementJacobian computes the measurement Jacobian numerically.
func (e *EKF) computeNumericalMeasurementJacobian(x vec.Vector) {
	epsilon := e.JacobianEpsilon

	// Compute Jacobian using central differences
	for j := 0; j < e.n; j++ {
		// Perturb x[j]
		xPlus := vec.New(e.n)
		copy(xPlus, x)
		xPlus[j] += epsilon

		xMinus := vec.New(e.n)
		copy(xMinus, x)
		xMinus[j] -= epsilon

		hPlus := e.hFunc(xPlus)
		hMinus := e.hFunc(xMinus)

		// Central difference: ∂h/∂x[j] = (h(x+ε) - h(x-ε)) / (2ε)
		for i := 0; i < e.m; i++ {
			e.H[i][j] = (hPlus[i] - hMinus[i]) / (2.0 * epsilon)
		}
	}
}

// zeroMatrix zeros out a matrix.
func zeroMatrix(m mat.Matrix) {
	for i := range m {
		for j := range m[i] {
			m[i][j] = 0
		}
	}
}

// Predict performs the prediction step without control input.
// Computes: x_pred = f(x, dt), P_pred = F * P * F^T + Q
func (e *EKF) Predict(dt float32) *EKF {
	// x_pred = f(x, nil, dt) (nonlinear prediction)
	xPred := e.fFunc(e.x, nil, dt)
	copy(e.x, xPred)

	// Compute Jacobian F = ∂f/∂x at current state
	e.computeStateJacobian(e.x, nil, dt)

	// P_pred = F * P * F^T + Q
	// First compute F * P
	zeroMatrix(e.tempM)
	e.tempM.Mul(e.F, e.P)

	// Then compute (F * P) * F^T
	zeroMatrix(e.tempP)
	e.tempP.Transpose(e.F)
	zeroMatrix(e.P)
	e.P.Mul(e.tempM, e.tempP)

	// Add Q: P_pred = F * P * F^T + Q
	e.P.Add(e.Q)
	copy(e.Output, e.x)

	return e
}

// PredictWithControl performs the prediction step with control input.
// Computes: x_pred = f(x, u, dt), P_pred = F * P * F^T + Q
func (e *EKF) PredictWithControl(u vec.Vector, dt float32) *EKF {
	if u == nil || len(u) != e.k {
		panic("ekf: control input dimension mismatch")
	}

	// x_pred = f(x, u, dt) (nonlinear prediction)
	xPred := e.fFunc(e.x, u, dt)
	copy(e.x, xPred)

	// Compute Jacobian F = ∂f/∂x at current state
	e.computeStateJacobian(e.x, u, dt)

	// P_pred = F * P * F^T + Q
	// First compute F * P
	zeroMatrix(e.tempM)
	e.tempM.Mul(e.F, e.P)

	// Then compute (F * P) * F^T
	zeroMatrix(e.tempP)
	e.tempP.Transpose(e.F)
	zeroMatrix(e.P)
	e.P.Mul(e.tempM, e.tempP)

	// Add Q: P_pred = F * P * F^T + Q
	e.P.Add(e.Q)
	copy(e.Output, e.x)

	return e
}

// UpdateMeasurement performs the measurement update step.
// Computes Kalman gain and updates state and covariance.
// z: measurement vector (m dimensions)
func (e *EKF) UpdateMeasurement(z vec.Vector) *EKF {
	if len(z) != e.m {
		panic("ekf: measurement vector dimension mismatch")
	}

	// Copy measurement to Input for Filter interface
	copy(e.Input, z)

	// Predicted measurement: z_pred = h(x_pred)
	zPred := e.hFunc(e.x)

	// Innovation: y = z - z_pred
	innovation := vec.New(e.m)
	copy(innovation, z)
	innovation.Sub(zPred)

	// Compute Jacobian H = ∂h/∂x at predicted state
	e.computeMeasurementJacobian(e.x)

	// Innovation covariance: S = H * P_pred * H^T + R
	zeroMatrix(e.tempHP)
	e.tempHP.Mul(e.H, e.P)

	zeroMatrix(e.tempHT)
	e.tempHT.Transpose(e.H)

	zeroMatrix(e.S)
	e.S.Mul(e.tempHP, e.tempHT)

	// Add R: S = H * P_pred * H^T + R
	e.S.Add(e.R)

	// Kalman gain: K = P_pred * H^T * S^-1
	zeroMatrix(e.tempPH)
	e.tempPH.Mul(e.P, e.tempHT)

	// Compute S^-1
	zeroMatrix(e.tempN)
	if err := e.S.Inverse(e.tempN); err != nil {
		// If inversion fails, use identity (fallback)
		e.tempN.Eye()
	}

	// K = (P_pred * H^T) * S^-1
	zeroMatrix(e.K)
	e.K.Mul(e.tempPH, e.tempN)

	// Update state: x = x_pred + K * y
	e.tempV.FillC(0)
	e.K.MulVec(innovation, e.tempV)
	e.x.Add(e.tempV)

	// Update covariance: P = (I - K * H) * P_pred
	// First compute K * H
	zeroMatrix(e.tempM)
	e.tempM.Mul(e.K, e.H)

	// Compute I - K * H
	e.tempN2.Eye()
	e.tempN2.Sub(e.tempM)

	// P = (I - K * H) * P_pred
	zeroMatrix(e.tempM)
	e.tempM.Mul(e.tempN2, e.P)
	copy(e.P, e.tempM)

	// Copy state to Output for Filter interface
	copy(e.Output, e.x)

	return e
}

// Update implements the Filter interface.
// This method performs prediction and expects measurement to be set in Input.
func (e *EKF) Update(timestep float32) filter.Filter {
	// Predict step
	e.Predict(timestep)

	// If measurement is available (non-zero), perform update
	hasMeasurement := false
	for i := range e.Input {
		if e.Input[i] != 0 {
			hasMeasurement = true
			break
		}
	}

	if hasMeasurement {
		e.UpdateMeasurement(e.Input)
	}

	return e
}

// GetInput returns the measurement input vector.
func (e *EKF) GetInput() vec.Vector {
	return e.Input
}

// GetOutput returns the estimated state vector.
func (e *EKF) GetOutput() vec.Vector {
	return e.Output
}

// GetTarget returns the target state vector.
func (e *EKF) GetTarget() vec.Vector {
	return e.Target
}

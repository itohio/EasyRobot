package kalman

import (
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// Kalman implements a linear Kalman filter for state estimation.
// It uses mat.Matrix and vec.Vector for all computations.
type Kalman struct {
	// State
	x vec.Vector // State vector (n dimensions)

	// Covariances
	P mat.Matrix // State covariance (n x n)
	Q mat.Matrix // Process noise covariance (n x n)
	R mat.Matrix // Measurement noise covariance (m x m)

	// Matrices
	F mat.Matrix // State transition matrix (n x n)
	H mat.Matrix // Measurement matrix (m x n)
	B mat.Matrix // Control input matrix (n x k) - nil if not used

	// Temporary matrices for computations
	K      mat.Matrix // Kalman gain (n x m)
	S      mat.Matrix // Innovation covariance (m x m)
	tempP  mat.Matrix // Temporary for transposes (n x n)
	tempM  mat.Matrix // Temporary for P-related products (n x n)
	tempHP mat.Matrix // Temporary for H*P (m x n)
	tempHT mat.Matrix // Temporary for H^T (n x m)
	tempN  mat.Matrix // Temporary for matrix operations
	tempN2 mat.Matrix // Temporary for matrix operations (n x n)
	tempV  vec.Vector // Temporary for vector operations
	tempV2 vec.Vector // Temporary for vector operations (control input size)

	// Dimensions
	n int // State dimension
	m int // Measurement dimension
	k int // Control input dimension (0 if not used)

	// Filter interface
	inputVec  vec.Vector // Measurement input
	outputVec vec.Vector // Estimated state output
	targetVec vec.Vector // Target state (optional)
}

// New creates a new Kalman filter without control input.
// n: state dimension
// m: measurement dimension
// F: state transition matrix (n x n)
// H: measurement matrix (m x n)
// Q: process noise covariance (n x n)
// R: measurement noise covariance (m x m)
func New(n, m int, F, H, Q, R mat.Matrix) *Kalman {
	if len(F) != n || (len(F) > 0 && len(F[0]) != n) {
		panic("kalman: F must be n x n")
	}
	if len(H) != m || (len(H) > 0 && len(H[0]) != n) {
		panic("kalman: H must be m x n")
	}
	if len(Q) != n || (len(Q) > 0 && len(Q[0]) != n) {
		panic("kalman: Q must be n x n")
	}
	if len(R) != m || (len(R) > 0 && len(R[0]) != m) {
		panic("kalman: R must be m x m")
	}

	kf := &Kalman{
		n:         n,
		m:         m,
		k:         0,
		F:         F,
		H:         H,
		B:         nil,
		Q:         Q,
		R:         R,
		x:         vec.New(n),
		P:         mat.New(n, n),
		K:         mat.New(n, m),
		S:         mat.New(m, m),
		tempP:     mat.New(n, n),
		tempM:     mat.New(n, n),
		tempHP:    mat.New(m, n),
		tempHT:    mat.New(n, m),
		tempN:     mat.New(m, m),
		tempN2:    mat.New(n, n),
		tempV:     vec.New(n),
		inputVec:  vec.New(m),
		outputVec: vec.New(n),
		targetVec: vec.New(n),
	}

	// Initialize P as identity
	kf.P.Eye()
	kf.x.FillC(0)

	return kf
}

// NewWithControl creates a new Kalman filter with control input.
// n: state dimension
// m: measurement dimension
// k: control input dimension
// F: state transition matrix (n x n)
// H: measurement matrix (m x n)
// B: control input matrix (n x k)
// Q: process noise covariance (n x n)
// R: measurement noise covariance (m x m)
func NewWithControl(n, m, k int, F, H, B, Q, R mat.Matrix) *Kalman {
	if len(F) != n || (len(F) > 0 && len(F[0]) != n) {
		panic("kalman: F must be n x n")
	}
	if len(H) != m || (len(H) > 0 && len(H[0]) != n) {
		panic("kalman: H must be m x n")
	}
	if len(B) != n || (len(B) > 0 && len(B[0]) != k) {
		panic("kalman: B must be n x k")
	}
	if len(Q) != n || (len(Q) > 0 && len(Q[0]) != n) {
		panic("kalman: Q must be n x n")
	}
	if len(R) != m || (len(R) > 0 && len(R[0]) != m) {
		panic("kalman: R must be m x m")
	}

	kf := &Kalman{
		n:         n,
		m:         m,
		k:         k,
		F:         F,
		H:         H,
		B:         B,
		Q:         Q,
		R:         R,
		x:         vec.New(n),
		P:         mat.New(n, n),
		K:         mat.New(n, m),
		S:         mat.New(m, m),
		tempP:     mat.New(n, n),
		tempM:     mat.New(n, n),
		tempHP:    mat.New(m, n),
		tempHT:    mat.New(n, m),
		tempN:     mat.New(m, m),
		tempN2:    mat.New(n, n),
		tempV:     vec.New(n),
		tempV2:    vec.New(k),
		inputVec:  vec.New(m),
		outputVec: vec.New(n),
		targetVec: vec.New(n),
	}

	// Initialize P as identity
	kf.P.Eye()
	kf.x.FillC(0)

	return kf
}

// zeroMatrix zeros out a matrix.
func zeroMatrix(m mat.Matrix) {
	for i := range m {
		for j := range m[i] {
			m[i][j] = 0
		}
	}
}

// SetState sets the initial state vector.
func (k *Kalman) SetState(x vec.Vector) *Kalman {
	if len(x) != k.n {
		panic("kalman: state vector dimension mismatch")
	}
	copy(k.x, x)
	copy(k.outputVec, x)
	return k
}

// SetCovariance sets the initial state covariance matrix.
func (k *Kalman) SetCovariance(P mat.Matrix) *Kalman {
	if len(P) != k.n || (len(P) > 0 && len(P[0]) != k.n) {
		panic("kalman: covariance matrix dimension mismatch")
	}
	for i := range P {
		copy(k.P[i], P[i])
	}
	return k
}

// Reset resets the filter state to zero.
func (k *Kalman) Reset() {
	k.x.FillC(0)
	k.P.Eye()
	copy(k.inputVec, k.x)
	copy(k.outputVec, k.x)
	copy(k.targetVec, k.x)
}

// Predict performs the prediction step without control input.
// Computes: x_pred = F * x, P_pred = F * P * F^T + Q
func (k *Kalman) Predict() *Kalman {
	// x_pred = F * x
	k.tempV.FillC(0)
	k.F.MulVec(k.x, k.tempV)
	copy(k.x, k.tempV)

	// P_pred = F * P * F^T + Q
	// First compute F * P
	zeroMatrix(k.tempM)
	k.tempM.Mul(k.F, k.P)

	// Then compute (F * P) * F^T
	zeroMatrix(k.tempP)
	k.tempP.Transpose(k.F)
	zeroMatrix(k.P)
	k.P.Mul(k.tempM, k.tempP)

	// Add Q: P_pred = F * P * F^T + Q
	k.P.Add(k.Q)

	// Expose predicted state via Filter interface
	copy(k.outputVec, k.x)

	return k
}

// PredictWithControl performs the prediction step with control input.
// Computes: x_pred = F * x + B * u, P_pred = F * P * F^T + Q
func (k *Kalman) PredictWithControl(u vec.Vector) *Kalman {
	if k.B == nil {
		panic("kalman: control matrix B not initialized")
	}
	if len(u) != k.k {
		panic("kalman: control input dimension mismatch")
	}

	// x_pred = F * x
	k.tempV.FillC(0)
	k.F.MulVec(k.x, k.tempV)

	// x_pred = F * x + B * u
	k.tempV2.FillC(0)
	k.B.MulVec(u, k.tempV2)
	k.tempV.Add(k.tempV2)
	copy(k.x, k.tempV)

	// P_pred = F * P * F^T + Q
	// First compute F * P
	zeroMatrix(k.tempM)
	k.tempM.Mul(k.F, k.P)

	// Then compute (F * P) * F^T
	zeroMatrix(k.tempP)
	k.tempP.Transpose(k.F)
	zeroMatrix(k.P)
	k.P.Mul(k.tempM, k.tempP)

	// Add Q: P_pred = F * P * F^T + Q
	k.P.Add(k.Q)

	// Expose predicted state via Filter interface
	copy(k.outputVec, k.x)

	return k
}

// UpdateMeasurement performs the measurement update step.
// This is a separate method to avoid name conflict with Filter interface.
func (k *Kalman) UpdateMeasurement(z vec.Vector) *Kalman {
	return k.update(z)
}

// update is the internal implementation of the update step.
func (k *Kalman) update(z vec.Vector) *Kalman {
	if len(z) != k.m {
		panic("kalman: measurement vector dimension mismatch")
	}

	// Copy measurement to Input for Filter interface
	copy(k.inputVec, z)

	// Innovation: y = z - H * x_pred
	// First compute H * x_pred
	k.tempV.FillC(0)
	k.H.MulVec(k.x, k.tempV)

	// y = z - H * x_pred
	innovation := vec.New(k.m)
	copy(innovation, z)
	innovation.Sub(k.tempV)

	// Innovation covariance: S = H * P_pred * H^T + R
	// First compute H * P_pred
	zeroMatrix(k.tempHP)
	k.tempHP.Mul(k.H, k.P)

	// Then compute (H * P_pred) * H^T
	zeroMatrix(k.tempHT)
	k.tempHT.Transpose(k.H)
	zeroMatrix(k.S)
	k.S.Mul(k.tempHP, k.tempHT)

	// Add R: S = H * P_pred * H^T + R
	k.S.Add(k.R)

	// Kalman gain: K = P_pred * H^T * S^-1
	// First compute P_pred * H^T
	zeroMatrix(k.tempM)
	k.tempM.Mul(k.P, k.tempHT)

	// Compute S^-1
	zeroMatrix(k.tempN)
	if err := k.S.Inverse(k.tempN); err != nil {
		// If inversion fails, use identity (fallback)
		k.tempN.Eye()
	}

	// K = (P_pred * H^T) * S^-1
	zeroMatrix(k.K)
	k.K.Mul(k.tempM, k.tempN)

	// Update state: x = x_pred + K * y
	k.tempV.FillC(0)
	k.K.MulVec(innovation, k.tempV)
	k.x.Add(k.tempV)

	// Update covariance: P = (I - K * H) * P_pred
	// First compute K * H
	zeroMatrix(k.tempM)
	k.tempM.Mul(k.K, k.H)

	// Compute I - K * H
	k.tempN2.Eye()
	k.tempN2.Sub(k.tempM)

	// P = (I - K * H) * P_pred
	zeroMatrix(k.tempM)
	k.tempM.Mul(k.tempN2, k.P)
	copy(k.P, k.tempM)

	// Copy state to Output for Filter interface
	copy(k.outputVec, k.x)

	return k
}

// Update implements the Filter interface.
// This method performs prediction and update with the given measurement.
func (k *Kalman) Update(timestep float32, measurement vec.Vector) {
	// Predict step
	k.Predict()

	// If measurement is provided and non-zero, perform update
	if measurement != nil {
		hasMeasurement := false
		for i := range measurement {
			if measurement[i] != 0 {
				hasMeasurement = true
				break
			}
		}

		if hasMeasurement {
			k.UpdateMeasurement(measurement)
		}
	}

	// Update output
	copy(k.outputVec, k.x)
}

// Input returns the measurement input vector.
func (k *Kalman) Input() vec.Vector {
	return k.inputVec
}

// Output returns the estimated state vector.
func (k *Kalman) Output() vec.Vector {
	return k.outputVec
}

// GetTarget returns the target state vector.
func (k *Kalman) GetTarget() vec.Vector {
	return k.targetVec
}

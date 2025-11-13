package ekalman

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// TestEKF_SimplePendulum tests EKF with a simple pendulum model.
func TestEKF_SimplePendulum(t *testing.T) {
	// Nonlinear pendulum: θ'' = -(g/L)*sin(θ)
	// State: [angle, angular_velocity]
	// Measurement: angle (could be nonlinear, but using linear for simplicity)
	n, m := 2, 1
	dt := float32(0.01)

	// State transition: θ' = θ + ω*dt, ω' = ω - (g/L)*sin(θ)*dt
	fFunc := func(x, u vec.Vector, dt float32) vec.Vector {
		theta := x[0]
		omega := x[1]
		g := float32(9.81)
		L := float32(1.0)

		next := vec.New(2)
		next[0] = theta + omega*dt
		next[1] = omega - (g/L)*math32.Sin(theta)*dt
		return next
	}

	// Measurement: h(x) = angle
	hFunc := func(x vec.Vector) vec.Vector {
		z := vec.New(1)
		z[0] = x[0]
		return z
	}

	// Jacobian of state transition
	fJacobian := func(x, u vec.Vector, dt float32) mat.Matrix {
		theta := x[0]
		g := float32(9.81)
		L := float32(1.0)

		F := mat.New(2, 2)
		F[0][0] = 1
		F[0][1] = dt
		F[1][0] = -(g / L) * math32.Cos(theta) * dt
		F[1][1] = 1
		return F
	}

	// Jacobian of measurement
	hJacobian := func(x vec.Vector) mat.Matrix {
		H := mat.New(1, 2)
		H[0][0] = 1 // measure angle
		H[0][1] = 0
		return H
	}

	// Process noise
	Q := mat.New(2, 2)
	Q.Eye()
	Q.MulC(0.01)

	// Measurement noise
	R := mat.New(1, 1)
	R[0][0] = 0.1

	ekf := New(n, m, fFunc, hFunc, fJacobian, hJacobian, Q, R)
	ekf.SetState(vec.NewFrom(0.1, 0)) // Start at small angle

	// Initial covariance
	initialP := mat.New(2, 2)
	initialP.Eye()
	initialP.MulC(0.1)
	ekf.SetCovariance(initialP)

	// Predict
	ekf.Predict(dt)
	state := ekf.GetOutput()

	// State should have advanced
	if state[0] < -1 || state[0] > 1 {
		t.Errorf("Expected angle reasonable, got %f", state[0])
	}

	// Update with measurement
	measurement := vec.NewFrom(0.09) // slightly smaller angle
	ekf.UpdateMeasurement(measurement)
	state = ekf.GetOutput()

	// State should be updated toward measurement
	if state[0] < -1 || state[0] > 1 {
		t.Errorf("Expected angle reasonable after update, got %f", state[0])
	}
}

// TestEKF_NumericalJacobian tests EKF with numerical Jacobian computation.
func TestEKF_NumericalJacobian(t *testing.T) {
	n, m := 2, 1
	dt := float32(0.01)

	// Simple nonlinear function
	fFunc := func(x, u vec.Vector, dt float32) vec.Vector {
		next := vec.New(2)
		next[0] = x[0] + x[1]*dt
		next[1] = x[1] - x[0]*x[0]*dt // nonlinear: x[0]^2
		return next
	}

	hFunc := func(x vec.Vector) vec.Vector {
		z := vec.New(1)
		z[0] = x[0] * x[0] // nonlinear measurement
		return z
	}

	// No Jacobian functions provided - will use numerical
	Q := mat.New(2, 2)
	Q.Eye()
	Q.MulC(0.01)

	R := mat.New(1, 1)
	R[0][0] = 0.1

	ekf := New(n, m, fFunc, hFunc, nil, nil, Q, R)
	ekf.SetState(vec.NewFrom(1.0, 0.5))
	ekf.SetCovariance(mat.New(2, 2))
	ekf.P.Eye()
	ekf.P.MulC(0.1)

	// Predict should work with numerical Jacobian
	ekf.Predict(dt)
	state := ekf.GetOutput()

	if state[0] < -10 || state[0] > 10 {
		t.Errorf("Expected state reasonable, got %f", state[0])
	}

	// Update should work with numerical Jacobian
	measurement := vec.NewFrom(1.2)
	ekf.UpdateMeasurement(measurement)
	state = ekf.GetOutput()

	if state[0] < -10 || state[0] > 10 {
		t.Errorf("Expected state reasonable after update, got %f", state[0])
	}
}

// TestEKF_WithControl tests EKF with control input.
func TestEKF_WithControl(t *testing.T) {
	n, m, k := 2, 1, 1
	dt := float32(0.01)

	// State transition with control
	fFunc := func(x, u vec.Vector, dt float32) vec.Vector {
		next := vec.New(2)
		next[0] = x[0] + x[1]*dt
		control := float32(0)
		if u != nil && len(u) > 0 {
			control = u[0]
		}
		next[1] = x[1] + control*dt
		return next
	}

	hFunc := func(x vec.Vector) vec.Vector {
		z := vec.New(1)
		z[0] = x[0]
		return z
	}

	fJacobian := func(x, u vec.Vector, dt float32) mat.Matrix {
		F := mat.New(2, 2)
		F[0][0] = 1
		F[0][1] = dt
		F[1][0] = 0
		F[1][1] = 1
		return F
	}

	hJacobian := func(x vec.Vector) mat.Matrix {
		H := mat.New(1, 2)
		H[0][0] = 1
		H[0][1] = 0
		return H
	}

	Q := mat.New(2, 2)
	Q.Eye()
	Q.MulC(0.01)

	R := mat.New(1, 1)
	R[0][0] = 0.1

	ekf := NewWithControl(n, m, k, fFunc, hFunc, fJacobian, hJacobian, Q, R)
	ekf.SetState(vec.NewFrom(0, 0))

	// Predict with control
	control := vec.NewFrom(1.0)
	ekf.PredictWithControl(control, dt)

	// State should advance with control
	state := ekf.GetOutput()
	expectedVelocity := control[0] * dt
	if diff := math32.Abs(state[1] - expectedVelocity); diff > 1e-3 {
		t.Errorf("Expected velocity ~%0.2f after control, got %f", expectedVelocity, state[1])
	}
}

// TestEKF_FilterInterface tests Filter interface implementation.
func TestEKF_FilterInterface(t *testing.T) {
	n, m := 1, 1

	fFunc := func(x, u vec.Vector, dt float32) vec.Vector {
		next := vec.New(1)
		next[0] = x[0]
		return next
	}

	hFunc := func(x vec.Vector) vec.Vector {
		z := vec.New(1)
		z[0] = x[0]
		return z
	}

	fJacobian := func(x, u vec.Vector, dt float32) mat.Matrix {
		F := mat.New(1, 1)
		F[0][0] = 1
		return F
	}

	hJacobian := func(x vec.Vector) mat.Matrix {
		H := mat.New(1, 1)
		H[0][0] = 1
		return H
	}

	Q := mat.New(1, 1)
	Q.Eye()
	Q.MulC(0.01)

	R := mat.New(1, 1)
	R[0][0] = 0.1

	ekf := New(n, m, fFunc, hFunc, fJacobian, hJacobian, Q, R)

	// Test Filter interface methods
	input := ekf.GetInput()
	output := ekf.GetOutput()
	target := ekf.GetTarget()

	if len(input) != m {
		t.Errorf("Expected input length %d, got %d", m, len(input))
	}
	if len(output) != n {
		t.Errorf("Expected output length %d, got %d", n, len(output))
	}
	if len(target) != n {
		t.Errorf("Expected target length %d, got %d", n, len(target))
	}

	// Test Update method (Filter interface)
	copy(ekf.GetInput(), vec.NewFrom(1.0))
	ekf.Update(0.01) // timestep

	// State should have been updated
	state := ekf.GetOutput()
	if state[0] < -10 || state[0] > 10 {
		t.Errorf("Expected state reasonable after Update, got %f", state[0])
	}
}

// TestEKF_Reset tests filter reset functionality.
func TestEKF_Reset(t *testing.T) {
	n, m := 2, 1

	fFunc := func(x, u vec.Vector, dt float32) vec.Vector {
		next := vec.New(2)
		next[0] = x[0] + dt
		next[1] = x[1] + dt
		return next
	}

	hFunc := func(x vec.Vector) vec.Vector {
		z := vec.New(1)
		z[0] = x[0]
		return z
	}

	fJacobian := func(x, u vec.Vector, dt float32) mat.Matrix {
		F := mat.New(2, 2)
		F.Eye()
		return F
	}

	hJacobian := func(x vec.Vector) mat.Matrix {
		H := mat.New(1, 2)
		H[0][0] = 1
		H[0][1] = 0
		return H
	}

	Q := mat.New(2, 2)
	Q.Eye()
	Q.MulC(0.01)

	R := mat.New(1, 1)
	R[0][0] = 0.1

	ekf := New(n, m, fFunc, hFunc, fJacobian, hJacobian, Q, R)
	ekf.SetState(vec.NewFrom(1, 1))

	// Update state
	ekf.Predict(0.01)
	ekf.UpdateMeasurement(vec.NewFrom(2))

	// Reset
	ekf.Reset()

	// State should be zero after reset
	state := ekf.GetOutput()
	for i := range state {
		if state[i] != 0 {
			t.Errorf("Expected state[%d] = 0 after reset, got %f", i, state[i])
		}
	}

	// Covariance should be identity
	P := ekf.P
	for i := range P {
		for j := range P[i] {
			expected := float32(0)
			if i == j {
				expected = 1
			}
			if P[i][j] != expected {
				t.Errorf("Expected P[%d][%d] = %f after reset, got %f", i, j, expected, P[i][j])
			}
		}
	}
}

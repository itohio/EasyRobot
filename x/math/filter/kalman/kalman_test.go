package kalman

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/filter"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// TestKalmanFilter_Simple1D tests a simple 1D position tracking filter.
func TestKalmanFilter_Simple1D(t *testing.T) {
	// State: [position, velocity]
	// Measurement: [position]
	n, m := 2, 1

	// State transition: position = position + velocity*dt, velocity = velocity
	// For dt = 1:
	F := mat.New(2, 2,
		1, 1, // position update
		0, 1, // velocity constant
	)

	// Measurement: we only measure position
	H := mat.New(1, 2,
		1, 0, // measure position only
	)

	// Process noise: small uncertainty in velocity
	Q := mat.New(2, 2,
		0.01, 0,
		0, 0.01,
	)

	// Measurement noise: some uncertainty in position measurement
	R := mat.New(1, 1,
		0.1,
	)

	filter := New(n, m, F, H, Q, R)

	// Initial state: at position 0, velocity 1
	initialState := vec.NewFrom(0, 1)
	filter.SetState(initialState)

	// Initial covariance: some uncertainty
	initialP := mat.New(2, 2,
		1, 0,
		0, 1,
	)
	filter.SetCovariance(initialP)

	// Predict: state should advance
	filter.Predict()
	state := filter.Output()

	// After prediction: position should be 1 (0 + 1), velocity should be 1
	if state[0] < 0.9 || state[0] > 1.1 {
		t.Errorf("Expected position ~1 after prediction, got %f", state[0])
	}
	if state[1] < 0.9 || state[1] > 1.1 {
		t.Errorf("Expected velocity ~1 after prediction, got %f", state[1])
	}

	// Update with measurement: position = 1.0
	measurement := vec.NewFrom(1.0)
	filter.UpdateMeasurement(measurement)
	state = filter.Output()

	// State should be updated toward measurement
	if state[0] < 0.8 || state[0] > 1.2 {
		t.Errorf("Expected position ~1 after update, got %f", state[0])
	}
}

// TestKalmanFilter_PredictUpdate tests predict and update cycle.
func TestKalmanFilter_PredictUpdate(t *testing.T) {
	// Simple 1D position filter
	n, m := 1, 1

	// State transition: position stays the same
	F := mat.New(1, 1, 1)
	H := mat.New(1, 1, 1)
	Q := mat.New(1, 1, 0.01)
	R := mat.New(1, 1, 0.1)

	filter := New(n, m, F, H, Q, R)
	filter.SetState(vec.NewFrom(0))
	filter.SetCovariance(mat.New(1, 1, 1))

	// Predict
	filter.Predict()

	// Update with measurement
	measurement := vec.NewFrom(1.0)
	filter.UpdateMeasurement(measurement)

	// State should move toward measurement
	state := filter.Output()
	if state[0] < 0.5 || state[0] > 1.0 {
		t.Errorf("Expected state between 0.5 and 1.0, got %f", state[0])
	}
}

// TestKalmanFilter_Reset tests filter reset functionality.
func TestKalmanFilter_Reset(t *testing.T) {
	n, m := 2, 1

	F := mat.New(2, 2, 1, 1, 0, 1)
	H := mat.New(1, 2, 1, 0)
	Q := mat.New(2, 2, 0.01, 0, 0, 0.01)
	R := mat.New(1, 1, 0.1)

	filter := New(n, m, F, H, Q, R)
	filter.SetState(vec.NewFrom(1, 1))

	// Update state
	filter.Predict()
	filter.UpdateMeasurement(vec.NewFrom(2))

	// Reset
	filter.Reset()

	// State should be zero after reset
	state := filter.Output()
	for i := range state {
		if state[i] != 0 {
			t.Errorf("Expected state[%d] = 0 after reset, got %f", i, state[i])
		}
	}

	// Covariance should be identity
	P := filter.P
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

// TestKalmanFilter_FilterInterface tests Filter interface implementation.
func TestKalmanFilter_FilterInterface(t *testing.T) {
	n, m := 1, 1

	F := mat.New(1, 1, 1)
	H := mat.New(1, 1, 1)
	Q := mat.New(1, 1, 0.01)
	R := mat.New(1, 1, 0.1)

	filter := New(n, m, F, H, Q, R)

	// Test Filter interface methods
	input := filter.Input()
	output := filter.Output()
	target := filter.GetTarget()

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
	measurement := vec.NewFrom(1.0)
	filter.Update(1.0, measurement) // timestep and measurement

	// State should have been updated
	state := filter.Output()
	if state[0] < 0.5 || state[0] > 1.0 {
		t.Errorf("Expected state between 0.5 and 1.0 after Update, got %f", state[0])
	}
}

// TestKalmanFilter_WithControl tests Kalman filter with control input.
func TestKalmanFilter_WithControl(t *testing.T) {
	// State: [position]
	// Control: [acceleration]
	n, m, k := 1, 1, 1

	// State transition: position = position + velocity (with dt=1)
	// Control: position += acceleration
	F := mat.New(1, 1, 1)
	B := mat.New(1, 1, 1) // control affects position directly
	H := mat.New(1, 1, 1)
	Q := mat.New(1, 1, 0.01)
	R := mat.New(1, 1, 0.1)

	filter := NewWithControl(n, m, k, F, H, B, Q, R)
	filter.SetState(vec.NewFrom(0))

	// Predict with control input
	control := vec.NewFrom(1.0) // acceleration = 1
	filter.PredictWithControl(control)

	// State should advance
	state := filter.Output()
	if state[0] < 0.9 || state[0] > 1.1 {
		t.Errorf("Expected position ~1 after control prediction, got %f", state[0])
	}
}

func TestKalmanFilterInterface(t *testing.T) {
	var _ filter.Filter[vec.Vector, vec.Vector] = (*Kalman)(nil)
}

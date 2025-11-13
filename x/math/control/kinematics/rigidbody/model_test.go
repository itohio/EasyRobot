package rigidbody

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func TestForwardProducesWrench(t *testing.T) {
	cfg := Config{
		LinearGain:  vec.Vector3D{2, 2, 2},
		AngularGain: vec.Vector3D{3, 3, 3},
		MaxForce:    vec.Vector3D{100, 100, 100},
		MaxTorque:   vec.Vector3D{50, 50, 50},
	}
	inertia := mat.Matrix3x3{
		{0.6, 0, 0},
		{0, 0.8, 0},
		{0, 0, 1.1},
	}
	model, err := NewModel(1.5, inertia, cfg)
	if err != nil {
		t.Fatalf("NewModel failed: %v", err)
	}

	state := mat.New(stateSize, 1)
	state[0][0] = 1
	state[3][0] = 0.5
	dest := mat.New(stateSize, 1)

	if err := model.Forward(state, dest, nil); err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	if math32.Abs(dest[0][0]-3) > 1e-5 {
		t.Fatalf("expected force x 3, got %f", dest[0][0])
	}
	if math32.Abs(dest[3][0]-0.9) > 1e-5 {
		t.Fatalf("expected torque x 0.9, got %f", dest[3][0])
	}
}

func TestBackwardUsesError(t *testing.T) {
	cfg := Config{
		LinearGain:  vec.Vector3D{1, 1, 1},
		AngularGain: vec.Vector3D{2, 2, 2},
		MaxForce:    vec.Vector3D{10, 10, 10},
		MaxTorque:   vec.Vector3D{6, 6, 6},
	}
	inertia := mat.Matrix3x3{
		{1, 0, 0},
		{0, 1.2, 0},
		{0, 0, 1.5},
	}
	model, err := NewModel(1, inertia, cfg)
	if err != nil {
		t.Fatalf("NewModel failed: %v", err)
	}

	state := mat.New(stateSize, 1)
	state[0][0] = 0.5
	state[3][0] = 0.25
	dest := mat.New(stateSize, 1)
	dest[0][0] = 1.0
	dest[3][0] = 0.5
	controls := mat.New(stateSize, 1)

	if err := model.Backward(state, dest, controls); err != nil {
		t.Fatalf("Backward failed: %v", err)
	}

	if math32.Abs(controls[0][0]-0.5) > 1e-5 {
		t.Fatalf("expected force delta 0.5, got %f", controls[0][0])
	}
	if math32.Abs(controls[3][0]-0.5) > 1e-5 {
		t.Fatalf("expected torque delta 0.5, got %f", controls[3][0])
	}
}

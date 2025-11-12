package planner

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func TestNewMotionValidation(t *testing.T) {
	_, err := NewMotion(Constraints{}, Parameters{})
	if err == nil {
		t.Fatalf("expected error for invalid config")
	}
}

func TestForwardStraightPath(t *testing.T) {
	m := mustPlanner(t)

	state := State{
		Position:  vec.Vector3D{0, 0, 0},
		Yaw:       0,
		Speed:     0,
		Timestamp: 0,
	}
	path := []vec.Vector3D{
		{0, 0, 0},
		{5, 0, 0},
		{10, 0, 0},
	}

	traj, err := m.Forward(state, Controls{}, path)
	if err != nil {
		t.Fatalf("Forward error: %v", err)
	}
	if len(traj.States) != 1 {
		t.Fatalf("expected single state in trajectory, got %d", len(traj.States))
	}
	next := traj.States[0]
	if next.Position[0] <= state.Position[0] {
		t.Errorf("expected progress along path, got %v", next.Position)
	}
	if math32.Abs(next.Yaw) > 1e-3 {
		t.Errorf("expected yaw near zero, got %f", next.Yaw)
	}
	if next.Speed <= 0 || next.Speed > m.constraints.MaxSpeed {
		t.Errorf("speed out of bounds: %f", next.Speed)
	}
}

func TestBackwardControlClamping(t *testing.T) {
	m := mustPlanner(t)
	state := State{
		Position: vec.Vector3D{0, 0, 0},
		Yaw:      0,
		Speed:    0,
	}
	controls := Controls{}
	traj := Trajectory{
		States: []State{
			{
				Position: vec.Vector3D{1, 0, 0},
				Yaw:      0,
				Speed:    m.constraints.MaxSpeed * 2,
			},
		},
	}

	cmd, err := m.Backward(traj, state, controls)
	if err != nil {
		t.Fatalf("Backward error: %v", err)
	}
	if cmd.Linear > m.constraints.MaxSpeed {
		t.Errorf("linear exceeded max speed: %f", cmd.Linear)
	}
	if math32.Abs(cmd.Angular) > m.constraints.MaxTurnRate {
		t.Errorf("angular exceeded max turn rate: %f", cmd.Angular)
	}
}

func mustPlanner(t *testing.T) *Motion {
	t.Helper()
	constraints := Constraints{
		MaxSpeed:               2,
		MaxAcceleration:        1,
		MaxDeceleration:        1,
		MaxJerk:                5,
		MaxTurnRate:            math32.Pi / 2,
		MaxTurnAcceleration:    math32.Pi,
		MaxLateralAcceleration: 3,
	}
	params := Parameters{
		SamplePeriod:      0.02,
		LookaheadDistance: 0.5,
		SpeedPID:          PIDGains{P: 1.0, I: 0.0, D: 0.1},
		HeadingPID:        PIDGains{P: 1.0, I: 0.0, D: 0.1},
		LateralPID:        PIDGains{P: 0.5, I: 0.0, D: 0.1},
	}

	m, err := NewMotion(constraints, params)
	if err != nil {
		t.Fatalf("failed to create motion planner: %v", err)
	}
	return m
}

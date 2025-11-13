package rigidbody

import (
	"errors"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

const (
	testStateRows   = stateVectorSize
	testControlRows = controlBaseSize
)

func TestNewMotionValidation(t *testing.T) {
	_, err := NewMotion(Constraints{}, Parameters{})
	if err == nil {
		t.Fatalf("expected error for invalid config")
	}
}

func TestForwardStraightPath(t *testing.T) {
	m := mustPlanner(t)
	path := []vec.Vector3D{{0, 0, 0}, {5, 0, 0}, {10, 0, 0}}
	if err := m.SetPath(path); err != nil {
		t.Fatalf("SetPath error: %v", err)
	}

	state := mat.New(testStateRows, 1)
	controls := mat.New(testControlRows, 1)
	destination := mat.New(testStateRows, 1)

	if err := m.Forward(state, destination, controls); err != nil {
		t.Fatalf("Forward error: %v", err)
	}

	if destination[0][0] <= state[0][0] {
		t.Errorf("expected forward progress along X, got %f", destination[0][0])
	}
	if math32.Abs(destination[3][0]) > 1e-3 {
		t.Errorf("expected yaw near zero, got %f", destination[3][0])
	}
	if destination[4][0] <= 0 || destination[4][0] > m.constraints.MaxSpeed {
		t.Errorf("speed out of bounds: %f", destination[4][0])
	}
	if math32.Abs(controls[1][0]) > m.constraints.MaxTurnRate {
		t.Errorf("angular control exceeded limits: %f", controls[1][0])
	}
	expectedTs := state[5][0] + m.params.SamplePeriod
	if math32.Abs(destination[5][0]-expectedTs) > 1e-6 {
		t.Errorf("timestamp mismatch: got %f want %f", destination[5][0], expectedTs)
	}
}

func TestForwardRequiresPath(t *testing.T) {
	m := mustPlanner(t)
	state := mat.New(testStateRows, 1)
	destination := mat.New(testStateRows, 1)
	controls := mat.New(testControlRows, 1)

	err := m.Forward(state, destination, controls)
	if !errors.Is(err, ErrInvalidPath) {
		t.Fatalf("expected ErrInvalidPath, got %v", err)
	}
}

func TestBackwardControlClamping(t *testing.T) {
	m := mustPlanner(t)

	state := mat.New(testStateRows, 1)
	destination := mat.New(testStateRows, 1)
	destination[0][0] = 1
	destination[4][0] = m.constraints.MaxSpeed * 2

	controls := mat.New(testControlRows, 1)

	if err := m.Backward(state, destination, controls); err != nil {
		t.Fatalf("Backward error: %v", err)
	}
	if controls[0][0] > m.constraints.MaxSpeed {
		t.Errorf("linear exceeded max speed: %f", controls[0][0])
	}
	if math32.Abs(controls[1][0]) > m.constraints.MaxTurnRate {
		t.Errorf("angular exceeded max turn rate: %f", controls[1][0])
	}
}

func TestMotionMetadata(t *testing.T) {
	m := mustPlanner(t)

	dims := m.Dimensions()
	if dims.StateRows != testStateRows || dims.ControlSize != testControlRows {
		t.Fatalf("unexpected dimensions: %+v", dims)
	}

	caps := m.Capabilities()
	if caps.Holonomic {
		t.Errorf("expected non-holonomic capability")
	}
	if caps.ConstraintRank != testControlRows {
		t.Errorf("unexpected constraint rank: %d", caps.ConstraintRank)
	}

	cs := m.ConstraintSet()
	if cs.ControlLower == nil || cs.ControlUpper == nil {
		t.Fatalf("expected constraint matrices to be populated")
	}
	if cs.ControlLower.Rows() != testControlRows {
		t.Fatalf("unexpected control lower rows: %d", cs.ControlLower.Rows())
	}
}

func TestSetWaypointMatrixPosition(t *testing.T) {
	m := mustPlanner(t)
	path := mat.New(3, 2)
	path[0][0], path[1][0], path[2][0] = 0, 0, 0
	path[0][1], path[1][1], path[2][1] = 5, 0, 0
	if err := m.SetWaypointMatrix(path); err != nil {
		t.Fatalf("SetWaypointMatrix position failed: %v", err)
	}
}

func TestSetWaypointMatrixPose(t *testing.T) {
	m := customPlanner(t, Constraints{
		MaxSpeed:               5,
		MaxAcceleration:        5,
		MaxDeceleration:        5,
		MaxJerk:                10,
		MaxTurnRate:            math32.Pi,
		MaxTurnAcceleration:    math32.Pi * 4,
		MaxLateralAcceleration: 10,
	}, defaultParams())
	path := mat.New(7, 2)
	path[0][0], path[1][0], path[2][0] = 0, 0, 0
	path[3][0], path[4][0], path[5][0], path[6][0] = 1, 0, 0, 0
	path[0][1], path[1][1], path[2][1] = 0, 5, 0
	yaw := math32.Pi / 2
	path[3][1] = math32.Cos(yaw / 2)
	path[4][1] = 0
	path[5][1] = 0
	path[6][1] = math32.Sin(yaw / 2)

	if err := m.SetWaypointMatrix(path); err != nil {
		t.Fatalf("SetWaypointMatrix pose failed: %v", err)
	}

	planner, ok := m.planner.(*posePlanner)
	if !ok {
		t.Fatalf("expected posePlanner, got %T", m.planner)
	}
	sample := planner.Sample(planner.Length())
	if math32.Abs(yawFromQuaternion(sample.orientation)-yaw) > 1e-4 {
		t.Fatalf("expected terminal yaw %.2f, got %.2f", yaw, yawFromQuaternion(sample.orientation))
	}
}

func TestSetWaypointMatrixPositionVelocity(t *testing.T) {
	m := customPlanner(t, Constraints{
		MaxSpeed:               10,
		MaxAcceleration:        10,
		MaxDeceleration:        10,
		MaxJerk:                20,
		MaxTurnRate:            math32.Pi,
		MaxTurnAcceleration:    math32.Pi * 2,
		MaxLateralAcceleration: 10,
	}, defaultParams())
	path := mat.New(6, 2)
	path[0][0], path[1][0], path[2][0] = 0, 0, 0
	path[3][0], path[4][0], path[5][0] = 0, 0, 0
	path[0][1], path[1][1], path[2][1] = 5, 0, 0
	path[3][1], path[4][1], path[5][1] = 4, 0, 0

	if err := m.SetWaypointMatrix(path); err != nil {
		t.Fatalf("SetWaypointMatrix position velocity failed: %v", err)
	}

	planner, ok := m.planner.(*positionVelocityPlanner)
	if !ok {
		t.Fatalf("expected positionVelocityPlanner, got %T", m.planner)
	}
	sample := planner.Sample(planner.Length())
	if sample.linearHint[0] != 4 {
		t.Fatalf("expected terminal linear hint 4, got %.2f", sample.linearHint[0])
	}
}

func TestSetWaypointMatrixPoseVelocity(t *testing.T) {
	m := customPlanner(t, Constraints{
		MaxSpeed:               10,
		MaxAcceleration:        10,
		MaxDeceleration:        10,
		MaxJerk:                20,
		MaxTurnRate:            math32.Pi,
		MaxTurnAcceleration:    math32.Pi * 4,
		MaxLateralAcceleration: 10,
	}, defaultParams())
	path := mat.New(14, 2)
	path[0][0], path[1][0], path[2][0] = 0, 0, 0
	path[3][0], path[4][0], path[5][0], path[6][0] = 1, 0, 0, 0
	path[7][0], path[8][0], path[9][0] = 0, 0, 0
	path[10][0], path[11][0], path[12][0] = 0, 0, 0
	path[13][0] = 0

	path[0][1], path[1][1], path[2][1] = 0, 5, 0
	yaw := math32.Pi / 2
	path[3][1] = math32.Cos(yaw / 2)
	path[4][1] = 0
	path[5][1] = 0
	path[6][1] = math32.Sin(yaw / 2)
	path[7][1], path[8][1], path[9][1] = 3, 0, 0
	path[10][1], path[11][1], path[12][1] = 0, 0, 0.5
	path[13][1] = 0

	if err := m.SetWaypointMatrix(path); err != nil {
		t.Fatalf("SetWaypointMatrix pose velocity failed: %v", err)
	}

	planner, ok := m.planner.(*poseVelocityPlanner)
	if !ok {
		t.Fatalf("expected poseVelocityPlanner, got %T", m.planner)
	}
	sample := planner.Sample(planner.Length())
	if math32.Abs(yawFromQuaternion(sample.orientation)-yaw) > 1e-4 {
		t.Fatalf("expected terminal yaw %.2f, got %.2f", yaw, yawFromQuaternion(sample.orientation))
	}
	if sample.linearHint[0] != 3 {
		t.Fatalf("expected terminal linear hint 3, got %.2f", sample.linearHint[0])
	}
	if sample.angularHint[2] != 0.5 {
		t.Fatalf("expected terminal angular hint 0.5, got %.2f", sample.angularHint[2])
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

func customPlanner(t *testing.T, constraints Constraints, params Parameters) *Motion {
	m, err := NewMotion(constraints, params)
	if err != nil {
		t.Fatalf("failed to create motion planner: %v", err)
	}
	return m
}

func defaultParams() Parameters {
	return Parameters{
		SamplePeriod:      0.02,
		LookaheadDistance: 0.5,
		SpeedPID:          PIDGains{P: 1.0, I: 0.0, D: 0.1},
		HeadingPID:        PIDGains{P: 1.0, I: 0.0, D: 0.1},
		LateralPID:        PIDGains{P: 0.5, I: 0.0, D: 0.1},
	}
}

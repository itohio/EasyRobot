package rigidbody

import (
	"fmt"

	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const (
	stateIdxX = iota
	stateIdxY
	stateIdxZ
	stateIdxYaw
	stateIdxSpeed
	stateIdxTimestamp
	stateVectorSize
)

const (
	controlIdxLinear = iota
	controlIdxAngular
	controlBaseSize
)

func ensureStateMatrix(m mattype.Matrix) error {
	if m == nil {
		return fmt.Errorf("planner: %w (state nil)", kintypes.ErrInvalidDimensions)
	}
	if m.Rows() != stateVectorSize || m.Cols() < 1 {
		return fmt.Errorf("planner: %w (state %dx%d)", kintypes.ErrInvalidDimensions, m.Rows(), m.Cols())
	}
	return nil
}

func ensureDestinationMatrix(m mattype.Matrix) error {
	return ensureStateMatrix(m)
}

func ensureControlsMatrix(m mattype.Matrix, expected int) error {
	if m == nil {
		return fmt.Errorf("planner: %w (controls nil)", kintypes.ErrInvalidDimensions)
	}
	if m.Rows() != expected || m.Cols() < 1 {
		return fmt.Errorf("planner: %w (controls %dx%d expected %d rows)", kintypes.ErrInvalidDimensions, m.Rows(), m.Cols(), expected)
	}
	return nil
}

func stateFromMatrix(m mattype.Matrix) (State, error) {
	if err := ensureStateMatrix(m); err != nil {
		return State{}, err
	}
	flat := m.Flat()
	if len(flat) < stateVectorSize {
		return State{}, fmt.Errorf("planner: %w (state flat len %d)", kintypes.ErrInvalidDimensions, len(flat))
	}
	return State{
		Position: vec.Vector3D{
			flat[stateIdxX],
			flat[stateIdxY],
			flat[stateIdxZ],
		},
		Yaw:       flat[stateIdxYaw],
		Speed:     flat[stateIdxSpeed],
		Timestamp: flat[stateIdxTimestamp],
	}, nil
}

func populateStateMatrix(dst mattype.Matrix, state State) error {
	if err := ensureStateMatrix(dst); err != nil {
		return err
	}
	flat := dst.Flat()
	if len(flat) < stateVectorSize {
		return fmt.Errorf("planner: %w (state flat len %d)", kintypes.ErrInvalidDimensions, len(flat))
	}
	flat[stateIdxX] = state.Position[0]
	flat[stateIdxY] = state.Position[1]
	flat[stateIdxZ] = state.Position[2]
	flat[stateIdxYaw] = state.Yaw
	flat[stateIdxSpeed] = state.Speed
	flat[stateIdxTimestamp] = state.Timestamp
	return nil
}

func controlsFromMatrix(m mattype.Matrix, expected int) (Controls, error) {
	if err := ensureControlsMatrix(m, expected); err != nil {
		return Controls{}, err
	}
	flat := m.Flat()
	if len(flat) < expected {
		return Controls{}, fmt.Errorf("planner: %w (controls flat len %d)", kintypes.ErrInvalidDimensions, len(flat))
	}
	ctrl := Controls{
		Linear:  flat[controlIdxLinear],
		Angular: flat[controlIdxAngular],
	}
	if expected > controlBaseSize {
		effort := make(vec.Vector, expected-controlBaseSize)
		copy(effort, flat[controlBaseSize:expected])
		ctrl.Effort = effort
	}
	return ctrl, nil
}

func populateControlsMatrix(dst mattype.Matrix, ctrl Controls, expected int) error {
	if err := ensureControlsMatrix(dst, expected); err != nil {
		return err
	}
	flat := dst.Flat()
	if len(flat) < expected {
		return fmt.Errorf("planner: %w (controls flat len %d)", kintypes.ErrInvalidDimensions, len(flat))
	}
	flat[controlIdxLinear] = ctrl.Linear
	flat[controlIdxAngular] = ctrl.Angular
	if expected > controlBaseSize {
		if len(ctrl.Effort) < expected-controlBaseSize {
			return fmt.Errorf("planner: %w (effort len %d expected %d)", kintypes.ErrInvalidDimensions, len(ctrl.Effort), expected-controlBaseSize)
		}
		copy(flat[controlBaseSize:expected], ctrl.Effort[:expected-controlBaseSize])
	}
	return nil
}

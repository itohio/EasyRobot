package thrusters

import (
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const wrenchSize = 6

type Model struct {
	body      Body
	thrusters []Thruster
	commands  []Command
	cache     []ThrusterCommand
	state     BodyState

	dimensions   kintypes.Dimensions
	capabilities kintypes.Capabilities
}

var _ kintypes.Bidirectional = (*Model)(nil)

func NewModel(body Body, thrusters []Thruster) (*Model, error) {
	if err := validateBody(body); err != nil {
		return nil, err
	}
	if len(thrusters) == 0 {
		return nil, ErrInfeasible
	}
	cmds := make([]Command, len(thrusters))
	cache := make([]ThrusterCommand, len(thrusters))
	for i := range cache {
		cache[i].Thruster = thrusters[i]
	}
	return &Model{
		body:      body,
		thrusters: thrusters,
		commands:  cmds,
		cache:     cache,
		dimensions: kintypes.Dimensions{
			StateRows:    len(thrusters) * 2,
			StateCols:    1,
			ControlSize:  len(thrusters) * 2,
			ActuatorSize: len(thrusters) * 2,
		},
		capabilities: kintypes.Capabilities{
			Holonomic:       true,
			Omnidirectional: true,
			ConstraintRank:  wrenchSize,
		},
	}, nil
}

func (m *Model) Dimensions() kintypes.Dimensions {
	return m.dimensions
}

func (m *Model) Capabilities() kintypes.Capabilities {
	return m.capabilities
}

func (*Model) ConstraintSet() kintypes.Constraints {
	return kintypes.Constraints{}
}

// Forward reads the actuator command column (`state`), where each thruster is
// represented as `[thrust, torque]`, and writes the resulting wrench
// `[Fx, Fy, Fz, Tx, Ty, Tz]` into `destination`.
func (m *Model) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := ensureColumn(state, m.dimensions.StateRows); err != nil {
		return err
	}
	if err := ensureColumn(destination, wrenchSize); err != nil {
		return err
	}

	stateView := state.View().(mat.Matrix)
	for i := range m.thrusters {
		cmd := Command{
			Thrust: stateView[2*i][0],
			Torque: stateView[2*i+1][0],
		}
		applied, ok := clampCommand(cmd, m.thrusters[i])
		if !ok {
			return ErrCommandLimit
		}
		m.commands[i] = applied
		m.cache[i].Command = applied
	}

	result, err := applyForward(m.body, BodyState{}, m.cache)
	if err != nil {
		return err
	}
	m.state = result

	destView := destination.View().(mat.Matrix)
	destView[0][0] = result.Force[0]
	destView[1][0] = result.Force[1]
	destView[2][0] = result.Force[2]
	destView[3][0] = result.Torque[0]
	destView[4][0] = result.Torque[1]
	destView[5][0] = result.Torque[2]

	return nil
}

// Backward consumes a desired wrench from `destination` (`[Fx, Fy, Fz, Tx, Ty,
// Tz]`) and solves thruster commands, writing `[thrust_i, torque_i]` pairs into
// the `controls` column vector. Optional `state` seeds the current command
// vector prior to solving.
func (m *Model) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := ensureColumn(destination, wrenchSize); err != nil {
		return err
	}
	if err := ensureColumn(controls, m.dimensions.ControlSize); err != nil {
		return err
	}
	if state != nil {
		if err := ensureColumn(state, m.dimensions.StateRows); err != nil {
			return err
		}
		stateView := state.View().(mat.Matrix)
		for i := range m.commands {
			m.commands[i] = Command{
				Thrust: stateView[2*i][0],
				Torque: stateView[2*i+1][0],
			}
			m.cache[i].Command = m.commands[i]
		}
	}

	destView := destination.View().(mat.Matrix)
	desired := BodyState{
		Force:  vec.Vector3D{destView[0][0], destView[1][0], destView[2][0]},
		Torque: vec.Vector3D{destView[3][0], destView[4][0], destView[5][0]},
	}

	solution, err := allocate(m.body, m.thrusters, BodyState{}, desired)
	if err != nil {
		return err
	}
	for i := range solution {
		m.commands[i] = solution[i]
		m.cache[i].Command = solution[i]
	}

	result, err := applyForward(m.body, BodyState{}, m.cache)
	if err != nil {
		return err
	}
	m.state = result

	controlView := controls.View().(mat.Matrix)
	for i := range m.commands {
		controlView[2*i][0] = m.commands[i].Thrust
		controlView[2*i+1][0] = m.commands[i].Torque
	}

	return nil
}

func ensureColumn(m mattype.Matrix, rows int) error {
	if m == nil {
		return kintypes.ErrInvalidDimensions
	}
	if m.Rows() != rows || m.Cols() < 1 {
		return kintypes.ErrInvalidDimensions
	}
	return nil
}

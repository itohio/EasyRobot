package differential

import (
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

type drive struct {
	wR float32
	wB float32

	wheels [2]float32 // left, right
	speed  [2]float32 // forward speed, angular speed

	dimensions   kintypes.Dimensions
	capabilities kintypes.Capabilities
}

var _ kintypes.Bidirectional = (*drive)(nil)

// New returns a kinematics model for a differential-drive base.
// wheelRadius defines the wheel radius and base is the distance between wheels.
func New(wheelRadius, base float32) *drive {
	return &drive{
		wR: wheelRadius,
		wB: base,
		dimensions: kintypes.Dimensions{
			StateRows:    2,
			StateCols:    1,
			ControlSize:  2,
			ActuatorSize: 2,
		},
		capabilities: kintypes.Capabilities{
			Holonomic:       false,
			Omnidirectional: false,
			ConstraintRank:  2,
		},
	}
}

func (d *drive) Dimensions() kintypes.Dimensions {
	return d.dimensions
}

func (d *drive) Capabilities() kintypes.Capabilities {
	return d.capabilities
}

func (*drive) ConstraintSet() kintypes.Constraints {
	return kintypes.Constraints{}
}

// Params returns the current wheel rate parameters (left, right).
func (d *drive) Params() vec.Vector {
	return d.wheels[:]
}

// Effector returns the last computed chassis twist [v, ω].
func (d *drive) Effector() vec.Vector {
	return d.speed[:]
}

// Forward maps wheel rates provided in `state` (left, right) into chassis twist
// `[v, ω]`, writing the result into the supplied `destination` column matrix.
func (d *drive) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := ensureColumn(state, len(d.wheels)); err != nil {
		return err
	}
	if err := ensureColumn(destination, len(d.speed)); err != nil {
		return err
	}

	stateView := state.View().(mat.Matrix)
	for i := range d.wheels {
		d.wheels[i] = stateView[i][0]
	}

	d.speed[0] = (d.wheels[0] + d.wheels[1]) * 0.5
	d.speed[1] = (d.wheels[1] - d.wheels[0]) / d.wB

	destView := destination.View().(mat.Matrix)
	for i := range d.speed {
		destView[i][0] = d.speed[i]
	}

	return nil
}

// Backward maps a desired chassis twist from `destination` (`[v, ω]`) into the
// wheel-rate column vector stored in `controls` (left, right). Optional `state`
// input seeds the current wheel parameters before solving.
func (d *drive) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := ensureColumn(destination, len(d.speed)); err != nil {
		return err
	}
	if err := ensureColumn(controls, len(d.wheels)); err != nil {
		return err
	}
	if state != nil {
		if err := ensureColumn(state, len(d.wheels)); err != nil {
			return err
		}
		stateView := state.View().(mat.Matrix)
		for i := range d.wheels {
			d.wheels[i] = stateView[i][0]
		}
	}

	destView := destination.View().(mat.Matrix)
	for i := range d.speed {
		d.speed[i] = destView[i][0]
	}

	d.wheels[0] = d.speed[0] - d.wB*d.speed[1]*0.5
	d.wheels[1] = d.speed[0] + d.wB*d.speed[1]*0.5

	controlView := controls.View().(mat.Matrix)
	for i := range d.wheels {
		controlView[i][0] = d.wheels[i]
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

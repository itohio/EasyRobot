package thrusters

import (
	"errors"

	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

var (
	ErrInvalidMass     = errors.New("thrusters: body mass must be positive")
	ErrInvalidInertia  = errors.New("thrusters: inertia matrix is singular")
	ErrMismatchedInput = errors.New("thrusters: thruster and command counts do not match")
	ErrInfeasible      = errors.New("thrusters: desired wrench infeasible under limits")
	ErrCommandLimit    = errors.New("thrusters: command violates thruster limits")
)

type Range struct {
	Min float32
	Max float32
}

func (r Range) Clamp(v float32) float32 {
	if v < r.Min {
		return r.Min
	}
	if v > r.Max {
		return r.Max
	}
	return v
}

func (r Range) Contains(v float32) bool {
	return v >= r.Min && v <= r.Max
}

type Thruster struct {
	Position    vec.Vector3D
	Direction   vec.Vector3D
	TorqueAxis  vec.Vector3D
	ThrustLimit Range
	TorqueLimit Range
}

type Command struct {
	Thrust float32
	Torque float32
}

type ThrusterCommand struct {
	Thruster Thruster
	Command  Command
}

type Body struct {
	Mass    float32
	Inertia mat.Matrix3x3
}

type BodyState struct {
	Force        vec.Vector3D
	Torque       vec.Vector3D
	LinearAccel  vec.Vector3D
	AngularAccel vec.Vector3D
}

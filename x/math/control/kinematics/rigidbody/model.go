package rigidbody

import (
	"fmt"

	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const (
	stateSize   = 6
	controlSize = 6
)

type Config struct {
	LinearGain  vec.Vector3D
	AngularGain vec.Vector3D
	MaxForce    vec.Vector3D
	MaxTorque   vec.Vector3D
}

type Model struct {
	mass       float32
	inertia    mat.Matrix3x3
	invInertia mat.Matrix3x3
	config     Config
	dims       kintypes.Dimensions
	caps       kintypes.Capabilities
	constSet   kintypes.Constraints
}

var _ kintypes.Bidirectional = (*Model)(nil)

func NewModel(mass float32, inertia mat.Matrix3x3, cfg Config) (*Model, error) {
	if mass <= 0 {
		return nil, fmt.Errorf("rigidbody: invalid mass")
	}
	for i := 0; i < 3; i++ {
		if inertia[i][i] <= 0 {
			return nil, fmt.Errorf("rigidbody: inertia diagonal must be positive")
		}
	}
	inertiaCopy := inertia
	var inv mat.Matrix3x3
	if err := inertia.Inverse(&inv); err != nil {
		return nil, fmt.Errorf("rigidbody: inertia not invertible: %w", err)
	}
	m := &Model{
		mass:       mass,
		inertia:    inertiaCopy,
		invInertia: inv,
		config:     cfg,
		dims: kintypes.Dimensions{
			StateRows:    stateSize,
			StateCols:    1,
			ControlSize:  controlSize,
			ActuatorSize: controlSize,
		},
		caps: kintypes.Capabilities{
			Holonomic:        true,
			Omnidirectional:  true,
			SupportsLateral:  true,
			SupportsVertical: true,
			ConstraintRank:   controlSize,
		},
	}
	m.constSet = m.buildConstraints()
	return m, nil
}

func (m *Model) Dimensions() kintypes.Dimensions {
	return m.dims
}

func (m *Model) Capabilities() kintypes.Capabilities {
	return m.caps
}

func (m *Model) ConstraintSet() kintypes.Constraints {
	return m.constSet
}

func (m *Model) Inertia() mat.Matrix3x3 {
	return m.inertia
}

func (m *Model) InvInertia() mat.Matrix3x3 {
	return m.invInertia
}

func (m *Model) AngularAcceleration(torque vec.Vector3D) vec.Vector3D {
	return m.invInertia.MulVec(torque, nil).(vec.Vector3D)
}

func (m *Model) buildConstraints() kintypes.Constraints {
	lower := mat.New(controlSize, 1)
	upper := mat.New(controlSize, 1)
	for i := 0; i < 3; i++ {
		lower[i][0] = -m.config.MaxForce[i]
		upper[i][0] = m.config.MaxForce[i]
		lower[i+3][0] = -m.config.MaxTorque[i]
		upper[i+3][0] = m.config.MaxTorque[i]
	}
	return kintypes.Constraints{
		ControlLower: lower,
		ControlUpper: upper,
	}
}

// Forward expects state and destination matrices shaped 6x1, encoding linear (0..2) and angular (3..5) velocities.
func (m *Model) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	force, torque := m.velocityToWrench(state)
	if destination != nil {
		writeWrench(destination, force, torque)
	}
	if controls != nil {
		writeWrench(controls, force, torque)
	}
	return nil
}

// Backward expects destination velocities shaped 6x1 and writes the resulting wrench into controls (6x1).
func (m *Model) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	var current mattype.Matrix
	if state != nil {
		current = state
	}

	force, torque := m.velocityCommandToWrench(current, destination)
	if controls != nil {
		writeWrench(controls, force, torque)
	}
	return nil
}

func (m *Model) velocityToWrench(state mattype.Matrix) (vec.Vector3D, vec.Vector3D) {
	force := vec.Vector3D{}
	torque := vec.Vector3D{}
	if state == nil {
		return force, torque
	}
	for i := 0; i < 3; i++ {
		vel := state.Flat()[i]
		force[i] = clamp(vel*m.mass*m.config.LinearGain[i], -m.config.MaxForce[i], m.config.MaxForce[i])
	}
	omegaVec := vec.Vector3D{
		state.Flat()[3],
		state.Flat()[4],
		state.Flat()[5],
	}
	temp := m.inertia.MulVec(omegaVec, nil).(vec.Vector3D)
	for i := 0; i < 3; i++ {
		raw := temp[i] * m.config.AngularGain[i]
		torque[i] = clamp(raw, -m.config.MaxTorque[i], m.config.MaxTorque[i])
	}
	return force, torque
}

func (m *Model) velocityCommandToWrench(state mattype.Matrix, destination mattype.Matrix) (vec.Vector3D, vec.Vector3D) {
	force := vec.Vector3D{}
	torque := vec.Vector3D{}
	if destination == nil {
		return force, torque
	}
	for i := 0; i < 3; i++ {
		target := destination.Flat()[i]
		current := float32(0)
		if state != nil {
			current = state.Flat()[i]
		}
		error := target - current
		force[i] = clamp(error*m.mass*m.config.LinearGain[i], -m.config.MaxForce[i], m.config.MaxForce[i])
	}

	errorOmega := vec.Vector3D{}
	for i := 0; i < 3; i++ {
		targetOmega := destination.Flat()[i+3]
		currentOmega := float32(0)
		if state != nil {
			currentOmega = state.Flat()[i+3]
		}
		errorOmega[i] = targetOmega - currentOmega
	}
	torqueVec := m.inertia.MulVec(errorOmega, nil).(vec.Vector3D)
	for i := 0; i < 3; i++ {
		raw := torqueVec[i] * m.config.AngularGain[i]
		torque[i] = clamp(raw, -m.config.MaxTorque[i], m.config.MaxTorque[i])
	}
	return force, torque
}

func writeWrench(dst mattype.Matrix, force, torque vec.Vector3D) {
	if dst == nil {
		return
	}
	flat := dst.Flat()
	for i := 0; i < 3; i++ {
		flat[i] = force[i]
		flat[i+3] = torque[i]
	}
}

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

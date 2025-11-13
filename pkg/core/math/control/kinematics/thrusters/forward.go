package thrusters

import (
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func applyForward(body Body, base BodyState, cmds []ThrusterCommand) (BodyState, error) {
	if err := validateBody(body); err != nil {
		return BodyState{}, err
	}
	invInertia, err := invertInertia(body.Inertia)
	if err != nil {
		return BodyState{}, err
	}
	state := base
	var (
		force  = state.Force
		torque = state.Torque
	)
	for _, tc := range cmds {
		command, ok := clampCommand(tc.Command, tc.Thruster)
		if !ok {
			return BodyState{}, ErrCommandLimit
		}
		thrustVec := scaled(command.Thrust, tc.Thruster.Direction)
		addVec(&force, thrustVec)

		moment := cross(tc.Thruster.Position, thrustVec)
		if command.Torque != 0 {
			reaction := scaled(command.Torque, tc.Thruster.TorqueAxis)
			addVec(&moment, reaction)
		}
		addVec(&torque, moment)
	}
	state.Force = force
	state.Torque = torque
	state.LinearAccel = acceleration(force, body.Mass)
	state.AngularAccel = angularAcceleration(torque, invInertia)
	return state, nil
}

func scaled(scale float32, dir vec.Vector3D) vec.Vector3D {
	var out vec.Vector3D
	scaledVec(&out, dir, scale)
	return out
}

func acceleration(force vec.Vector3D, mass float32) vec.Vector3D {
	var acc vec.Vector3D
	if mass == 0 {
		return acc
	}
	inv := 1 / mass
	acc[0] = force[0] * inv
	acc[1] = force[1] * inv
	acc[2] = force[2] * inv
	return acc
}

func angularAcceleration(torque vec.Vector3D, inv mat.Matrix3x3) vec.Vector3D {
	var result vec.Vector3D
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			result[row] += inv[row][col] * torque[col]
		}
	}
	return result
}

package thrusters

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

func validateBody(body Body) error {
	if body.Mass <= 0 {
		return ErrInvalidMass
	}
	var det float32 = body.Inertia.Det()
	if math32.Abs(det) <= mat.SingularityTolerance {
		return ErrInvalidInertia
	}
	return nil
}

func invertInertia(inertia mat.Matrix3x3) (mat.Matrix3x3, error) {
	var inv mat.Matrix3x3
	if err := inertia.Inverse(&inv); err != nil {
		return mat.Matrix3x3{}, ErrInvalidInertia
	}
	return inv, nil
}

func scaledVec(dst *vec.Vector3D, src vec.Vector3D, scale float32) {
	for i := range dst {
		dst[i] = src[i] * scale
	}
}

func addVec(dst *vec.Vector3D, src vec.Vector3D) {
	for i := range dst {
		dst[i] += src[i]
	}
}

func cross(a, b vec.Vector3D) vec.Vector3D {
	return vec.Vector3D{
		a[1]*b[2] - a[2]*b[1],
		a[2]*b[0] - a[0]*b[2],
		a[0]*b[1] - a[1]*b[0],
	}
}

func clampCommand(cmd Command, limits Thruster) (Command, bool) {
	applied := cmd
	applied.Thrust = limits.ThrustLimit.Clamp(applied.Thrust)
	applied.Torque = limits.TorqueLimit.Clamp(applied.Torque)
	exact := limits.ThrustLimit.Contains(cmd.Thrust) && limits.TorqueLimit.Contains(cmd.Torque)
	return applied, exact
}

func deltaState(desired, current BodyState) BodyState {
	return BodyState{
		Force: vec.Vector3D{
			desired.Force[0] - current.Force[0],
			desired.Force[1] - current.Force[1],
			desired.Force[2] - current.Force[2],
		},
		Torque: vec.Vector3D{
			desired.Torque[0] - current.Torque[0],
			desired.Torque[1] - current.Torque[1],
			desired.Torque[2] - current.Torque[2],
		},
		LinearAccel: vec.Vector3D{
			desired.LinearAccel[0] - current.LinearAccel[0],
			desired.LinearAccel[1] - current.LinearAccel[1],
			desired.LinearAccel[2] - current.LinearAccel[2],
		},
		AngularAccel: vec.Vector3D{
			desired.AngularAccel[0] - current.AngularAccel[0],
			desired.AngularAccel[1] - current.AngularAccel[1],
			desired.AngularAccel[2] - current.AngularAccel[2],
		},
	}
}

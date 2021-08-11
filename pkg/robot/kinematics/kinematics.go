package kinematics

import "github.com/foxis/EasyRobot/pkg/core/math/vec"

type Kinematics interface {
	DOF() int
	Params() vec.Vector
	Effector() vec.Vector

	Forward() bool
	Inverse() bool
}

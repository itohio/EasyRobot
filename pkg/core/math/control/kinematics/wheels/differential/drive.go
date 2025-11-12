package differential

import (
	"github.com/itohio/EasyRobot/pkg/control/kinematics"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

type drive struct {
	wR float32
	wB float32

	wheels [2]float32 // left, right
	speed  [2]float32 // forward speed, angular speed
}

var _ kinematics.Kinematics = (*drive)(nil)

// New returns a kinematics model for a differential-drive base.
// wheelRadius defines the wheel radius and base is the distance between wheels.
func New(wheelRadius, base float32) *drive {
	return &drive{
		wR: wheelRadius,
		wB: base,
	}
}

func (*drive) DOF() int {
	return 2
}

func (d *drive) Params() vec.Vector {
	return d.wheels[:]
}

func (d *drive) Effector() vec.Vector {
	return d.speed[:]
}

func (d *drive) Forward() bool {
	d.speed[0] = (d.wheels[0] + d.wheels[1]) * 0.5
	d.speed[1] = (d.wheels[1] - d.wheels[0]) / d.wB
	return true
}

func (d *drive) Inverse() bool {
	d.wheels[0] = d.speed[0] - d.wB*d.speed[1]*0.5
	d.wheels[1] = d.speed[0] + d.wB*d.speed[1]*0.5
	return true
}

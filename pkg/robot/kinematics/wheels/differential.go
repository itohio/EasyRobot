package wheels

import (
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
)

type diff struct {
	wR float32
	wB float32

	wheels [2]float32 // left, right
	speed  [2]float32 // forward speed, angular speed
}

func NewDifferential(wheelRadius, base float32) kinematics.Kinematics {
	return &diff{
		wR: wheelRadius,
		wB: base,
	}
}

func (*diff) DOF() int {
	return 2
}

func (p *diff) Params() vec.Vector {
	return p.wheels[:]
}

func (p *diff) Effector() vec.Vector {
	return p.speed[:]
}

func (p *diff) Forward() bool {
	p.speed[0] = (p.wheels[0] + p.wheels[1]) / 2
	p.speed[1] = (p.wheels[1] - p.wheels[0]) / p.wB
	return true
}

func (p *diff) Inverse() bool {
	p.wheels[0] = p.speed[0] - p.wB*p.speed[1]/2
	p.wheels[1] = p.speed[0] + p.wB*p.speed[1]/2
	return true
}

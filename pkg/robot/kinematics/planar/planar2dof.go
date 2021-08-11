package planar

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
)

type p2d struct {
	c      [2]Config
	params [2]float32
	pos    [6]float32
}

func New2DOF(cfg [2]Config) kinematics.Kinematics {
	return &p2d{
		c: cfg,
	}
}

func (*p2d) DOF() int {
	return 2
}

func (p *p2d) Params() vec.Vector {
	return p.params[:]
}

func (p *p2d) Effector() vec.Vector {
	return p.pos[:]
}

func (p *p2d) Forward() bool {
	a0 := p.c[0].Limit(p.params[0])
	a1 := p.c[1].Limit(p.params[1])
	l0 := p.c[0].Length
	l1 := p.c[1].Length

	x := l0 + l1*math32.Cos(a1)
	z := l1 * math32.Sin(a1)

	p.pos[0] = x * math32.Cos(a0)
	p.pos[1] = x * math32.Sin(a0)
	p.pos[2] = z

	p.pos[4] = a1
	p.pos[5] = a0

	return true
}

func (p *p2d) Inverse() bool {
	x_prime := math32.Sqrt(math.SQR(p.pos[0])+math.SQR(p.pos[1])) - p.c[0].Length

	p.params[0] = p.c[0].Limit(math32.Atan2(p.pos[1], p.pos[0]))
	p.params[1] = p.c[1].Limit(math32.Atan2(p.pos[2], x_prime))

	return true
}

package planar

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
)

type p3d struct {
	c      [3]Config
	params [3]float32
	pos    [6]float32
}

func New3DOF(cfg [3]Config) kinematics.Kinematics {
	return &p3d{
		c: cfg,
	}
}

func (*p3d) DOF() int {
	return 3
}

func (p *p3d) Params() vec.Vector {
	return p.params[:]
}

func (p *p3d) Effector() vec.Vector {
	return p.pos[:]
}

func (p *p3d) Forward() bool {
	a0 := p.c[0].Limit(p.params[0])
	a1 := p.c[1].Limit(p.params[1])
	a2 := p.c[2].Limit(p.params[2]) + a1
	l0 := p.c[0].Length
	l1 := p.c[1].Length
	l2 := p.c[2].Length

	x := l0 + l1*math32.Cos(a1) + l2*math32.Cos(a2)
	z := l1*math32.Sin(a1) + l2*math32.Sin(a2)
	p.pos[0] = x * math32.Cos(a0)
	p.pos[1] = x * math32.Sin(a0)
	p.pos[2] = z
	p.pos[4] = a2
	p.pos[5] = a0

	return true
}

func (p *p3d) Inverse() bool {
	l0 := p.c[0].Length
	l1 := p.c[1].Length
	l2 := p.c[2].Length
	x_prime := math32.Sqrt(math.SQR(p.pos[0])+math.SQR(p.pos[1])) - l0
	gamma := math32.Atan2(p.pos[2], x_prime)
	beta := math32.Acos((math.SQR(l1) + math.SQR(l2) - math.SQR(x_prime) - math.SQR(p.pos[2])) / (2 * l1 * l2))
	alpha := math32.Acos((math.SQR(x_prime) + math.SQR(p.pos[2]) + math.SQR(l1) - math.SQR(l2)) / (2 * l1 * math32.Sqrt(math.SQR(p.pos[2])+math.SQR(x_prime))))

	p.params[0] = p.c[0].Limit(math32.Atan2(p.pos[1], p.pos[0]))
	p.params[1] = gamma + alpha
	p.params[2] = beta - math32.Pi

	return true
}

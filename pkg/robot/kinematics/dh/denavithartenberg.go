package dh

import (
	"github.com/foxis/EasyRobot/pkg/core/math/mat"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
)

type DenavitHartenberg struct {
	c             []Config
	eps           float32
	maxIterations int
	params        []float32
	pos           [7]float32
	H0i           []mat.Matrix4x4 // TODO replace with Matrix4x4
}

func New(eps float32, maxIterations int, cfg ...Config) kinematics.Kinematics {
	return &DenavitHartenberg{
		eps:           eps,
		maxIterations: maxIterations,
		c:             cfg,
		params:        make([]float32, len(cfg)),
	}
}

func (p *DenavitHartenberg) DOF() int {
	return len(p.c)
}

func (p *DenavitHartenberg) Params() vec.Vector {
	return p.params[:]
}

func (p *DenavitHartenberg) Effector() vec.Vector {
	return p.pos[:]
}

func (p *DenavitHartenberg) Forward() bool {
	H := mat.Matrix4x4{}
	H.Eye()
	p.H0i[0].Eye()
	for i, cfg := range p.c {
		if cfg.CalculateTransform(p.params[i], &H) {
			return false
		}
		p.H0i[i].MulTo(H, &(p.H0i[i+1]))
	}

	copy(p.pos[:3], p.H0i[len(p.c)].Col(3))
	copy(p.pos[3:], p.H0i[len(p.c)].Quaternion().Vector())

	return true
}

func (p *DenavitHartenberg) Inverse() bool {

	return false
}

package wheels

import (
	"github.com/foxis/EasyRobot/pkg/core/math/mat"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
	"github.com/foxis/EasyRobot/pkg/robot/kinematics"
)

type mechanum struct {
	wR  float32
	wBX float32
	wBY float32

	params [4]float32
	pos    [3]float32
}

func NewMechanum(wheelRadius, baseX, baseY float32) kinematics.Kinematics {
	return &mechanum{
		wR:  wheelRadius,
		wBX: baseX,
		wBY: baseY,
	}
}

func (*mechanum) DOF() int {
	return 4
}

func (p *mechanum) Params() vec.Vector {
	return p.params[:]
}

func (p *mechanum) Effector() vec.Vector {
	return p.pos[:]
}

func (p *mechanum) Forward() bool {
	c := 2 / (p.wBX + p.wBY)
	m := mat.New3x4(
		1, 1, 1, 1,
		1, -1, -1, 1,
		-c, c, -c, c,
	)
	m.MulC(p.wR/4).MulVec(p.params, p.pos[:])

	return true
}

func (p *mechanum) Inverse() bool {
	c := (p.wBX + p.wBY) / 2
	m := mat.New4x3(
		1, 1, -c,
		1, -1, c,
		1, -1, -c,
		1, 1, c,
	)
	m.DivC(p.wR).MulVec(p.pos, p.params[:])
	return true
}

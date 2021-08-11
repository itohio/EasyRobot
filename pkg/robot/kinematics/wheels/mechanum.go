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
	m := mat.NewBacked(3, 4, []float32{
		1, 1, 1, 1,
		1, -1, -1, 1,
		-c, c, -c, c,
	}).MulC(p.wR / 4)

	m.MulVTo(p.params[:], p.pos[:])

	return true
}

func (p *mechanum) Inverse() bool {
	c := (p.wBX + p.wBY) / 2
	m := mat.NewBacked(4, 3, []float32{
		1, 1, -c,
		1, -1, c,
		1, -1, -c,
		1, 1, c,
	}).DivC(p.wR)

	m.MulVTo(p.pos[:], p.params[:])
	return true
}

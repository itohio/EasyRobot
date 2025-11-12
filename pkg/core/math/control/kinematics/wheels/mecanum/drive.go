package mecanum

import (
	"github.com/itohio/EasyRobot/pkg/control/kinematics"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

type drive struct {
	wR  float32
	wBX float32
	wBY float32

	params [4]float32
	pos    [3]float32
}

var _ kinematics.Kinematics = (*drive)(nil)

// New returns a kinematics model for a mecanum-drive platform.
// wheelRadius is the radius of each wheel, while baseX and baseY represent
// the chassis dimensions between wheel centers along the X and Y axes.
func New(wheelRadius, baseX, baseY float32) *drive {
	return &drive{
		wR:  wheelRadius,
		wBX: baseX,
		wBY: baseY,
	}
}

func (*drive) DOF() int {
	return 4
}

func (d *drive) Params() vec.Vector {
	return d.params[:]
}

func (d *drive) Effector() vec.Vector {
	return d.pos[:]
}

func (d *drive) Forward() bool {
	transform3x4(d.wR, d.wBX, d.wBY).
		MulVec(d.params, d.pos[:])

	return true
}

func (d *drive) Inverse() bool {
	transform4x3(d.wR, d.wBX, d.wBY).
		MulVec(d.pos, d.params[:])
	return true
}

func transform3x4(wR, baseX, baseY float32) *mat.Matrix3x4 {
	c := 2 / (baseX + baseY)
	m := mat.New3x4(
		1, 1, 1, 1,
		1, -1, -1, 1,
		-c, c, -c, c,
	)
	return (&m).MulC(wR / 4)
}

func transform4x3(wR, baseX, baseY float32) *mat.Matrix4x3 {
	c := (baseX + baseY) * 0.5
	m := mat.New4x3(
		1, 1, -c,
		1, -1, c,
		1, -1, -c,
		1, 1, c,
	)
	return (&m).DivC(wR)
}

package planar

import (
	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/pkg/core/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/pkg/core/math/mat"
	mattype "github.com/itohio/EasyRobot/pkg/core/math/mat/types"
)

var _ kintypes.Bidirectional = (*p3d)(nil)

type p3d struct {
	c            [3]Config
	params       [3]float32
	pos          [planarEffectorSize]float32
	dimensions   kintypes.Dimensions
	capabilities kintypes.Capabilities
}

func New3DOF(cfg [3]Config) *p3d {
	return &p3d{
		c: cfg,
		dimensions: kintypes.Dimensions{
			StateRows:    3,
			StateCols:    1,
			ControlSize:  3,
			ActuatorSize: 3,
		},
		capabilities: kintypes.Capabilities{
			Holonomic:        true,
			SupportsLateral:  true,
			SupportsVertical: true,
			ConstraintRank:   3,
		},
	}
}

func (p *p3d) Dimensions() kintypes.Dimensions {
	return p.dimensions
}

func (p *p3d) Capabilities() kintypes.Capabilities {
	return p.capabilities
}

func (p *p3d) ConstraintSet() kintypes.Constraints {
	return kintypes.Constraints{}
}

// Forward interprets the 3×1 joint column `[a0, a1, a2]` and writes the
// effector `[x, y, z, roll, pitch, yaw]` into `destination`, with orientation
// entries reflecting accumulated joint angles.
func (p *p3d) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := ensureColumn(state, len(p.params)); err != nil {
		return err
	}
	if err := ensureColumn(destination, planarEffectorSize); err != nil {
		return err
	}

	stateView := state.View().(mat.Matrix)
	for i := range p.params {
		p.params[i] = stateView[i][0]
	}

	a0 := p.c[0].Limit(p.params[0])
	p.params[0] = a0
	a1 := p.c[1].Limit(p.params[1])
	p.params[1] = a1
	limitedA2 := p.c[2].Limit(p.params[2])
	p.params[2] = limitedA2
	a2 := limitedA2 + a1
	l0 := p.c[0].Length
	l1 := p.c[1].Length
	l2 := p.c[2].Length

	x := l0 + l1*math32.Cos(a1) + l2*math32.Cos(a2)
	z := l1*math32.Sin(a1) + l2*math32.Sin(a2)

	p.pos[0] = x * math32.Cos(a0)
	p.pos[1] = x * math32.Sin(a0)
	p.pos[2] = z
	p.pos[3] = 0
	p.pos[4] = a2
	p.pos[5] = a0

	destView := destination.View().(mat.Matrix)
	for i := 0; i < planarEffectorSize; i++ {
		destView[i][0] = p.pos[i]
	}
	return nil
}

// Backward consumes a desired effector column (positions used, orientation slots
// ignored) and writes joint angles back into the 3×1 `controls` column.
func (p *p3d) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := ensureColumn(destination, planarEffectorSize); err != nil {
		return err
	}
	if err := ensureColumn(controls, len(p.params)); err != nil {
		return err
	}
	if state != nil {
		if err := ensureColumn(state, len(p.params)); err != nil {
			return err
		}
		stateView := state.View().(mat.Matrix)
		for i := range p.params {
			p.params[i] = stateView[i][0]
		}
	}

	destView := destination.View().(mat.Matrix)
	x := destView[0][0]
	y := destView[1][0]
	z := destView[2][0]

	l0 := p.c[0].Length
	l1 := p.c[1].Length
	l2 := p.c[2].Length

	xPrime := math32.Sqrt(x*x+y*y) - l0
	rSquared := xPrime*xPrime + z*z
	r := math32.Sqrt(rSquared)

	denomBeta := 2 * l1 * l2
	if denomBeta == 0 {
		return kintypes.ErrUnsupportedOperation
	}
	betaCos := (l1*l1 + l2*l2 - rSquared) / denomBeta
	betaCos = clamp(betaCos, -1, 1)
	beta := math32.Acos(betaCos)

	denomAlpha := 2 * l1 * r
	if denomAlpha == 0 {
		return kintypes.ErrUnsupportedOperation
	}
	alphaCos := (rSquared + l1*l1 - l2*l2) / denomAlpha
	alphaCos = clamp(alphaCos, -1, 1)
	alpha := math32.Acos(alphaCos)

	p.params[0] = p.c[0].Limit(math32.Atan2(y, x))
	p.params[1] = p.c[1].Limit(math32.Atan2(z, xPrime) + alpha)
	p.params[2] = p.c[2].Limit(beta - math32.Pi)

	controlsView := controls.View().(mat.Matrix)
	for i := range p.params {
		controlsView[i][0] = p.params[i]
	}

	return nil
}

func clamp(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

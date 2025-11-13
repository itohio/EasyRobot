package planar

import (
	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
)

const planarEffectorSize = 6

var _ kintypes.Bidirectional = (*p2d)(nil)

type p2d struct {
	c            [2]Config
	params       [2]float32
	pos          [planarEffectorSize]float32
	dimensions   kintypes.Dimensions
	capabilities kintypes.Capabilities
}

func New2DOF(cfg [2]Config) *p2d {
	return &p2d{
		c: cfg,
		dimensions: kintypes.Dimensions{
			StateRows:    2,
			StateCols:    1,
			ControlSize:  2,
			ActuatorSize: 2,
		},
		capabilities: kintypes.Capabilities{
			Holonomic:        true,
			SupportsLateral:  true,
			SupportsVertical: true,
			ConstraintRank:   2,
		},
	}
}

func (p *p2d) Dimensions() kintypes.Dimensions {
	return p.dimensions
}

func (p *p2d) Capabilities() kintypes.Capabilities {
	return p.capabilities
}

func (p *p2d) ConstraintSet() kintypes.Constraints {
	return kintypes.Constraints{}
}

// Forward reads the 2×1 joint column `[a0, a1]` and writes the effector
// `[x, y, z, roll, pitch, yaw]` (with unused orientation slots zeroed) into
// `destination`.
func (p *p2d) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
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
	a1 := p.c[1].Limit(p.params[1])
	p.params[0] = a0
	p.params[1] = a1

	l0 := p.c[0].Length
	l1 := p.c[1].Length

	x := l0 + l1*math32.Cos(a1)
	z := l1 * math32.Sin(a1)

	p.pos[0] = x * math32.Cos(a0)
	p.pos[1] = x * math32.Sin(a0)
	p.pos[2] = z
	p.pos[3] = 0
	p.pos[4] = a1
	p.pos[5] = a0

	destView := destination.View().(mat.Matrix)
	for i := 0; i < planarEffectorSize; i++ {
		destView[i][0] = p.pos[i]
	}
	return nil
}

// Backward consumes a desired effector column `[x, y, z, …]` where only
// positions are used and writes joint angles into the 2×1 `controls` column.
func (p *p2d) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
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

	xPrime := math32.Sqrt(x*x+y*y) - l0

	p.params[0] = p.c[0].Limit(math32.Atan2(y, x))
	p.params[1] = p.c[1].Limit(math32.Atan2(z, xPrime))

	controlsView := controls.View().(mat.Matrix)
	for i := range p.params {
		controlsView[i][0] = p.params[i]
	}

	return nil
}

func ensureColumn(m mattype.Matrix, rows int) error {
	if m == nil {
		return kintypes.ErrInvalidDimensions
	}
	if m.Rows() != rows || m.Cols() < 1 {
		return kintypes.ErrInvalidDimensions
	}
	return nil
}

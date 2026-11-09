package planar

import (
	"fmt"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
)

var _ kintypes.Bidirectional = (*p3d)(nil)

// reachTol is the relative slack on the femur–tibia reach annulus that absorbs
// float32 rounding of targets computed by Forward.
const reachTol = 1e-5

// p3d holds per-solve scratch state (params, pos), so an instance is not safe
// for concurrent use: use one instance per goroutine (per leg).
type p3d struct {
	c            [3]Config
	params       [3]float32
	pos          [planarEffectorSize]float32
	dimensions   kintypes.Dimensions
	capabilities kintypes.Capabilities
}

// New3DOF builds a coxa–femur–tibia leg: joint 0 yaws about +z, joints 1 and 2
// pitch in the vertical plane through the yaw angle (tibia relative to femur,
// positive up). Every Config must Validate, and the femur and tibia lengths
// must be > 0. Errors wrap kintypes.ErrInvalidConfig.
func New3DOF(cfg [3]Config) (*p3d, error) {
	for i := range cfg {
		if err := cfg[i].Validate(); err != nil {
			return nil, fmt.Errorf("planar: joint %d: %w", i, err)
		}
	}
	if !(cfg[1].Length > 0 && cfg[2].Length > 0) {
		return nil, fmt.Errorf("%w: planar: femur and tibia lengths must be > 0, got %g, %g", kintypes.ErrInvalidConfig, cfg[1].Length, cfg[2].Length)
	}
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
	}, nil
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

	stateView := column(state)
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

	destView := column(destination)
	for i := range planarEffectorSize {
		destView[i][0] = p.pos[i]
	}
	return nil
}

// Backward consumes a desired effector column (positions used, orientation
// slots ignored) and writes joint angles into the 3×1 `controls` column.
//
// Candidates are tried in order: coxa towards the foot, then coxa turned by π
// with the foot behind the yaw axis (signed reach); for each, knee down
// (tibia ≤ 0) then knee up. Angles are wrapped into the limits mod 2π. The
// first in-limit candidate is exact and returns nil. Otherwise `controls`
// still receives a best-effort, in-limit solution and the error is
// kintypes.ErrUnreachable (outside the femur–tibia annulus for both coxa
// directions: the leg is stretched or folded towards the target) or
// kintypes.ErrJointLimit (reachable only past a limit: the in-limit pose
// nearest the target over both coxa directions, see closestInLimits).
// A NaN/Inf target returns kintypes.ErrNonFinite with `controls` untouched.
// `state`, if given, must be 3×1; it is not needed by the analytic solution.
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
	}

	destView := column(destination)
	x := destView[0][0]
	y := destView[1][0]
	z := destView[2][0]
	for _, v := range [3]float32{x, y, z} {
		if math32.IsNaN(v) || math32.IsInf(v, 0) {
			return kintypes.ErrNonFinite
		}
	}

	err := p.solve(x, y, z)

	controlsView := column(controls)
	for i := range p.params {
		controlsView[i][0] = p.params[i]
	}
	return err
}

// solve writes the solution (or best effort) into p.params.
func (p *p3d) solve(x, y, z float32) error {
	l0, l1, l2 := p.c[0].Length, p.c[1].Length, p.c[2].Length
	rMax := l1 + l2
	rMin := math32.Abs(l1 - l2)
	tol := reachTol * rMax

	rho := math32.Sqrt(x*x + y*y)
	yaw := math32.Atan2(y, x)

	reachable := false
	for dir := range 2 {
		a0, s := yaw, rho
		if dir == 1 {
			a0, s = yaw+math32.Pi, -rho // foot behind the yaw axis
		}
		xp := s - l0
		r := math32.Sqrt(xp*xp + z*z)
		if r > rMax+tol || r < rMin-tol {
			continue
		}
		reachable = true
		knee := math32.Acos(clamp((r*r-l1*l1-l2*l2)/(2*l1*l2), -1, 1))
		for _, a2 := range [2]float32{-knee, knee} {
			a1 := math32.Atan2(z, xp) - math32.Atan2(l2*math32.Sin(a2), l1+l2*math32.Cos(a2))
			w0, v0 := p.c[0].wrap(a0)
			w1, v1 := p.c[1].wrap(a1)
			w2, v2 := p.c[2].wrap(a2)
			if v0+v1+v2 == 0 {
				p.params = [3]float32{w0, w1, w2}
				return nil
			}
		}
	}

	if reachable {
		p.params = p.closestInLimits(yaw, x, y, z)
		return kintypes.ErrJointLimit
	}

	// Unreachable: stretch (or fold) towards the target, coxa towards the foot
	// unless only the reversed coxa fits the limits.
	a0, s := yaw, rho
	if _, v := p.c[0].wrap(yaw); v > 0 {
		if _, vb := p.c[0].wrap(yaw + math32.Pi); vb < v {
			a0, s = yaw+math32.Pi, -rho
		}
	}
	xp := s - l0
	r := math32.Sqrt(xp*xp + z*z)
	a2 := -math32.Acos(clamp((r*r-l1*l1-l2*l2)/(2*l1*l2), -1, 1))
	a1 := math32.Atan2(z, xp) - math32.Atan2(l2*math32.Sin(a2), l1+l2*math32.Cos(a2))
	w0, _ := p.c[0].wrap(a0)
	w1, _ := p.c[1].wrap(a1)
	w2, _ := p.c[2].wrap(a2)
	p.params = [3]float32{w0, w1, w2}
	return kintypes.ErrUnreachable
}

// closestInLimits is the ErrJointLimit best effort: for the coxa clamped
// towards the foot and turned by π, the in-limit femur–tibia pose nearest the
// target. With no exact solution the in-plane minimum lies on an edge of the
// femur × tibia limit box or at the stretched/folded tibia, so it evaluates
// each tibia candidate (both limits, 0, π) with the femur re-aimed and each
// femur limit with the tibia re-aimed, everything clamped.
func (p *p3d) closestInLimits(yaw, x, y, z float32) [3]float32 {
	l0, l1, l2 := p.c[0].Length, p.c[1].Length, p.c[2].Length
	bestErr2 := math32.Inf(1)
	var best [3]float32
	try := func(a0, a1, a2 float32) {
		if e2 := p.residual2(a0, a1, a2, x, y, z); e2 < bestErr2 {
			bestErr2, best = e2, [3]float32{a0, a1, a2}
		}
	}
	for _, c := range [2]float32{yaw, yaw + math32.Pi} {
		a0, _ := p.c[0].wrap(c)
		xc := math32.Cos(a0)*x + math32.Sin(a0)*y - l0 // target in the leg plane
		for _, t := range [4]float32{p.c[2].Min, p.c[2].Max, 0, math32.Pi} {
			a2, _ := p.c[2].wrap(t)
			a1, _ := p.c[1].wrap(math32.Atan2(z, xc) - math32.Atan2(l2*math32.Sin(a2), l1+l2*math32.Cos(a2)))
			try(a0, a1, a2)
		}
		for _, a1 := range [2]float32{p.c[1].Min, p.c[1].Max} {
			a2, _ := p.c[2].wrap(math32.Atan2(z-l1*math32.Sin(a1), xc-l1*math32.Cos(a1)) - a1)
			try(a0, a1, a2)
		}
	}
	return best
}

// residual2 is |FK(a0, a1, a2) − (x, y, z)|² (angles in the limits).
func (p *p3d) residual2(a0, a1, a2, x, y, z float32) float32 {
	l0, l1, l2 := p.c[0].Length, p.c[1].Length, p.c[2].Length
	s := l0 + l1*math32.Cos(a1) + l2*math32.Cos(a1+a2)
	dx := s*math32.Cos(a0) - x
	dy := s*math32.Sin(a0) - y
	dz := l1*math32.Sin(a1) + l2*math32.Sin(a1+a2) - z
	return dx*dx + dy*dy + dz*dz
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

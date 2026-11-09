package dh

import (
	"fmt"

	"github.com/chewxy/math32"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/mat"
	mattype "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const effectorSize = 7

// Solver tuning. λ and linear step caps scale with the chain size (sum of
// link offsets), so the solver behaves the same in metres or millimetres.
const (
	lambdaInit     = 1e-2     // initial damping, × chain scale
	lambdaMin      = 3e-3     // damping floor, × chain scale: singular values σ ≪ λmin·scale are damped (relative σ cutoff)
	lambdaMax      = 1e2      // damping ceiling, × chain scale: steps become negligible, the solve has stalled
	lambdaDown     = 0.25     // λ factor after an accepted step
	lambdaUp       = 4        // λ factor after a rejected step
	maxAngleStep   = 0.35     // rad per iteration for θ/α joints
	maxLinearStep  = 0.1      // × chain scale per iteration for r/d joints
	maxTargetStep  = 0.2      // × chain scale: Cartesian error used per step is clamped to this length
	crawlWindow    = 8        // iterations per progress check while a joint is blocked
	crawlRatio     = 0.81     // residual² ratio over a window above which a blocked solve counts as stalled (< 10 % distance gain)
	liftIterations = 30       // extra iterations with limits lifted to classify a blocked failure
	liftGain       = 1.0 / 16 // residual² ratio (4× closer) lifting the limits must reach to report ErrJointLimit
	maxRestarts    = 4        // restarts (reshape at a limit-blocked joint, or perturb off a stationary point) within the shared iteration budget
	perturbAngle   = 0.2      // rad: restart nudge for θ/α joints
	perturbLinear  = 0.05     // × chain scale: restart nudge for r/d joints
	stationaryTol  = 1e-2     // |Jᵀe| ≤ tol·‖J‖·|e|: the lifted solve sits at a residual minimum (unreachable)
)

var (
	_ kintypes.Bidirectional = (*DenavitHartenberg)(nil)

	identity4 = mat.Matrix4x4{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}
)

// DenavitHartenberg is a serial chain of DH joints. It owns preallocated
// solver workspaces, so an instance is not safe for concurrent use: use one
// instance per goroutine (per leg).
type DenavitHartenberg struct {
	c             []Config
	eps           float32
	maxIterations int
	params        []float32
	pos           [effectorSize]float32
	H0i           []mat.Matrix4x4
	constraints   kintypes.Constraints
	dimensions    kintypes.Dimensions
	capabilities  kintypes.Capabilities

	// Inverse-kinematics workspace, allocated once in New.
	jac   []float32 // 3×dof position Jacobian, row-major
	dq    []float32 // joint step
	q     []float32 // last accepted iterate
	best  []float32 // lowest-residual iterate over all restarts
	scale float32   // Σ Config.maxOffset: bound on |effector − base| and unit for λ

	lifted bool // limits ignored (failure classification only)
}

// New validates every joint (see Config.Validate), eps > 0 (position
// tolerance in the chain's length unit) and maxIterations >= 0, and returns a
// chain with all solver buffers preallocated. Errors wrap
// kintypes.ErrInvalidConfig.
func New(eps float32, maxIterations int, cfg ...Config) (*DenavitHartenberg, error) {
	if !(eps > 0) || math32.IsInf(eps, 0) {
		return nil, fmt.Errorf("%w: dh: eps must be finite and > 0, got %g", kintypes.ErrInvalidConfig, eps)
	}
	if maxIterations < 0 {
		return nil, fmt.Errorf("%w: dh: maxIterations must be >= 0, got %d", kintypes.ErrInvalidConfig, maxIterations)
	}
	dof := len(cfg)
	if dof == 0 {
		return nil, fmt.Errorf("%w: dh: at least one joint required", kintypes.ErrInvalidConfig)
	}
	var scale float32
	for i := range cfg {
		if err := cfg[i].Validate(); err != nil {
			return nil, fmt.Errorf("dh: joint %d: %w", i, err)
		}
		scale += cfg[i].maxOffset()
	}
	if scale == 0 {
		scale = 1
	}

	return &DenavitHartenberg{
		eps:           eps,
		maxIterations: maxIterations,
		c:             append([]Config(nil), cfg...),
		params:        make([]float32, dof),
		H0i:           make([]mat.Matrix4x4, dof+1),
		jac:           make([]float32, 3*dof),
		dq:            make([]float32, dof),
		q:             make([]float32, dof),
		best:          make([]float32, dof),
		scale:         scale,
		dimensions: kintypes.Dimensions{
			StateRows:    dof,
			StateCols:    1,
			ControlSize:  dof,
			ActuatorSize: dof,
		},
		capabilities: kintypes.Capabilities{
			Holonomic:      true,
			Underactuated:  false,
			ConstraintRank: dof,
		},
	}, nil
}

func (p *DenavitHartenberg) Dimensions() kintypes.Dimensions {
	return p.dimensions
}

func (p *DenavitHartenberg) Capabilities() kintypes.Capabilities {
	return p.capabilities
}

func (p *DenavitHartenberg) ConstraintSet() kintypes.Constraints {
	return p.constraints
}

// Forward interprets `state` as a DOF×1 joint parameter column vector and
// writes the end-effector pose `[x, y, z, qx, qy, qz, qw]` into `destination`.
// Joint values are clamped to their limits.
func (p *DenavitHartenberg) Forward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := p.loadState(state); err != nil {
		return err
	}
	if err := ensureColumn(destination, effectorSize); err != nil {
		return err
	}

	if err := p.forwardInternal(); err != nil {
		return err
	}

	view := column(destination)
	for i := range effectorSize {
		view[i][0] = p.pos[i]
	}
	return nil
}

// Backward consumes a desired position `[x, y, z, …]` from `destination`
// (orientation rows are ignored) and writes joint parameters into the
// `controls` column vector; `state` seeds the solver.
//
// The solver is damped least squares, Δq = Jᵀ(JJᵀ + λ²I)⁻¹e, with adaptive λ
// (Levenberg–Marquardt accept/reject), the Cartesian error clamped per step, a
// per-joint step cap, angle wrapping into the limits, and joints that sit on a
// limit while the step pushes them outward removed from the step (active set).
// If a solve ends blocked by a limit or stalled, it restarts (at most
// maxRestarts times, sharing the maxIterations budget) from a reshaped pose
// at the most proximal blocked angular joint (elbow flipped and the joint
// re-aimed at the target, re-aimed, or aimed away from it), or nudged off the
// stationary point. It does not allocate.
//
// Input errors (dimensions; kintypes.ErrNonFinite for a NaN/Inf target
// position or seed) return before solving and leave `controls` untouched.
// Otherwise `controls` is always written, with the best (lowest-residual)
// in-limit solution found. On failure the error is kintypes.ErrUnreachable (target
// farther from the base than the chain can extend, or a residual minimum with
// no joint blocked), kintypes.ErrJointLimit (the best solution is blocked by a
// joint limit) or kintypes.ErrNoConvergence (iteration budget spent).
func (p *DenavitHartenberg) Backward(state mattype.Matrix, destination mattype.Matrix, controls mattype.Matrix) error {
	if err := p.loadState(state); err != nil {
		return err
	}
	if err := p.loadEffector(destination); err != nil {
		return err
	}
	if err := ensureColumn(controls, len(p.params)); err != nil {
		return err
	}
	if !finite(p.pos[:3]) || !finite(p.params) {
		return kintypes.ErrNonFinite
	}

	err := p.inverseInternal()

	view := column(controls)
	for i, val := range p.params {
		view[i][0] = val
	}
	return err
}

// computeFrames clamps p.params to the limits (unless p.lifted), fills H0i
// and the effector position p.pos[0:3]. It does not allocate.
func (p *DenavitHartenberg) computeFrames() error {
	p.H0i[0] = identity4
	var A mat.Matrix4x4
	for i := range p.c {
		cfg := &p.c[i]
		if !p.lifted {
			p.params[i] = cfg.Limit(p.params[i])
		}
		if !cfg.transform(p.params[i], &A) {
			return kintypes.ErrUnsupportedOperation
		}
		mul4(&p.H0i[i+1], &p.H0i[i], &A)
	}
	n := len(p.c)
	p.pos[0] = p.H0i[n][0][3]
	p.pos[1] = p.H0i[n][1][3]
	p.pos[2] = p.H0i[n][2][3]
	return nil
}

func (p *DenavitHartenberg) forwardInternal() error {
	if err := p.computeFrames(); err != nil {
		return err
	}
	quat := p.H0i[len(p.c)].View().Quaternion()
	copy(p.pos[3:], quat.View().(vec.Vector))
	return nil
}

// mul4 computes dst = a·b; dst must not alias a or b.
func mul4(dst, a, b *mat.Matrix4x4) {
	for r := range 4 {
		for c := range 4 {
			dst[r][c] = a[r][0]*b[0][c] + a[r][1]*b[1][c] + a[r][2]*b[2][c] + a[r][3]*b[3][c]
		}
	}
}

func (p *DenavitHartenberg) residual2(tx, ty, tz float32) (ex, ey, ez, e2 float32) {
	ex, ey, ez = tx-p.pos[0], ty-p.pos[1], tz-p.pos[2]
	return ex, ey, ez, ex*ex + ey*ey + ez*ez
}

func (p *DenavitHartenberg) inverseInternal() error {
	tx, ty, tz := p.pos[0], p.pos[1], p.pos[2]
	eps2 := p.eps * p.eps

	for i := range p.params {
		p.params[i] = p.c[i].wrapLimit(p.params[i])
	}
	budget := p.maxIterations
	// Best effort starts at this call's seed: a residual that overflows to
	// +Inf never beats bestErr2, and p.best must not keep the last call's q.
	copy(p.best, p.params)
	bestErr2 := math32.Inf(1)
	bestStalled := false
	for attempt := 0; ; attempt++ {
		if err := p.computeFrames(); err != nil {
			return err
		}
		used, ex, ey, ez, err2, stalled, err := p.descend(tx, ty, tz, eps2, budget)
		if err != nil {
			return err
		}
		budget -= used
		if err2 < bestErr2 {
			copy(p.best, p.q)
			bestErr2, bestStalled = err2, stalled
		}
		if err2 < eps2 || budget <= 0 || attempt == maxRestarts {
			break
		}
		if !p.reshape(tx, ty, tz, ex, ey, ez) && !(stalled && p.perturb()) {
			break
		}
	}

	copy(p.params, p.best)
	if err := p.computeFrames(); err != nil {
		return err
	}
	ex, ey, ez, err2 := p.residual2(tx, ty, tz)
	if err2 < eps2 {
		return nil
	}
	return p.failureReason(tx, ty, tz, ex, ey, ez, eps2, bestStalled)
}

// descend runs at most `budget` damped least-squares iterations from p.params
// (frames already computed). On return p.params == p.q is the lowest-residual
// iterate and H0i is evaluated there; e and err2 are its residual.
func (p *DenavitHartenberg) descend(tx, ty, tz, eps2 float32, budget int) (used int, ex, ey, ez, err2 float32, stalled bool, err error) {
	n := len(p.c)
	ex, ey, ez, err2 = p.residual2(tx, ty, tz)
	copy(p.q, p.params)

	lambda := lambdaInit * p.scale
	window2 := err2
	for ; used < budget && err2 >= eps2; used++ {
		p.jacobian()
		// Clamp-and-zero-column: a joint on a limit whose gradient points
		// outward takes no part in the step.
		anyBlocked := false
		for i := range n {
			if p.blocked(i, ex, ey, ez) {
				p.jac[i], p.jac[n+i], p.jac[2*n+i] = 0, 0, 0
				anyBlocked = true
			}
		}
		// Crawling along a limit: end the attempt so a restart can try another
		// branch instead of spending the budget here.
		if used > 0 && used%crawlWindow == 0 {
			if anyBlocked && err2 > crawlRatio*window2 {
				stalled = true
				return
			}
			window2 = err2
		}
		// Aim at most maxTargetStep·scale towards the target per iteration:
		// the linearisation only holds near the current pose (Buss' clamp).
		sx, sy, sz := ex, ey, ez
		if lim := maxTargetStep * p.scale; err2 > lim*lim {
			s := lim / math32.Sqrt(err2)
			sx, sy, sz = ex*s, ey*s, ez*s
		}
		if p.limitAwareStep(sx, sy, sz, lambda) {
			p.capStep()
			for i := range p.params {
				p.params[i] = p.move(i, p.q[i]+p.dq[i])
			}
			if err = p.computeFrames(); err != nil {
				return
			}
			nex, ney, nez, ne2 := p.residual2(tx, ty, tz)
			if ne2 < err2 {
				copy(p.q, p.params)
				ex, ey, ez, err2 = nex, ney, nez, ne2
				lambda = math32.Max(lambda*lambdaDown, lambdaMin*p.scale)
				continue
			}
			copy(p.params, p.q)
			if err = p.computeFrames(); err != nil {
				return
			}
		}
		lambda *= lambdaUp
		if lambda > lambdaMax*p.scale {
			stalled = true // no damping level improves the residual: local minimum
			used++
			return
		}
	}
	return
}

// perturb prepares a restart from p.q nudged off a stationary point (e.g. a
// stretched, singular pose whose Jacobian is orthogonal to the error): joints
// alternately move by ±perturbAngle / ±perturbLinear·scale (wrapped/clamped;
// the other sign when a limit swallows the nudge).
func (p *DenavitHartenberg) perturb() bool {
	for i := range p.c {
		c := &p.c[i]
		d := float32(perturbAngle)
		if !c.angular() {
			d = perturbLinear * p.scale
		}
		if i%2 == 1 {
			d = -d
		}
		v := p.move(i, p.q[i]+d)
		if math32.Abs(v-p.q[i]) < 0.5*math32.Abs(d) {
			v = p.move(i, p.q[i]-d)
		}
		p.params[i] = v
	}
	return true
}

// reshape prepares a restart from p.q for the most proximal limit-blocked
// angular joint i, taking the first option that changes the pose:
//   - if the next angular joint j turns about a parallel axis (a planar
//     elbow), flip the elbow (q_j → −q_j) and re-aim joint i at the target
//     (a femur pinned while the knee bends the wrong way);
//   - re-aim joint i at the target (a yaw pinned while the folded foot is
//     already behind its axis);
//   - aim joint i directly away from the target (a yaw pinned while aiming at
//     a target behind its axis: the folded leg then reaches it in-limit).
//
// Angles are wrapped/clamped into the limits. Frames must be at p.q.
func (p *DenavitHartenberg) reshape(tx, ty, tz, ex, ey, ez float32) bool {
	p.jacobian()
	for i := range p.c {
		if !p.c[i].angular() || !p.blocked(i, ex, ey, ez) {
			continue
		}
		if j := p.nextAngular(i); j >= 0 && p.parallelAxes(i, j) && p.flipElbow(i, j, tx, ty, tz) {
			return true
		}
		aim := p.aimAngle(i, tx, ty, tz)
		for _, off := range [2]float32{0, math32.Pi} {
			if v := p.c[i].wrapLimit(p.q[i] + aim + off); math32.Abs(v-p.q[i]) > 1e-3 {
				p.params[i] = v
				return true
			}
		}
	}
	return false
}

func (p *DenavitHartenberg) nextAngular(i int) int {
	for j := i + 1; j < len(p.c); j++ {
		if p.c[j].angular() {
			return j
		}
	}
	return -1
}

// axis returns the rotation axis and a point on it for angular joint i from
// the current frames (θ: z and origin of H0i[i]; α: x and origin of H0i[i+1]).
func (p *DenavitHartenberg) axis(i int) (ax, ay, az, ox, oy, oz float32) {
	if p.c[i].Index == 1 {
		H := &p.H0i[i+1]
		return H[0][0], H[1][0], H[2][0], H[0][3], H[1][3], H[2][3]
	}
	H := &p.H0i[i]
	return H[0][2], H[1][2], H[2][2], H[0][3], H[1][3], H[2][3]
}

func (p *DenavitHartenberg) parallelAxes(i, j int) bool {
	ax, ay, az, _, _, _ := p.axis(i)
	bx, by, bz, _, _, _ := p.axis(j)
	cx, cy, cz := ay*bz-az*by, az*bx-ax*bz, ax*by-ay*bx
	return cx*cx+cy*cy+cz*cz < 1e-6
}

// flipElbow sets params to p.q with q_j negated and q_i re-aimed at the
// target. Frames must be at p.q on entry; they are left at the new pose.
func (p *DenavitHartenberg) flipElbow(i, j int, tx, ty, tz float32) bool {
	v := p.c[j].wrapLimit(-p.q[j])
	if math32.Abs(v-p.q[j]) < 1e-3 {
		return false // straight elbow: nothing to flip
	}
	p.params[j] = v
	if p.computeFrames() != nil {
		copy(p.params, p.q)
		return false
	}
	p.params[i] = p.c[i].wrapLimit(p.q[i] + p.aimAngle(i, tx, ty, tz))
	return true
}

// aimAngle is the rotation of angular joint i (about its axis, current
// frames) that turns the effector direction, seen from the axis, towards the
// target.
func (p *DenavitHartenberg) aimAngle(i int, tx, ty, tz float32) float32 {
	ax, ay, az, ox, oy, oz := p.axis(i)
	ux, uy, uz := p.pos[0]-ox, p.pos[1]-oy, p.pos[2]-oz
	wx, wy, wz := tx-ox, ty-oy, tz-oz
	du, dw := ax*ux+ay*uy+az*uz, ax*wx+ay*wy+az*wz
	ux, uy, uz = ux-du*ax, uy-du*ay, uz-du*az
	wx, wy, wz = wx-dw*ax, wy-dw*ay, wz-dw*az
	sin := ax*(uy*wz-uz*wy) + ay*(uz*wx-ux*wz) + az*(ux*wy-uy*wx)
	cos := ux*wx + uy*wy + uz*wz
	return math32.Atan2(sin, cos)
}

// move maps a joint value after a step: wrapped/clamped into the limits, or
// unchanged while the limits are lifted.
func (p *DenavitHartenberg) move(i int, v float32) float32 {
	if p.lifted {
		return v
	}
	return p.c[i].wrapLimit(v)
}

// failureReason classifies a failed solve; p.params/H0i hold the best
// in-limit iterate p.best:
//   - target beyond the bounding sphere → ErrUnreachable;
//   - a joint blocked by its limit → classifyBlocked (ErrJointLimit or
//     ErrNoConvergence, or a solution found with the limits lifted);
//   - stalled at a residual minimum with free joints (the target is off the
//     reachable surface, e.g. out of a planar arm's plane) → ErrUnreachable;
//   - iteration budget spent while still improving → ErrNoConvergence.
func (p *DenavitHartenberg) failureReason(tx, ty, tz, ex, ey, ez, eps2 float32, stalled bool) error {
	if math32.Sqrt(tx*tx+ty*ty+tz*tz) > p.scale*(1+1e-5) {
		return kintypes.ErrUnreachable
	}
	p.jacobian()
	for i := range p.c {
		if p.blocked(i, ex, ey, ez) {
			return p.classifyBlocked(tx, ty, tz, eps2, ex*ex+ey*ey+ez*ez)
		}
	}
	if stalled {
		return kintypes.ErrUnreachable
	}
	return kintypes.ErrNoConvergence
}

// classifyBlocked tells a target that needs a joint beyond its limits from a
// blocked local minimum. From p.best (nudged off it: a blocked best pose is
// often symmetric about the target) it descends with the limits lifted for up
// to liftIterations extra iterations, nudging off stationary points:
//   - converged inside the limits (mod 2π) → that solution is taken (nil);
//   - converged, or ≥ 4× closer than p.best (liftGain), only past a limit →
//     ErrJointLimit;
//   - lifted solve stationary at a nonzero residual (no pose, even past the
//     limits, gets closer nearby: e.g. inside the inner workspace hole) →
//     ErrUnreachable;
//   - otherwise → ErrNoConvergence (a blocked local minimum the lifted solve
//     is still descending from).
//
// This is a local test: an in-limit solution on a branch the restarts did not
// reach may still exist. Unless a solution is taken, p.params/H0i are
// restored to p.best.
func (p *DenavitHartenberg) classifyBlocked(tx, ty, tz, eps2, bestErr2 float32) error {
	p.lifted = true
	copy(p.q, p.best)
	p.perturb()
	err := p.computeFrames()
	err2 := math32.Inf(1)
	for budget := liftIterations; err == nil && budget > 0; {
		var used int
		var stalled bool
		used, _, _, _, err2, stalled, err = p.descend(tx, ty, tz, eps2, budget)
		budget -= used
		if err != nil || err2 < eps2 || !stalled {
			break
		}
		p.perturb()
		err = p.computeFrames()
	}
	stationary := false
	if err == nil && err2 >= eps2 {
		copy(p.params, p.q)
		if err = p.computeFrames(); err == nil {
			stationary = p.stationary(tx, ty, tz)
		}
	}
	p.lifted = false
	if err != nil {
		return err
	}

	inLimits := true
	for i := range p.c {
		v := p.c[i].wrapLimit(p.q[i])
		d := v - p.q[i]
		if p.c[i].angular() {
			d -= twoPi * math32.Round(d/twoPi)
		}
		if math32.Abs(d) > 1e-5*math32.Max(1, p.colScale(i)) {
			inLimits = false
		}
		p.params[i] = v
	}
	if err2 < eps2 && inLimits {
		if err := p.computeFrames(); err != nil {
			return err
		}
		if _, _, _, e2 := p.residual2(tx, ty, tz); e2 < eps2 {
			return nil
		}
	}
	result := kintypes.ErrNoConvergence
	switch {
	case err2 < eps2 || err2 < liftGain*bestErr2:
		result = kintypes.ErrJointLimit
	case stationary:
		result = kintypes.ErrUnreachable
	}
	copy(p.params, p.best)
	if err := p.computeFrames(); err != nil {
		return err
	}
	return result
}

// stationary reports the frames' pose as a stationary point of the residual:
// the gradient Jᵀe is negligible, |Jᵀe| ≤ stationaryTol·‖J‖_F·|e|. It
// ignores limits (blocked joints are not removed).
func (p *DenavitHartenberg) stationary(tx, ty, tz float32) bool {
	ex, ey, ez, e2 := p.residual2(tx, ty, tz)
	p.jacobian()
	n := len(p.c)
	var g2, j2 float32
	for i := range n {
		a, b, c := p.jac[i], p.jac[n+i], p.jac[2*n+i]
		g := a*ex + b*ey + c*ez
		g2 += g * g
		j2 += a*a + b*b + c*c
	}
	return g2 <= stationaryTol*stationaryTol*j2*e2
}

// blocked reports joint i sitting on a limit while the steepest-descent
// direction Jᵢ·e pushes it further out. Continuous joints wrap and never block.
func (p *DenavitHartenberg) blocked(i int, ex, ey, ez float32) bool {
	c := &p.c[i]
	if p.lifted || c.continuous() {
		return false
	}
	n := len(p.c)
	g := p.jac[i]*ex + p.jac[n+i]*ey + p.jac[2*n+i]*ez
	return (p.params[i] >= c.Max && g > 0) || (p.params[i] <= c.Min && g < 0)
}

// jacobian fills p.jac with the position Jacobian from H0i (standard DH):
// θ: zᵢ × (pₙ − oᵢ); α: xᵢ₊₁ × (pₙ − oᵢ₊₁); r: xᵢ₊₁; d: zᵢ, where zᵢ, oᵢ are
// the z axis and origin of H0i[i] and xᵢ₊₁, oᵢ₊₁ the x axis and origin of
// H0i[i+1]. Columns are scaled by colScale (J·S) so every column is a length.
func (p *DenavitHartenberg) jacobian() {
	n := len(p.c)
	Hn := &p.H0i[n]
	pnx, pny, pnz := Hn[0][3], Hn[1][3], Hn[2][3]
	for i := range p.c {
		var cx, cy, cz float32
		switch p.c[i].Index {
		case 0, 1:
			ax, ay, az, ox, oy, oz := p.axis(i)
			dx, dy, dz := pnx-ox, pny-oy, pnz-oz
			cx, cy, cz = ay*dz-az*dy, az*dx-ax*dz, ax*dy-ay*dx
		case 2:
			H := &p.H0i[i+1]
			cx, cy, cz = H[0][0], H[1][0], H[2][0]
		case 3:
			H := &p.H0i[i]
			cx, cy, cz = H[0][2], H[1][2], H[2][2]
		}
		s := p.colScale(i)
		p.jac[i], p.jac[n+i], p.jac[2*n+i] = cx*s, cy*s, cz*s
	}
}

// colScale is S in J·S: 1 for angular joints (columns already lengths), the
// chain scale for r/d joints (unitless columns), so the damping λ — a length —
// weighs all joints alike in any length unit. Steps solved in scaled
// coordinates are mapped back by the same factor.
func (p *DenavitHartenberg) colScale(i int) float32 {
	if p.c[i].angular() {
		return 1
	}
	return p.scale
}

// limitAwareStep computes a damped least-squares step in which joints sitting
// on a limit never step outward: such a joint's Jacobian column is zeroed and
// the step recomputed (active set, at most one pass per joint).
func (p *DenavitHartenberg) limitAwareStep(ex, ey, ez, lambda float32) bool {
	if p.lifted {
		return p.dlsStep(ex, ey, ez, lambda)
	}
	n := len(p.c)
	for range n + 1 {
		if !p.dlsStep(ex, ey, ez, lambda) {
			return false
		}
		changed := false
		for i := range n {
			c := &p.c[i]
			if c.continuous() || p.dq[i] == 0 {
				continue
			}
			if (p.params[i] >= c.Max && p.dq[i] > 0) || (p.params[i] <= c.Min && p.dq[i] < 0) {
				p.jac[i], p.jac[n+i], p.jac[2*n+i] = 0, 0, 0
				changed = true
			}
		}
		if !changed {
			return true
		}
	}
	return true
}

// dlsStep solves (J'J'ᵀ + λ²I)y = e by 3×3 Cholesky for the scaled Jacobian
// J' = J·S and sets p.dq = S·J'ᵀy.
// It reports false if the system is numerically not positive definite.
func (p *DenavitHartenberg) dlsStep(ex, ey, ez, lambda float32) bool {
	n := len(p.c)
	j0, j1, j2 := p.jac[:n], p.jac[n:2*n], p.jac[2*n:3*n]
	var a00, a01, a02, a11, a12, a22 float32
	for i := range n {
		a00 += j0[i] * j0[i]
		a01 += j0[i] * j1[i]
		a02 += j0[i] * j2[i]
		a11 += j1[i] * j1[i]
		a12 += j1[i] * j2[i]
		a22 += j2[i] * j2[i]
	}
	l2 := lambda * lambda
	a00 += l2
	a11 += l2
	a22 += l2

	if !(a00 > 0) {
		return false
	}
	l00 := math32.Sqrt(a00)
	l10 := a01 / l00
	l20 := a02 / l00
	d1 := a11 - l10*l10
	if !(d1 > 0) {
		return false
	}
	l11 := math32.Sqrt(d1)
	l21 := (a12 - l20*l10) / l11
	d2 := a22 - l20*l20 - l21*l21
	if !(d2 > 0) {
		return false
	}
	l22 := math32.Sqrt(d2)

	y0 := ex / l00
	y1 := (ey - l10*y0) / l11
	y2 := (ez - l20*y0 - l21*y1) / l22
	x2 := y2 / l22
	x1 := (y1 - l21*x2) / l11
	x0 := (y0 - l10*x1 - l20*x2) / l00

	for i := range n {
		p.dq[i] = (j0[i]*x0 + j1[i]*x1 + j2[i]*x2) * p.colScale(i)
	}
	return true
}

// capStep clamps each joint step to its cap (per joint: one poorly
// conditioned joint must not freeze the others, as uniform scaling would).
func (p *DenavitHartenberg) capStep() {
	for i := range p.dq {
		limit := float32(maxAngleStep)
		if !p.c[i].angular() {
			limit = maxLinearStep * p.scale
		}
		p.dq[i] = math32.Max(-limit, math32.Min(limit, p.dq[i]))
	}
}

func (p *DenavitHartenberg) loadState(state mattype.Matrix) error {
	if err := ensureColumn(state, len(p.params)); err != nil {
		return err
	}

	view := column(state)
	for i := range p.params {
		p.params[i] = view[i][0]
	}
	return nil
}

func (p *DenavitHartenberg) loadEffector(destination mattype.Matrix) error {
	if err := ensureColumn(destination, effectorSize); err != nil {
		return err
	}
	view := column(destination)
	for i := range effectorSize {
		p.pos[i] = view[i][0]
	}
	return nil
}

func finite(v []float32) bool {
	for _, x := range v {
		if math32.IsNaN(x) || math32.IsInf(x, 0) {
			return false
		}
	}
	return true
}

// column returns a row view of m without copying when m is already a
// mat.Matrix (Matrix.View allocates).
func column(m mattype.Matrix) mat.Matrix {
	if v, ok := m.(mat.Matrix); ok {
		return v
	}
	return m.View().(mat.Matrix)
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

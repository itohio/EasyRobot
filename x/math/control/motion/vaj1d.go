// Package vaj provides jerk-limited motion profiles with bounded velocity
// and acceleration for one-dimensional trajectories.
package vaj

import (
	"math"

	"github.com/chewxy/math32"
)

// VAJ1D implements a jerk-limited (velocity–acceleration–jerk) trajectory
// generator for one-dimensional motion. The profile respects configured
// limits while tracking a target position that may change over time.
//
// Velocity and Acceleration are signed. Each Update picks the largest jerk
// toward the target for which, after this tick, a time-optimal brake (jerk to
// full deceleration, hold, jerk back to zero) still stops at or before the
// target and the speed limit can still be met. Following that rule yields the
// time-optimal S-curve: accelerate, cruise, then ride the braking curve onto
// the target. A target change mid-move brakes smoothly instead of snapping.
//
// Once the brake would finish within a few ticks, the remaining motion is
// solved as an exact landing: a sequence of per-tick jerks that reaches the
// target at rest after an integer number of ticks, within all limits and
// without overshoot or velocity reversal. That plan is then followed, which
// removes the chatter a per-tick bang-bang choice shows around the target.
// The plan is a continuous-time jerk profile on the tick grid of the Update
// that solved it; later Updates advance along it by their own sample period,
// so a measured, jittering period (time.Since per tick) still lands exactly
// without reversal.
//
// There is no deadband: every Target change, however small, is tracked to an
// exact landing, so a Target carrying noise keeps the output moving. Callers
// that feed noisy targets apply their own deadband.
//
// Limits: a zero, negative or NaN limit makes Update a no-op (Output = Input,
// state untouched), like an invalid sample period. +Inf disables a limit. An
// unlimited jerk is taken as the jerk that reaches the acceleration limit
// within one tick (the speed limit, when the acceleration is unlimited too);
// with all three limits unlimited the output jumps onto the target.
//
// A NaN or infinite Target is replaced by the position where a brake from the
// current state comes to rest, so the profile stops there and holds.
//
// The state is integrated in float64 (pos, vel, acc); the exported float32
// fields are its rounded mirrors. Writing Input, Velocity or Acceleration
// from outside replaces the corresponding internal value on the next Update
// and drops the landing plan (Input only when it moves by more than float32
// noise). float32 integration biased moves whose braking spans thousands of
// ticks into overshoots and reversals.
type VAJ1D struct {
	maxV, maxA, maxJ       float32
	Velocity, Acceleration float32
	Input, Output, Target  float32

	// pos, vel and acc are the float64 state; last, lastV and lastA are the
	// float32 values published for them, used to detect external writes.
	// posCarry holds the part of the integrated position pos cannot
	// represent (compensated summation), so slow creeps far from the origin
	// still progress.
	pos, vel, acc      float64
	posCarry           float64
	last, lastV, lastA float32

	// limV, limA and limJ are the effective limits of the current Update
	// (see limits).
	limV, limA, limJ float64

	// tol is the rounding slack of the feasibility tests, refreshed every
	// Update; margin keeps the per-tick controller this far inside the
	// braking curve (see Update).
	tol, margin float64

	// plan holds the per-tick jerks of the terminal landing on a grid of
	// planH; piece planIdx has run for planOff so far. planLen is 0 when no
	// landing is under way. planTarget is the target it was solved for, so a
	// target change discards it.
	plan             [landMaxTicks]float64
	planLen, planIdx int
	planH, planOff   float64
	planTarget       float32
}

const (
	// jerkIterations bisection steps resolve the jerk well below the
	// float32 resolution of the published state.
	jerkIterations = 32
	// landHorizonTicks: a landing is attempted once the brake would finish
	// within this many ticks.
	landHorizonTicks = 6
	// landExtraTicks: landings up to this many ticks beyond the brake time
	// are tried before falling back to the per-tick controller.
	landExtraTicks = 3
	// landMaxTicks bounds the landing length (and the plan storage).
	landMaxTicks = landHorizonTicks + landExtraTicks + 1
	// landMarginTicks scales the slack (in units of maxJ*dt³) the per-tick
	// controller keeps inside the braking curve for the landing.
	landMarginTicks = 3
	// landStallRatio and landStallRounds end a landing search whose box
	// violation shrinks by less than 5% for 4 rounds in a row.
	landStallRatio  = 0.95
	landStallRounds = 4
	// brakeJerkDerate scales the jerk of the planned final deceleration ramp.
	brakeJerkDerate = 0.9
	// landProjections bounds the box/affine projection rounds per landing.
	landProjections = 64
	// landLimitSlack is the relative slack of the landing limit checks.
	landLimitSlack = 1e-6
	// cruiseSnap is the relative rounding below which a state at the speed
	// limit with no acceleration left is taken as exact cruise.
	cruiseSnap = 1e-12
)

// New1D constructs a VAJ1D profile with the supplied kinematic limits.
// The returned profile is ready to track targets via successive Update calls.
// Limits must be positive; +Inf disables a limit and a zero, negative or NaN
// limit makes Update a no-op (see VAJ1D).
func New1D(maxVelocity, maxAcceleration, jerk float32) VAJ1D {
	return VAJ1D{
		maxV: maxVelocity,
		maxA: maxAcceleration,
		maxJ: jerk,
	}
}

// Reset clears the velocity and acceleration state without altering limits.
func (l *VAJ1D) Reset() *VAJ1D {
	l.Velocity, l.Acceleration = 0, 0
	l.vel, l.acc = 0, 0
	l.lastV, l.lastA = 0, 0
	l.planLen = 0
	return l
}

// Update advances the internal profile by the supplied sample period and moves
// the Output (starting from Input) toward Target while respecting jerk,
// acceleration, and velocity constraints.
func (l *VAJ1D) Update(samplePeriod float32) *VAJ1D {
	// NaN, non-positive and infinite periods and invalid limits leave the
	// state untouched.
	if !(samplePeriod > 0) || math32.IsInf(samplePeriod, 1) || !l.limits(float64(samplePeriod)) {
		l.Output = l.Input
		return l
	}
	dt := float64(samplePeriod)
	l.syncExternal()
	if math32.IsNaN(l.Target) || math32.IsInf(l.Target, 0) {
		l.Target = l.restPoint()
	}
	if math.IsInf(l.limJ, 1) {
		// Every limit is unlimited.
		l.settle()
		return l
	}
	if l.Target != l.planTarget {
		l.planLen = 0
	}
	J, A, V := l.limJ, l.limA, l.limV
	target := float64(l.Target)
	e := target - l.pos - l.posCarry
	v, a := l.vel, l.acc
	stop, tStop := l.brake(v, a)

	// The float64 rounding of e (relative to |target|) and of the brake
	// distance (relative to |e|) bounds how finely the target can be
	// resolved; it is far below the float32 resolution of Output.
	l.tol = 4*ulp64(math.Abs(target)) + 16*ulp64(math.Abs(e))

	// Settle when the brake ends within this tick on the target, up to what
	// one tick can resolve within the jerk, acceleration and speed limits,
	// and the settling step itself stays within one tick of maxV. Within the
	// rounding floor of the target, settle as soon as dropping the remaining
	// velocity and acceleration to zero fits one tick of acceleration and
	// jerk.
	floor := 2 * l.tol
	tol := math.Min(math.Min(J*dt*dt*dt/6, A*dt*dt/2), V*dt) + floor
	if (tStop <= dt && math.Abs(e-stop) <= tol && math.Abs(e) <= V*dt) ||
		(math.Abs(e) <= 4*floor && math.Abs(v) <= A*dt && math.Abs(a) <= J*dt) {
		l.settle()
		return l
	}
	if math.Abs(e) <= floor {
		// The remaining error is rounding noise: aim at the current stop
		// point, i.e. just brake to rest, instead of chasing the noise.
		e = stop
	}

	if l.planLen == 0 && tStop <= landHorizonTicks*dt {
		l.planLanding(e, v, a, tStop, dt)
	}
	if l.planLen > 0 {
		dp, vn, an, done := l.followPlan(v, a, dt)
		if done {
			// The plan ends at rest on the target within this tick.
			l.settle()
			return l
		}
		l.publish(dp, vn, an)
		return l
	}

	// Work in the frame where the target lies ahead. If even the hardest
	// brake overshoots, it is applied and the frame flips once past the target.
	s := 1.0
	if e < 0 || (e == 0 && stop > 0) {
		s = -1
	}
	e, v, a = s*e, s*v, s*a

	// Keep about one tick of jerk inside the braking curve so that a whole-
	// tick landing without overshoot still exists when the brake nears its
	// end. Never demand more slack than the current state has; drop it if
	// it cannot be honoured.
	slack := e - s*stop
	l.margin = math.Max(math.Min(math.Min(landMarginTicks*J*dt*dt*dt, e/4), slack), 0)
	// The rounding slack is harmless while the margin (much larger) keeps
	// the stop short of the target. Without the margin, aim exactly at the
	// target first and accept the slack only if nothing else is feasible, so
	// it cannot bias the approach past the target.
	slackTol := l.tol
	if l.margin == 0 {
		l.tol = 0
	}
	j, ok := l.chooseJerk(e, v, a, dt)
	if !ok && l.margin > 0 {
		l.margin, l.tol = 0, 0
		j, ok = l.chooseJerk(e, v, a, dt)
	}
	if !ok {
		l.tol = slackTol
		j, _ = l.chooseJerk(e, v, a, dt)
	}

	dp, vn, an := l.step(v, a, j, dt)
	if math.Abs(vn-V) <= cruiseSnap*V && math.Abs(an) <= cruiseSnap*J*dt {
		// At the speed limit up to rounding: cruise exactly at it, so that
		// chooseJerk skips the bisection on the following ticks.
		vn, an = V, 0
	}
	l.publish(s*dp, s*vn, s*math.Max(math.Min(an, A), -A))
	return l
}

// chooseJerk returns the largest feasible jerk in [-maxJ, maxJ], or -maxJ
// with ok=false when none is feasible.
func (l *VAJ1D) chooseJerk(e, v, a, dt float64) (float64, bool) {
	jmax := l.limJ
	jmin := -jmax
	switch {
	case l.feasible(e, v, a, jmax, dt):
		return jmax, true
	case !l.feasible(e, v, a, jmin, dt):
		return jmin, false
	case a == 0 && v >= l.limV && l.feasible(e, v, a, 0, dt):
		// Cruise: every positive jerk exceeds the speed limit, so the
		// bisection below would return its first midpoint, zero.
		return 0, true
	}
	lo, hi := jmin, jmax
	for range jerkIterations {
		mid := 0.5 * (lo + hi)
		if l.feasible(e, v, a, mid, dt) {
			lo = mid
		} else {
			hi = mid
		}
	}
	return lo, true
}

// limits refreshes the effective limits for a tick of dt and reports whether
// the configured limits are valid (all positive, +Inf meaning unlimited). An
// unlimited jerk becomes the jerk that reaches the acceleration limit within
// one tick, or the speed limit when the acceleration is unlimited too; it
// stays +Inf only when every limit is unlimited.
func (l *VAJ1D) limits(dt float64) bool {
	V, A, J := float64(l.maxV), float64(l.maxA), float64(l.maxJ)
	if !(V > 0 && A > 0 && J > 0) {
		return false
	}
	if math.IsInf(J, 1) {
		switch {
		case !math.IsInf(A, 1):
			J = A / dt
		case !math.IsInf(V, 1):
			J = V / (dt * dt)
		}
	}
	l.limV, l.limA, l.limJ = V, A, J
	return true
}

// syncExternal takes over Input, Velocity and Acceleration written from
// outside since the last Update and drops the landing plan for any of them
// except float32 noise in an externally fed-back Input.
func (l *VAJ1D) syncExternal() {
	if l.Input != l.last {
		ref := math32.Abs(l.Input)
		if t := math32.Abs(l.Target); t > ref && !math32.IsInf(t, 1) {
			ref = t
		}
		if math.Abs(float64(l.Input)-float64(l.last)) > 4*float64(ulp(ref)) {
			l.planLen = 0
		}
		l.pos, l.posCarry = float64(l.Input), 0
	}
	if l.Velocity != l.lastV {
		l.vel = float64(l.Velocity)
		l.planLen = 0
	}
	if l.Acceleration != l.lastA {
		l.acc = float64(l.Acceleration)
		l.planLen = 0
	}
}

// restPoint returns the float32 position where a brake from the current
// state comes to rest, rounded away from the current position so that the
// brake does not pass it.
func (l *VAJ1D) restPoint() float32 {
	var stop float64
	if !math.IsInf(l.limJ, 1) {
		stop, _ = l.brake(l.vel, l.acc)
	}
	goal := l.pos + l.posCarry + stop
	g := float32(goal)
	switch {
	case stop > 0 && float64(g) < goal:
		g = math32.Nextafter(g, math32.Inf(1))
	case stop < 0 && float64(g) > goal:
		g = math32.Nextafter(g, math32.Inf(-1))
	}
	return g
}

// followPlan advances (v, a) by dt along the landing plan and returns the
// displacement and the new velocity and acceleration. The plan is a
// continuous-time jerk profile with pieces of planH, so it lands exactly
// whatever sample periods follow it; done reports that it ends within this
// tick, at rest on the target.
func (l *VAJ1D) followPlan(v, a, dt float64) (dp, vn, an float64, done bool) {
	for {
		j := l.plan[l.planIdx]
		left := l.planH - l.planOff
		var d float64
		if dt < left {
			d, v, a = integrate(v, a, j, dt)
			l.planOff += dt
			return dp + d, v, a, false
		}
		d, v, a = integrate(v, a, j, left)
		dp += d
		dt -= left
		l.planIdx++
		l.planOff = 0
		if l.planIdx == l.planLen {
			return dp, v, a, true
		}
		if dt <= 0 {
			return dp, v, a, false
		}
	}
}

// publish moves the internal state by dp to velocity v and acceleration a
// and mirrors it into the exported float32 fields.
func (l *VAJ1D) publish(dp, v, a float64) {
	move := l.posCarry + dp
	next := l.pos + move
	l.posCarry = move - (next - l.pos)
	l.pos = next
	l.vel, l.acc = v, a
	l.mirror()
}

// settle places the profile at rest exactly on Target.
func (l *VAJ1D) settle() {
	l.pos, l.posCarry, l.vel, l.acc = float64(l.Target), 0, 0, 0
	l.planLen, l.planIdx = 0, 0
	l.mirror()
	l.Output = l.Target
	l.Input, l.last = l.Output, l.Output
}

// mirror publishes the float64 state into the exported fields.
func (l *VAJ1D) mirror() {
	l.Output = float32(l.pos)
	l.Input, l.last = l.Output, l.Output
	l.Velocity, l.lastV = float32(l.vel), float32(l.vel)
	l.Acceleration, l.lastA = float32(l.acc), float32(l.acc)
}

// planLanding searches the shortest feasible exact landing, starting at the
// brake duration rounded up to whole ticks (at least 3, the fewest ticks
// that can match position, velocity and acceleration), and stores it.
func (l *VAJ1D) planLanding(e, v, a, tStop, dt float64) {
	l.planLen, l.planIdx, l.planOff = 0, 0, 0
	first := max(3, int(math.Ceil(tStop/dt)))
	for n := first; n <= first+landExtraTicks && n <= landMaxTicks; n++ {
		if l.solveLanding(n, e, v, a, dt) {
			l.planLen = n
			l.planTarget = l.Target
			l.planH = dt
			return
		}
	}
}

// solveLanding computes per-tick jerks j[0..n) that move (e, v, a) to rest at
// the target after n ticks with |j| <= maxJ and stores them in plan if the
// resulting motion respects the remaining limits (also inside each tick),
// never passes the target and does not reverse velocity (beyond turning back
// once if it starts moving away).
//
// With m = n-k ticks between tick k and the end, jerk j_k contributes h*j_k
// to the final acceleration, h²(2m-1)/2*j_k to the velocity and
// h³(3m²-3m+1)/6*j_k to the position. In units x_k = h³ j_k the three end
// conditions form M x = b. Starting from the minimum-norm solution
// x = Mᵀ (M Mᵀ)⁻¹ b, alternating projections onto the jerk box and back
// onto the affine set find a box-feasible solution when one exists; the
// final iterate satisfies the end conditions exactly.
func (l *VAJ1D) solveLanding(n int, e, v, a, h float64) bool {
	nh := float64(n) * h
	b := [3]float64{
		-a * h * h,
		-(v + nh*a) * h,
		e - nh*v - nh*nh*a/2,
	}
	var rows [landMaxTicks][3]float64
	var g [3][3]float64
	for k := range n {
		m := float64(n - k)
		rows[k] = [3]float64{1, (2*m - 1) / 2, (3*m*m - 3*m + 1) / 6}
		for r := range 3 {
			for c := range 3 {
				g[r][c] += rows[k][r] * rows[k][c]
			}
		}
	}
	gi, ok := inverse3(g)
	if !ok {
		return false
	}

	var x [landMaxTicks]float64
	projectAffine(&x, &rows, &gi, &b, n)
	h3 := h * h * h
	box := l.limJ * h3
	prev, stalls := -1.0, 0
	for range landProjections {
		var over float64
		for k := range n {
			if x[k] > box {
				over += x[k] - box
				x[k] = box
			} else if x[k] < -box {
				over += -box - x[k]
				x[k] = -box
			}
		}
		if over == 0 {
			break
		}
		// Infeasible landings stop shrinking the box violation quickly.
		if prev >= 0 && over > landStallRatio*prev {
			if stalls++; stalls >= landStallRounds {
				return false
			}
		} else {
			stalls = 0
		}
		prev = over
		projectAffine(&x, &rows, &gi, &b, n)
	}

	s := 1.0
	if e < 0 || (e == 0 && v > 0) {
		s = -1
	}
	reversals := 0
	if s*v < 0 {
		reversals = -1
	}
	jLim := l.limJ * (1 + landLimitSlack)
	aLim := l.limA * (1 + landLimitSlack)
	vLim := l.limV * (1 + landLimitSlack)
	posTol := landLimitSlack * math.Abs(e)
	rest := e
	prevSign := math.Copysign(1, s*v)
	for k := range n {
		j := x[k] / h3
		if math.Abs(j) > jLim {
			return false
		}
		l.plan[k] = j
		// Velocity extremum inside the tick, where the acceleration
		// crosses zero.
		if t := -a / j; j != 0 && t > 0 && t < h {
			vx := v + a*t/2
			if math.Abs(vx) > vLim || (k < n-1 && s*vx < 0 && prevSign > 0) {
				return false
			}
		}
		var dp float64
		dp, v, a = integrate(v, a, j, h)
		rest -= dp
		if math.Abs(a) > aLim || math.Abs(v) > vLim || (k < n-1 && s*rest < -posTol) {
			return false
		}
		if k < n-1 && s*v != 0 {
			if sg := math.Copysign(1, s*v); sg != prevSign {
				reversals++
				prevSign = sg
			}
		}
	}
	return reversals <= 0
}

// projectAffine projects x[0..n) onto the affine set M x = b in place, where
// rows[k] is column k of M and gi = (M Mᵀ)⁻¹.
func projectAffine(x *[landMaxTicks]float64, rows *[landMaxTicks][3]float64, gi *[3][3]float64, b *[3]float64, n int) {
	var res [3]float64
	for r := range 3 {
		res[r] = -b[r]
		for k := range n {
			res[r] += rows[k][r] * x[k]
		}
	}
	var y [3]float64
	for r := range 3 {
		y[r] = gi[r][0]*res[0] + gi[r][1]*res[1] + gi[r][2]*res[2]
	}
	for k := range n {
		x[k] -= rows[k][0]*y[0] + rows[k][1]*y[1] + rows[k][2]*y[2]
	}
}

// inverse3 inverts a 3x3 matrix by its adjugate.
func inverse3(g [3][3]float64) ([3][3]float64, bool) {
	var inv [3][3]float64
	inv[0][0] = g[1][1]*g[2][2] - g[1][2]*g[2][1]
	inv[0][1] = g[0][2]*g[2][1] - g[0][1]*g[2][2]
	inv[0][2] = g[0][1]*g[1][2] - g[0][2]*g[1][1]
	inv[1][0] = g[1][2]*g[2][0] - g[1][0]*g[2][2]
	inv[1][1] = g[0][0]*g[2][2] - g[0][2]*g[2][0]
	inv[1][2] = g[0][2]*g[1][0] - g[0][0]*g[1][2]
	inv[2][0] = g[1][0]*g[2][1] - g[1][1]*g[2][0]
	inv[2][1] = g[0][1]*g[2][0] - g[0][0]*g[2][1]
	inv[2][2] = g[0][0]*g[1][1] - g[0][1]*g[1][0]
	det := g[0][0]*inv[0][0] + g[0][1]*inv[1][0] + g[0][2]*inv[2][0]
	if det == 0 {
		return inv, false
	}
	for r := range 3 {
		for c := range 3 {
			inv[r][c] /= det
		}
	}
	return inv, true
}

// feasible reports whether applying jerk j for dt from (v, a) keeps the
// position at or before the remaining distance e through this tick and the
// following brake, and keeps the velocity within maxV inside the tick and at
// the peak reached by ramping the acceleration back to zero.
func (l *VAJ1D) feasible(e, v, a, j, dt float64) bool {
	J, V := l.limJ, l.limV
	dp, vn, an := l.step(v, a, j, dt)
	if dp > e+l.tol {
		return false
	}
	peak := vn
	if an > 0 {
		// Ramping the acceleration back to zero takes at least an/J, and at
		// least one tick at tick-constant jerk.
		peak += an * math.Max(an/(2*J), dt/2)
	}
	if a > 0 && an < 0 {
		// The acceleration crosses zero inside the tick at the velocity peak.
		peak = math.Max(peak, v-a*a/(2*j))
	}
	if peak > V {
		return false
	}
	rest := e - dp + l.tol
	if vn > 0 && an < 0 && an*an > 2*J*vn {
		// The brake reverses the velocity: it first ramps the deceleration
		// back up and crosses v=0 before stopping, so its farthest point is
		// that crossing, not the final rest position.
		t0 := (-an - math.Sqrt(an*an-2*J*vn)) / J
		x0, _, _ := integrate(vn, an, J, t0)
		if x0 > rest {
			return false
		}
	}
	stop, _ := l.brake(vn, an)
	return stop <= rest-l.margin
}

// step advances (v, a) for dt under jerk j, holding the acceleration once it
// reaches ±maxA within the tick.
func (l *VAJ1D) step(v, a, j, dt float64) (float64, float64, float64) {
	A := l.limA
	an := a + j*dt
	var limit float64
	switch {
	case an > A:
		limit = A
	case an < -A:
		limit = -A
	default:
		return integrate(v, a, j, dt)
	}
	t1 := math.Max((limit-a)/j, 0)
	x1, v1, _ := integrate(v, a, j, t1)
	x2, v2, _ := integrate(v1, limit, 0, dt-t1)
	return x1 + x2, v2, limit
}

// brake returns the signed displacement and duration of the jerk-limited stop
// from (v, a) to rest with zero acceleration: time-optimal except that the
// final ramp of the deceleration back to zero uses brakeJerkDerate*maxJ.
func (l *VAJ1D) brake(v, a float64) (float64, float64) {
	J, A := l.limJ, l.limA
	// The final ramp of the deceleration back to zero is planned at a
	// derated jerk: the remaining headroom lets the whole-tick landing
	// absorb the mismatch between tick-constant jerk and the continuous
	// switching times.
	Jb := brakeJerkDerate * J
	// Velocity reached by ramping the acceleration straight to zero.
	vz := v + a*math.Abs(a)/(2*J)
	if vz == 0 {
		t := math.Abs(a) / J
		dp, _, _ := integrate(v, a, -math.Copysign(J, a), t)
		return dp, t
	}
	sg := math.Copysign(1, vz)
	v, a = sg*v, sg*a
	if a < 0 && v < a*a/(2*Jb) {
		// Between the derated and the full jerk: a single ramp with the jerk
		// that zeroes velocity and acceleration together.
		j := a * a / (2 * v)
		t := -a / j
		dp, _, _ := integrate(v, a, j, t)
		return sg * dp, t
	}
	// Decelerate with peak ap (triangular), or hold -A when ap would exceed it:
	// v + (a²-ap²)/(2J) - ap²/(2Jb) - ap*t2 = 0.
	k := 1/(2*J) + 1/(2*Jb)
	ap := math.Sqrt(math.Max((v+a*a/(2*J))/k, 0))
	var t2 float64
	if ap > A {
		ap = A
		t2 = math.Max((v+a*a/(2*J)-A*A*k)/A, 0)
	}
	t1 := math.Max((a+ap)/J, 0)
	t3 := ap / Jb
	x1, v1, _ := integrate(v, a, -J, t1)
	x2, v2, _ := integrate(v1, -ap, 0, t2)
	x3, _, _ := integrate(v2, -ap, Jb, t3)
	return sg * (x1 + x2 + x3), t1 + t2 + t3
}

// integrate advances (v, a) under constant jerk j for t and returns the
// displacement and final velocity and acceleration.
func integrate(v, a, j, t float64) (float64, float64, float64) {
	return (v + (a/2+j*t/6)*t) * t, v + (a+j*t/2)*t, a + j*t
}

// ulp64 returns the spacing of float64 values at magnitude x.
func ulp64(x float64) float64 {
	return math.Nextafter(x, math.Inf(1)) - x
}

// ulp returns the spacing of float32 values at magnitude x.
func ulp(x float32) float32 {
	return math32.Nextafter(x, math32.Inf(1)) - x
}

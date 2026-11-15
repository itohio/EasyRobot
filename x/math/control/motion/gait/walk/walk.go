// Package walk turns a body twist into servo commands for a legged robot, one
// tick at a time: gait scheduler → footholds → stance/swing feet in the world →
// body→hip transforms → analytic leg IK → ServoMap degrees. Step does not
// allocate.
//
// Footholds (Raibert-style): leg i with neutral foot nᵢ
// (body frame) touches down at pᵢ* = nᵢ + ½·T_st·(v + ω×nᵢ), so its stance
// sweeps symmetrically through the neutral point. Stance feet stay fixed in
// the world while the body moves.
//
// Reach clipping: every sub-step the slew-limited twist is scaled uniformly
// for all legs (never distorted) to the largest scale at which, continuing at
// that twist, every leg's plan keeps its required margin inside its reachable
// annulus sector. A leg's plan is: a planted foot from where it really is
// through the rest of its stance and the swing from that lift-off point to its
// next foothold, a swinging foot through the rest of its swing (and, if its
// foothold is frozen, the stance on it), and the steady cycle after that.
// The required margin is min(Margin, the plan under a safe fallback): standing
// still, or holding the previously applied twist, whichever falls less below
// Margin. The fallback always qualifies, so a robot that cannot stop safely
// slows down along its previous twist as far as the plan allows, or keeps it
// (Applied, Scale), and the margin never ratchets down through a choice. A
// NaN plan margin (e.g. a huge rotation over a very long gait cycle) is
// infeasible. Margin is therefore kept from any state
// that has it, up to the plan check's sampling error (planSamplingError,
// 0.2 mm, asserted against a dense check); a state that has already lost it
// (e.g. Margin raised, or a twist step without slew limit landing a frozen
// foothold short) is not driven below its best fallback's plan. The scale can
// drop within one sub-step (reach takes precedence over the slew limit).
//
// Swing (§3): from the lift-off point to the predicted touchdown point, the
// horizontal motion is minimum jerk (10τ³ − 15τ⁴ + 6τ⁵), delayed by
// LiftFraction at both ends so the foot lifts and lands vertically, and the
// height is the lift-to-touchdown line plus H·(1 − cos 2πτ)/2, apex = H at
// mid-swing. The body-frame foothold p* is re-planned from the current twist
// each sub-step until the horizontal progress reaches freezeAt (½), then
// frozen; the world target is p* on the touchdown pose predicted from the
// current twist. The horizontal update covers the remaining min-jerk fraction
// of the remaining distance, so a twist change moves a swinging foot
// continuously, and late in the swing only by the change of the predicted
// body motion over the remaining swing time.
//
// Step sub-steps dt (at most maxSubStep = 25 ms, and at most half the
// shortest stance or swing phase, so no phase is skipped and the clip acts
// often enough; New rejects phases below MinPhaseTime = 20 ms, so a Step runs
// at most 100 sub-steps), processes at most MaxStepDt = 1 s per call (longer
// dt is clamped and reported with ErrDtTooLong), rejects non-finite commands
// or dt with kintypes.ErrNonFinite and commands above MaxCommandSpeed or
// MaxCommandYawRate with ErrCommandTooLarge. The yaw is kept in (−π, π].
// A Walker, like its chain.LegSolver, is not safe for concurrent use.
package walk

import (
	"errors"
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/control/kinematics/chain"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/control/motion/gait/scheduler"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// Twist is the body velocity in the body frame: VX forward, VY left (m/s), W
// yaw rate about z (rad/s, counter-clockwise).
type Twist struct {
	VX, VY, W float32
}

func (t Twist) scaled(s float32) Twist { return Twist{t.VX * s, t.VY * s, t.W * s} }

// Config configures a Walker.
type Config struct {
	Body    chain.Body        // every leg must be coxa–femur–tibia shaped (chain.Planar3)
	Pattern scheduler.Pattern // Pattern.Legs == len(Body.Mounts), same leg order
	Period  float32           // gait cycle, s
	// SwingHeight is the swing apex above the lift-off→touchdown line, m.
	SwingHeight float32
	// LiftFraction ∈ [0, 0.4) is the share of the swing at each end with
	// vertical motion only (lift-off and landing without scuffing).
	LiftFraction float32
	// Margin is the reach margin (m, see legReach) every planned stance foot
	// keeps; the twist is scaled down to honour it.
	Margin float32
	// MaxAccel (m/s²) and MaxAngAccel (rad/s²) slew-limit the commanded twist;
	// 0 disables the limit. A legged body cannot change speed instantly: stance
	// feet planned for the old twist must still reach their lift-off point.
	MaxAccel, MaxAngAccel float32
}

// hipFrame caches a mount: hip position and the cosine/sine of its yaw.
type hipFrame struct {
	x, y, z, cos, sin float32
}

// legState is the per-leg contact bookkeeping.
type legState struct {
	stance   bool
	lift     vec.Vector3D // world lift-off point of the current swing
	target   vec.Vector3D // world touchdown target, re-predicted every sub-step
	foothold vec.Vector3D // body-frame touchdown point p*, re-planned until frozen
	frozen   bool         // foothold no longer re-planned (see freezeAt)
	tau      float32      // swing progress at the previous sub-step
}

// Walker runs the per-tick pipeline. It is not safe for concurrent use.
type Walker struct {
	cfg     Config
	body    chain.Body
	sched   *scheduler.Scheduler
	solver  *chain.LegSolver
	reach   []legReach
	hips    []hipFrame
	legs    []legState
	feet    []vec.Vector3D // world foot targets
	q       [][]float32    // joint angles per leg (IK seeds and results)
	errs    []error
	req     []float32 // per-leg margin the clip requires (see clip)
	held    []float32 // per-leg plan margin when holding the applied twist
	servo   []int     // first servo index of each leg
	nServo  int
	pose    chain.Pose
	raw     Twist // slew-limited command, before reach scaling
	scale   float32
	applied Twist   // raw·scale: the twist the body moves with
	maxSub  float32 // longest sub-step, s (see maxSubStep)
	bind    int     // leg that failed the last feasibility check
}

// Plan check and clip search parameters.
const (
	stanceSamples = 4    // stance segment sampled at stanceSamples+1 points
	swingSamples  = 12   // swing segment sampled at swingSamples+1 points
	refineIters   = 8    // golden-section steps per refined dip
	refineBand    = 5e-3 // m: sampled dips this close to the smallest are refined
	clipGrid      = 16   // scale grid k/clipGrid scanned downwards
	clipIters     = 8    // bisection steps inside one grid cell
	clipFastStep  = 1.0 / 256
	// maxSubStep caps a sub-step (s); sub-steps are also at most half the
	// shortest stance or swing phase.
	maxSubStep = 0.025
	// freezeAt is the horizontal swing progress after which the body-frame
	// foothold p* is no longer re-planned from the current twist.
	freezeAt = 0.5
)

// MaxStepDt is the longest dt (s) one Step processes; a longer dt is clamped
// to it and Step returns ErrDtTooLong (bounded work per call).
const MaxStepDt = 1

// MinPhaseTime (s) is the shortest stance or swing phase New accepts. Step
// sub-steps at most half the shortest phase, so one Step runs at most
// 2·MaxStepDt/MinPhaseTime = 100 sub-steps.
const MinPhaseTime = 0.02

// MaxCommandSpeed (m/s, |VX| and |VY|) and MaxCommandYawRate (rad/s, |W|)
// bound the command Step accepts. They are far beyond any legged robot (the
// reach clip scales real commands down long before) and keep the plan
// arithmetic finite; Step rejects a larger command with ErrCommandTooLarge.
const (
	MaxCommandSpeed   = 100
	MaxCommandYawRate = 100
)

// planSamplingError (m) bounds how far the sampled plan check (legMargin) can
// overestimate a plan's true minimum margin; the kept margin can fall short of
// Margin by this much.
const planSamplingError = 2e-4

var (
	// ErrDtTooLong reports a dt above MaxStepDt; Step processed MaxStepDt.
	ErrDtTooLong = errors.New("walk: dt longer than MaxStepDt, clamped")
	// ErrCommandTooLarge reports a command above MaxCommandSpeed or
	// MaxCommandYawRate; Step changed nothing.
	ErrCommandTooLarge = errors.New("walk: command above MaxCommandSpeed or MaxCommandYawRate")
)

// New validates the configuration and places the body at the origin, yaw 0,
// at the height that puts the neutral feet on the ground plane z = 0, with all
// feet at their neutral points. The walker keeps a deep copy of cfg.Body.
// Errors wrap kintypes.ErrInvalidConfig (a leg that is not coxa–femur–tibia
// shaped also kintypes.ErrUnsupportedOperation), or are the scheduler's
// errors for a bad pattern or period.
func New(cfg Config) (*Walker, error) {
	b := cfg.Body.Clone()
	cfg.Body = b
	if err := b.Validate(); err != nil {
		return nil, err
	}
	n := len(b.Mounts)
	if cfg.Pattern.Legs != n {
		return nil, fmt.Errorf("%w: walk: pattern %q has %d legs, body %d", kintypes.ErrInvalidConfig, cfg.Pattern.Name, cfg.Pattern.Legs, n)
	}
	if !finiteNonNeg(cfg.SwingHeight, cfg.Margin, cfg.MaxAccel, cfg.MaxAngAccel) {
		return nil, fmt.Errorf("%w: walk: SwingHeight, Margin, MaxAccel and MaxAngAccel must be finite and >= 0", kintypes.ErrInvalidConfig)
	}
	if !(cfg.LiftFraction >= 0 && cfg.LiftFraction < 0.4) {
		return nil, fmt.Errorf("%w: walk: LiftFraction %g not in [0, 0.4)", kintypes.ErrInvalidConfig, cfg.LiftFraction)
	}
	sched, err := scheduler.New(cfg.Pattern, cfg.Period)
	if err != nil {
		return nil, fmt.Errorf("walk: %w", err)
	}
	solver, err := b.NewLegSolver()
	if err != nil {
		return nil, err
	}
	w := &Walker{
		cfg:    cfg,
		body:   b,
		sched:  sched,
		solver: solver,
		reach:  make([]legReach, n),
		hips:   make([]hipFrame, n),
		legs:   make([]legState, n),
		feet:   make([]vec.Vector3D, n),
		q:      make([][]float32, n),
		errs:   make([]error, n),
		req:    make([]float32, n),
		held:   make([]float32, n),
		servo:  make([]int, n),
		scale:  1,
	}
	var height float32
	for i, m := range b.Mounts {
		p3, err := m.Leg.Planar3()
		if err != nil {
			return nil, fmt.Errorf("%w: walk: leg %d: %w", kintypes.ErrInvalidConfig, i, err)
		}
		if w.reach[i], err = newLegReach(p3); err != nil {
			return nil, fmt.Errorf("walk: leg %d: %w", i, err)
		}
		height -= b.Neutral[i][2] / float32(n)
		w.q[i] = make([]float32, len(m.Leg.Joints))
		w.servo[i] = w.nServo
		w.hips[i] = hipFrame{x: m.Hip[0], y: m.Hip[1], z: m.Hip[2], cos: math32.Cos(m.Yaw), sin: math32.Sin(m.Yaw)}
		w.nServo += len(m.Leg.Joints)
		phase := min(sched.StanceTime(i), sched.SwingTime(i))
		if !(phase >= MinPhaseTime) {
			return nil, fmt.Errorf("%w: walk: leg %d: shortest phase %g s < MinPhaseTime %g s", kintypes.ErrInvalidConfig, i, phase, float32(MinPhaseTime))
		}
		if i == 0 || phase/2 < w.maxSub {
			w.maxSub = phase / 2
		}
		w.maxSub = min(w.maxSub, maxSubStep)
	}
	w.pose = chain.Pose{Z: height}
	for i := range b.Mounts {
		if m := w.planMargin(i, Twist{}); m < cfg.Margin {
			return nil, fmt.Errorf("%w: walk: leg %d: standing and stepping in place keep a reach margin of %.4f m < Margin %.4f m (neutral foot or SwingHeight out of reach)", kintypes.ErrInvalidConfig, i, m, cfg.Margin)
		}
		w.feet[i] = w.pose.ToWorld(b.Neutral[i])
		w.legs[i] = legState{stance: true, target: w.feet[i]}
	}
	if err := w.solver.SolveLegs(w.pose, w.feet, w.q, w.errs); err != nil {
		return nil, fmt.Errorf("walk: neutral stance: %w", err)
	}
	return w, nil
}

func finiteNonNeg(vs ...float32) bool {
	for _, v := range vs {
		if !(v >= 0) || math32.IsInf(v, 1) {
			return false
		}
	}
	return true
}

// Foothold returns the touchdown point p* = n + ½·T_st·(v + ω×n) for neutral
// foot n (body frame) and body twist tw; z is kept.
func Foothold(n vec.Vector3D, tw Twist, stanceTime float32) vec.Vector3D {
	h := stanceTime / 2
	return vec.Vector3D{n[0] + h*(tw.VX-tw.W*n[1]), n[1] + h*(tw.VY+tw.W*n[0]), n[2]}
}

// displacement is the body motion over t at constant twist, in the body frame
// at the start: translation (dx, dy) and rotation θ (SE(2) exponential).
func displacement(tw Twist, t float32) (dx, dy, theta float32) {
	theta = tw.W * t
	if math32.Abs(theta) < 1e-6 {
		return tw.VX * t, tw.VY * t, theta
	}
	s := math32.Sin(theta) / tw.W
	half := math32.Sin(theta / 2)
	c := 2 * half * half / tw.W // (1 − cos θ)/ω
	return s*tw.VX - c*tw.VY, c*tw.VX + s*tw.VY, theta
}

// advance moves a level pose by twist tw for t seconds.
func advance(p chain.Pose, tw Twist, t float32) chain.Pose {
	dx, dy, theta := displacement(tw, t)
	c, s := math32.Cos(p.Yaw), math32.Sin(p.Yaw)
	p.X += c*dx - s*dy
	p.Y += s*dx + c*dy
	p.Yaw = wrapPi(p.Yaw + theta)
	return p
}

// wrapPi wraps any finite angle a to (−π, π].
func wrapPi(a float32) float32 {
	if a > -math32.Pi && a <= math32.Pi {
		return a
	}
	if a = math32.Remainder(a, 2*math32.Pi); a <= -math32.Pi {
		a += 2 * math32.Pi
	}
	return a
}

// afterMotion returns where a world-fixed point at body-frame p appears in the
// body frame after the body moved by tw for t.
func afterMotion(p vec.Vector3D, tw Twist, t float32) vec.Vector3D {
	dx, dy, theta := displacement(tw, t)
	c, s := math32.Cos(theta), math32.Sin(theta)
	x, y := p[0]-dx, p[1]-dy
	return vec.Vector3D{c*x + s*y, -s*x + c*y, p[2]}
}

// beforeMotion maps a point given in the body frame after the body moved by
// tw for t back to the body frame at the start (inverse of afterMotion).
func beforeMotion(p vec.Vector3D, tw Twist, t float32) vec.Vector3D {
	dx, dy, theta := displacement(tw, t)
	c, s := math32.Cos(theta), math32.Sin(theta)
	return vec.Vector3D{dx + c*p[0] - s*p[1], dy + s*p[0] + c*p[1], p[2]}
}

// Plans below are expressed in the body frame now (N) for a body that keeps
// moving at constant twist tw; marginAt is leg i's reach margin of the
// world-fixed point p (frame N) seen from the body at time t.
func (w *Walker) marginAt(i int, p vec.Vector3D, tw Twist, t float32) float32 {
	b := afterMotion(p, tw, t)
	h := &w.hips[i]
	x, y := b[0]-h.x, b[1]-h.y
	return w.reach[i].margin(vec.Vector3D{h.cos*x + h.sin*y, -h.sin*x + h.cos*y, b[2] - h.z})
}

// segment is one piece of a leg's plan in frame N, parametrised by u ∈ [0, 1].
// A stance segment is the planted point x seen from time t0 to t0 + dur. A
// swing segment runs as updateFoot does: from progress tau0 (foot at x at time
// t0, lift-off point lift) to touchdown at td; dur is the swing time.
type segment struct {
	swing       bool
	tw          Twist
	x, lift, td vec.Vector3D
	t0, dur     float32
	tau0, h0    float32
	samples     int
}

// at is leg i's reach margin at parameter u of segment s.
func (w *Walker) at(i int, s *segment, u float32) float32 {
	if !s.swing {
		return w.marginAt(i, s.x, s.tw, s.t0+u*s.dur)
	}
	tau := s.tau0 + u*(1-s.tau0)
	f := s.td
	if s.h0 < 1 {
		r := (w.horizontal(tau) - s.h0) / (1 - s.h0)
		f[0], f[1] = s.x[0]+r*(s.td[0]-s.x[0]), s.x[1]+r*(s.td[1]-s.x[1])
	}
	f[2] = s.lift[2] + (s.td[2]-s.lift[2])*minJerk(tau) + w.cfg.SwingHeight*bump(tau)
	return w.marginAt(i, f, s.tw, s.t0+(tau-s.tau0)*s.dur)
}

// segMin is the smallest margin along segment s: samples at u = k/n, then a
// golden-section refinement between the neighbours of every sampled local
// minimum within refineBand of the smallest sample. The margin is a minimum
// of smooth terms, so a swing can have two dips and a dip can sit at a kink
// between samples. If the smallest sample exceeds floor + refineBand the
// refinement is skipped: the result then only certifies "≥ floor".
func (w *Walker) segMin(i int, s *segment, floor float32) float32 {
	n := s.samples
	var v [swingSamples + 1]float32
	best := math32.Inf(1)
	for k := 0; k <= n; k++ {
		v[k] = w.at(i, s, float32(k)/float32(n))
		best = math32.Min(best, v[k])
	}
	if best > floor+refineBand {
		return best
	}
	m := best
	for k := 0; k <= n; k++ {
		if v[k] > best+refineBand || (k > 0 && v[k-1] < v[k]) || (k < n && v[k+1] < v[k]) {
			continue
		}
		m = math32.Min(m, w.refine(i, s, float32(max(k-1, 0))/float32(n), float32(min(k+1, n))/float32(n)))
	}
	return m
}

// refine is a golden-section search for the smallest margin of s on [a, b].
func (w *Walker) refine(i int, s *segment, a, b float32) float32 {
	const g = 0.618034
	c, d := b-g*(b-a), a+g*(b-a)
	fc, fd := w.at(i, s, c), w.at(i, s, d)
	for range refineIters {
		if fc < fd {
			b, d, fd = d, c, fc
			c = b - g*(b-a)
			fc = w.at(i, s, c)
		} else {
			a, c, fc = c, d, fd
			d = a + g*(b-a)
			fd = w.at(i, s, d)
		}
	}
	return math32.Min(fc, fd)
}

// Plan margins below take a floor: a result below floor + refineBand is the
// plan's minimum (to the sampling error), a larger one only certifies that the
// minimum is above floor. With early set, evaluation stops at the first part
// of the plan below floor. Pass +Inf for the exact value.

// cycleMargin covers a stance of duration tst on the planted point x from time
// t0, then the swing from that lift-off point to the next foothold.
func (w *Walker) cycleMargin(i int, tw Twist, x vec.Vector3D, t0, tst, floor float32, early bool) float32 {
	s := segment{tw: tw, x: x, t0: t0, dur: tst, samples: stanceSamples}
	m := w.segMin(i, &s, floor)
	if early && m < floor {
		return m
	}
	tl := t0 + tst
	td := beforeMotion(Foothold(w.body.Neutral[i], tw, w.sched.StanceTime(i)), tw, tl+w.sched.SwingTime(i))
	return math32.Min(m, w.swingMargin(i, tw, x, x, td, 0, tl, floor))
}

// swingMargin covers a swing as updateFoot runs it: the foot at x with swing
// progress tau0 at time t0, lift-off point lift, touchdown point td.
func (w *Walker) swingMargin(i int, tw Twist, lift, x, td vec.Vector3D, tau0, t0, floor float32) float32 {
	s := segment{swing: true, tw: tw, x: x, lift: lift, td: td, t0: t0, dur: w.sched.SwingTime(i),
		tau0: tau0, h0: w.horizontal(tau0), samples: swingSamples}
	return w.segMin(i, &s, floor)
}

// planMargin is leg i's smallest reach margin over the steady cycle at
// constant twist tw: stance from the foothold to lift-off, then the swing to
// the next foothold. During the vertical lift and landing the body keeps
// moving, so the swing reaches beyond the stance ends.
func (w *Walker) planMargin(i int, tw Twist) float32 {
	return w.steadyMargin(i, tw, math32.Inf(1), false)
}

func (w *Walker) steadyMargin(i int, tw Twist, floor float32, early bool) float32 {
	tst := w.sched.StanceTime(i)
	return w.cycleMargin(i, tw, Foothold(w.body.Neutral[i], tw, tst), 0, tst, floor, early)
}

// legMargin is leg i's smallest reach margin if the body continues at twist
// tw from the current state: a planted foot from where it is through the rest
// of its stance and the swing from there; a swinging foot through the rest of
// its swing and, if its foothold is frozen, the stance on it; then the steady
// cycle.
func (w *Walker) legMargin(i int, tw Twist) float32 {
	return w.legMarginAbove(i, tw, math32.Inf(1), false)
}

// legMarginAbove is legMargin with a floor; early stops at the first part of
// the plan below floor (feasibility checks).
func (w *Walker) legMarginAbove(i int, tw Twist, floor float32, early bool) float32 {
	ls, st := w.sched.Leg(i), &w.legs[i]
	x := w.pose.ToBody(w.feet[i])
	if ls.Stance {
		m := w.cycleMargin(i, tw, x, 0, (1-ls.Progress)*w.sched.StanceTime(i), floor, early)
		if early && m < floor {
			return m
		}
		return math32.Min(m, w.steadyMargin(i, tw, floor, early))
	}
	lift := x // lift-off at the next sub-step (only right after New)
	if !st.stance {
		lift = w.pose.ToBody(st.lift)
	}
	remaining := (1 - ls.Progress) * w.sched.SwingTime(i)
	p := Foothold(w.body.Neutral[i], tw, w.sched.StanceTime(i))
	if st.frozen {
		p = st.foothold
	}
	td := beforeMotion(p, tw, remaining)
	m := w.swingMargin(i, tw, lift, x, td, ls.Progress, 0, floor)
	if early && m < floor {
		return m
	}
	if st.frozen {
		if m = math32.Min(m, w.cycleMargin(i, tw, td, remaining, w.sched.StanceTime(i), floor, early)); early && m < floor {
			return m
		}
	}
	return math32.Min(m, w.steadyMargin(i, tw, floor, early))
}

// feasible reports whether every leg's plan at twist tw keeps its required
// margin; a NaN plan margin is infeasible. The leg that failed last is
// checked first.
func (w *Walker) feasible(tw Twist) bool {
	n := len(w.reach)
	for k := range n {
		i := (w.bind + k) % n
		if m := w.legMarginAbove(i, tw, w.req[i], true); !(m >= w.req[i]) {
			w.bind = i
			return false
		}
	}
	return true
}

// clip picks the twist to move with for the slew-limited command tw. It
// returns the scale s ∈ [0, 1] and the applied twist, normally tw·s with the
// largest s at which every leg's plan from the current state (legMargin)
// keeps its required margin req.
//
// req comes from a safe fallback: standing still, or holding the previously
// applied twist (whose plan was checked last sub-step), whichever falls less
// below Margin; req_i = min(Margin, leg i's plan under that fallback). The
// fallback is feasible by construction, so the margin never ratchets down
// through a choice and a robot that cannot stop safely keeps moving instead.
// If no s is feasible (the fallback is the held twist and stopping is not
// safe), clip returns s = 0 and decelerates along the previous twist: the
// smallest feasible fraction k/clipGrid of it, or the full previous twist if
// none is feasible.
//
// Feasibility need not be monotone in s (a swing or a frozen foothold can
// favour the current speed). Fast path: if the previous scale is still
// feasible and 1/256 more is not, keep it. Otherwise scan the grid k/clipGrid
// downwards from 1 for the largest feasible point (the previous scale counts
// if feasible and higher) and bisect up to the next grid point.
func (w *Walker) clip(tw Twist) (float32, Twist) {
	// Holding the applied twist: if every leg keeps Margin, req = Margin
	// whatever standing still gives, so the stop plan is not needed.
	hold := math32.Inf(1)
	for i := range w.req {
		w.held[i] = w.legMarginAbove(i, w.applied, w.cfg.Margin, false)
		hold = math32.Min(hold, w.held[i])
	}
	holding := false
	if !(hold >= w.cfg.Margin) { // NaN: the held twist is no fallback
		stop := math32.Inf(1)
		for i := range w.req {
			w.req[i] = w.legMarginAbove(i, Twist{}, w.cfg.Margin, false)
			stop = math32.Min(stop, w.req[i])
		}
		holding = stop < w.cfg.Margin && hold > stop
		if holding {
			copy(w.req, w.held)
		}
	} else {
		copy(w.req, w.held)
		holding = true // the held twist keeps Margin: it is the fallback
	}
	heldOK := true // holding the applied twist meets req (true unless stopping was chosen)
	for i := range w.req {
		w.req[i] = math32.Min(w.cfg.Margin, w.req[i]) - 1e-6
		heldOK = heldOK && w.held[i] >= w.req[i]
	}
	if tw == w.applied && heldOK || tw != w.applied && w.feasible(tw) {
		return 1, tw
	}
	prev := w.scale
	scaled := tw.scaled(prev)
	prevOK := prev < 1 && (scaled == w.applied && heldOK || scaled != w.applied && w.feasible(scaled))
	if prevOK && !w.feasible(tw.scaled(math32.Min(prev+clipFastStep, 1))) {
		return prev, scaled
	}
	lo, found := float32(0), !holding // s = 0 is the stop fallback
	for k := clipGrid - 1; k >= 1; k-- {
		if s := float32(k) / clipGrid; w.feasible(tw.scaled(s)) {
			lo, found = s, true
			break
		}
	}
	hi := lo + 1.0/clipGrid
	if prevOK && prev > lo {
		lo, hi, found = prev, math32.Min(1, (math32.Floor(prev*clipGrid)+1)/clipGrid), true
	}
	for range clipIters {
		mid := (lo + hi) / 2
		if w.feasible(tw.scaled(mid)) {
			lo, found = mid, true
		} else {
			hi = mid
		}
	}
	if !found {
		for k := 1; k < clipGrid; k++ {
			if a := w.applied.scaled(float32(k) / clipGrid); w.feasible(a) {
				return 0, a
			}
		}
		return 0, w.applied
	}
	return lo, tw.scaled(lo)
}

// slew moves the raw twist towards cmd within the acceleration limits.
func (w *Walker) slew(cmd Twist, dt float32) {
	dvx, dvy := cmd.VX-w.raw.VX, cmd.VY-w.raw.VY
	if a := w.cfg.MaxAccel; a > 0 {
		if d, lim := math32.Hypot(dvx, dvy), a*dt; d > lim {
			dvx, dvy = dvx*lim/d, dvy*lim/d
		}
	}
	dw := cmd.W - w.raw.W
	if a := w.cfg.MaxAngAccel; a > 0 {
		dw = math32.Max(-a*dt, math32.Min(a*dt, dw))
	}
	w.raw = Twist{w.raw.VX + dvx, w.raw.VY + dvy, w.raw.W + dw}
}

// bump is the swing height profile (1 − cos 2πτ)/2: 0 at both ends, 1 at τ = ½.
func bump(tau float32) float32 {
	return (1 - math32.Cos(2*math32.Pi*tau)) / 2
}

// minJerk is 10τ³ − 15τ⁴ + 6τ⁵ for τ clamped to [0, 1].
func minJerk(tau float32) float32 {
	tau = math32.Max(0, math32.Min(1, tau))
	return tau * tau * tau * (10 + tau*(-15+6*tau))
}

// horizontal is the horizontal swing progress: min jerk between the vertical
// lift and landing phases.
func (w *Walker) horizontal(tau float32) float32 {
	a := w.cfg.LiftFraction
	return minJerk((tau - a) / (1 - 2*a))
}

// Step advances the gait by dt seconds with commanded twist cmd and writes
// servo degrees for every joint into servoDeg (legs in body order, each leg's
// joints base to tip; len == total joint count, else
// kintypes.ErrInvalidDimensions). A non-finite cmd or dt returns
// kintypes.ErrNonFinite, and |cmd.VX| or |cmd.VY| above MaxCommandSpeed or
// |cmd.W| above MaxCommandYawRate returns ErrCommandTooLarge; both change
// nothing and write no servo. Non-positive dt holds the gait
// and the body; dt above MaxStepDt is clamped to it and, absent an IK or servo
// error, returns ErrDtTooLong. dt is split into sub-steps of at most
// maxSubStep and half the shortest stance or swing phase. Each sub-step
// slew-limits the command and scales it to the legs' reach from the current
// state (Applied, Scale). On an IK or servo error every servo still receives
// the best-effort, clamped command and the first error is returned. No
// allocation.
func (w *Walker) Step(dt float32, cmd Twist, servoDeg []float32) error {
	if len(servoDeg) != w.nServo {
		return kintypes.ErrInvalidDimensions
	}
	if !finite(dt, cmd.VX, cmd.VY, cmd.W) {
		return kintypes.ErrNonFinite
	}
	if math32.Abs(cmd.VX) > MaxCommandSpeed || math32.Abs(cmd.VY) > MaxCommandSpeed || math32.Abs(cmd.W) > MaxCommandYawRate {
		return ErrCommandTooLarge
	}
	dt = math32.Max(dt, 0)
	var long error
	if dt > MaxStepDt {
		dt, long = MaxStepDt, ErrDtTooLong
	}
	n := max(1, int(math32.Ceil(dt/w.maxSub)))
	h := dt / float32(n)
	for range n {
		w.subStep(h, cmd)
	}

	err := w.solver.SolveLegs(w.pose, w.feet, w.q, w.errs)
	for i, m := range w.body.Mounts {
		off := w.servo[i]
		if serr := m.Leg.ToServo(w.q[i], servoDeg[off:off+len(m.Leg.Joints)]); err == nil {
			err = serr
		}
	}
	if err == nil {
		err = long
	}
	return err
}

func finite(vs ...float32) bool {
	for _, v := range vs {
		if math32.IsNaN(v) || math32.IsInf(v, 0) {
			return false
		}
	}
	return true
}

func (w *Walker) subStep(dt float32, cmd Twist) {
	w.slew(cmd, dt)
	w.scale, w.applied = w.clip(w.raw)
	w.sched.Advance(dt)
	w.pose = advance(w.pose, w.applied, dt)
	for i := range w.legs {
		w.updateFoot(i)
	}
}

func (w *Walker) updateFoot(i int) {
	ls := w.sched.Leg(i)
	st := &w.legs[i]
	if ls.Stance {
		if !st.stance { // touchdown: land on the last predicted target
			w.feet[i] = st.target
			st.stance = true
		}
		return
	}
	if st.stance { // lift-off
		st.stance = false
		st.frozen = false
		st.lift = w.feet[i]
		st.tau = 0
	}
	tau := ls.Progress
	if !st.frozen {
		st.foothold = Foothold(w.body.Neutral[i], w.applied, w.sched.StanceTime(i))
		st.frozen = w.horizontal(tau) >= freezeAt
	}
	remaining := (1 - tau) * w.sched.SwingTime(i)
	st.target = advance(w.pose, w.applied, remaining).ToWorld(st.foothold)

	foot := &w.feet[i]
	if h0 := w.horizontal(st.tau); h0 >= 1 {
		foot[0], foot[1] = st.target[0], st.target[1]
	} else {
		k := (w.horizontal(tau) - h0) / (1 - h0)
		foot[0] += k * (st.target[0] - foot[0])
		foot[1] += k * (st.target[1] - foot[1])
	}
	foot[2] = st.lift[2] + (st.target[2]-st.lift[2])*minJerk(tau) + w.cfg.SwingHeight*bump(tau)
	st.tau = tau
}

// Body returns a deep copy of the body the walker drives (allocates).
func (w *Walker) Body() chain.Body { return w.body.Clone() }

// Pose returns the body pose in the world.
func (w *Walker) Pose() chain.Pose { return w.pose }

// Applied returns the twist the body moved with at the last Step: the
// slew-limited command times Scale, or, when no scale keeps the margin and
// stopping would not either, a fraction of the previously applied twist
// (slowing down) or the previous twist itself (held).
func (w *Walker) Applied() Twist { return w.applied }

// Scale returns the reach scale applied to the slew-limited command, in
// [0, 1]; 0 also while slowing down along or holding the previous twist (see
// Applied).
func (w *Walker) Scale() float32 { return w.scale }

// State returns the scheduler state of the last Step.
func (w *Walker) State() scheduler.State { return w.sched.State() }

// Feet returns the world foot targets of the last Step. The slice is owned by
// the walker; do not modify it.
func (w *Walker) Feet() []vec.Vector3D { return w.feet }

// Joints returns the joint angles (radians) of the last Step, per leg. The
// slices are owned by the walker; do not modify them.
func (w *Walker) Joints() [][]float32 { return w.q }

// margin is the reach margin of leg i's current foot target.
func (w *Walker) margin(i int) float32 {
	return w.reach[i].margin(w.body.FootInLeg(i, w.pose, w.feet[i]))
}

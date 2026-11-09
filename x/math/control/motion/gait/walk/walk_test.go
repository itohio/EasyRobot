package walk

import (
	"math"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/control/kinematics/chain"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/control/motion/gait/scheduler"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// probeR is the ER-R2 probe-R leg (39/45/100 mm); chain.Hexapod applies the
// probe-R limits (coxa ±60°, femur −60…72°, tibia −149…−20°).
var probeR = chain.LegGeometry{Coxa: 0.039, Femur: 0.045, Tibia: 0.100}

const tick = 0.02 // s

func hexConfig() Config {
	return Config{
		Body:         chain.Hexapod(0.10, 0, -0.10, 0.06, probeR),
		Pattern:      scheduler.Tripod(),
		Period:       1.0,
		SwingHeight:  0.03,
		LiftFraction: 0.1,
		Margin:       0.008,
		MaxAccel:     0.5,
		MaxAngAccel:  1.5,
	}
}

func newWalker(t testing.TB, cfg Config) *Walker {
	t.Helper()
	w, err := New(cfg)
	require.NoError(t, err)
	return w
}

// simStats are measured from the servo commands: servo degrees → FromServo →
// leg FK → FootInWorld, i.e. where the commanded robot actually puts its feet.
type simStats struct {
	ticks        int
	maxDrift     float32    // m, stance foot vs its touchdown position, world
	minMargin    float32    // m, reach margin of every foot target
	maxSpeed     [3]float32 // rad/s per joint (coxa, femur, tibia), max over legs
	maxSpeedLeg  [3]int
	ikErrors     int
	limitErrors  int // joint outside its limits
	servoErrors  int // servo degrees outside the travel
	maxApex      float32
	minStanceCnt int
	maxStanceCnt int
	simTrack
}

func legFK(c chain.Chain, q []float32) vec.Vector3D {
	l0, l1, l2 := c.Joints[0].A, c.Joints[1].A, c.Joints[2].A
	x := l0 + l1*math32.Cos(q[1]) + l2*math32.Cos(q[1]+q[2])
	z := l1*math32.Sin(q[1]) + l2*math32.Sin(q[1]+q[2])
	return vec.Vector3D{x * math32.Cos(q[0]), x * math32.Sin(q[0]), z}
}

func dist(a, b vec.Vector3D) float32 {
	dx, dy, dz := a[0]-b[0], a[1]-b[1], a[2]-b[2]
	return math32.Sqrt(dx*dx + dy*dy + dz*dz)
}

// run steps the walker for duration seconds with twist(t) and checks every tick.
func run(t testing.TB, w *Walker, duration float32, twist func(t float32) Twist, st *simStats) {
	t.Helper()
	body := w.Body()
	n := len(body.Mounts)
	servo := make([]float32, 3*n)
	q := make([]float32, 3)
	if st.ticks == 0 {
		st.minMargin = math32.Inf(1)
		st.minStanceCnt = n
		for i := range n {
			st.lastQ = append(st.lastQ, append([]float32(nil), w.Joints()[i]...))
			st.touch = append(st.touch, vec.Vector3D{})
			st.stance = append(st.stance, false)
		}
	}
	steps := int(math.Round(float64(duration / tick)))
	for range steps {
		now := float32(st.ticks) * tick
		err := w.Step(tick, twist(now), servo)
		st.ticks++
		if err != nil {
			st.ikErrors++
			t.Errorf("tick %d (t=%.2fs): Step: %v", st.ticks, now, err)
		}
		pose := w.Pose()
		sched := w.State()
		cnt := sched.StanceCount()
		st.minStanceCnt = min(st.minStanceCnt, cnt)
		st.maxStanceCnt = max(st.maxStanceCnt, cnt)
		for i, m := range body.Mounts {
			deg := servo[3*i : 3*i+3]
			if ferr := m.Leg.FromServo(deg, q); ferr != nil {
				st.limitErrors++
				t.Errorf("tick %d leg %d: joints %v outside limits", st.ticks, i, q)
			}
			for j, jt := range m.Leg.Joints {
				lo, hi := jt.Servo.Range()
				if deg[j] < lo || deg[j] > hi {
					st.servoErrors++
				}
				if q[j] < jt.Min || q[j] > jt.Max {
					st.limitErrors++
				}
				sp := math32.Abs(q[j]-st.lastQ[i][j]) / tick
				if sp > st.maxSpeed[j] {
					st.maxSpeed[j], st.maxSpeedLeg[j] = sp, i
				}
				st.lastQ[i][j] = q[j]
			}
			foot := body.FootInWorld(i, pose, legFK(m.Leg, q))
			if d := dist(foot, w.Feet()[i]); d > 1e-4 {
				t.Errorf("tick %d leg %d: commanded foot %v misses target %v by %g m", st.ticks, i, foot, w.Feet()[i], d)
			}
			st.minMargin = min(st.minMargin, w.margin(i))
			if sched.Leg[i].Stance {
				if !st.stance[i] {
					st.touch[i] = foot
				}
				st.maxDrift = max(st.maxDrift, dist(foot, st.touch[i]))
			} else {
				st.maxApex = max(st.maxApex, foot[2])
			}
			st.stance[i] = sched.Leg[i].Stance
		}
	}
}

// Fields used only by run.
type simTrack struct {
	lastQ  [][]float32
	touch  []vec.Vector3D
	stance []bool
}

func TestFoothold(t *testing.T) {
	n := vec.Vector3D{0.2, 0.1, -0.1}
	got := Foothold(n, Twist{VX: 0.1}, 0.5)
	assert.InDelta(t, 0.225, got[0], 1e-6)
	assert.InDelta(t, 0.1, got[1], 1e-6)
	assert.InDelta(t, -0.1, got[2], 1e-6)
	// ω × n = (−ω·n_y, ω·n_x, 0).
	got = Foothold(n, Twist{VY: 0.02, W: 0.4}, 0.5)
	assert.InDelta(t, 0.2+0.25*(-0.4*0.1), got[0], 1e-6)
	assert.InDelta(t, 0.1+0.25*(0.02+0.4*0.2), got[1], 1e-6)
	assert.InDelta(t, -0.1, got[2], 1e-6)
}

func TestNewValidates(t *testing.T) {
	cfg := hexConfig()
	_, err := New(cfg)
	require.NoError(t, err)

	bad := hexConfig()
	bad.Pattern = scheduler.Walk()
	_, err = New(bad)
	assert.ErrorIs(t, err, kintypes.ErrInvalidConfig, "4-leg pattern on 6 legs")

	bad = hexConfig()
	bad.Margin = 0.05
	_, err = New(bad)
	assert.ErrorIs(t, err, kintypes.ErrInvalidConfig, "margin larger than the neutral stance allows")

	bad = hexConfig()
	bad.LiftFraction = 0.5
	_, err = New(bad)
	assert.ErrorIs(t, err, kintypes.ErrInvalidConfig)

	bad = hexConfig()
	bad.SwingHeight = 0.2
	_, err = New(bad)
	assert.ErrorIs(t, err, kintypes.ErrInvalidConfig, "apex out of reach")

	bad = hexConfig()
	bad.Period = 0
	_, err = New(bad)
	assert.Error(t, err)

	w := newWalker(t, cfg)
	assert.ErrorIs(t, w.Step(tick, Twist{}, make([]float32, 17)), kintypes.ErrInvalidDimensions)
}

// marginTol is the float/sampling slack on the reach margin: the clip checks
// each leg's plan at sampled instants, so a tick can fall between samples.
const marginTol = 5e-4 // m

func checkWalk(t *testing.T, cmd Twist, seconds float32) (*Walker, *simStats) {
	t.Helper()
	return checkRun(t, hexConfig(), seconds, func(float32) Twist { return cmd })
}

// checkRun runs a walker and asserts the per-tick invariants: stance feet
// fixed, no IK/limit/servo error, every foot target keeps the margin.
func checkRun(t *testing.T, cfg Config, seconds float32, twist func(float32) Twist) (*Walker, *simStats) {
	t.Helper()
	w := newWalker(t, cfg)
	st := &simStats{}
	run(t, w, seconds, twist, st)
	t.Logf("cmd %+v: applied %+v scale %.3f, pose %+v; max stance drift %.2e m, min margin %.4f m, max joint speed %.2f/%.2f/%.2f rad/s, apex %.4f m",
		twist(seconds), w.Applied(), w.Scale(), w.Pose(), st.maxDrift, st.minMargin, st.maxSpeed[0], st.maxSpeed[1], st.maxSpeed[2], st.maxApex)
	assert.Less(t, st.maxDrift, float32(1e-3), "stance feet fixed in the world")
	assert.Zero(t, st.ikErrors)
	assert.Zero(t, st.limitErrors)
	assert.Zero(t, st.servoErrors)
	assert.GreaterOrEqual(t, st.minMargin, cfg.Margin-marginTol, "every tick keeps the reach margin")
	assert.Equal(t, 3, st.minStanceCnt)
	assert.Equal(t, 3, st.maxStanceCnt)
	return w, st
}

// T-IK-10: straight walk at constant v keeps stance feet fixed in the world
// and moves the body by v·t.
func TestStraightWalk(t *testing.T) {
	w, st := checkWalk(t, Twist{VX: 0.1}, 6)
	// Ramp 0 → 0.1 m/s at 0.5 m/s² takes 0.2 s (0.01 m), then 5.8 s at 0.1 m/s.
	assert.InDelta(t, 0.01+0.58, w.Pose().X, 2e-3)
	assert.InDelta(t, 0, w.Pose().Y, 1e-5)
	assert.InDelta(t, 0, w.Pose().Yaw, 1e-5)
	assert.InDelta(t, 1, w.Scale(), 0)
	assert.InDelta(t, 0.03, st.maxApex, 1e-3, "swing apex = SwingHeight")
}

// T-IK-10: turning in place.
func TestTurnInPlace(t *testing.T) {
	w, _ := checkWalk(t, Twist{W: 0.5}, 6)
	// Ramp 0 → 0.5 rad/s at 1.5 rad/s² takes 1/3 s.
	assert.InDelta(t, 0.5*(6-1.0/6), w.Pose().Yaw, 2e-2)
	assert.InDelta(t, 0, w.Pose().X, 1e-5)
	assert.InDelta(t, 0, w.Pose().Y, 1e-5)
}

// T-IK-10: crab walking (v_y only) and a combined twist.
func TestCrabWalk(t *testing.T) {
	w, _ := checkWalk(t, Twist{VY: 0.08}, 6)
	assert.InDelta(t, 0, w.Pose().X, 1e-5)
	assert.InDelta(t, 0.08*(6-0.08), w.Pose().Y, 2e-3)
	checkWalk(t, Twist{VX: 0.06, VY: -0.05, W: 0.2}, 6)
}

// T-IK-10: a twist beyond the legs' reach is scaled uniformly for all legs;
// no leg breaks, every foot keeps the margin, the direction is kept.
func TestClipScalesTwist(t *testing.T) {
	cmd := Twist{VX: 1.0, VY: 0.5, W: 0.5}
	w, st := checkWalk(t, cmd, 6)
	s := w.Scale()
	assert.Greater(t, s, float32(0.05))
	assert.Less(t, s, float32(0.5))
	a := w.Applied()
	assert.InDelta(t, cmd.VX*s, a.VX, 1e-6)
	assert.InDelta(t, cmd.VY*s, a.VY, 1e-6)
	assert.InDelta(t, cmd.W*s, a.W, 1e-6)
	assert.GreaterOrEqual(t, st.minMargin, hexConfig().Margin-marginTol, "every foot target keeps the margin")

	// The steady cycle at the applied twist keeps the margin, and the
	// state-dependent clip is not much stricter than the steady-cycle bound.
	cfg := hexConfig()
	for i := range w.Body().Mounts {
		assert.GreaterOrEqual(t, w.planMargin(i, a), cfg.Margin-1e-4, "leg %d", i)
	}
	steadyOK := func(s float32) bool {
		for i := range w.Body().Mounts {
			if w.planMargin(i, cmd.scaled(s)) < cfg.Margin {
				return false
			}
		}
		return true
	}
	lo, hi := float32(0), float32(1)
	for range 20 {
		if mid := (lo + hi) / 2; steadyOK(mid) {
			lo = mid
		} else {
			hi = mid
		}
	}
	t.Logf("scale %.4f, steady-cycle bound %.4f", s, lo)
	assert.LessOrEqual(t, s, lo+1e-3)
	assert.GreaterOrEqual(t, s, 0.8*lo)
}

// Review P1: the clip must account for where the planted feet really are.
// Starting from rest, accelerating into the clip, and reversing at the clip
// keep the margin on every tick with no IK error.
func TestClipKeepsMarginFromCurrentState(t *testing.T) {
	for _, tc := range []struct {
		name string
		cmd  Twist
	}{
		{"from rest VX 0.17", Twist{VX: 0.17}},
		{"from rest VX 0.15", Twist{VX: 0.15}},
		{"clipped VX 1", Twist{VX: 1}},
		{"clipped VY 1", Twist{VY: 1}},
		{"clipped VX -1", Twist{VX: -1}},
	} {
		t.Run(tc.name, func(t *testing.T) { checkWalk(t, tc.cmd, 4) })
	}
	t.Run("forward to backward at the clip", func(t *testing.T) {
		checkRun(t, hexConfig(), 6, func(now float32) Twist {
			if now < 3 {
				return Twist{VX: 1}
			}
			return Twist{VX: -1}
		})
	})
	t.Run("forward to backward at the clip, no slew limit", func(t *testing.T) {
		cfg := hexConfig()
		cfg.MaxAccel, cfg.MaxAngAccel = 0, 0
		checkRun(t, cfg, 6, func(now float32) Twist {
			if now < 3 {
				return Twist{VX: 1, W: 0.5}
			}
			return Twist{VX: -1, W: -0.5}
		})
	})
}

// Glue: one tick after warm-up allocates nothing.
func TestStepAllocs(t *testing.T) {
	w := newWalker(t, hexConfig())
	servo := make([]float32, 18)
	cmd := Twist{VX: 0.1, W: 0.3}
	for range 100 {
		require.NoError(t, w.Step(tick, cmd, servo))
	}
	allocs := testing.AllocsPerRun(200, func() {
		if err := w.Step(tick, cmd, servo); err != nil {
			t.Fatal(err)
		}
	})
	assert.Zero(t, allocs, "allocations per Step")
	// A clipped command (scale < 1, bisection every tick), still no allocation.
	for range 100 {
		_ = w.Step(tick, Twist{VX: 1, W: 0.3}, servo)
	}
	require.Less(t, w.Scale(), float32(1))
	allocs = testing.AllocsPerRun(200, func() {
		_ = w.Step(tick, Twist{VX: 1, W: 0.3}, servo)
	})
	assert.Zero(t, allocs, "allocations per Step with a clipped twist")
	// A long tick is sub-stepped, still without allocation.
	allocs = testing.AllocsPerRun(50, func() {
		_ = w.Step(0.6, Twist{VX: 0.1}, servo)
	})
	assert.Zero(t, allocs, "allocations per sub-stepped Step")
}

// Review P2: a non-finite command or dt is rejected before it reaches the
// walker state; the walker keeps working afterwards.
func TestNonFiniteInputRejected(t *testing.T) {
	w := newWalker(t, hexConfig())
	servo := make([]float32, 18)
	for range 10 {
		require.NoError(t, w.Step(tick, Twist{VX: 0.05}, servo))
	}
	pose := w.Pose()
	nan, inf := math32.NaN(), math32.Inf(1)
	for _, c := range []struct {
		dt  float32
		cmd Twist
	}{
		{tick, Twist{VX: nan}},
		{tick, Twist{VY: inf}},
		{tick, Twist{W: -inf}},
		{nan, Twist{VX: 0.05}},
		{inf, Twist{VX: 0.05}},
	} {
		assert.ErrorIs(t, w.Step(c.dt, c.cmd, servo), kintypes.ErrNonFinite, "%+v", c)
		assert.Equal(t, pose, w.Pose(), "rejected step leaves the pose")
	}
	for range 50 {
		require.NoError(t, w.Step(tick, Twist{VX: 0.05}, servo))
	}
	p, a := w.Pose(), w.Applied()
	for _, v := range []float32{p.X, p.Y, p.Yaw, a.VX, a.VY, a.W, w.Scale()} {
		assert.False(t, math32.IsNaN(v) || math32.IsInf(v, 0), "walker state finite: pose %+v applied %+v", p, a)
	}
	assert.InDelta(t, 0.05, a.VX, 1e-6)
	for _, d := range servo {
		assert.False(t, math32.IsNaN(d))
	}
}

// Review P3: a tick longer than a swing or stance phase is sub-stepped, so no
// phase transition is skipped.
func TestLongTickSubsteps(t *testing.T) {
	w := newWalker(t, hexConfig())
	servo := make([]float32, 18)
	cfg := hexConfig()
	minM := math32.Inf(1)
	step := func(dt float32) {
		t.Helper()
		if err := w.Step(dt, Twist{VX: 0.1}, servo); err != nil {
			t.Errorf("dt %g at pose %+v: %v", dt, w.Pose(), err)
		}
		for i := range w.Body().Mounts {
			minM = min(minM, w.margin(i))
		}
	}
	for range 100 {
		step(tick)
	}
	x := w.Pose().X
	step(0.6) // longer than the 0.5 s swing and stance
	assert.InDelta(t, x+0.06, w.Pose().X, 1e-4, "the long tick moves the body by v·dt")
	step(0.95)
	for range 100 {
		step(tick)
	}
	t.Logf("min margin %.4f m", minM)
	assert.GreaterOrEqual(t, minM, cfg.Margin-marginTol)
}

// Review P4: a twist step late in a swing must not make the descending foot
// jump. With no slew limit, a 0 → 0.1 m/s step at any tick moves no foot by
// more than 1.5× the steady 0.1 m/s per-tick maximum.
func TestSwingFootNoJumpOnTwistStep(t *testing.T) {
	cfg := hexConfig()
	cfg.MaxAccel, cfg.MaxAngAccel = 0, 0
	maxMove := func(cmd func(k int) Twist) float32 {
		w := newWalker(t, cfg)
		servo := make([]float32, 18)
		prev := append([]vec.Vector3D(nil), w.Feet()...)
		var worst float32
		for k := range 200 {
			require.NoError(t, w.Step(tick, cmd(k), servo))
			for i, f := range w.Feet() {
				worst = max(worst, math32.Hypot(f[0]-prev[i][0], f[1]-prev[i][1]))
				prev[i] = f
			}
		}
		return worst
	}
	steady := maxMove(func(int) Twist { return Twist{VX: 0.1} })
	var worst float32
	worstAt := 0
	for s := 50; s < 100; s++ {
		if m := maxMove(func(k int) Twist {
			if k < s {
				return Twist{}
			}
			return Twist{VX: 0.1}
		}); m > worst {
			worst, worstAt = m, s
		}
	}
	t.Logf("steady max per-tick foot move %.4f m; worst under a step %.4f m (step at tick %d)", steady, worst, worstAt)
	assert.LessOrEqual(t, worst, 1.5*steady)
}

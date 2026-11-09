package walk

import (
	"math"
	"testing"
	"time"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/control/kinematics/chain"
	kintypes "github.com/itohio/EasyRobot/x/math/control/kinematics/types"
	"github.com/itohio/EasyRobot/x/math/control/motion/gait/scheduler"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Review R-ER-017-018 G1: a huge finite yaw command without slew limit used
// to turn the pose NaN (the NaN plan margin passed the feasibility check), so
// every later Step failed, stop commands included. The walker must recover
// as soon as it is told to stop: no error, finite pose, servos moving.
func TestHugeCommandRecovers(t *testing.T) {
	for name, cfg := range map[string]Config{"hex": noSlew(hexConfig()), "quad": noSlew(quadConfig())} {
		t.Run(name, func(t *testing.T) {
			w := newWalker(t, cfg)
			servo := make([]float32, w.nServo)
			require.NoError(t, w.Step(tick, Twist{}, servo))
			pose := w.Pose()
			for range 50 {
				assert.ErrorIs(t, w.Step(tick, Twist{W: 1e10}, servo), ErrCommandTooLarge)
			}
			assert.Equal(t, pose, w.Pose(), "a rejected command changes nothing")
			prev := append([]float32(nil), servo...)
			var moved float32
			for k := range 100 {
				require.NoError(t, w.Step(tick, Twist{}, servo), "stop tick %d", k)
				for j, d := range servo {
					moved = max(moved, math32.Abs(d-prev[j]))
				}
				copy(prev, servo)
			}
			p, a := w.Pose(), w.Applied()
			for _, v := range []float32{p.X, p.Y, p.Z, p.Yaw, a.VX, a.VY, a.W, w.Scale()} {
				require.False(t, math32.IsNaN(v) || math32.IsInf(v, 0), "walker state finite: pose %+v applied %+v", p, a)
			}
			assert.Equal(t, Twist{}, a, "stopped")
			assert.Greater(t, moved, float32(1), "servos keep moving (stepping in place), max change per tick in degrees")
			for i := range w.legs {
				assert.GreaterOrEqual(t, w.margin(i), cfg.Margin-marginTol, "leg %d margin", i)
			}
		})
	}
}

// Review G1: within the command bound a plan margin can still be NaN when the
// plan spans a huge rotation (here a 1e9 s gait cycle at the yaw-rate bound).
// Step must treat it as infeasible and keep the walker finite and standing.
func TestNaNPlanThroughStep(t *testing.T) {
	cfg := noSlew(hexConfig())
	cfg.Period = 1e9
	w := newWalker(t, cfg)
	cmd := Twist{W: MaxCommandYawRate}
	require.True(t, math32.IsNaN(w.legMargin(0, cmd)), "precondition: NaN plan margin at %+v", cmd)
	servo := make([]float32, w.nServo)
	for k := range 50 {
		require.NoError(t, w.Step(tick, cmd, servo), "tick %d", k)
	}
	p := w.Pose()
	require.False(t, math32.IsNaN(p.X+p.Y+p.Yaw), "pose %+v", p)
	for _, d := range servo {
		require.False(t, math32.IsNaN(d))
	}
	for i := range w.legs {
		assert.False(t, math32.IsNaN(w.legMargin(i, w.Applied())), "applied twist %+v has a NaN plan", w.Applied())
	}
	require.NoError(t, w.Step(tick, Twist{}, servo))
}

// Review G1: a plan whose margin is NaN is infeasible, whichever leg the
// check starts from.
func TestNaNPlanMarginInfeasible(t *testing.T) {
	w := newWalker(t, noSlew(hexConfig()))
	tw := Twist{W: 1e10}
	for i := range w.legs {
		require.True(t, math32.IsNaN(w.legMargin(i, tw)), "precondition: leg %d plan margin NaN at %+v", i, tw)
	}
	for bind := range w.legs {
		for i := range w.req {
			w.req[i] = -1
		}
		w.bind = bind
		assert.False(t, w.feasible(tw), "bind %d: NaN plan accepted", bind)
	}
	s, applied := w.clip(tw)
	assert.Zero(t, s)
	assert.Equal(t, Twist{}, applied, "standing walker stays standing")
	// A previous twist whose plan is NaN is no fallback: the clip stops.
	w.applied = tw
	s, applied = w.clip(Twist{})
	assert.Equal(t, float32(1), s)
	assert.Equal(t, Twist{}, applied, "stop instead of holding a NaN-plan twist")
}

// Review G2: a phase far shorter than a sub-step used to be accepted, and
// Step then ran millions of sub-steps (period 1e-20: never returned). New
// rejects it promptly.
func TestNewRejectsTinyPhases(t *testing.T) {
	for _, tc := range []struct {
		name    string
		pattern scheduler.Pattern
		period  float32
	}{
		{"tripod 1 ms", scheduler.Tripod(), 1e-3},
		{"tripod 1e-20 s", scheduler.Tripod(), 1e-20},
		{"wave 0.1 s (swing 16.7 ms)", scheduler.Wave(), 0.1},
	} {
		cfg := hexConfig()
		cfg.Pattern, cfg.Period = tc.pattern, tc.period
		start := time.Now()
		_, err := New(cfg)
		assert.Less(t, time.Since(start), time.Second, tc.name)
		assert.ErrorIs(t, err, kintypes.ErrInvalidConfig, tc.name)
	}
	// The floor itself is accepted, and a 1 s Step there returns promptly.
	cfg := hexConfig()
	cfg.Period = 2 * MinPhaseTime
	w := newWalker(t, cfg)
	assert.Equal(t, float32(MinPhaseTime/2), w.maxSub)
	servo := make([]float32, w.nServo)
	start := time.Now()
	assert.NoError(t, w.Step(MaxStepDt, Twist{}, servo))
	assert.Less(t, time.Since(start), 2*time.Second)
	cfg.Period = math32.Nextafter(2*MinPhaseTime, 0)
	_, err := New(cfg)
	assert.ErrorIs(t, err, kintypes.ErrInvalidConfig, "just below the floor")
}

// Review G1: commands at the bound are processed (scaled by the reach clip),
// just above it rejected.
func TestCommandBound(t *testing.T) {
	w := newWalker(t, noSlew(hexConfig()))
	servo := make([]float32, w.nServo)
	for _, c := range []Twist{{VX: MaxCommandSpeed}, {VY: -MaxCommandSpeed}, {W: MaxCommandYawRate}, {MaxCommandSpeed, MaxCommandSpeed, -MaxCommandYawRate}} {
		for range 20 {
			require.NoError(t, w.Step(tick, c, servo), "%+v", c)
		}
		p := w.Pose()
		require.False(t, math32.IsNaN(p.X+p.Y+p.Yaw), "%+v: pose %+v", c, p)
		for i := range w.legs {
			assert.GreaterOrEqual(t, w.margin(i), hexConfig().Margin-marginTol, "%+v: leg %d", c, i)
		}
	}
	big := math32.Nextafter(MaxCommandSpeed, math32.Inf(1))
	bigW := math32.Nextafter(MaxCommandYawRate, math32.Inf(1))
	for _, c := range []Twist{{VX: big}, {VY: -big}, {W: -bigW}} {
		assert.ErrorIs(t, w.Step(tick, c, servo), ErrCommandTooLarge, "%+v", c)
	}
}

// stopResult is what stopLatency measures after the stop command.
type stopResult struct {
	ticks        int     // ticks the body still moved with a zero slew-limited command
	travel, full float32 // travel over those ticks, and at the speed before them
	margin       float32 // smallest margin
}

// stopLatency drives a fresh walker with a seeded random command sequence for
// warm ticks, then commands a stop. Travel is |v|·dt + |ω|·dt·0.1 m.
func stopLatency(t *testing.T, cfg Config, seed uint64, warm int) stopResult {
	t.Helper()
	w := newWalker(t, cfg)
	servo := make([]float32, w.nServo)
	cmd := randomCommands(seed, warm)
	for k := range warm {
		require.NoError(t, w.Step(tick, cmd(float32(k)*tick), servo))
	}
	speed := func(a Twist) float32 { return math32.Hypot(a.VX, a.VY) + 0.1*math32.Abs(a.W) }
	r := stopResult{margin: math32.Inf(1)}
	var first float32
	for k := 0; k < 500 && (w.applied != Twist{}); k++ {
		before := w.applied
		require.NoError(t, w.Step(tick, Twist{}, servo))
		if (w.raw == Twist{}) && (w.applied != Twist{}) {
			if r.ticks == 0 {
				first = speed(before)
			}
			r.ticks++
			r.travel += speed(w.applied)
		}
		for i := range w.legs {
			r.margin = min(r.margin, w.margin(i))
		}
	}
	require.Equal(t, Twist{}, w.applied, "seed %d warm %d: never stopped", seed, warm)
	r.full = float32(r.ticks) * first
	return r
}

// tightQuad is the adverse stop case: a quadruped without slew limit whose
// Margin is 95 % of what standing and stepping in place keep.
func tightQuad(t *testing.T) Config {
	cfg := noSlew(quadConfig())
	cfg.Margin = 0
	w := newWalker(t, cfg)
	standing := math32.Inf(1)
	for i := range w.legs {
		standing = min(standing, w.planMargin(i, Twist{}))
	}
	cfg.Margin = 0.95 * standing
	return cfg
}

// Review G3: once the slew-limited command is zero, the walker stops within
// one gait cycle, and while it cannot stop yet (a swing planned for the old
// twist must land first) it slows down along the previous twist instead of
// holding it. Measured over 160 stops per configuration (stopSeeds 40,
// 2026-11-09, arm64), worst latency / coasting speed as a fraction of the
// speed before / travel after the stop, against holding the previous twist
// (before the fix):
//
//	hex            9 ticks / 0.85 / 0.007 m   (hold:  8 / 1 / 0.008 m)
//	hex no slew   15 ticks / 0.48 / 0.054 m   (hold: 15 / 1 / 0.115 m)
//	quad           5 ticks / 0.98 / 0.005 m   (hold:  5 / 1 / 0.006 m)
//	quad no slew  45 ticks / 0.14 / 0.024 m   (hold: 12 / 1 / 0.041 m)
//	tightQuad     36 ticks / 0.45 / 0.173 m   (hold: 33 / 1 / 0.267 m)
//
// Slowing down can take longer to reach a safe stop but always travels less.
// The coasting bound is asserted only where the configuration decelerates.
func TestStopLatency(t *testing.T) {
	for _, tc := range []struct {
		name  string
		cfg   Config
		coast float32
	}{
		{"hex", hexConfig(), 1.001}, // 1 + rounding: not asserted
		{"hex no slew", noSlew(hexConfig()), 0.6},
		{"quad", quadConfig(), 1.001},
		{"quad no slew", noSlew(quadConfig()), 0.6},
		{"quad no slew, Margin 95 % of standing", tightQuad(t), 0.6},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			worst, total, minM := 0, 0, math32.Inf(1)
			var travel, full float32
			for seed := range uint64(stopSeeds) {
				for _, warm := range []int{23, 61, 97, 140} {
					r := stopLatency(t, tc.cfg, 100+seed, warm)
					worst, total, minM = max(worst, r.ticks), total+r.ticks, min(minM, r.margin)
					travel, full = travel+r.travel, full+r.full
				}
			}
			coast := float32(0)
			if full > 0 {
				coast = travel / full
			}
			t.Logf("%s: stop latency worst %d ticks (%.2f s), total %d; coasting at %.3f of the speed before (travel %.4f m-equivalent), min margin %.4f m (Margin %.4f m)",
				tc.name, worst, float32(worst)*tick, total, coast, travel*tick, minM, tc.cfg.Margin)
			assert.LessOrEqual(t, float32(worst)*tick, tc.cfg.Period, "stop latency within one gait cycle")
			assert.LessOrEqual(t, coast, tc.coast, "coasting speed while it cannot stop yet")
			assert.GreaterOrEqual(t, minM, tc.cfg.Margin-marginTol)
		})
	}
}

// Review G4: the walker keeps a deep copy of Config.Body; the caller's later
// edits (and edits to what Body returns) have no effect.
func TestBodyCopied(t *testing.T) {
	cfg := hexConfig()
	w := newWalker(t, cfg)
	ref := newWalker(t, hexConfig())
	cfg.Body.Neutral[0][0] += 0.03
	cfg.Body.Mounts[0].Hip[0] += 0.02
	cfg.Body.Mounts[0].Leg.Joints[1].Min = 0
	got := w.Body()
	got.Neutral[1][0] += 0.03
	got.Mounts[1].Hip[1] += 0.02
	got.Mounts[1].Leg.Joints[2].Max = -1
	a, b := make([]float32, 18), make([]float32, 18)
	for k := range 100 {
		require.NoError(t, w.Step(tick, Twist{VX: 0.1, W: 0.2}, a), "tick %d", k)
		require.NoError(t, ref.Step(tick, Twist{VX: 0.1, W: 0.2}, b))
		require.Equal(t, b, a, "tick %d: servo commands differ from an untouched walker", k)
	}
	assert.Equal(t, hexConfig().Body, w.Body())
}

// Review G5: a leg that is not coxa–femur–tibia shaped is an invalid walker
// configuration and an unsupported leg shape.
func TestNewWrapsUnsupportedLeg(t *testing.T) {
	cfg := hexConfig()
	leg := &cfg.Body.Mounts[0].Leg
	leg.Joints = append([]chain.Joint(nil), leg.Joints...)
	leg.Joints[2].Offset = 0.1 // valid chain, not leg-shaped: DH model, no Planar3
	_, err := New(cfg)
	assert.ErrorIs(t, err, kintypes.ErrInvalidConfig)
	assert.ErrorIs(t, err, kintypes.ErrUnsupportedOperation)
}

func probeRReach(t *testing.T) (legReach, chain.Chain) {
	t.Helper()
	leg := chain.Hexapod(0.1, 0, -0.1, 0.06, probeR).Mounts[0].Leg
	p3, err := leg.Planar3()
	require.NoError(t, err)
	r, err := newLegReach(p3)
	require.NoError(t, err)
	return r, leg
}

// Review G9 (R2, R3): the reach margin is bound by the femur upper limit and
// by the inner radius (tibia folded to its limit), each as slack × lever arm.
func TestReachMarginLimits(t *testing.T) {
	r, leg := probeRReach(t)
	femurMax, tibiaMin := leg.Joints[1].Max, leg.Joints[2].Min
	// Femur 2° below its upper limit, tibia at −90°: every other slack is
	// larger, so the margin is 2° × the femur-pivot-to-foot distance.
	q := []float32{0, femurMax - chain.Deg(2), -math32.Pi / 2}
	d := r.knee(q[2])
	assert.InDelta(t, chain.Deg(2)*d, r.margin(legFK(leg, q)), 2e-5, "femur upper limit")
	// Femur level, tibia 1° short of fully folded: the margin is the distance
	// to the inner radius.
	q = []float32{0, 0.2, tibiaMin + chain.Deg(1)}
	want := r.knee(q[2]) - r.knee(tibiaMin)
	require.Less(t, want, float32(2e-3))
	assert.InDelta(t, want, r.margin(legFK(leg, q)), 2e-5, "inner radius")
	// Inside the inner radius the margin is negative.
	q[2] = tibiaMin - chain.Deg(3)
	assert.Less(t, r.margin(legFK(leg, q)), float32(0))
}

// Review G1/G9 (W5): the yaw is wrapped to (−π, π] for any finite angle.
func TestWrapPi(t *testing.T) {
	for _, a := range []float32{0, 1, -1, math32.Pi, -math32.Pi, 3*math32.Pi + 0.1, -3*math32.Pi - 0.1, 7, -7, 1e3, -1e6, 1e10} {
		got := wrapPi(a)
		assert.True(t, got > -math32.Pi && got <= math32.Pi, "wrapPi(%g) = %g", a, got)
		d := math.Remainder(float64(got)-float64(a), 2*math.Pi)
		assert.InDelta(t, 0, d, 1e-6*max(1, math.Abs(float64(a))), "wrapPi(%g) = %g is not a + 2πk", a, got)
	}
	assert.Equal(t, float32(math32.Pi), wrapPi(-math32.Pi))
}

// Review G9 (W5): a walker turning for several turns keeps its yaw in
// (−π, π] and equal to the integrated yaw rate modulo 2π.
func TestYawWrapsWhileTurning(t *testing.T) {
	w := newWalker(t, hexConfig())
	servo := make([]float32, 18)
	var total float64
	for k := range 600 {
		require.NoError(t, w.Step(tick, Twist{W: 1.5}, servo))
		total += float64(w.Applied().W) * tick
		yaw := w.Pose().Yaw
		require.True(t, yaw > -math32.Pi && yaw <= math32.Pi, "tick %d: yaw %g", k, yaw)
		require.InDelta(t, 0, math.Remainder(float64(yaw)-total, 2*math.Pi), 1e-3, "tick %d", k)
	}
	require.Greater(t, total, 2*math.Pi, "turned more than once")
}

// Review G9 (W20): a sub-step is at most maxSubStep and at most half the
// shortest stance or swing phase.
func TestSubStepRule(t *testing.T) {
	for _, tc := range []struct {
		name    string
		pattern scheduler.Pattern
		period  float32
		want    float32
	}{
		{"tripod 1 s: 25 ms cap", scheduler.Tripod(), 1, maxSubStep},
		{"tripod 0.08 s: half of 40 ms", scheduler.Tripod(), 0.08, 0.02},
		{"wave 0.18 s: half the 30 ms swing", scheduler.Wave(), 0.18, 0.015},
		{"wave 0.24 s: half the 40 ms swing", scheduler.Wave(), 0.24, 0.02},
	} {
		cfg := hexConfig()
		cfg.Pattern, cfg.Period = tc.pattern, tc.period
		w := newWalker(t, cfg)
		assert.InDelta(t, tc.want, w.maxSub, 1e-6, tc.name)
	}
}

// Review G9 (W15): a state that has lost Margin (here Margin raised to 1 m
// mid-walk, which no plan keeps) is held to its best fallback's plan instead:
// no leg drops below what standing still or holding the previous twist would
// keep, and the walker still follows the command as far as that allows
// rather than freezing.
func TestLostMarginFallback(t *testing.T) {
	for name, cfg := range map[string]Config{"hex": hexConfig(), "quad no slew": noSlew(quadConfig())} {
		t.Run(name, func(t *testing.T) {
			w := newWalker(t, cfg)
			servo := make([]float32, w.nServo)
			cmd := Twist{VX: 0.08, W: 0.2}
			for range 100 {
				require.NoError(t, w.Step(tick, cmd, servo))
			}
			fallback := func() float32 {
				hold, stop := math32.Inf(1), math32.Inf(1)
				for i := range w.legs {
					hold = min(hold, w.legMargin(i, w.applied))
					stop = min(stop, w.legMargin(i, Twist{}))
				}
				return max(hold, stop)
			}
			floor := fallback()
			w.cfg.Margin = 1
			x0, y0, yaw0 := w.Pose().X, w.Pose().Y, w.Pose().Yaw
			minM := math32.Inf(1)
			for k := range 200 {
				require.NoError(t, w.Step(tick, cmd, servo), "tick %d", k)
				for i := range w.legs {
					minM = min(minM, w.margin(i))
				}
			}
			p := w.Pose()
			moved := math32.Hypot(p.X-x0, p.Y-y0) + 0.1*math32.Abs(p.Yaw-yaw0)
			t.Logf("fallback plan %.4f m, min margin after the raise %.4f m, moved %.4f m-equivalent", floor, minM, moved)
			assert.GreaterOrEqual(t, minM, floor-marginTol, "never below the fallback plan")
			assert.Greater(t, moved, float32(1e-3), "still follows the command (measured 3.4 mm quad, 24 mm hex)")
		})
	}
}

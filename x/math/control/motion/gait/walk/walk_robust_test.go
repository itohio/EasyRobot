package walk

import (
	"errors"
	"math/rand/v2"
	"testing"
	"time"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/control/kinematics/chain"
	"github.com/itohio/EasyRobot/x/math/control/motion/gait/scheduler"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func quadConfig() Config {
	return Config{
		Body:         chain.Quadruped(0.12, 0.06, probeR),
		Pattern:      scheduler.Walk(),
		Period:       1.6,
		SwingHeight:  0.03,
		LiftFraction: 0.1,
		Margin:       0.008,
		MaxAccel:     0.3,
		MaxAngAccel:  1,
	}
}

func noSlew(cfg Config) Config {
	cfg.MaxAccel, cfg.MaxAngAccel = 0, 0
	return cfg
}

// randomCommands returns a seeded piecewise-constant command sequence over
// ticks: moderate, far-beyond-reach, reversed and pure-turn twists, each held
// 1…30 ticks.
func randomCommands(seed uint64, ticks int) func(now float32) Twist {
	rng := rand.New(rand.NewPCG(seed, 7))
	cmds := make([]Twist, 0, ticks)
	var last Twist
	for len(cmds) < ticks {
		var tw Twist
		u := func(a float32) float32 { return (rng.Float32()*2 - 1) * a }
		switch rng.IntN(4) {
		case 0:
			tw = Twist{u(0.3), u(0.3), u(2)}
		case 1:
			tw = Twist{u(1.5), u(1.5), u(4)}
		case 2:
			tw = Twist{-last.VX, -last.VY, -last.W}
		case 3:
			tw = Twist{W: u(4)}
		}
		last = tw
		for n := 1 + rng.IntN(30); n > 0 && len(cmds) < ticks; n-- {
			cmds = append(cmds, tw)
		}
	}
	return func(now float32) Twist {
		return cmds[min(len(cmds)-1, int(math32.Round(now/tick)))]
	}
}

// Review round 3: random command sequences keep the margin and never fail
// the IK, with and without the slew limit, hexapod and quadruped. randomTicks
// is shorter under the race detector (race_on_test.go).
func TestRandomCommandSequences(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
		seed uint64
	}{
		{"hex", hexConfig(), 1},
		{"hex no slew", noSlew(hexConfig()), 2},
		{"quad", quadConfig(), 3},
		{"quad no slew", noSlew(quadConfig()), 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			checkRun(t, tc.cfg, randomTicks*tick, randomCommands(tc.seed, randomTicks))
		})
	}
}

// zeroRun steps w with cmd(k) and returns the longest run of ticks the body
// stood still (applied twist 0) while a non-zero command was slewed in, plus
// the minimum margin and the number
// of IK errors.
func zeroRun(t *testing.T, w *Walker, ticks int, dt float32, cmd func(k int) Twist) (longest int, minM float32, errs int) {
	t.Helper()
	servo := make([]float32, w.nServo)
	minM = math32.Inf(1)
	run := 0
	for k := range ticks {
		if err := w.Step(dt, cmd(k), servo); err != nil {
			errs++
		}
		if (w.applied == Twist{}) && (w.raw != Twist{}) {
			run++
			longest = max(longest, run)
		} else {
			run = 0
		}
		for i := range w.legs {
			minM = min(minM, w.margin(i))
		}
	}
	return longest, minM, errs
}

// Review round 3: standing still must stay feasible so the clip can climb
// back from 0 (quadruped VX 0.1 → W 3 without slew stalled for 70 ticks), and
// the clip must find a feasible scale above an infeasible previous one (hex
// VY 1 → −1 without slew stalled for 37 ticks).
func TestClipNeverStalls(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
		a, b Twist
	}{
		{"quad no slew VX 0.1 → W 3", noSlew(quadConfig()), Twist{VX: 0.1}, Twist{W: 3}},
		{"hex no slew VY 1 → −1", noSlew(hexConfig()), Twist{VY: 1}, Twist{VY: -1}},
		{"hex no slew VX 1 W 0.5 → reverse", noSlew(hexConfig()), Twist{VX: 1, W: 0.5}, Twist{VX: -1, W: -0.5}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			for shift := 0; shift < 50; shift += 9 {
				w := newWalker(t, tc.cfg)
				longest, minM, errs := zeroRun(t, w, 400+shift, tick, func(k int) Twist {
					switch {
					case k < shift:
						return Twist{}
					case k < shift+150:
						return tc.a
					}
					return tc.b
				})
				assert.LessOrEqual(t, longest, 5, "shift %d: ticks stopped while commanded", shift)
				assert.Zero(t, errs, "shift %d: IK errors", shift)
				assert.GreaterOrEqual(t, minM, tc.cfg.Margin-marginTol, "shift %d: margin", shift)
			}
		})
	}
}

// Review round 3: long ticks without a slew limit (random commands) stay
// feasible: sub-steps are short enough for the reach clip to act.
func TestLongTickNoSlew(t *testing.T) {
	for _, dt := range []float32{0.1, 0.26, 0.6} {
		w := newWalker(t, noSlew(hexConfig()))
		rng := rand.New(rand.NewPCG(3, 3))
		cmd := func(int) Twist {
			return Twist{(rng.Float32()*2 - 1) * 1, (rng.Float32()*2 - 1) * 1, (rng.Float32()*2 - 1) * 3}
		}
		_, minM, errs := zeroRun(t, w, 100, dt, cmd)
		t.Logf("dt %.2f: min margin %.4f m, %d error ticks", dt, minM, errs)
		assert.Zero(t, errs, "dt %g", dt)
		assert.GreaterOrEqual(t, minM, hexConfig().Margin-marginTol, "dt %g", dt)
	}
}

// Review round 3: a huge finite dt is clamped to MaxStepDt (bounded work) and
// reported with ErrDtTooLong; the walker keeps working.
func TestHugeDtClamped(t *testing.T) {
	w := newWalker(t, hexConfig())
	servo := make([]float32, 18)
	for range 50 {
		require.NoError(t, w.Step(tick, Twist{VX: 0.1}, servo))
	}
	x := w.Pose().X
	start := time.Now()
	err := w.Step(1e4, Twist{VX: 0.1}, servo)
	elapsed := time.Since(start)
	assert.ErrorIs(t, err, ErrDtTooLong)
	assert.Less(t, elapsed, 2*time.Second)
	assert.InDelta(t, x+0.1*MaxStepDt, w.Pose().X, 1e-3, "processed exactly MaxStepDt")
	for range 50 {
		require.NoError(t, w.Step(tick, Twist{VX: 0.1}, servo))
	}
	// A dt at the limit is processed without the error.
	assert.False(t, errors.Is(w.Step(MaxStepDt, Twist{VX: 0.1}, servo), ErrDtTooLong))
}

// denseLegMargin mirrors legMargin's plan with 400 samples per segment.
func denseLegMargin(w *Walker, i int, tw Twist) float32 {
	const n = 400
	tst, tsw := w.sched.StanceTime(i), w.sched.SwingTime(i)
	stance := func(x vec.Vector3D, t0, d float32) float32 {
		m := math32.Inf(1)
		for j := 0; j <= n; j++ {
			m = min(m, w.marginAt(i, x, tw, t0+d*float32(j)/n))
		}
		return m
	}
	swing := func(lift, x, td vec.Vector3D, tau0, t0 float32) float32 {
		h0 := w.horizontal(tau0)
		m := math32.Inf(1)
		for j := 1; j < n; j++ {
			tau := tau0 + (1-tau0)*float32(j)/n
			f := td
			if h0 < 1 {
				r := (w.horizontal(tau) - h0) / (1 - h0)
				f[0], f[1] = x[0]+r*(td[0]-x[0]), x[1]+r*(td[1]-x[1])
			}
			f[2] = lift[2] + (td[2]-lift[2])*minJerk(tau) + w.cfg.SwingHeight*bump(tau)
			m = min(m, w.marginAt(i, f, tw, t0+(tau-tau0)*tsw))
		}
		return m
	}
	cycle := func(x vec.Vector3D, t0, d float32) float32 {
		tl := t0 + d
		td := beforeMotion(Foothold(w.body.Neutral[i], tw, tst), tw, tl+tsw)
		return min(stance(x, t0, d), swing(x, x, td, 0, tl))
	}
	m := cycle(Foothold(w.body.Neutral[i], tw, tst), 0, tst)
	ls, st := w.sched.Leg(i), &w.legs[i]
	x := w.pose.ToBody(w.feet[i])
	if ls.Stance {
		return min(m, cycle(x, 0, (1-ls.Progress)*tst))
	}
	lift := x
	if !st.stance {
		lift = w.pose.ToBody(st.lift)
	}
	remaining := (1 - ls.Progress) * tsw
	p := Foothold(w.body.Neutral[i], tw, tst)
	if st.frozen {
		p = st.foothold
	}
	td := beforeMotion(p, tw, remaining)
	if st.frozen {
		m = min(m, cycle(td, remaining, tst))
	}
	return min(m, swing(lift, x, td, ls.Progress, 0))
}

// Review round 3: the plan check's sampling error (legMargin vs a dense check
// of the same plan) is what the margin guarantee may fall short by; the
// package documents the bound asserted here.
func TestPlanSamplingError(t *testing.T) {
	for _, tc := range []struct {
		name  string
		cfg   Config
		shift int
		a, b  Twist
	}{
		{"quad VX 1 W 1", quadConfig(), 36, Twist{VX: 1, W: 1}, Twist{VX: 1, W: 1}},
		{"quad VX −1", quadConfig(), 18, Twist{VX: -1}, Twist{VX: -1}},
		{"hex no slew VY 1 → −1", noSlew(hexConfig()), 9, Twist{VY: 1}, Twist{VY: -1}},
		{"hex VX 0.1 W 0.3", hexConfig(), 0, Twist{VX: 0.1, W: 0.3}, Twist{W: 0.5}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			w := newWalker(t, tc.cfg)
			servo := make([]float32, w.nServo)
			var worst float32
			for k := range 300 + tc.shift {
				cmd := Twist{}
				if k >= tc.shift {
					cmd = tc.a
					if k >= tc.shift+150 {
						cmd = tc.b
					}
				}
				_ = w.Step(tick, cmd, servo)
				for i := range w.legs {
					worst = max(worst, w.legMargin(i, w.applied)-denseLegMargin(w, i, w.applied))
				}
			}
			t.Logf("%s: worst sampling error %.3f mm", tc.name, worst*1000)
			assert.LessOrEqual(t, worst, float32(planSamplingError))
		})
	}
}

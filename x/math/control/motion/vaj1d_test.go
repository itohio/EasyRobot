package vaj

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/chewxy/math32"
)

const (
	testMaxV = 100
	testMaxA = 100
	testMaxJ = 1000
	testDt   = 0.01
)

// run steps the profile until it settles or the step budget is exhausted and
// returns the visited outputs (including the start position).
func run(p *VAJ1D, steps int) []float32 {
	return runDt(p, steps, testDt)
}

func runDt(p *VAJ1D, steps int, dt float32) []float32 {
	out := make([]float32, 0, steps+1)
	out = append(out, p.Output)
	for range steps {
		p.Update(dt)
		out = append(out, p.Output)
	}
	return out
}

func TestVAJ1D_ReachesTarget(t *testing.T) {
	tests := []struct {
		name          string
		start, target float32
	}{
		{"0 to 0.5", 0, 0.5},
		{"0 to 100", 0, 100},
		{"0 to -0.5", 0, -0.5},
		{"0 to -100", 0, -100},
		{"10 to 10.002", 10, 10.002},
		{"5 to -7.3", 5, -7.3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := New1D(testMaxV, testMaxA, testMaxJ)
			p.Input, p.Output, p.Target = tt.start, tt.start, tt.target
			run(&p, 2000)
			if d := math32.Abs(p.Output - tt.target); d > 1e-3 {
				t.Fatalf("final output %v, target %v, error %v > 1e-3", p.Output, tt.target, d)
			}
			if math32.Abs(p.Velocity) > 1e-2 {
				t.Fatalf("final velocity %v not settled", p.Velocity)
			}
		})
	}
}

// timeOptimal returns the duration of the time-optimal rest-to-rest
// jerk-limited move over distance d with limits v, a, j.
func timeOptimal(v, a, j, d float64) float64 {
	if d < 0 {
		d = -d
	}
	// rampTime and rampDist: rest to cruise speed vp (and acceleration back to 0).
	rampTime := func(vp float64) float64 {
		if vp*j >= a*a {
			return vp/a + a/j
		}
		return 2 * math.Sqrt(vp/j)
	}
	rampDist := func(vp float64) float64 { return vp * rampTime(vp) / 2 }
	if 2*rampDist(v) <= d {
		return 2*rampTime(v) + (d-2*rampDist(v))/v
	}
	lo, hi := 0.0, v
	for range 200 {
		mid := (lo + hi) / 2
		if 2*rampDist(mid) > d {
			hi = mid
		} else {
			lo = mid
		}
	}
	return 2 * rampTime(lo)
}

type stepCase struct {
	name        string
	v, a, j, dt float32
	from, to    float32
	steps       int
}

// stepCases covers the unit-test profile, the manipulator firmware profile
// (New1D(100, 100, 100) at a 100 ms tick) and a slow-jerk profile.
var stepCases = []stepCase{
	{"fast 10ms 0.5", testMaxV, testMaxA, testMaxJ, testDt, 0, 0.5, 2000},
	{"fast 10ms 3", testMaxV, testMaxA, testMaxJ, testDt, 0, 3, 2000},
	{"fast 10ms 100", testMaxV, testMaxA, testMaxJ, testDt, 0, 100, 2000},
	{"fast 10ms -100", testMaxV, testMaxA, testMaxJ, testDt, 0, -100, 2000},
	{"firmware 100ms 5", 100, 100, 100, 0.1, 0, 5, 400},
	{"firmware 100ms 30", 100, 100, 100, 0.1, 0, 30, 400},
	{"firmware 100ms -30", 100, 100, 100, 0.1, 0, -30, 400},
	{"firmware 100ms 200", 100, 100, 100, 0.1, 0, 200, 400},
	{"firmware 10ms 5", 100, 100, 100, 0.01, 0, 5, 4000},
	{"firmware 10ms 200", 100, 100, 100, 0.01, 0, 200, 4000},
	{"large magnitude 1000 to 1500", testMaxV, testMaxA, testMaxJ, testDt, 1000, 1500, 4000},
	{"slow jerk 1ms 100", 1000, 100, 10, 0.001, 0, 100, 20000},
}

func TestVAJ1D_StepProfile(t *testing.T) {
	for _, tc := range stepCases {
		t.Run(tc.name, func(t *testing.T) {
			p := New1D(tc.v, tc.a, tc.j)
			p.Input, p.Output, p.Target = tc.from, tc.from, tc.to
			pos := make([]float32, 0, tc.steps+1)
			acc := make([]float32, 0, tc.steps+1)
			pos, acc = append(pos, p.Output), append(acc, p.Acceleration)
			for range tc.steps {
				p.Update(tc.dt)
				pos, acc = append(pos, p.Output), append(acc, p.Acceleration)
			}
			d := tc.to - tc.from
			dir := math32.Copysign(1, d)

			settled := -1
			for i, x := range pos {
				if x == tc.to {
					settled = i
					break
				}
			}
			if settled < 0 || p.Velocity != 0 || p.Acceleration != 0 {
				t.Fatalf("never settled exactly on %v: final %v v=%v a=%v", tc.to, p.Output, p.Velocity, p.Acceleration)
			}

			t1 := -1
			// Output is float32: one ulp of position noise shows up as up to
			// 8 ulp/dt^3 in a third difference, so the finite-difference jerk
			// gets that slack; the state jerk (from Acceleration) gets none.
			jerkSlack := 8 * ulp(math32.Max(math32.Abs(tc.from), math32.Abs(tc.to))) / (tc.dt * tc.dt * tc.dt)
			var maxV, maxA, maxJ, maxStateJ, over float32
			for i, x := range pos {
				if i+1 < len(acc) && i+1 != settled {
					maxStateJ = math32.Max(maxStateJ, math32.Abs(acc[i+1]-acc[i])/tc.dt)
				}
				if t1 < 0 && math32.Abs(tc.to-x) <= 0.01*math32.Abs(d) {
					t1 = i
				}
				over = math32.Max(over, dir*(x-tc.to))
				if i+1 < len(pos) {
					maxV = math32.Max(maxV, math32.Abs(pos[i+1]-x)/tc.dt)
				}
				if i >= 1 && i+1 < len(pos) {
					maxA = math32.Max(maxA, math32.Abs(pos[i+1]-2*x+pos[i-1])/(tc.dt*tc.dt))
				}
				// Third difference over pos[i-1..i+2], excluding windows that
				// contain the final snap sample.
				if i >= 1 && i+2 < len(pos) && (settled < i-1 || settled > i+2) {
					jerk := (pos[i+2] - 3*pos[i+1] + 3*x - pos[i-1]) / (tc.dt * tc.dt * tc.dt)
					maxJ = math32.Max(maxJ, math32.Abs(jerk))
				}
			}
			opt := timeOptimal(float64(tc.v), float64(tc.a), float64(tc.j), float64(d))
			got := float64(t1) * float64(tc.dt)
			t.Logf("time to within 1%%: %.3fs, time-optimal %.3fs, settled %.3fs", got, opt, float64(settled)*float64(tc.dt))
			if got > 1.5*opt {
				t.Errorf("time to within 1%%: %.3fs > 1.5 x time-optimal %.3fs", got, opt)
			}
			if over > 1e-3 {
				t.Errorf("overshoot %v", over)
			}
			if maxV > 1.05*tc.v {
				t.Errorf("finite-difference |v| peak %v > 1.05*maxV", maxV)
			}
			if maxA > 1.05*tc.a {
				t.Errorf("finite-difference |a| peak %v > 1.05*maxA", maxA)
			}
			if maxJ > 1.05*tc.j+jerkSlack {
				t.Errorf("finite-difference |jerk| peak %v > 1.05*maxJ + float32 slack %v", maxJ, jerkSlack)
			}
			if maxStateJ > 1.05*tc.j {
				t.Errorf("state |jerk| peak %v > 1.05*maxJ", maxStateJ)
			}
		})
	}
}

// idealStop simulates a time-optimal jerk-limited brake from (v, a) with v>0
// and returns the distance travelled until v reaches 0.
func idealStop(v, a float32) float32 {
	const h = 1e-5
	var s float32
	for v > 0 {
		// Release braking in time to land at v=0 with a=0: ramping a back to
		// 0 at +J still removes a^2/(2J) of velocity.
		var j float32
		switch {
		case a < 0 && v <= a*a/(2*testMaxJ):
			j = testMaxJ
		case a > -testMaxA:
			j = -testMaxJ
		}
		a += j * h
		if a < -testMaxA {
			a = -testMaxA
		}
		v += a * h
		s += v * h
	}
	return s
}

func TestVAJ1D_ReversalStopsWithinStoppingDistance(t *testing.T) {
	for _, steps := range []int{20, 50, 100} {
		p := New1D(testMaxV, testMaxA, testMaxJ)
		p.Target = 100
		run(&p, steps)
		revPos, v, a := p.Output, p.Velocity, p.Acceleration
		if v <= 0 {
			t.Fatalf("after %d steps velocity %v must be positive", steps, v)
		}
		bound := revPos + 1.05*idealStop(v, a) + 1e-3
		p.Target = 0
		pos := run(&p, 3000)
		var peak float32
		for _, x := range pos {
			peak = math32.Max(peak, x)
		}
		if peak > bound {
			t.Errorf("reversal after %d steps at %v (v=%v a=%v): peak %v > bound %v", steps, revPos, v, a, peak, bound)
		}
		if math32.Abs(p.Output) > 1e-3 {
			t.Errorf("reversal after %d steps: final %v not at 0", steps, p.Output)
		}
	}
}

// TestVAJ1D_ShortMoveSweep sweeps 200 log-spaced rest-to-rest move sizes from
// 0.1 mm to 100 units for two limit sets. Each move must settle exactly with
// no velocity sign reversal, within 1.5x the time-optimal
// jerk-limited duration plus 2 ticks, without overshoot (0.1% of the move)
// and within 5% of the limits.
func TestVAJ1D_ShortMoveSweep(t *testing.T) {
	const sizes = 200
	limits := []struct {
		name        string
		v, a, j, dt float32
	}{
		{"V1 A1 J10 20ms", 1, 1, 10, 0.02},
		{"V100 A100 J1000 10ms", testMaxV, testMaxA, testMaxJ, testDt},
	}
	for _, lim := range limits {
		var failures, maxReversals int
		var worstRatio float64
		for k := range sizes {
			d := float32(math.Pow(10, -4+6*float64(k)/float64(sizes-1)))
			opt := timeOptimal(float64(lim.v), float64(lim.a), float64(lim.j), float64(d))
			bound := 1.5*opt + 2*float64(lim.dt)
			steps := int(3*opt/float64(lim.dt)) + 100

			p := New1D(lim.v, lim.a, lim.j)
			p.Target = d
			prevPos, prevAcc := p.Output, p.Acceleration
			var prevSign float32
			reversals, settled := 0, -1
			var over, maxAJump, maxStateJ float32
			for i := 1; i <= steps; i++ {
				prevV := (p.Output - prevPos)
				prevPos = p.Output
				p.Update(lim.dt)
				if p.Velocity != 0 {
					sg := math32.Copysign(1, p.Velocity)
					if prevSign != 0 && sg != prevSign {
						reversals++
					}
					prevSign = sg
				}
				over = math32.Max(over, p.Output-d)
				if p.Output != d || p.Velocity != 0 {
					maxStateJ = math32.Max(maxStateJ, math32.Abs(p.Acceleration-prevAcc)/lim.dt)
				}
				prevAcc = p.Acceleration
				maxAJump = math32.Max(maxAJump, math32.Abs(p.Output-prevPos-prevV)/(lim.dt*lim.dt))
				if settled < 0 && p.Output == d && p.Velocity == 0 {
					settled = i
				}
			}
			got := float64(settled) * float64(lim.dt)
			maxReversals = max(maxReversals, reversals)
			worstRatio = math.Max(worstRatio, (got-2*float64(lim.dt))/opt)
			var msgs []string
			if settled < 0 {
				msgs = append(msgs, fmt.Sprintf("never settled (final %v)", p.Output))
			} else if got > bound {
				msgs = append(msgs, fmt.Sprintf("settled %.3fs > bound %.3fs (opt %.3fs)", got, bound, opt))
			}
			if reversals > 0 {
				msgs = append(msgs, fmt.Sprintf("%d velocity reversals", reversals))
			}
			if over > 1e-3*d {
				msgs = append(msgs, fmt.Sprintf("overshoot %v", over))
			}
			if maxAJump > 1.05*lim.a {
				msgs = append(msgs, fmt.Sprintf("finite-difference |a| %v", maxAJump))
			}
			if maxStateJ > 1.05*lim.j {
				msgs = append(msgs, fmt.Sprintf("state |jerk| %v", maxStateJ))
			}
			if len(msgs) > 0 {
				failures++
				if failures <= 8 {
					t.Errorf("%s move %.4g: %v", lim.name, d, msgs)
				}
			}
		}
		t.Logf("%s: max reversals %d, worst (settle - 2 ticks)/time-optimal %.3f", lim.name, maxReversals, worstRatio)
		if failures > 0 {
			t.Errorf("%s: %d of %d move sizes failed", lim.name, failures, sizes)
		}
	}
}

func TestVAJ1D_InvalidSamplePeriodIsNoOp(t *testing.T) {
	for _, dt := range []float32{0, -0.01, math32.NaN(), math32.Inf(1), math32.Inf(-1)} {
		p := New1D(testMaxV, testMaxA, testMaxJ)
		p.Target = 10
		run(&p, 20)
		before := p
		p.Update(dt)
		if p.Output != before.Output || p.Input != before.Input || p.Velocity != before.Velocity || p.Acceleration != before.Acceleration {
			t.Errorf("dt %v changed state: before %+v after %+v", dt, before, p)
		}
		run(&p, 2000)
		if p.Output != 10 || p.Velocity != 0 {
			t.Errorf("dt %v: profile did not recover, final %v v=%v", dt, p.Output, p.Velocity)
		}
	}
}

func TestVAJ1D_UpdateDoesNotAllocate(t *testing.T) {
	p := New1D(1, 1, 10)
	target := float32(0.5)
	allocs := testing.AllocsPerRun(200, func() {
		if p.Output == target && p.Velocity == 0 {
			target = -target
		}
		p.Target = target
		p.Update(0.02)
	})
	if allocs != 0 {
		t.Fatalf("Update allocates %v times per call", allocs)
	}
}

// tracker steps a profile and records per-tick metrics. Its last speed
// carries across external writes between ticks.
type tracker struct {
	p         *VAJ1D
	dir       float32 // direction of the move, for the overshoot
	ticks     int
	settled   int     // first tick with Output == Target at rest, -1 if never
	elapsed   float64 // summed sample periods
	reversals int     // sign changes of the per-tick displacement
	overshoot float32 // largest excursion past the target along dir
	// Per-tick |dx|/dt, finite-difference |a| (change of the tick-average
	// speed between tick midpoints) and state |jerk|, settle tick included.
	maxStep, maxFDAcc, maxSJ float32
	// maxResidual is the largest |dx - v dt - a dt²/2| (v, a published
	// before the tick) in units of maxJ dt³/6: a tick that is consistent
	// with jerk-limited motion from the published state stays within 1.
	maxResidual       float64
	prevSpeed, prevDt float64
	prevDir           float32
}

func newTracker(p *VAJ1D) *tracker {
	return &tracker{p: p, settled: -1, dir: math32.Copysign(1, p.Target-p.Output)}
}

// step advances the profile by one tick of dt and returns its displacement.
func (tr *tracker) step(dt float32) float32 {
	p := tr.p
	in, v, a := p.Input, float64(p.Velocity), float64(p.Acceleration)
	p.Update(dt)
	tr.ticks++
	h := float64(dt)
	tr.elapsed += h
	dx := p.Output - in
	speed := float64(dx) / h
	tr.maxStep = math32.Max(tr.maxStep, float32(math.Abs(speed)))
	if tr.prevDt > 0 {
		tr.maxFDAcc = math32.Max(tr.maxFDAcc, float32(math.Abs(speed-tr.prevSpeed)/((h+tr.prevDt)/2)))
	}
	tr.prevSpeed, tr.prevDt = speed, h
	tr.maxSJ = math32.Max(tr.maxSJ, float32(math.Abs(float64(p.Acceleration)-a)/h))
	residual := math.Abs(float64(dx)-v*h-a*h*h/2) / (float64(p.maxJ) * h * h * h / 6)
	tr.maxResidual = math.Max(tr.maxResidual, residual)
	if dx != 0 {
		sg := math32.Copysign(1, dx)
		if tr.prevDir != 0 && sg != tr.prevDir {
			tr.reversals++
		}
		tr.prevDir = sg
	}
	tr.overshoot = math32.Max(tr.overshoot, tr.dir*(p.Output-p.Target))
	if tr.settled < 0 && p.Output == p.Target && p.Velocity == 0 {
		tr.settled = tr.ticks
	}
	return dx
}

// run steps with the sample period returned by dt until the profile settles
// or steps ticks pass, and reports whether it settled.
func (tr *tracker) run(steps int, dt func() float32) bool {
	for range steps {
		tr.step(dt())
		if tr.settled >= 0 {
			return true
		}
	}
	return false
}

// drive runs a profile at a constant dt until it settles.
func drive(p *VAJ1D, dt float32, steps int) *tracker {
	tr := newTracker(p)
	tr.run(steps, func() float32 { return dt })
	return tr
}

// TestVAJ1D_SettleRespectsLimits covers profiles whose jerk is so large that
// one tick of jerk spans more than the acceleration or speed limit allows:
// the final settle must not jump farther than the limits permit.
func TestVAJ1D_SettleRespectsLimits(t *testing.T) {
	cases := []struct {
		v, a, j, dt, d float32
	}{
		{1, 1, 1e5, 0.02, 0.13},
		{1, 1, 1000, 0.02, 0.0013},
		{1, 1, 1e6, 0.5, 1000},
		{0.5, 2, 400, 0.01, 0.0005},
		{1, 1, 1e4, 0.1, 1},
	}
	for _, c := range cases {
		p := New1D(c.v, c.a, c.j)
		p.Target = c.d
		opt := timeOptimal(float64(c.v), float64(c.a), float64(c.j), float64(c.d))
		r := drive(&p, c.dt, int(3*opt/float64(c.dt))+200)
		name := fmt.Sprintf("V=%v A=%v J=%v dt=%v d=%v", c.v, c.a, c.j, c.dt, c.d)
		if r.settled < 0 {
			t.Errorf("%s: never settled, final %v", name, p.Output)
		}
		if r.maxStep > 1.05*c.v {
			t.Errorf("%s: per-tick speed %v > 1.05*V", name, r.maxStep)
		}
		if r.maxFDAcc > 1.05*c.a {
			t.Errorf("%s: finite-difference |a| %v > 1.05*A", name, r.maxFDAcc)
		}
	}
}

// TestVAJ1D_LongBrakeNoOvershoot covers moves whose braking phase spans 1e4+
// ticks, where float32 rounding of the integrated velocity used to bias the
// state into an overshoot, and a final creep near 1e6 that moves less than a
// float64 ulp per tick.
func TestVAJ1D_LongBrakeNoOvershoot(t *testing.T) {
	cases := []struct {
		v, a, j, dt, from, d float32
	}{
		{11.162597, 0.49838075, 11.845336, 1e-4, -1000, 20.820007},
		{1.7447788, 0.026597047, 0.15868635, 1e-3, 0, 40.589207},
		{1.8583897, 0.005283287, 352.69788, 1e-3, 1, -37.827988},
		{13.008156, 0.0759522, 64.550606, 1e-3, 1, -32.67151},
		// Creeps below one float64 ulp per tick near 1e6 at the end.
		{3.6827369, 0.009180717, 0.013251612, 1e-4, -1e6, -0.0625},
		// Rounding slack used to bias the approach into a noise-level limit
		// cycle around the target.
		{0.9824892, 0.0056840573, 13.649135, 1e-4, 1e6, -0.125},
		// Final deceleration ramp shorter than one tick (J*dt > A).
		{363.09476, 2.1139417, 2324.6284, 1e-3, 1, 2.303213},
	}
	for _, c := range cases {
		p := New1D(c.v, c.a, c.j)
		p.Input, p.Output, p.Target = c.from, c.from, c.from+c.d
		opt := timeOptimal(float64(c.v), float64(c.a), float64(c.j), float64(c.d))
		r := drive(&p, c.dt, int(1.2*opt/float64(c.dt))+1000)
		name := fmt.Sprintf("V=%v A=%v J=%v dt=%v %v->%v", c.v, c.a, c.j, c.dt, c.from, c.from+c.d)
		if r.settled < 0 {
			t.Errorf("%s: never settled, final %v", name, p.Output)
		}
		if r.reversals > 0 {
			t.Errorf("%s: %d reversals", name, r.reversals)
		}
		if tol := 4 * ulp(math32.Abs(p.Target)); r.overshoot > tol {
			t.Errorf("%s: overshoot %v > 4 ulp (%v)", name, r.overshoot, tol)
		}
	}
}

// TestVAJ1D_RandomRestToRest drives 400 seeded random rest-to-rest moves
// (limits over several decades, small to coarse ticks, targets near and far
// from the origin; at most 2e4 time-optimal ticks each). Every move must
// settle within the limits with no reversal and overshoot <= 4 ulp of the
// target.
func TestVAJ1D_RandomRestToRest(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 9))
	const runs = 400
	var failures int
	for n := 0; n < runs; {
		v := math32.Pow(10, rng.Float32()*6-3)
		a := math32.Pow(10, rng.Float32()*6-3)
		j := math32.Pow(10, rng.Float32()*8-3)
		dt := []float32{1e-4, 1e-3, 0.01, 0.02, 0.1, 0.5}[rng.IntN(6)]
		from := []float32{0, 1, -1000, 1e6, -1e6}[rng.IntN(5)]
		to := from + (rng.Float32()-0.5)*[]float32{1e-4, 0.01, 1, 100}[rng.IntN(4)]
		if to == from {
			continue
		}
		opt := timeOptimal(float64(v), float64(a), float64(j), float64(to-from))
		if opt/float64(dt) > 2e4 {
			continue
		}
		n++
		p := New1D(v, a, j)
		p.Input, p.Output, p.Target = from, from, to
		r := drive(&p, dt, int(2*opt/float64(dt))+2000)
		var msgs []string
		if r.settled < 0 {
			msgs = append(msgs, fmt.Sprintf("never settled (final %v)", p.Output))
		}
		if r.reversals > 0 {
			msgs = append(msgs, fmt.Sprintf("%d reversals", r.reversals))
		}
		if tol := 4 * ulp(math32.Abs(to)); r.overshoot > tol {
			msgs = append(msgs, fmt.Sprintf("overshoot %v > %v", r.overshoot, tol))
		}
		// Output moves in float32 steps: allow one quantum per tick.
		if quantum := 2 * ulp(math32.Max(math32.Abs(from), math32.Abs(to))) / dt; r.maxStep > 1.05*v+quantum {
			msgs = append(msgs, fmt.Sprintf("speed %v", r.maxStep))
		}
		if r.maxSJ > 1.05*j {
			msgs = append(msgs, fmt.Sprintf("state jerk %v", r.maxSJ))
		}
		if len(msgs) > 0 {
			failures++
			if failures <= 10 {
				t.Errorf("V=%v A=%v J=%v dt=%v %v->%v: %v", v, a, j, dt, from, to, msgs)
			}
		}
	}
	if failures > 0 {
		t.Errorf("%d of %d random moves failed", failures, runs)
	}
}

// Manipulator firmware profile: New1D(100, 100, 100) at a nominal 100 ms tick.
const (
	fwV, fwA, fwJ = 100, 100, 100
	fwDt          = 0.1
)

func constDt(dt float32) func() float32 { return func() float32 { return dt } }

// TestVAJ1D_JitteredSamplePeriod reproduces the manipulator firmware, which
// passes the measured time.Since period to Update: 200 log-spaced
// rest-to-rest moves from 1e-4 to 100 with V=A=J=100 at
// dt = 0.1*(1+jitter*u), u uniform in [-1, 1) (seeded), for jitter up to 10%.
// Every move must land exactly on the target at rest without a direction
// reversal or overshoot, every tick must be consistent with jerk-limited
// motion from the published state, within the limits, and the move must
// settle within 1.5x the time-optimal duration plus 3 ticks.
func TestVAJ1D_JitteredSamplePeriod(t *testing.T) {
	const sizes = 200
	for _, jitter := range []float64{1e-4, 1e-3, 0.02, 0.1} {
		t.Run(fmt.Sprintf("jitter %g", jitter), func(t *testing.T) {
			rng := rand.New(rand.NewPCG(18, math.Float64bits(jitter)))
			dt := func() float32 { return float32(fwDt * (1 + jitter*(2*rng.Float64()-1))) }
			var failures, reversed int
			var worstResidual, worstRatio float64
			for k := range sizes {
				d := float32(math.Pow(10, -4+6*float64(k)/float64(sizes-1)))
				opt := timeOptimal(fwV, fwA, fwJ, float64(d))
				p := New1D(fwV, fwA, fwJ)
				p.Target = d
				tr := newTracker(&p)
				tr.run(int(3*opt/fwDt)+100, dt)
				worstResidual = math.Max(worstResidual, tr.maxResidual)
				var msgs []string
				if tr.settled < 0 {
					msgs = append(msgs, fmt.Sprintf("never settled exactly (final %v, v=%v)", p.Output, p.Velocity))
				} else {
					worstRatio = math.Max(worstRatio, tr.elapsed/opt)
					if bound := 1.5*opt + 3*fwDt*(1+jitter); tr.elapsed > bound {
						msgs = append(msgs, fmt.Sprintf("settled after %.3fs > %.3fs", tr.elapsed, bound))
					}
				}
				if tr.reversals > 0 {
					reversed++
					msgs = append(msgs, fmt.Sprintf("%d direction reversals", tr.reversals))
				}
				if tol := 4 * ulp(d); tr.overshoot > tol {
					msgs = append(msgs, fmt.Sprintf("overshoot %v", tr.overshoot))
				}
				if tr.maxResidual > 1.05 {
					msgs = append(msgs, fmt.Sprintf("tick inconsistent with jerk-limited motion: residual %.3g x J dt³/6", tr.maxResidual))
				}
				if tr.maxStep > 1.05*fwV {
					msgs = append(msgs, fmt.Sprintf("speed %v", tr.maxStep))
				}
				if tr.maxFDAcc > 1.05*fwA {
					msgs = append(msgs, fmt.Sprintf("finite-difference |a| %v", tr.maxFDAcc))
				}
				if tr.maxSJ > 1.05*fwJ {
					msgs = append(msgs, fmt.Sprintf("state |jerk| %v", tr.maxSJ))
				}
				if len(msgs) > 0 {
					failures++
					if failures <= 8 {
						t.Errorf("move %.4g: %v", d, msgs)
					}
				}
			}
			t.Logf("jitter %g: %d/%d moves reversed, worst residual %.3f x J dt³/6, worst settle/time-optimal %.3f",
				jitter, reversed, sizes, worstResidual, worstRatio)
			if failures > 0 {
				t.Errorf("%d of %d moves failed", failures, sizes)
			}
		})
	}
}

// TestVAJ1D_NonFiniteTargetBrakes sets a NaN or infinite Target (an unchecked
// serial command) while the profile accelerates: it must brake to rest within
// its stopping distance without reversing, replace Target with that rest
// position, hold there while Target stays non-finite, and track a finite
// Target again afterwards.
func TestVAJ1D_NonFiniteTargetBrakes(t *testing.T) {
	for _, bad := range []float32{math32.NaN(), math32.Inf(1), math32.Inf(-1)} {
		t.Run(fmt.Sprint(bad), func(t *testing.T) {
			p := New1D(testMaxV, testMaxA, testMaxJ)
			p.Target = 100
			run(&p, 30)
			revPos, v, a := p.Output, p.Velocity, p.Acceleration
			if v <= 0 {
				t.Fatalf("velocity %v must be positive before the bad target", v)
			}
			bound := revPos + 1.05*idealStop(v, a) + 1e-3
			p.Target = bad
			tr := newTracker(&p)
			tr.dir = 1
			if !tr.run(1000, constDt(testDt)) {
				t.Fatalf("never came to rest: Output %v Velocity %v Acceleration %v Target %v", p.Output, p.Velocity, p.Acceleration, p.Target)
			}
			if math32.IsNaN(p.Target) || math32.IsInf(p.Target, 0) {
				t.Errorf("Target %v not replaced by the rest position", p.Target)
			}
			if p.Output < revPos || p.Output > bound {
				t.Errorf("rest at %v outside [%v, stopping bound %v]", p.Output, revPos, bound)
			}
			if tr.reversals > 0 {
				t.Errorf("%d reversals while braking", tr.reversals)
			}
			if tr.maxStep > 1.05*testMaxV {
				t.Errorf("speed %v > 1.05*V", tr.maxStep)
			}
			if tr.maxResidual > 1.05 {
				t.Errorf("tick inconsistent with jerk-limited motion: residual %.3g x J dt³/6", tr.maxResidual)
			}
			rest := p.Output
			p.Target = bad
			run(&p, 50)
			if p.Output != rest || p.Velocity != 0 || p.Acceleration != 0 {
				t.Errorf("did not hold at rest %v: Output %v v=%v a=%v", rest, p.Output, p.Velocity, p.Acceleration)
			}
			p.Target = 50
			if r := drive(&p, testDt, 2000); r.settled < 0 || p.Output != 50 {
				t.Errorf("finite target afterwards not reached: Output %v v=%v", p.Output, p.Velocity)
			}
		})
	}

	// Far from the origin a slow brake ends within one float32 ulp, below
	// what Output resolves: the rest position must be rounded ahead of the
	// motion, so the velocity never turns back.
	p := New1D(1, 1, 10)
	p.Input, p.Output, p.Target = 1e6, 1e6, 1e6+1
	p.Update(0.02)
	start := p.Output
	p.Target = math32.NaN()
	minV := p.Velocity
	for range 100 {
		p.Update(0.02)
		minV = math32.Min(minV, p.Velocity)
	}
	if p.Velocity != 0 || minV < 0 || p.Output < start || p.Output > start+ulp(start) {
		t.Errorf("far from origin: rest at %v (v=%v), lowest velocity %v, want rest within [%v, %v] without turning back",
			p.Output, p.Velocity, minV, start, start+ulp(start))
	}
}

// TestVAJ1D_InvalidLimits: a zero, negative or NaN limit makes Update hold
// (Output = Input, state untouched); a +Inf limit means unlimited, for the
// speed, acceleration and jerk alike, while the finite limits still hold.
func TestVAJ1D_InvalidLimits(t *testing.T) {
	nan, inf := math32.NaN(), math32.Inf(1)
	hold := []struct {
		name    string
		v, a, j float32
	}{
		{"zero value", 0, 0, 0},
		{"V=0", 0, 100, 1000}, {"A=0", 100, 0, 1000}, {"J=0", 100, 100, 0},
		{"V=NaN", nan, 100, 1000}, {"A=NaN", 100, nan, 1000}, {"J=NaN", 100, 100, nan},
		{"V<0", -100, 100, 1000}, {"A<0", 100, -100, 1000}, {"J<0", 100, 100, -1000},
		{"V=-Inf", -inf, 100, 1000}, {"A=-Inf", 100, -inf, 1000}, {"J=-Inf", 100, 100, -inf},
	}
	for _, c := range hold {
		p := New1D(c.v, c.a, c.j)
		p.Input, p.Output, p.Target = 1, 1, 10
		for range 100 {
			p.Update(testDt)
		}
		if p.Output != 1 || p.Input != 1 || p.Velocity != 0 || p.Acceleration != 0 {
			t.Errorf("%s: did not hold: Output %v Input %v v=%v a=%v", c.name, p.Output, p.Input, p.Velocity, p.Acceleration)
		}
	}

	unlimited := []struct {
		name    string
		v, a, j float32
	}{
		{"V=Inf", inf, 100, 1000}, {"A=Inf", 100, inf, 1000}, {"J=Inf", 100, 100, inf},
		{"V=A=Inf", inf, inf, 1000}, {"V=J=Inf", inf, 100, inf}, {"A=J=Inf", 100, inf, inf},
		{"all Inf", inf, inf, inf},
	}
	finite := func(x float32) float64 {
		if math32.IsInf(x, 1) {
			return 1e12
		}
		return float64(x)
	}
	for _, c := range unlimited {
		for _, d := range []float32{0.001, 9, 1000} {
			name := fmt.Sprintf("%s d=%v", c.name, d)
			p := New1D(c.v, c.a, c.j)
			p.Input, p.Output, p.Target = 1, 1, 1+d
			opt := timeOptimal(finite(c.v), finite(c.a), finite(c.j), float64(d))
			r := drive(&p, testDt, int(3*opt/testDt)+100)
			if r.settled < 0 {
				t.Errorf("%s: never settled: Output %v v=%v a=%v", name, p.Output, p.Velocity, p.Acceleration)
				continue
			}
			if bound := 1.5*opt + 3*testDt; r.elapsed > bound {
				t.Errorf("%s: settled after %.3fs > %.3fs", name, r.elapsed, bound)
			}
			if r.reversals > 0 || r.overshoot > 4*ulp(1+d) {
				t.Errorf("%s: %d reversals, overshoot %v", name, r.reversals, r.overshoot)
			}
			if !math32.IsInf(c.v, 1) && r.maxStep > 1.05*c.v {
				t.Errorf("%s: speed %v > 1.05*V", name, r.maxStep)
			}
			if !math32.IsInf(c.a, 1) && r.maxFDAcc > 1.05*c.a {
				t.Errorf("%s: finite-difference |a| %v > 1.05*A", name, r.maxFDAcc)
			}
			if !math32.IsInf(c.j, 1) && r.maxSJ > 1.05*c.j {
				t.Errorf("%s: state |jerk| %v > 1.05*J", name, r.maxSJ)
			}
		}
	}
}

// toMidLanding steps the profile until a landing plan has executed at least
// one tick and still has ticks left.
func toMidLanding(t *testing.T, tr *tracker, dt float32) {
	t.Helper()
	p := tr.p
	for range 1000 {
		if p.planIdx >= 1 && p.planIdx < p.planLen {
			return
		}
		tr.step(dt)
		if tr.settled >= 0 {
			t.Fatalf("settled before a landing was under way")
		}
	}
	t.Fatalf("no landing under way after 1000 ticks")
}

// checkFinish runs the firmware profile to rest and requires an exact landing
// on Target with every tick consistent with jerk-limited motion from the
// published state (also across the external change just made).
func checkFinish(t *testing.T, name string, tr *tracker) {
	t.Helper()
	p := tr.p
	if !tr.run(400, constDt(fwDt)) {
		t.Errorf("%s: never settled: Output %v v=%v target %v", name, p.Output, p.Velocity, p.Target)
		return
	}
	if tr.maxResidual > 1.05 {
		t.Errorf("%s: tick inconsistent with jerk-limited motion: residual %.3g x J dt³/6 (bound 1.05)", name, tr.maxResidual)
	}
	if tr.maxStep > 1.05*fwV {
		t.Errorf("%s: speed %v > 1.05*V", name, tr.maxStep)
	}
}

// TestVAJ1D_TargetChangeMidLanding changes Target while a landing plan is
// executing. The plan must be dropped: a stale plan finishes on the old
// target and then snaps the output by the whole target change.
func TestVAJ1D_TargetChangeMidLanding(t *testing.T) {
	for _, target := range []float32{8, 5.5, 4.9, 2} {
		p := New1D(fwV, fwA, fwJ)
		p.Target = 5
		tr := newTracker(&p)
		toMidLanding(t, tr, fwDt)
		p.Target = target
		checkFinish(t, fmt.Sprintf("5 -> %v", target), tr)
		if tr.maxFDAcc > 1.05*fwA {
			t.Errorf("5 -> %v: finite-difference |a| %v > 1.05*A", target, tr.maxFDAcc)
		}
	}
}

// TestVAJ1D_ResetMidMove calls Reset while cruising and while landing: the
// profile must restart from rest at its current position (rigidbody calls
// Reset on every path change), drop any plan and land exactly.
func TestVAJ1D_ResetMidMove(t *testing.T) {
	cases := []struct {
		name   string
		target float32
		mid    func(*testing.T, *tracker)
	}{
		{"cruise", 200, func(_ *testing.T, tr *tracker) { tr.run(20, constDt(fwDt)) }},
		{"landing", 5, func(t *testing.T, tr *tracker) { toMidLanding(t, tr, fwDt) }},
	}
	for _, c := range cases {
		p := New1D(fwV, fwA, fwJ)
		p.Target = c.target
		tr := newTracker(&p)
		c.mid(t, tr)
		if p.Velocity == 0 {
			t.Fatalf("%s: not moving before Reset", c.name)
		}
		pos := p.Output
		p.Reset()
		if p.Velocity != 0 || p.Acceleration != 0 || p.Output != pos {
			t.Errorf("%s: Reset left v=%v a=%v Output %v (was %v)", c.name, p.Velocity, p.Acceleration, p.Output, pos)
		}
		tr.prevDt = 0 // the finite-difference acceleration restarts from rest
		checkFinish(t, c.name, tr)
		if p.Output != c.target || tr.reversals > 0 {
			t.Errorf("%s: final %v (target %v), %d reversals", c.name, p.Output, c.target, tr.reversals)
		}
	}
}

// TestVAJ1D_ExternalStateWrites writes Velocity or Acceleration from outside
// mid-move and mid-landing: the next tick must start from the written state
// and any plan must be dropped.
func TestVAJ1D_ExternalStateWrites(t *testing.T) {
	cases := []struct {
		name   string
		target float32
		mid    func(*testing.T, *tracker)
		write  func(*VAJ1D)
	}{
		{"velocity mid-move", 30, func(_ *testing.T, tr *tracker) { tr.run(5, constDt(fwDt)) },
			func(p *VAJ1D) { p.Velocity, p.Acceleration = p.Velocity/2, 0 }},
		{"acceleration mid-move", 30, func(_ *testing.T, tr *tracker) { tr.run(5, constDt(fwDt)) },
			func(p *VAJ1D) { p.Acceleration = -50 }},
		{"velocity mid-landing", 5, func(t *testing.T, tr *tracker) { toMidLanding(t, tr, fwDt) },
			func(p *VAJ1D) { p.Velocity /= 2 }},
		{"acceleration mid-landing", 5, func(t *testing.T, tr *tracker) { toMidLanding(t, tr, fwDt) },
			func(p *VAJ1D) { p.Acceleration -= 10 }},
	}
	for _, c := range cases {
		p := New1D(fwV, fwA, fwJ)
		p.Target = c.target
		tr := newTracker(&p)
		c.mid(t, tr)
		before := p
		c.write(&p)
		if p.Velocity == before.Velocity && p.Acceleration == before.Acceleration {
			t.Fatalf("%s: write changed nothing (v=%v a=%v)", c.name, p.Velocity, p.Acceleration)
		}
		checkFinish(t, c.name, tr)
		if p.Output != c.target {
			t.Errorf("%s: final %v, target %v", c.name, p.Output, c.target)
		}
	}
}

// TestVAJ1D_InputWriteMidLanding writes Input mid-landing. A position
// correction must drop the plan (a stale plan snaps the output by the
// correction on its last tick); float32 noise of a fed-back Input (1 ulp on
// every tick, as a clamped feedback produces) keeps it: over 200 log-spaced
// moves the noisy run stays within the accumulated drift of the clean run and
// settles on the same tick.
func TestVAJ1D_InputWriteMidLanding(t *testing.T) {
	for _, shift := range []float32{-2, -0.3, 0.3} {
		p := New1D(fwV, fwA, fwJ)
		p.Target = 5
		tr := newTracker(&p)
		toMidLanding(t, tr, fwDt)
		p.Input += shift
		checkFinish(t, fmt.Sprintf("shift %v", shift), tr)
		if p.Output != 5 {
			t.Errorf("shift %v: final %v, target 5", shift, p.Output)
		}
	}

	var failures int
	for k := range 200 {
		d := float32(math.Pow(10, -4+6*float64(k)/199))
		ref := New1D(fwV, fwA, fwJ)
		ref.Target = d
		p := ref
		var msgs []string
		for i := 1; i <= 1000; i++ {
			ref.Update(fwDt)
			p.Input = math32.Nextafter(p.Input, 0)
			p.Update(fwDt)
			if dev, drift := math32.Abs(p.Output-ref.Output), float32(i+1)*ulp(d); dev > drift && len(msgs) == 0 {
				msgs = append(msgs, fmt.Sprintf("tick %d off the clean run by %v > drift %v", i, dev, drift))
			}
			refDone, done := ref.Output == d && ref.Velocity == 0, p.Output == d && p.Velocity == 0
			if refDone != done {
				msgs = append(msgs, fmt.Sprintf("settled on a different tick than the clean run (%d)", i))
				break
			}
			if done {
				break
			}
		}
		if len(msgs) > 0 {
			if failures++; failures <= 5 {
				t.Errorf("move %.4g with 1 ulp Input noise per tick: %v", d, msgs)
			}
		}
	}
	if failures > 0 {
		t.Errorf("%d of 200 moves: Input noise changed the landing", failures)
	}
}

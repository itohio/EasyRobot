package eskf

import (
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nav/calib"
)

// quat is a float64 quaternion [w,x,y,z] for the truth.
type quat [4]float64

func axisAngle(ax [3]float64, a float64) quat {
	n := math.Sqrt(ax[0]*ax[0] + ax[1]*ax[1] + ax[2]*ax[2])
	if n == 0 || a == 0 {
		return quat{1, 0, 0, 0}
	}
	s := math.Sin(a/2) / n
	return quat{math.Cos(a / 2), ax[0] * s, ax[1] * s, ax[2] * s}
}

func (p quat) mul(q quat) quat {
	return quat{
		p[0]*q[0] - p[1]*q[1] - p[2]*q[2] - p[3]*q[3],
		p[0]*q[1] + p[1]*q[0] + p[2]*q[3] - p[3]*q[2],
		p[0]*q[2] - p[1]*q[3] + p[2]*q[0] + p[3]*q[1],
		p[0]*q[3] + p[1]*q[2] - p[2]*q[1] + p[3]*q[0],
	}
}

func (p quat) conj() quat { return quat{p[0], -p[1], -p[2], -p[3]} }

func (p quat) rot(v [3]float64) [3]float64 {
	r := p.mul(quat{0, v[0], v[1], v[2]}).mul(p.conj())
	return [3]float64{r[1], r[2], r[3]}
}

// rng is a seeded generator (the same on every machine).
type rng uint64

func (r *rng) gauss() float64 {
	*r = *r*6364136223846793005 + 1442695040888963407
	u1 := (float64(*r>>11) + 1) / (1 << 53)
	*r = *r*6364136223846793005 + 1442695040888963407
	u2 := float64(*r>>11) / (1 << 53)
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

const rate = 100.0

// upright is the badge held upright, screen to the user: badge y up.
var upright = axisAngle([3]float64{1, 0, 0}, math.Pi/2)

// hand is a model of a hand holding the IMU, fitted to a recorded
// hand-held MPU6050: the gyro's tremor is a real rotation θ (θ' = w − θ/s, w an
// AR(1) of lag-1 0.92) that the accelerometer sees; the translational
// tremor a damped 8 Hz oscillator; white noise 0.016 m/s² and 0.0006 rad/s;
// the accelerometer offset off (zero once a multi-pose calibration is
// applied) and the gyro
// bias gb.
type hand struct {
	r       rng
	tA, tG  float64
	tg, th  [3]float64
	ox, ov  [3]float64
	unit    float64
	off, gb [3]float64
}

const w0, zeta = 2 * math.Pi * 8, 0.3

func newHand(seed uint64, tA, tG float64) *hand {
	h := &hand{r: rng(seed), tA: tA, tG: tG, off: [3]float64{0.04, -0.06, 0.05}, gb: [3]float64{0.01, -0.006, 0.008}}
	r := rng(99)
	var x, v, sq float64
	for i := range 100000 {
		a := -w0*w0*x - 2*zeta*w0*v
		v += (a + r.gauss()) / rate
		x += v / rate
		if i > 1000 {
			sq += a * a
		}
	}
	h.unit = math.Sqrt(sq / (100000 - 1001))
	return h
}

// stillHand is a steady hand (the capture's 20 s held still).
func stillHand(seed uint64) *hand { return newHand(seed, 0.031, 0.016) }

// calibrated: a multi-pose calibration's fit applied to the samples (no
// offset left), and the
// fit as the filter's prior.
func calibrated(h *hand, f *Filter) {
	h.off = [3]float64{}
	fit := calib.Fit{GyroBias: [3]float32{float32(h.gb[0]), float32(h.gb[1]), float32(h.gb[2])}, RMS: 0.01, Scale: [3]float32{1, 1, 1}}
	f.Prior(&fit)
}

// sample: the body's attitude q (sensor to world), world acceleration a
// and body rate w; returns the readings and the true attitude with the
// tremor's rotation.
func (h *hand) sample(q quat, a, w [3]float64) (acc, gyro [3]float32, qt quat) {
	var wt, ta [3]float64
	for k := range 3 {
		h.tg[k] = 0.92*h.tg[k] + math.Sqrt(1-0.92*0.92)*h.r.gauss()*h.tG
		wt[k] = h.tg[k] - h.th[k]
		h.th[k] += wt[k] / rate
		x := -w0*w0*h.ox[k] - 2*zeta*w0*h.ov[k]
		h.ov[k] += (x + h.r.gauss()*h.tA/h.unit) / rate
		h.ox[k] += h.ov[k] / rate
		ta[k] = x
	}
	th := math.Sqrt(h.th[0]*h.th[0] + h.th[1]*h.th[1] + h.th[2]*h.th[2])
	qt = q
	if th > 0 {
		qt = q.mul(axisAngle(h.th, th))
	}
	f := qt.conj().rot([3]float64{a[0] + ta[0], a[1] + ta[1], a[2] + ta[2] + StandardGravity})
	for i := range 3 {
		acc[i] = float32(f[i] + h.off[i] + 0.016*h.r.gauss())
		gyro[i] = float32(w[i] + wt[i] + h.gb[i] + 0.0006*h.r.gauss())
	}
	return
}

// hold feeds s seconds of the hand at attitude q.
func hold(h *hand, f *Filter, q quat, s float64) {
	for range int(s * rate) {
		a, g, _ := h.sample(q, [3]float64{}, [3]float64{})
		f.Update(1/rate, a, g)
	}
}

func newFilter(t *testing.T) Filter {
	t.Helper()
	f, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	return f
}

// tiltErr is the angle between the filter's up and the true up (degrees).
func tiltErr(f *Filter, q quat) float64 {
	fq := quat{float64(f.Q[0]), float64(f.Q[1]), float64(f.Q[2]), float64(f.Q[3])}
	a := fq.conj().rot([3]float64{0, 0, 1})
	b := q.conj().rot([3]float64{0, 0, 1})
	return math.Acos(math.Min(1, a[0]*b[0]+a[1]*b[1]+a[2]*b[2])) * 180 / math.Pi
}

func dist(a, b [3]float32) float64 {
	var s float64
	for i := range 3 {
		d := float64(a[i] - b[i])
		s += d * d
	}
	return math.Sqrt(s)
}

// quintic is a minimum-jerk move of A over T: position and rate at t.
func quintic(t, A, T float64) (x, v float64) {
	if t <= 0 {
		return 0, 0
	}
	if t >= T {
		return A, 0
	}
	u := t / T
	return A * (10*u*u*u - 15*u*u*u*u + 6*u*u*u*u*u), A / T * (30*u*u - 60*u*u*u + 30*u*u*u*u)
}

func TestNewRejectsConfig(t *testing.T) {
	for name, mod := range map[string]func(*Config){
		"G 0":              func(c *Config) { c.G = 0 },
		"noise NaN":        func(c *Config) { c.AccelNoise = float32(math.NaN()) },
		"gate negative":    func(c *Config) { c.Gate = -1 },
		"TiltEvery 0":      func(c *Config) { c.TiltEvery = 0 },
		"StillTime +Inf":   func(c *Config) { c.StillTime = float32(math.Inf(1)) },
		"RestTime 0":       func(c *Config) { c.RestTime = 0 },
		"StillAccel zero":  func(c *Config) { c.StillAccel = 0 },
		"bias walk absent": func(c *Config) { c.GyroBiasWalk = 0 },
	} {
		c := DefaultConfig()
		mod(&c)
		if _, err := New(c); !errors.Is(err, ErrConfig) {
			t.Errorf("%s: New returned %v, want ErrConfig", name, err)
		}
	}
}

// Until its first rest the filter holds still (Ready false, nothing moves);
// the first rest sets the tilt from the force, in any pose.
func TestFirstRestSetsTheTilt(t *testing.T) {
	for _, pose := range []quat{upright, quat{1, 0, 0, 0}, upright.mul(axisAngle([3]float64{0, 0, 1}, 0.7)), axisAngle([3]float64{1, 1, 0}, 2.5)} {
		h := stillHand(3)
		f := newFilter(t)
		calibrated(h, &f)
		a, g, _ := h.sample(pose, [3]float64{}, [3]float64{})
		f.Update(1/rate, a, g)
		if f.Ready || f.P != [3]float32{} || f.V != [3]float32{} {
			t.Fatalf("pose %v: ready or moving after one sample", pose)
		}
		var qt quat
		for range int(2 * rate) {
			a, g, qt = h.sample(pose, [3]float64{}, [3]float64{})
			f.Update(1/rate, a, g)
		}
		if !f.Ready {
			t.Fatalf("pose %v: not ready after 2 s in a steady hand", pose)
		}
		if e := tiltErr(&f, qt); e > 0.2 {
			t.Errorf("pose %v: tilt %.2f° off the hand's true attitude after the first rest (want ≤ 0.2°)", pose, e)
		}
	}
}

// At rest the gyro's bias is what it reads: learned on a desk to 0.5
// mrad/s in 5 s; in a steady hand, whose tremor is real rotation with about
// a second of correlation, to what 5 s of it can average down to (0.016
// rad/s · √(2·1 s / 5 s) ≈ 0.01).
func TestRestLearnsTheGyroBias(t *testing.T) {
	for _, c := range []struct {
		name   string
		tA, tG float64
		within float64
	}{{"desk", 0, 0, 0.0005}, {"steady hand", 0.031, 0.016, 0.01}} {
		h := newHand(5, c.tA, c.tG)
		f := newFilter(t)
		hold(h, &f, upright, 5)
		var e float64
		for i := range 3 {
			e = math.Max(e, math.Abs(float64(f.GyroBias[i])-h.gb[i]))
		}
		if e > c.within {
			t.Errorf("%s: gyro bias %.4f rad/s off after 5 s at rest (want ≤ %.4f)", c.name, e, c.within)
		}
	}
}

// A gyro's bias wanders (0.7 mrad/s between still stretches in a hand-held
// capture; an
// MPU6050's moves by about 20 mrad/s, 1°/s, as it warms): once learned, a
// real change is still followed, though the zero rate's gate (which keeps
// slow turns out) refuses it at first. On a desk, after the bias has
// settled, it steps by 20 mrad/s; within 5 s at rest the filter has it to
// 0.5 mrad/s.
func TestRestFollowsABiasChange(t *testing.T) {
	for axis := range 3 {
		h := newHand(19, 0, 0)
		f := newFilter(t)
		hold(h, &f, upright, 10)
		h.gb[axis] += 0.02
		hold(h, &f, upright, 5)
		if e := math.Abs(float64(f.GyroBias[axis]) - h.gb[axis]); e > 0.0005 {
			t.Errorf("axis %d: the bias still %.4f rad/s off 5 s after a 20 mrad/s step (want ≤ 0.0005)", axis, e)
		}
	}
}

// Slow rotation drifting to one side (a rest test against a moving bias
// learns a slow turn as bias): a slow turn in a steady hand, calibrated,
// is a turn. The gyro bias doesn't
// absorb it (a rest test against a moving bias did), the tilt follows, and
// the position stays put through the turn and 10 s held after it.
func TestSlowTurnIsATurn(t *testing.T) {
	for _, ax := range [][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}} {
		for _, c := range []struct{ A, T float64 }{{20, 2}, {20, 5}, {45, 5}} {
			h := stillHand(7)
			f := newFilter(t)
			calibrated(h, &f)
			hold(h, &f, upright, 10)
			p0 := f.P
			var tilt, moved, bias float64
			for i := range int((c.T + 10) * rate) {
				ang, w := quintic(float64(i)/rate, c.A*math.Pi/180, c.T)
				a, g, qt := h.sample(upright.mul(axisAngle(ax, ang)), [3]float64{}, [3]float64{ax[0] * w, ax[1] * w, ax[2] * w})
				f.Update(1/rate, a, g)
				tilt = math.Max(tilt, tiltErr(&f, qt))
				moved = math.Max(moved, dist(f.P, p0))
				for k := range 3 {
					bias = math.Max(bias, math.Abs(float64(f.GyroBias[k])-h.gb[k]))
				}
			}
			if tilt > 0.25 || moved > 0.06 || bias > 0.004 {
				t.Errorf("%v° / %v s about %v: tilt %.2f°, moved %.1f cm, gyro bias %.4f rad/s off (want ≤ 0.25°, 6 cm, 0.004)", c.A, c.T, ax, tilt, moved*100, bias)
			}
		}
	}
}

// Uncalibrated (the accelerometer's offsets unseen) a turn leaves a tilt
// the filter can't tell from an offset, and the world acceleration it
// leaks keeps the body from looking calm; zero velocity must still come
// once the body is still, or the speed runs on for good. With an
// MPU6050's uncalibrated zero-g offsets (0.2–0.3 m/s²), after a 45° turn
// and 10 s held, the body has stopped and moved at most 1.5 m: the honest
// uncalibrated drift of a turn, against metres a second once it locks out.
func TestUncalibratedTurnComesToRest(t *testing.T) {
	for _, ax := range [][3]float64{{1, 0, 0}, {0, 0, 1}} {
		for _, tremor := range []bool{false, true} {
			h := newHand(7, 0, 0)
			if tremor {
				h = stillHand(7)
			}
			h.off = [3]float64{0.2, -0.3, 0.25}
			f := newFilter(t)
			hold(h, &f, upright, 10)
			p0 := f.P
			end := upright.mul(axisAngle(ax, math.Pi/4))
			for i := range int(5 * rate) {
				ang, w := quintic(float64(i)/rate, math.Pi/4, 5)
				a, g, _ := h.sample(upright.mul(axisAngle(ax, ang)), [3]float64{}, [3]float64{ax[0] * w, ax[1] * w, ax[2] * w})
				f.Update(1/rate, a, g)
			}
			hold(h, &f, end, 10)
			speed := math.Sqrt(float64(f.V[0]*f.V[0] + f.V[1]*f.V[1] + f.V[2]*f.V[2]))
			if moved := dist(f.P, p0); speed > 0.05 || moved > 1.5 {
				t.Errorf("45° about %v, tremor %v: speed %.3f m/s, moved %.1f cm after 10 s held (want ≤ 0.05, 150 cm)", ax, tremor, speed, moved*100)
			}
		}
	}
}

// Right after power-on, uncalibrated, the gyro's bias is still uncertain
// and the zero rate's gate wide: a slow turn begun then (it looks still at
// its slow start and end) must not be learned as bias. Only a quiet gyro
// is rest.
func TestEarlySlowTurnIsNotBias(t *testing.T) {
	for _, ax := range [][3]float64{{1, 0, 0}, {0, 0, 1}} {
		h := stillHand(13)
		f := newFilter(t)
		for !f.Ready {
			a, g, _ := h.sample(upright, [3]float64{}, [3]float64{})
			f.Update(1/rate, a, g)
		}
		var bias float64
		for i := range int(7 * rate) {
			ang, w := quintic(float64(i)/rate, 20*math.Pi/180, 5)
			a, g, _ := h.sample(upright.mul(axisAngle(ax, ang)), [3]float64{}, [3]float64{ax[0] * w, ax[1] * w, ax[2] * w})
			f.Update(1/rate, a, g)
			for k := range 3 {
				bias = math.Max(bias, math.Abs(float64(f.GyroBias[k])-h.gb[k]))
			}
		}
		if bias > 0.02 {
			t.Errorf("20°/5 s about %v right after the first rest: the gyro bias went %.3f rad/s off (want ≤ 0.02; the turn peaks at 0.13)", ax, bias)
		}
	}
}

// Powered on while turning slowly (0.04 rad/s about a horizontal axis: the
// force looks steady), the filter waits for the turn to end before its
// first rest sets the tilt; a rest mid-turn would set it 1–2° off.
func TestFirstRestWaitsForATurnToEnd(t *testing.T) {
	h := stillHand(17)
	h.tA, h.tG = 0, 0
	f := newFilter(t)
	ax := [3]float64{1, 0, 0}
	var qt quat
	for i := range int(4 * rate) {
		a, g, q := h.sample(upright.mul(axisAngle(ax, 0.04*float64(i)/rate)), [3]float64{}, [3]float64{0.04, 0, 0})
		qt = q
		f.Update(1/rate, a, g)
		if f.Ready {
			t.Fatalf("ready %.2f s into a 0.04 rad/s turn", float64(i)/rate)
		}
	}
	for range int(2 * rate) {
		var a, g [3]float32
		a, g, qt = h.sample(upright.mul(axisAngle(ax, 0.16)), [3]float64{}, [3]float64{})
		f.Update(1/rate, a, g)
	}
	if !f.Ready || tiltErr(&f, qt) > 0.5 {
		t.Errorf("after the turn: ready %v, tilt %.2f° off (want ready, ≤ 0.5°)", f.Ready, tiltErr(&f, qt))
	}
}

// Squares drawn by hand: 15 and 30 cm
// squares in the badge's plane (right, up, left, down; 1.5 s a side, 0.5 s
// pauses) in a steady hand, calibrated, seeds 7–10: the path stays in
// the square's box grown by 25 % of the side and closes within 30 % of it.
// Measured: 2–10 %, and 29 % for the 30 cm square of seed 8, whose last
// side tilts 0.26° (a corner's zero velocity cancelled what was left). A
// corner's zero velocity missed reads 200 % and more.
func TestSquaresClose(t *testing.T) {
	for _, side := range []float64{0.15, 0.30} {
		for seed := uint64(7); seed <= 10; seed++ {
			h := stillHand(seed)
			f := newFilter(t)
			calibrated(h, &f)
			hold(h, &f, upright, 10)
			p0 := f.P
			var out float64
			for _, d := range [4][2]float64{{1, 0}, {0, 1}, {-1, 0}, {0, -1}} { // world x right, z up
				for k := range 200 {
					tt := float64(k) / rate
					var acc float64
					if tt < 1.5 {
						u := tt / 1.5
						acc = side / (1.5 * 1.5) * (60*u - 180*u*u + 120*u*u*u)
					}
					a, g, _ := h.sample(upright, [3]float64{d[0] * acc, 0, d[1] * acc}, [3]float64{})
					f.Update(1/rate, a, g)
					x, z, m := float64(f.P[0]-p0[0]), float64(f.P[2]-p0[2]), 0.25*side
					out = math.Max(out, math.Hypot(math.Max(math.Max(-m-x, x-side-m), 0), math.Max(math.Max(-m-z, z-side-m), 0)))
				}
			}
			if c := dist(f.P, p0); c > 0.30*side || out > 0 {
				t.Errorf("%.0f cm square, seed %d: closes %.1f cm away, %.1f cm outside the box (want ≤ %.1f cm, 0)", side*100, seed, c*100, out*100, 30*side)
			}
		}
	}
}

// A stroke from rest in a steady hand, calibrated, over 40 seeds and two
// directions (80 strokes each): 10 cm in 1 s reads within 10 % in 61 of 80
// (0.50–1.24×). Slow strokes mostly don't register in a hand: their middle
// is as slow as the tremor, so zero velocity takes them (5 cm over 2 s
// within 0.5–1.5× in 16 of 80, 3 cm in 9 of 80; under 5 of 80 each with
// StillSpeed at 0.016). The floors keep those rates from falling.
func TestStrokeRegisters(t *testing.T) {
	// A slow stroke's middle has almost no acceleration while its speed is
	// highest (3 cm over 2 s: under 0.05 m/s² for 0.9 s at 0.028 m/s), so
	// it looks calm: zero velocity there would erase it.
	for _, s := range []struct {
		A, T, lo, hi float64
		floor        int
	}{{0.10, 1, 0.9, 1.1, 55}, {0.05, 2, 0.5, 1.5, 12}, {0.03, 2, 0.5, 1.5, 6}} {
		in, lo, hi := 0, math.Inf(1), math.Inf(-1)
		for seed := uint64(100); seed < 140; seed++ {
			for _, d := range [][2]float64{{1, 0}, {0, 1}} {
				h := stillHand(seed)
				f := newFilter(t)
				calibrated(h, &f)
				hold(h, &f, upright, 10)
				p0 := f.P
				for k := range int((s.T + 1) * rate) {
					u := float64(k) / rate / s.T
					var acc float64
					if u < 1 {
						acc = s.A / (s.T * s.T) * (60*u - 180*u*u + 120*u*u*u)
					}
					a, g, _ := h.sample(upright, [3]float64{d[0] * acc, 0, d[1] * acc}, [3]float64{})
					f.Update(1/rate, a, g)
				}
				r := (float64(f.P[0]-p0[0])*d[0] + float64(f.P[2]-p0[2])*d[1]) / s.A
				if r >= s.lo && r <= s.hi {
					in++
				}
				lo, hi = min(lo, r), max(hi, r)
			}
		}
		if in < s.floor {
			t.Errorf("%.0f cm in %.0f s in a hand: %d of 80 within %.1f–%.1f× (reads %.2f–%.2f×; want at least %d)", s.A*100, s.T, in, s.lo, s.hi, lo, hi, s.floor)
		}
	}
}

// On a desk (no tremor), uncalibrated, with an MPU6050's offsets and gyro
// biases still being learned after 10 s, strokes over 2 s, right or up,
// over 40 seeds (80 strokes each): 3 cm (0.028 m/s at most, under the calm
// limit throughout) within 0.5–1.5× in 64 of 80, 5 cm (whose slowing end
// breaks it) in 73 of 80, 10 cm (0.09 m/s: no rest in its middle) in 80
// of 80 (0.83–1.13×). Without the shadow a 3 or 5 cm stroke reads 0;
// StillSpeed at 0.022 loses the 3 cm strokes (52 of 80), at 0.025 nearly
// all (6 of 80).
func TestSlowStrokesOnADesk(t *testing.T) {
	for _, s := range []struct {
		A     float64
		floor int
	}{{0.03, 58}, {0.05, 68}, {0.10, 78}} {
		in, lo, hi := 0, math.Inf(1), math.Inf(-1)
		for seed := uint64(100); seed < 140; seed++ {
			for _, d := range [][2]float64{{1, 0}, {0, 1}} {
				h := newHand(seed, 0, 0)
				h.off = [3]float64{0.04, -0.06, 0.05}
				h.gb = [3]float64{0.01, -0.006, 0.008}
				f := newFilter(t)
				hold(h, &f, upright, 10)
				p0 := f.P
				for k := range int(2.5 * rate) {
					u := float64(k) / rate / 2
					var acc float64
					if u < 1 {
						acc = s.A / 4 * (60*u - 180*u*u + 120*u*u*u)
					}
					a, g, _ := h.sample(upright, [3]float64{d[0] * acc, 0, d[1] * acc}, [3]float64{})
					f.Update(1/rate, a, g)
				}
				r := (float64(f.P[0]-p0[0])*d[0] + float64(f.P[2]-p0[2])*d[1]) / s.A
				if r >= 0.5 && r <= 1.5 {
					in++
				}
				lo, hi = min(lo, r), max(hi, r)
			}
		}
		if in < s.floor {
			t.Errorf("%.0f cm in 2 s on a desk: %d of 80 within 0.5–1.5× (reads %.2f–%.2f×; want at least %d)", s.A*100, in, lo, hi, s.floor)
		}
	}
}

// The filter's σ is honest at rest and too small after a turn. Over 50
// seeds (calibrated, a steady hand, 10 s held), the mean normalised error
// squared lies in its χ² 95 % band at rest: tilt (2 degrees of freedom,
// band 1.45–2.55) measured 1.68, velocity (3, band 2.32–3.68) 2.94; the
// gyro bias reads 3.82 (just over its band). Right after a 30° turn the
// tilt reads 7.26 (σ about half the error's spread), the velocity 4.65 and
// the gyro bias 6.16; their bounds hold that measured over-confidence, so a
// change in either direction shows.
func TestSigmaIsHonestAtRest(t *testing.T) {
	for _, c := range []struct {
		turn            bool
		tilt, vel, bias [2]float64 // bounds on the mean
	}{
		{false, [2]float64{1.45, 2.55}, [2]float64{2.32, 3.68}, [2]float64{2.8, 4.5}},
		{true, [2]float64{5.5, 9}, [2]float64{3.5, 6}, [2]float64{4.5, 8}},
	} {
		var st, sv, sb float64
		const seeds = 50
		for seed := range seeds {
			h := stillHand(uint64(100 + seed))
			f := newFilter(t)
			calibrated(h, &f)
			hold(h, &f, upright, 10)
			q := upright
			if c.turn {
				for i := range int(3.2 * rate) {
					ang, w := quintic(float64(i)/rate, 30*math.Pi/180, 3)
					var a, g [3]float32
					a, g, q = h.sample(upright.mul(axisAngle([3]float64{1, 0, 0}, ang)), [3]float64{}, [3]float64{w, 0, 0})
					f.Update(1/rate, a, g)
				}
			} else {
				_, _, q = h.sample(upright, [3]float64{}, [3]float64{})
			}
			fq := quat{float64(f.Q[0]), float64(f.Q[1]), float64(f.Q[2]), float64(f.Q[3])}
			u := fq.rot(q.conj().rot([3]float64{0, 0, 1})) // the true up seen through the filter: (δθ_y, −δθ_x, ~1)
			att, vel, _ := f.Sigma()
			st += u[1]*u[1]/float64(att[0]*att[0]) + u[0]*u[0]/float64(att[1]*att[1])
			for k := range 3 {
				sv += float64(f.V[k]*f.V[k]) / float64(vel[k]*vel[k])
				e := float64(f.GyroBias[k]) - h.gb[k]
				sb += e * e / float64(f.cov[iBg+k][iBg+k])
			}
		}
		for _, m := range []struct {
			name string
			v    float64
			b    [2]float64
		}{{"tilt", st / seeds, c.tilt}, {"velocity", sv / seeds, c.vel}, {"gyro bias", sb / seeds, c.bias}} {
			if m.v < m.b[0] || m.v > m.b[1] {
				t.Errorf("turn %v: %s mean normalised error squared %.2f over %d seeds (want %.2f–%.2f)", c.turn, m.name, m.v, seeds, m.b[0], m.b[1])
			}
		}
	}
}

// Ten minutes of a hand moving (turns, strokes, rests) in float32 leave the
// covariance usable: finite, a positive diagonal and correlations within ±1
// (what guard holds, pair by pair), and the estimate finite and bounded.
// Not positive semidefinite: the smallest eigenvalue of the correlation
// matrix reaches −0.2 in such runs and −2.6 in some uncalibrated ones, in
// directions no measurement touches (the innovation variance stays above
// the measurement noise).
func TestTenMinutesStayAFilter(t *testing.T) {
	h := stillHand(11)
	f := newFilter(t)
	hold(h, &f, upright, 5)
	r := rng(5)
	q := upright
	for s := range 120 { // 5 s each: a turn or a stroke, then a pause
		ax := [3]float64{r.gauss(), r.gauss(), r.gauss()}
		A := 0.5 * r.gauss()
		d := [3]float64{r.gauss(), r.gauss(), r.gauss()}
		for i := range 500 {
			tt := float64(i) / rate
			var a [3]float64
			w := [3]float64{}
			if s%2 == 0 {
				_, v := quintic(tt, A, 2)
				n := math.Sqrt(ax[0]*ax[0] + ax[1]*ax[1] + ax[2]*ax[2])
				for k := range 3 {
					w[k] = ax[k] / n * v
				}
				q = q.mul(axisAngle(w, math.Sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])/rate))
			} else if tt < 1.5 {
				u := tt / 1.5
				for k := range 3 {
					a[k] = 0.1 * d[k] / 2.25 * (60*u - 180*u*u + 120*u*u*u)
				}
			}
			acc, gyro, _ := h.sample(q, a, w)
			f.Update(1/rate, acc, gyro)
		}
	}
	for i := range n {
		if !(f.cov[i][i] > 0) || math.IsInf(float64(f.cov[i][i]), 0) {
			t.Fatalf("variance %d = %g after 10 minutes", i, f.cov[i][i])
		}
		for j := range n {
			if rho := math.Abs(float64(f.cov[i][j])) / math.Sqrt(float64(f.cov[i][i])*float64(f.cov[j][j])); rho > 1.0001 || math.IsNaN(rho) {
				t.Fatalf("correlation %d,%d = %g", i, j, rho)
			}
		}
	}
	for _, v := range [][3]float32{f.V, f.P, f.GyroBias, f.AccelBias} {
		for _, x := range v {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("estimate not finite: V %v P %v bg %v ba %v", f.V, f.P, f.GyroBias, f.AccelBias)
			}
		}
	}
	if s := math.Sqrt(float64(f.V[0]*f.V[0] + f.V[1]*f.V[1] + f.V[2]*f.V[2])); s > 0.05 {
		t.Errorf("speed %.3f m/s at the last pause (want ≤ 0.05)", s)
	}
}

// guard, after each injection, floors the variances and clamps each
// correlation to ±1, symmetric: a covariance broken by rounding (a negative
// variance, a correlation over 1) is usable again after the next update.
func TestGuardKeepsTheCovarianceUsable(t *testing.T) {
	h := stillHand(3)
	f := newFilter(t)
	hold(h, &f, upright, 10)
	sd := math.Sqrt(float64(f.cov[iP][iP]) * float64(f.cov[iP+1][iP+1])) // two directions no measurement touches
	f.cov[iP][iP+1], f.cov[iP+1][iP] = float32(10*sd), float32(10*sd)
	f.cov[iBa+2][iBa+2] = -1
	for range 2 * f.cfg.TiltEvery {
		a, g, _ := h.sample(upright, [3]float64{}, [3]float64{})
		f.Update(1/rate, a, g)
	}
	for i := range n {
		if !(f.cov[i][i] > 0) {
			t.Fatalf("variance %d = %g after an update", i, f.cov[i][i])
		}
		for j := range n {
			if f.cov[i][j] != f.cov[j][i] {
				t.Fatalf("covariance not symmetric at %d,%d", i, j)
			}
			if rho := math.Abs(float64(f.cov[i][j])) / math.Sqrt(float64(f.cov[i][i])*float64(f.cov[j][j])); rho > 1.0001 {
				t.Fatalf("correlation %d,%d = %g after an update", i, j, rho)
			}
		}
	}
}

func TestUpdateDoesNotAllocate(t *testing.T) {
	h := stillHand(1)
	f := newFilter(t)
	a, g, _ := h.sample(upright, [3]float64{}, [3]float64{})
	if n := testing.AllocsPerRun(200, func() { f.Update(1/rate, a, g) }); n != 0 {
		t.Errorf("Update allocates %v times per call", n)
	}
}

func BenchmarkUpdate(b *testing.B) {
	h := stillHand(1)
	f, _ := New(DefaultConfig())
	hold(h, &f, upright, 2)
	var acc, gyro [100][3]float32
	for i := range 100 {
		acc[i], gyro[i], _ = h.sample(upright.mul(axisAngle([3]float64{1, 0, 0}, 0.3*math.Sin(float64(i)/16))), [3]float64{0.5 * math.Sin(float64(i)/8), 0, 0}, [3]float64{0.02 * math.Cos(float64(i)/16), 0, 0})
	}
	b.ResetTimer()
	for i := range b.N {
		f.Update(1/rate, acc[i%100], gyro[i%100])
	}
}

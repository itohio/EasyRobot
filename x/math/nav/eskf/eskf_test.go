package eskf

import (
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nav/calib"
	"github.com/itohio/EasyRobot/x/math/nav/estimator"
	"github.com/itohio/EasyRobot/x/math/nav/estimator/estimatortest"
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

// steadyHand is a steady hand (the capture's 20 s held still).
func steadyHand(seed uint64) *hand { return newHand(seed, 0.031, 0.016) }

// calibrated: a multi-pose calibration's fit applied to the samples (no
// offset left), and the
// fit as the filter's prior.
func calibrated(h *hand, f *Filter) {
	h.off = [3]float64{}
	fit := calib.Fit{GyroBias: [3]float32{float32(h.gb[0]), float32(h.gb[1]), float32(h.gb[2])}, RMS: 0.01, Scale: [3]float32{1, 1, 1}}
	f.Prior(&fit, 0)
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
	c := [3]float64{a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]}
	return math.Atan2(math.Sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]), a[0]*b[0]+a[1]*b[1]+a[2]*b[2]) * 180 / math.Pi
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

// exact: the force of attitude q with world acceleration a, the rate w:
// no noise, no bias.
func exact(q quat, a, w [3]float64) (acc, gyro [3]float32) {
	f := q.conj().rot([3]float64{a[0], a[1], a[2] + StandardGravity})
	for i := range 3 {
		acc[i], gyro[i] = float32(f[i]), float32(w[i])
	}
	return
}

// shake feeds s seconds of vigorous shaking: the hand swings the body
// through 3–8 Hz in changing directions, up to ~3 g and ~10 rad/s, and
// the accelerometer clips at its full scale (±2 g) as a real one does.
// update takes each sample; each, if not nil, runs after every sample.
func shake(h *hand, update func(acc, gyro [3]float32), q quat, s float64, seed uint64, each func()) quat {
	r := rng(seed)
	c := DefaultConfig()
	var amp, w [3]float64
	var fr [3]float64
	for k := range 3 {
		amp[k], w[k], fr[k] = 12+8*r.gauss(), 4*r.gauss(), 3+5*math.Abs(r.gauss())/2
	}
	for i := range int(s * rate) {
		tt := float64(i) / rate
		var a, om [3]float64
		for k := range 3 {
			a[k] = amp[k] * math.Sin(2*math.Pi*fr[k]*tt+float64(k))
			om[k] = w[k] * math.Cos(2*math.Pi*fr[(k+1)%3]*tt)
		}
		q = q.mul(axisAngle(om, math.Sqrt(om[0]*om[0]+om[1]*om[1]+om[2]*om[2])/rate))
		acc, gyro, _ := h.sample(q, a, om)
		for k := range 3 {
			acc[k] = min(max(acc[k], -c.AccelRange), c.AccelRange)
		}
		update(acc, gyro)
		if each != nil {
			each()
		}
	}
	return q
}

func TestNewRejectsConfig(t *testing.T) {
	for name, mod := range map[string]func(*Config){
		"G 0":                   func(c *Config) { c.Gravity = 0 },
		"noise NaN":             func(c *Config) { c.AccelNoise = float32(math.NaN()) },
		"TiltEvery 0":           func(c *Config) { c.TiltEvery = 0 },
		"TiltNoise +Inf":        func(c *Config) { c.TiltNoise = float32(math.Inf(1)) },
		"range negative":        func(c *Config) { c.GyroRange = -1 },
		"bias walk absent":      func(c *Config) { c.GyroBiasWalk = 0 },
		"LinAccelTime negative": func(c *Config) { c.LinAccelTime = -1 },
	} {
		c := DefaultConfig()
		mod(&c)
		if _, err := New(c); !errors.Is(err, ErrConfig) {
			t.Errorf("%s: New returned %v, want ErrConfig", name, err)
		}
	}
}

// The filter passes the motion estimator's conformance suite, set up as
// deployed: a calibration's prior (the gyro's bias known to 2 mrad/s).
// TranslationResponse runs one noise draw: its 3 cm strokes over 2 s pass
// with 15-20 % error, and across draws they are off by 0-41 % (a known
// limit, in the filter's note).
func TestConformance(t *testing.T) {
	estimatortest.Motion(t, func(t *testing.T) estimator.Motion {
		f := newFilter(t)
		fit := calib.Fit{RMS: 0.01, Scale: [3]float32{1, 1, 1}}
		f.Prior(&fit, 0)
		return &f
	})
}

// Init sets a filter in place without allocating: a used filter (a prior,
// another range, a minute of samples) initialised again equals a fresh one,
// bit for bit; an invalid configuration is refused and leaves it as it was.
func TestInit(t *testing.T) {
	fresh := newFilter(t)
	h := steadyHand(2)
	f := newFilter(t)
	calibrated(h, &f)
	if err := f.SetRange(8*StandardGravity, 4); err != nil {
		t.Fatal(err)
	}
	hold(h, &f, upright, 60)
	used := f
	bad := DefaultConfig()
	bad.TiltEvery = 0
	if err := Init(&f, bad); !errors.Is(err, ErrConfig) || f != used {
		t.Errorf("Init with an invalid configuration: %v, filter changed %v (want ErrConfig, false)", err, f != used)
	}
	if err := Init(&f, DefaultConfig()); err != nil || f != fresh {
		t.Errorf("Init of a used filter: %v, equals a fresh one %v (want nil, true)", err, f == fresh)
	}
	c := DefaultConfig()
	if n := testing.AllocsPerRun(100, func() { _ = Init(&f, c) }); n != 0 {
		t.Errorf("Init allocates %v times per call", n)
	}
}

// The first accepted sample sets roll and pitch from its force, in any
// pose: with an exact sample to 0.01°; a steady hand's tremor and noise,
// the tilt updates averaging them, is within 0.5° after 5 s.
func TestFirstSampleSetsTheTilt(t *testing.T) {
	for _, pose := range []quat{upright, {1, 0, 0, 0}, upright.mul(axisAngle([3]float64{0, 0, 1}, 0.7)), axisAngle([3]float64{1, 1, 0}, 2.5)} {
		f := newFilter(t)
		if f.Ready() {
			t.Fatal("Ready before any sample")
		}
		a, g := exact(pose, [3]float64{}, [3]float64{})
		f.Update(1/rate, a, g)
		if !f.Ready() || tiltErr(&f, pose) > 0.01 {
			t.Errorf("pose %v, exact: Ready %v, tilt %.3f° (want true, ≤ 0.01°)", pose, f.Ready(), tiltErr(&f, pose))
		}
		h := steadyHand(3)
		f = newFilter(t)
		calibrated(h, &f)
		var qt quat
		for range int(5 * rate) {
			a, g, qt = h.sample(pose, [3]float64{}, [3]float64{})
			f.Update(1/rate, a, g)
		}
		if e := tiltErr(&f, qt); e > 0.5 {
			t.Errorf("pose %v, a hand: tilt %.2f° after 5 s (want ≤ 0.5°)", pose, e)
		}
	}
}

// Nothing pulls the estimate: a body at rest, the
// samples exact, stays put (60 s: within 5 mm, float rounding), and a body
// set moving at 0.5 m/s (up: the tilt sees an acceleration along gravity
// in full) keeps moving, its velocity integrated, never zeroed.
func TestItOnlyIntegrates(t *testing.T) {
	f := newFilter(t)
	for range int(60 * rate) {
		a, g := exact(upright, [3]float64{}, [3]float64{})
		f.Update(1/rate, a, g)
	}
	if d, v := dist(f.P, [3]float32{}), dist(f.V, [3]float32{}); d > 0.005 || v > 0.001 {
		t.Errorf("exact samples at rest, 60 s: moved %.2f mm, speed %.4f m/s (want ≤ 5 mm, 0.001)", d*1000, v)
	}
	f = newFilter(t)
	a, g := exact(upright, [3]float64{}, [3]float64{})
	f.Update(1/rate, a, g)
	for i := range int(10 * rate) {
		var acc float64
		if i < int(rate) {
			acc = 0.5
		}
		a, g := exact(upright, [3]float64{0, 0, acc}, [3]float64{})
		f.Update(1/rate, a, g)
	}
	if math.Abs(float64(f.V[2])-0.5) > 0.01 || math.Abs(float64(f.P[2])-4.75) > 0.05 {
		t.Errorf("1 s at 0.5 m/s² up, then 9 s coasting: V %.3f m/s, P %.3f m (want 0.5, 4.75)", f.V[2], f.P[2])
	}
}

// The attitude stays a unit quaternion through two minutes of shaking.
func TestQuaternionStaysNormalised(t *testing.T) {
	h := steadyHand(6)
	f := newFilter(t)
	worst := 0.0
	shake(h, func(a, g [3]float32) { f.Update(1/rate, a, g) }, upright, 120, 6, func() {
		n := math.Sqrt(float64(f.Q[0]*f.Q[0] + f.Q[1]*f.Q[1] + f.Q[2]*f.Q[2] + f.Q[3]*f.Q[3]))
		worst = max(worst, math.Abs(n-1))
	})
	if worst > 1e-6 {
		t.Errorf("|q| off 1 by up to %.2e (want ≤ 1e-6)", worst)
	}
}

// The position only integrates: through 20 exact
// turns and strokes no update moves it by more than the step's own
// motion, the trapezoid of the velocity, to 0.1 mm. (When the tilt update
// corrected the velocity and position through their correlation with the
// attitude, one update moved it 9.9 m.)
func TestNoPositionStep(t *testing.T) {
	f := newFilter(t)
	var worst float64
	q := upright
	for s := range 20 {
		var ax [3]float64
		ax[s%3] = 1
		for i := 1; i <= int(3*rate); i++ {
			tt := float64(i) / rate
			var a, w [3]float64
			if s%2 == 0 {
				_, wv := quintic(tt, 0.8, 2)
				w = [3]float64{ax[0] * wv, ax[1] * wv, ax[2] * wv}
				q = q.mul(axisAngle(w, math.Sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])/rate))
			} else if tt < 1 {
				k := 0.2 * (60*tt - 180*tt*tt + 120*tt*tt*tt)
				a = [3]float64{ax[0] * k, ax[1] * k, ax[2] * k}
			}
			acc, gyro := exact(q, a, w)
			p0, v0 := f.P, f.V
			f.Update(1/rate, acc, gyro)
			var d float64
			for k := range 3 {
				e := float64(f.P[k]-p0[k]) - 0.5*float64(f.V[k]+v0[k])/rate
				d += e * e
			}
			worst = max(worst, math.Sqrt(d))
		}
	}
	if worst > 1e-4 {
		t.Errorf("the position stepped %.2e m beyond its integral (want ≤ 1e-4)", worst)
	}
}

// guard, after each injection, floors the variances and clamps each
// correlation to ±1, symmetric: a covariance broken by rounding (a negative
// variance, a correlation over 1) is usable again after the next update.
func TestGuardKeepsTheCovarianceUsable(t *testing.T) {
	h := steadyHand(3)
	f := newFilter(t)
	hold(h, &f, upright, 10)
	sd := math.Sqrt(float64(f.cov.p[iP][iP]) * float64(f.cov.p[iP+1][iP+1])) // two directions no measurement touches
	f.cov.p[iP][iP+1], f.cov.p[iP+1][iP] = float32(10*sd), float32(10*sd)
	f.cov.p[iBa+2][iBa+2] = -1
	for range 2 * f.cfg.TiltEvery {
		a, g, _ := h.sample(upright, [3]float64{}, [3]float64{})
		f.Update(1/rate, a, g)
	}
	for i := range n {
		if !(f.cov.p[i][i] > 0) {
			t.Fatalf("variance %d = %g after an update", i, f.cov.p[i][i])
		}
		for j := range n {
			if f.cov.p[i][j] != f.cov.p[j][i] {
				t.Fatalf("covariance not symmetric at %d,%d", i, j)
			}
			if rho := math.Abs(float64(f.cov.p[i][j])) / math.Sqrt(float64(f.cov.p[i][i])*float64(f.cov.p[j][j])); rho > 1.0001 {
				t.Fatalf("correlation %d,%d = %g after an update", i, j, rho)
			}
		}
	}
}

// Update takes only finite samples with a positive dt. A NaN or an infinity
// in any input, or a dt not above zero, is counted in Rejected and changes
// nothing else, before the first rest or after it (a NaN used to make the
// state NaN for good).
func TestNonFiniteSampleIsRejected(t *testing.T) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	up := [3]float32{0, 9.8, 0}
	for _, ready := range []bool{false, true} {
		for _, s := range []struct {
			name      string
			dt        float32
			acc, gyro [3]float32
		}{
			{"NaN accelerometer", 0.01, [3]float32{nan, 9.8, 0}, [3]float32{}},
			{"NaN gyro", 0.01, up, [3]float32{0, nan, 0}},
			{"infinite accelerometer", 0.01, [3]float32{0, inf, 0}, [3]float32{}},
			{"infinite gyro", 0.01, up, [3]float32{0, 0, -inf}},
			{"NaN dt", nan, up, [3]float32{}},
			{"infinite dt", inf, up, [3]float32{}},
			{"zero dt", 0, up, [3]float32{}},
			{"negative dt", -0.01, up, [3]float32{}},
		} {
			h := steadyHand(5)
			f := newFilter(t)
			if ready {
				hold(h, &f, upright, 5)
			} else {
				hold(h, &f, upright, 0.1)
			}
			before := f
			f.Update(s.dt, s.acc, s.gyro)
			if f.Rejected != before.Rejected+1 {
				t.Errorf("%s, ready %v: Rejected %d (want %d)", s.name, ready, f.Rejected, before.Rejected+1)
			}
			f.Rejected = before.Rejected
			if f != before {
				t.Errorf("%s, ready %v: the state changed", s.name, ready)
			}
		}
	}
}

// Prior's accelerometer-bias σ is the largest of the caller's
// accelBiasSigma, the fit's RMS and 0.01 m/s²: the first sample seeds the
// covariance with it. An exactly determined fit (RMS 0) with the caller's
// 0.05 gives 0.05; a fit's RMS above the caller's wins; a negative, NaN or
// infinite σ or RMS adds nothing (a saved fit can carry anything).
func TestPriorTakesTheLargestSigma(t *testing.T) {
	nan := float32(math.NaN())
	for _, c := range []struct {
		rms, sigma, want float32
	}{{0, 0, 0.01}, {0, 0.05, 0.05}, {0.08, 0.05, 0.08}, {0.02, 0, 0.02}, {0, -1, 0.01}, {0, nan, 0.01}, {0, float32(math.Inf(1)), 0.01}, {nan, 0.05, 0.05}, {float32(math.Inf(1)), 0, 0.01}, {float32(math.Inf(-1)), 0, 0.01}} {
		h := steadyHand(4)
		f := newFilter(t)
		fit := calib.Fit{GyroBias: [3]float32{float32(h.gb[0]), float32(h.gb[1]), float32(h.gb[2])}, RMS: c.rms, Scale: [3]float32{1, 1, 1}}
		h.off = [3]float64{}
		f.Prior(&fit, c.sigma)
		for !f.Ready() {
			a, g, _ := h.sample(upright, [3]float64{}, [3]float64{})
			f.Update(1/rate, a, g)
		}
		for i := range 3 {
			if got := math.Sqrt(float64(f.cov.p[iBa+i][iBa+i])); !(math.Abs(got-float64(c.want)) <= 1e-6) {
				t.Errorf("RMS %g, σ %g: the first sample's accelerometer-bias σ %.5f (want %.5f)", c.rms, c.sigma, got, c.want)
				break
			}
		}
	}
}

// After Prior the samples arrive through the fit's Apply, so a reading at
// the sensor's full scale arrives as (±range − offset)/scale: each axis and
// sign, it is counted in Clipped, at ±2 g and after SetRange to ±4 g, and
// one 0.5 % inside the full scale is not. (Against ±range itself, with a
// badge's fit only +x registered: −x arrived at −19.14 m/s², under the
// threshold.)
func TestClippedThroughTheCalibration(t *testing.T) {
	fit := calib.Fit{Model: calib.OffsetScale, Offset: [3]float32{-0.43, -0.17, -0.17}, Scale: [3]float32{1.002, 1.009, 1.031}}
	for _, ar := range []float32{2 * StandardGravity, 4 * StandardGravity} {
		for i := range 3 {
			for _, sign := range []float32{1, -1} {
				for _, k := range []float32{1, 0.995} {
					f := newFilter(t)
					f.Prior(&fit, 0)
					if err := f.SetRange(ar, f.cfg.GyroRange); err != nil {
						t.Fatal(err)
					}
					raw := [3]float32{0, 0, StandardGravity}
					raw[i] = sign * k * ar
					f.Update(1/rate, fit.Apply(raw), [3]float32{})
					want := uint32(0)
					if k == 1 {
						want = 1
					}
					if f.Clipped != want {
						t.Errorf("±%.0f g, axis %d, sign %+.0f, %.1f %% of the full scale through the fit: Clipped %d (want %d)", ar/StandardGravity, i, sign, 100*k, f.Clipped, want)
					}
				}
			}
		}
	}
}

// A fit whose offset isn't finite (NaN, ∞) can't break the clip check:
// its offset counts as 0, so a reading at the full scale is counted and
// one at rest is not (through a NaN offset nothing clipped, through an
// infinite one everything did).
func TestClippedWithANonFiniteOffset(t *testing.T) {
	for _, o := range []float32{float32(math.NaN()), float32(math.Inf(1))} {
		f := newFilter(t)
		fit := calib.Fit{Offset: [3]float32{o, 0, 0}, Scale: [3]float32{1, 1, 1}}
		f.Prior(&fit, 0)
		f.Update(1/rate, [3]float32{0, 0, StandardGravity}, [3]float32{})
		f.Update(1/rate, [3]float32{-2 * StandardGravity, 0, StandardGravity}, [3]float32{})
		if f.Clipped != 1 {
			t.Errorf("offset %v: a reading at rest and one at the full scale, Clipped %d (want 1)", o, f.Clipped)
		}
	}
}

// A reading at the sensor's full scale is clipped: its true value is beyond
// the scale. A clipped gyro sample is skipped, the attitude's variance
// grown by the full scale's angle over the step; a clipped accelerometer
// alone still turns the attitude by the gyro, the velocity's variance
// grown. Either way the sample is counted in Clipped, and calibrated and
// still the tilt a minute later is within 0.5°. (At the badge's ±2000 °/s
// a full-scale gyro sample once left a 20–35° tilt error for minutes.)
func TestClippedSample(t *testing.T) {
	c := DefaultConfig()
	gr, ar := float64(c.GyroRange), float64(c.AccelRange)
	for _, s := range []struct {
		name      string
		acc, gyro [3]float64 // replaced readings (0: the hand's)
	}{
		{"gyro at +full scale", [3]float64{}, [3]float64{gr, 0, 0}},
		{"gyro at −full scale", [3]float64{}, [3]float64{0, 0, -gr}},
		{"accelerometer at +full scale", [3]float64{0, ar, 0}, [3]float64{0.5, 0, 0}},
		{"accelerometer at −full scale", [3]float64{-ar, 0, 0}, [3]float64{0, 0, 0.5}},
	} {
		h := steadyHand(5)
		f := newFilter(t)
		calibrated(h, &f)
		hold(h, &f, upright, 10)
		a, g, _ := h.sample(upright, [3]float64{}, [3]float64{})
		for i := range 3 {
			if s.acc[i] != 0 {
				a[i] = float32(s.acc[i])
			}
			if s.gyro[i] != 0 {
				g[i] = float32(s.gyro[i])
			}
		}
		before := f
		f.Update(1/rate, a, g)
		if f.Clipped != before.Clipped+1 {
			t.Errorf("%s: Clipped %d (want %d)", s.name, f.Clipped, before.Clipped+1)
		}
		if s.gyro != [3]float64{} && s.acc == [3]float64{} {
			grew := f.cov.p[iTh][iTh] - before.cov.p[iTh][iTh]
			want := float32(gr / rate * gr / rate)
			f.Clipped, f.cov = before.Clipped, before.cov
			if f != before {
				t.Errorf("%s: the estimate changed", s.name)
			}
			if math.Abs(float64(grew-want)) > 1e-3*float64(want) {
				t.Errorf("%s: the attitude's variance grew by %.4g (want %.4g)", s.name, grew, want)
			}
		} else {
			if f.Q == before.Q {
				t.Errorf("%s: the gyro's rotation was lost", s.name)
			}
			if f.cov.p[iV][iV] <= before.cov.p[iV][iV] {
				t.Errorf("%s: the velocity's variance didn't grow", s.name)
			}
		}
		var qt quat
		for range int(60 * rate) {
			a, g, qt = h.sample(upright, [3]float64{}, [3]float64{})
			f.Update(1/rate, a, g)
		}
		if e := tiltErr(&f, qt); e > 0.5 {
			t.Errorf("%s: tilt %.2f° off a minute later (want ≤ 0.5°)", s.name, e)
		}
	}
}

// minEigCorr: the smallest eigenvalue of P's correlation matrix (Jacobi
// rotations in float64).
func minEigCorr(f *Filter) float64 {
	var a [n][n]float64
	for i := range n {
		for j := range n {
			a[i][j] = float64(f.cov.p[i][j]) / math.Sqrt(float64(f.cov.p[i][i])*float64(f.cov.p[j][j]))
		}
	}
	for range 100 {
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				off += a[i][j] * a[i][j]
			}
		}
		if off < 1e-20 {
			break
		}
		for p := range n {
			for q := p + 1; q < n; q++ {
				if math.Abs(a[p][q]) < 1e-300 {
					continue
				}
				th := (a[q][q] - a[p][p]) / (2 * a[p][q])
				t := math.Copysign(1, th) / (math.Abs(th) + math.Sqrt(th*th+1))
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := range n {
					akp, akq := a[k][p], a[k][q]
					a[k][p], a[k][q] = c*akp-s*akq, s*akp+c*akq
				}
				for k := range n {
					apk, aqk := a[p][k], a[q][k]
					a[p][k], a[q][k] = c*apk-s*aqk, s*apk+c*aqk
				}
			}
		}
	}
	m := math.Inf(1)
	for i := range n {
		m = min(m, a[i][i])
	}
	return m
}

func TestMinEigCorrIsRight(t *testing.T) {
	var f Filter
	for i := range n {
		f.cov.p[i][i] = float32(i + 1)
	}
	f.cov.p[0][1], f.cov.p[1][0] = 1.2, 1.2                           // correlation 1.2/√2 ≈ 0.849: eigenvalues 1 ± 0.849
	if m := minEigCorr(&f); math.Abs(m-(1-1.2/math.Sqrt(2))) > 1e-6 { // float32(1.2) is not 1.2
		t.Fatalf("smallest eigenvalue %g, want %g", m, 1-1.2/math.Sqrt(2))
	}
}

// The covariance stays positive semidefinite: over ten minutes of a hand
// moving (turns, strokes, rests), calibrated and not, five seeds, the
// smallest eigenvalue of its correlation matrix, checked every second,
// stays above −0.001. (Propagated to first order, P += dt(AP + PAᵀ),
// it dropped the term dt²·A·P·Aᵀ and reached −0.2, and −2.6 in some
// uncalibrated runs.)
func TestCovarianceStaysPositiveSemidefinite(t *testing.T) {
	for _, cal := range []bool{true, false} {
		for seed := uint64(11); seed <= 15; seed++ {
			h := steadyHand(seed)
			f := newFilter(t)
			if cal {
				calibrated(h, &f)
			}
			hold(h, &f, upright, 5)
			r := rng(seed)
			q := upright
			worst := math.Inf(1)
			for s := range 120 {
				ax := [3]float64{r.gauss(), r.gauss(), r.gauss()}
				A := 0.5 * r.gauss()
				d := [3]float64{r.gauss(), r.gauss(), r.gauss()}
				for i := range 500 {
					tt := float64(i) / rate
					var a [3]float64
					w := [3]float64{}
					if s%2 == 0 {
						_, v := quintic(tt, A, 2)
						nn := math.Sqrt(ax[0]*ax[0] + ax[1]*ax[1] + ax[2]*ax[2])
						for k := range 3 {
							w[k] = ax[k] / nn * v
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
					if i%100 == 99 {
						worst = min(worst, minEigCorr(&f))
					}
				}
			}
			if worst < -1e-3 {
				t.Errorf("calibrated %v, seed %d: smallest eigenvalue of the correlation matrix %.4f (want ≥ −0.001)", cal, seed, worst)
			}
		}
	}
}

// minPivotCorr: the smallest Cholesky pivot of P's correlation matrix (in
// float64); at or below zero, P is not positive definite.
func minPivotCorr(f *Filter) float64 {
	var a [n][n]float64
	for i := range n {
		for j := range n {
			a[i][j] = float64(f.cov.p[i][j]) / math.Sqrt(float64(f.cov.p[i][i])*float64(f.cov.p[j][j]))
		}
	}
	m := math.Inf(1)
	for k := range n {
		d := a[k][k]
		for j := range k {
			d -= a[k][j] * a[k][j]
		}
		if !(d > 0) {
			return d
		}
		m = min(m, d)
		s := math.Sqrt(d)
		for i := k + 1; i < n; i++ {
			v := a[i][k]
			for j := range k {
				v -= a[i][j] * a[k][j]
			}
			a[i][k] = v / s
		}
		a[k][k] = s
	}
	return m
}

// The covariance stays positive definite through vigorous shaking and the
// rest after it: 10 s of shaking (the accelerometer clipping), then 20 s
// held, calibrated and not, five shakes each; checked every sample.
// (Without a rest the yaw, velocity and position errors grew nearly
// collinear, correlations past 0.9998, and float32 rounded that block
// indefinite in three of the ten, first non-positive pivots down to −410.)
func TestCovarianceStaysPositiveDefiniteInShaking(t *testing.T) {
	for _, cal := range []bool{true, false} {
		for seed := uint64(1); seed <= 5; seed++ {
			h := steadyHand(seed)
			f := newFilter(t)
			if cal {
				calibrated(h, &f)
			}
			hold(h, &f, upright, 5)
			bad, worst := 0, math.Inf(1)
			check := func() {
				p := minPivotCorr(&f)
				if !(p > 0) {
					bad++
				}
				worst = min(worst, p)
			}
			shake(h, func(a, g [3]float32) { f.Update(1/rate, a, g) }, upright, 10, seed, check)
			for range int(20 * rate) {
				a, g, _ := h.sample(upright, [3]float64{}, [3]float64{})
				f.Update(1/rate, a, g)
				check()
			}
			if bad > 0 {
				t.Errorf("calibrated %v, shake %d: not positive definite in %d samples, first non-positive pivot down to %.4f (want > 0)", cal, seed, bad, worst)
			}
		}
	}
}

// SetRange changes the full scales in place (a range switched, or a
// calibration's scale put in use): after SetRange to ±4 g a 3 g reading is
// integrated, not clipped; an invalid range (negative, NaN, ∞) is refused
// with ErrConfig and changes nothing. No allocation.
func TestSetRangeTakesEffect(t *testing.T) {
	h := steadyHand(5)
	f := newFilter(t)
	calibrated(h, &f)
	hold(h, &f, upright, 10)
	before := f
	for _, bad := range [][2]float32{{-1, 30}, {40, float32(math.NaN())}, {float32(math.Inf(1)), 30}} {
		if err := f.SetRange(bad[0], bad[1]); err != ErrConfig {
			t.Errorf("SetRange(%v, %v): %v (want ErrConfig)", bad[0], bad[1], err)
		}
		if f != before {
			t.Fatalf("SetRange(%v, %v) changed the filter", bad[0], bad[1])
		}
	}
	a, g, _ := h.sample(upright, [3]float64{}, [3]float64{})
	a[1] = 3 * StandardGravity
	f.Update(1/rate, a, g)
	if f.Clipped != 1 {
		t.Fatalf("a 3 g reading at ±2 g: Clipped %d (want 1)", f.Clipped)
	}
	if err := f.SetRange(4*StandardGravity, 2000*math.Pi/180); err != nil {
		t.Fatal(err)
	}
	f.Update(1/rate, a, g)
	if f.Clipped != 1 {
		t.Errorf("a 3 g reading at ±4 g: Clipped %d (want 1: integrated)", f.Clipped)
	}
	if n := testing.AllocsPerRun(100, func() { _ = f.SetRange(4*StandardGravity, 2000*math.Pi/180) }); n != 0 {
		t.Errorf("SetRange allocates %v per call", n)
	}
}

// Update allocates nothing (a microcontroller's heap
// is small and its collector slow).
func TestUpdateDoesNotAllocate(t *testing.T) {
	h := steadyHand(1)
	f := newFilter(t)
	a, g, _ := h.sample(upright, [3]float64{}, [3]float64{})
	if n := testing.AllocsPerRun(200, func() { f.Update(1/rate, a, g) }); n != 0 {
		t.Errorf("Update allocates %v times per call", n)
	}
}

// A turn in place doesn't move the body: from upright, 45°, 90° and 180°
// about the screen's normal and 90° about the right and up axes in 2 s,
// exact samples (the force of the attitude at the sample, the gyro the
// rate at the sample), after 5 s held: the position moves at most 1 mm
// over the turn (measured 0.01–0.23 mm). With the force turned by the
// attitude at the step's start and the references turned by an Euler step
// it moved 2–17 cm. (After the turn the integration's own rounding, taught
// to the accelerometer bias by the tilt, drifts it: up to 2 cm in 3 s.)
func TestTurnInPlaceStaysPut(t *testing.T) {
	for _, c := range []struct {
		ax  [3]float64
		deg float64
	}{{[3]float64{0, 0, 1}, 45}, {[3]float64{0, 0, 1}, 90}, {[3]float64{0, 0, 1}, 180}, {[3]float64{1, 0, 0}, 90}, {[3]float64{0, 1, 0}, 90}} {
		f := newFilter(t)
		for range int(5 * rate) {
			a, g := exact(upright, [3]float64{}, [3]float64{})
			f.Update(1/rate, a, g)
		}
		p0 := f.P
		for i := 1; i <= int(2*rate); i++ {
			ang, w := quintic(float64(i)/rate, c.deg*math.Pi/180, 2)
			a, g := exact(upright.mul(axisAngle(c.ax, ang)), [3]float64{}, [3]float64{c.ax[0] * w, c.ax[1] * w, c.ax[2] * w})
			f.Update(1/rate, a, g)
		}
		if d := dist(f.P, p0); d > 0.001 {
			t.Errorf("%.0f° about %v: moved %.2f mm over the turn (want ≤ 1 mm)", c.deg, c.ax, d*1000)
		}
	}
}

// A slow turn in a steady hand, calibrated, is a turn: the gyro bias
// doesn't absorb it (≤ 4 mrad/s; measured ≤ 0.24) and the tilt follows
// (≤ 0.25°; measured ≤ 0.21°: the hand's tremor counts as the body's
// acceleration and slows the tilt), through the turn and 10 s held after it.
// (A rest test against a moving bias learned a slow turn as bias.)
func TestSlowTurnIsATurn(t *testing.T) {
	for _, ax := range [][3]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}} {
		for _, c := range []struct{ A, T float64 }{{20, 2}, {20, 5}, {45, 5}} {
			h := steadyHand(7)
			f := newFilter(t)
			calibrated(h, &f)
			hold(h, &f, upright, 10)
			var tilt, bias float64
			for i := range int((c.T + 10) * rate) {
				ang, w := quintic(float64(i)/rate, c.A*math.Pi/180, c.T)
				a, g, qt := h.sample(upright.mul(axisAngle(ax, ang)), [3]float64{}, [3]float64{ax[0] * w, ax[1] * w, ax[2] * w})
				f.Update(1/rate, a, g)
				tilt = math.Max(tilt, tiltErr(&f, qt))
				for k := range 3 {
					bias = math.Max(bias, math.Abs(float64(f.GyroBias[k])-h.gb[k]))
				}
			}
			if tilt > 0.25 || bias > 0.004 {
				t.Errorf("%v° / %v s about %v: tilt %.2f°, gyro bias %.4f rad/s off (want ≤ 0.25°, 0.004)", c.A, c.T, ax, tilt, bias)
			}
		}
	}
}

// After 10 s of shaking with the accelerometer clipping at ±2 g, put flat
// on a desk, a calibrated filter's tilt is within 0.5° 20 s later
// (measured 0.01–0.10°): a clipped accelerometer sample keeps the gyro's
// rotation, and the attitude's variance knows what was lost. (Skipping
// such samples whole left the attitude 80° off at σ 0.2°, and the tilt
// stuck 8.5° off on the desk.)
func TestTiltRecoversAfterShaking(t *testing.T) {
	for seed := uint64(1); seed <= 3; seed++ {
		h := steadyHand(seed)
		f := newFilter(t)
		calibrated(h, &f)
		hold(h, &f, upright, 5)
		shake(h, func(a, g [3]float32) { f.Update(1/rate, a, g) }, upright, 10, seed, nil)
		desk := newHand(seed+100, 0, 0)
		desk.off, desk.gb = h.off, h.gb
		var qt quat
		for range int(20 * rate) {
			var a, g [3]float32
			a, g, qt = desk.sample(upright, [3]float64{}, [3]float64{})
			f.Update(1/rate, a, g)
		}
		if e := tiltErr(&f, qt); e > 0.5 {
			t.Errorf("shake %d: tilt %.2f° off 20 s after the put-down (want ≤ 0.5°)", seed, e)
		}
	}
}

// The filter's σ is not too small: over 50 seeds (calibrated, a steady
// hand, 10 s held; and after a 30° turn) the mean normalised error squared
// of the tilt (2 degrees of freedom), the velocity (3) and the gyro bias
// (3) is under its χ² 95 % band's top. (It is far under: nothing observes
// the velocity, whose σ keeps the start's 0.1 m/s, and the tilt's noise
// counts a steady hand's tremor as the body's acceleration; measured tilt
// 0.01, velocity 0.04 and 0.02, gyro bias under 0.01.)
func TestSigmaIsNotTooSmall(t *testing.T) {
	for _, turn := range []bool{false, true} {
		var st, sv, sb float64
		const seeds = 50
		for seed := range seeds {
			h := steadyHand(uint64(100 + seed))
			f := newFilter(t)
			calibrated(h, &f)
			hold(h, &f, upright, 10)
			q := upright
			if turn {
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
				sb += e * e / float64(f.cov.p[iBg+k][iBg+k])
			}
		}
		for _, m := range []struct {
			name   string
			v, top float64
		}{{"tilt", st / seeds, 2.55}, {"velocity", sv / seeds, 3.68}, {"gyro bias", sb / seeds, 3.68}} {
			if m.v > m.top {
				t.Errorf("turn %v: %s mean normalised error squared %.2f over %d seeds (want ≤ %.2f)", turn, m.name, m.v, seeds, m.top)
			}
		}
	}
}

// Reset forgets the motion and returns the biases to the prior, so it
// recovers from what the range check can't see: a 34 rad/s glitch is
// inside ±2000 °/s, is integrated, and corrupts the attitude. After Reset
// the next sample makes the filter Ready again, the gyro bias the prior's,
// and 5 s held later its tilt is within 0.5°.
func TestResetStartsAgain(t *testing.T) {
	h := steadyHand(5)
	f := newFilter(t)
	calibrated(h, &f)
	hold(h, &f, upright, 10)
	a, _, _ := h.sample(upright, [3]float64{}, [3]float64{})
	f.Update(1/rate, a, [3]float32{34, 34, 34})
	hold(h, &f, upright, 10)
	f.Reset()
	if f.Ready() || f.P != [3]float32{} || f.V != [3]float32{} {
		t.Fatalf("after Reset: Ready %v, P %v, V %v (want false, 0, 0)", f.Ready(), f.P, f.V)
	}
	a, g, _ := h.sample(upright, [3]float64{}, [3]float64{})
	f.Update(1/rate, a, g)
	if !f.Ready() {
		t.Fatalf("not Ready after the first sample after Reset")
	}
	for i := range 3 {
		if math.Abs(float64(f.GyroBias[i])-h.gb[i]) > 1e-6 {
			t.Errorf("gyro bias %v after Reset (the prior %v)", f.GyroBias, h.gb)
			break
		}
	}
	var qt quat
	for range int(5 * rate) {
		a, g, qt = h.sample(upright, [3]float64{}, [3]float64{})
		f.Update(1/rate, a, g)
	}
	if e := tiltErr(&f, qt); e > 0.5 {
		t.Errorf("tilt %.2f° off 5 s after Reset (want ≤ 0.5°)", e)
	}
}

// The tilt update corrects the attitude and the biases, never the velocity
// or the position, even when the covariance correlates them (after exact
// turns and strokes): a tilt measurement 5° off leaves V, P and their
// variances bit for bit and moves the attitude.
func TestTiltUpdateLeavesVelocityAndPosition(t *testing.T) {
	f := newFilter(t)
	q := upright
	for s := range 4 {
		var ax [3]float64
		ax[s%3] = 1
		for i := 1; i <= int(2*rate); i++ {
			tt := float64(i) / rate
			_, wv := quintic(tt, 0.5, 2)
			w := [3]float64{ax[0] * wv, ax[1] * wv, ax[2] * wv}
			q = q.mul(axisAngle(w, math.Sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])/rate))
			k := 0.3 * math.Sin(math.Pi*tt)
			a, g := exact(q, [3]float64{k, 0, k}, w)
			f.Update(1/rate, a, g)
		}
	}
	if r := math.Abs(float64(f.cov.p[iTh][iV+1])) / math.Sqrt(float64(f.cov.p[iTh][iTh]*f.cov.p[iV+1][iV+1])); r < 0.01 {
		t.Fatalf("attitude–velocity correlation %.4f: the scenario doesn't correlate them", r)
	}
	a, _ := exact(q.mul(axisAngle([3]float64{1, 0, 0}, 5*math.Pi/180)), [3]float64{}, [3]float64{})
	v, p, q0 := f.V, f.P, f.Q
	var vv, pv [3]float32
	for i := range 3 {
		vv[i], pv[i] = f.cov.p[iV+i][iV+i], f.cov.p[iP+i][iP+i]
	}
	if !f.tiltUpdate(a) {
		t.Fatal("no tilt measured")
	}
	f.inject()
	for i := range 3 {
		if f.V[i] != v[i] || f.P[i] != p[i] || f.cov.p[iV+i][iV+i] != vv[i] || f.cov.p[iP+i][iP+i] != pv[i] {
			t.Fatalf("the tilt update moved V %v → %v or P %v → %v, or their variances", v, f.V, p, f.P)
		}
	}
	if f.Q == q0 {
		t.Errorf("a tilt 5° off left the attitude unchanged")
	}
}

// The accelerometer's bias is the calibration's: gravity's direction can't
// tell it from a horizontal acceleration, so the tilt doesn't touch it.
// Through 5 s of sustained sideways pushes (±2 m/s², 1 s each) while the
// body turns, AccelBias stays the prior's bit for bit, and 2–10 s into the
// rest after them the tilt is within 0.5° of the force. (With the bias in
// the tilt's measurement a hand's strokes moved it by up to 1 m/s², and the
// attitude settled 4–7° off the force at rest.)
func TestAccelBiasIsTheCalibrations(t *testing.T) {
	f := newFilter(t)
	fit := calib.Fit{RMS: 0.01, Scale: [3]float32{1, 1, 1}}
	f.Prior(&fit, 0.05)
	q := upright
	for range int(5 * rate) {
		a, g := exact(q, [3]float64{}, [3]float64{})
		f.Update(1/rate, a, g)
	}
	ba := f.AccelBias
	for i := range int(5 * rate) {
		tt := float64(i) / rate
		push := 2.0
		if int(tt)%2 == 1 {
			push = -2
		}
		w := [3]float64{0.3 * math.Sin(2*tt), 0.4 * math.Cos(1.3*tt), 0.2}
		q = q.mul(axisAngle(w, math.Sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])/rate))
		a, g := exact(q, [3]float64{push, 0.5 * push, 0}, w)
		f.Update(1/rate, a, g)
	}
	if f.AccelBias != ba {
		t.Errorf("the accelerometer bias moved from %v to %v through the pushes", ba, f.AccelBias)
	}
	worst := 0.0
	for i := range int(10 * rate) {
		a, g := exact(q, [3]float64{}, [3]float64{})
		f.Update(1/rate, a, g)
		if i >= int(2*rate) {
			worst = max(worst, tiltErr(&f, q))
		}
	}
	if worst > 0.5 {
		t.Errorf("at rest after the pushes the tilt is up to %.2f° off (want ≤ 0.5°)", worst)
	}
}

// The filter is deterministic: two fed the same two minutes (held, shaken
// with the accelerometer clipping, held) are identical after every sample.
func TestDeterministicReplay(t *testing.T) {
	h1, h2 := steadyHand(8), steadyHand(8)
	f1, f2 := newFilter(t), newFilter(t)
	calibrated(h1, &f1)
	calibrated(h2, &f2)
	var a1, a2 [][3]float32
	var g1, g2 [][3]float32
	rec := func(h *hand, a, g *[][3]float32) func(x, y [3]float32) {
		return func(x, y [3]float32) { *a, *g = append(*a, x), append(*g, y) }
	}
	for range int(30 * rate) {
		x, y, _ := h1.sample(upright, [3]float64{}, [3]float64{})
		a1, g1 = append(a1, x), append(g1, y)
		x, y, _ = h2.sample(upright, [3]float64{}, [3]float64{})
		a2, g2 = append(a2, x), append(g2, y)
	}
	shake(h1, rec(h1, &a1, &g1), upright, 60, 3, nil)
	shake(h2, rec(h2, &a2, &g2), upright, 60, 3, nil)
	for range int(30 * rate) {
		x, y, _ := h1.sample(upright, [3]float64{}, [3]float64{})
		a1, g1 = append(a1, x), append(g1, y)
		x, y, _ = h2.sample(upright, [3]float64{}, [3]float64{})
		a2, g2 = append(a2, x), append(g2, y)
	}
	for i := range a1 {
		if a1[i] != a2[i] || g1[i] != g2[i] {
			t.Fatalf("the two streams differ at sample %d", i)
		}
		f1.Update(1/rate, a1[i], g1[i])
		f2.Update(1/rate, a2[i], g2[i])
		if f1 != f2 {
			t.Fatalf("the filters differ after sample %d", i)
		}
	}
	if f1.Clipped == 0 {
		t.Errorf("no clipped sample in the replay")
	}
}

// Ten minutes of a hand moving (turns, strokes, pauses) in float32 leave the
// covariance usable: finite, a positive diagonal and correlations within ±1
// (what guard holds, pair by pair), and the estimate finite.
// Definiteness is TestCovarianceStaysPositiveSemidefinite's and
// TestCovarianceStaysPositiveDefiniteInShaking's.
func TestTenMinutesStayAFilter(t *testing.T) {
	h := steadyHand(11)
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
		if !(f.cov.p[i][i] > 0) || math.IsInf(float64(f.cov.p[i][i]), 0) {
			t.Fatalf("variance %d = %g after 10 minutes", i, f.cov.p[i][i])
		}
		for j := range n {
			if rho := math.Abs(float64(f.cov.p[i][j])) / math.Sqrt(float64(f.cov.p[i][i])*float64(f.cov.p[j][j])); rho > 1.0001 || math.IsNaN(rho) {
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
}

// The filter's σ is honest against its own priors: with the truth drawn from
// them (the velocity at the start 0.1 m/s, the calibration's residual 0.05
// m/s² on the accelerometer and 2 mrad/s on the gyro, per axis) on a still,
// level sensor with the Config's noise, the RMS over 40 draws of each error
// over its σ is within [1/3, 3] for the tilt about world x and y and the
// velocity along x, y and z, 10 s and 40 s after the start. (A residual bias
// tilts the estimate by b/g; it doesn't drift the velocity.)
func TestSigmaHonestAgainstThePriors(t *testing.T) {
	const rate, draws = 100.0, 40
	marks := [2]float64{10, 40}
	var z2 [2][5]float64
	c := DefaultConfig()
	sa, sg := float64(c.AccelNoise)*math.Sqrt(rate), float64(c.GyroNoise)*math.Sqrt(rate)
	for seed := range draws {
		r := rng(seed + 1)
		f := newFilter(t)
		fit := calib.Fit{RMS: 0.01, Scale: [3]float32{1, 1, 1}}
		f.Prior(&fit, 0.05)
		var v0, ba, bg [3]float64
		for i := range 3 {
			v0[i], ba[i], bg[i] = 0.1*r.gauss(), 0.05*r.gauss(), 0.002*r.gauss()
		}
		m := 0
		for k := 0; m < len(marks); k++ {
			var a, w [3]float32
			for i := range 3 {
				a[i] = float32(ba[i] + sa*r.gauss())
				w[i] = float32(bg[i] + sg*r.gauss())
			}
			a[2] += c.Gravity
			if err := f.Update(1/rate, a, w); err != nil {
				t.Fatal(err)
			}
			if float64(k)/rate < marks[m] {
				continue
			}
			att, vel, _ := f.Sigma()
			q, v := f.Q, f.Velocity()
			// the tilt error about world x and y: the estimate's up (world ẑ
			// in the sensor frame) against the truth's, level
			ux := 2 * (float64(q[1])*float64(q[3]) + float64(q[0])*float64(q[2]))
			uy := 2 * (float64(q[2])*float64(q[3]) - float64(q[0])*float64(q[1]))
			e := [5]float64{uy, ux, float64(v[0]) - v0[0], float64(v[1]) - v0[1], float64(v[2]) - v0[2]}
			s := [5]float64{float64(att[0]), float64(att[1]), float64(vel[0]), float64(vel[1]), float64(vel[2])}
			for i := range 5 {
				z2[m][i] += (e[i] / s[i]) * (e[i] / s[i])
			}
			m++
		}
	}
	names := [5]string{"tilt about x", "tilt about y", "velocity x", "velocity y", "velocity z"}
	for m, tm := range marks {
		for i := range 5 {
			rz := math.Sqrt(z2[m][i] / draws)
			if rz < 1.0/3 || rz > 3 {
				t.Errorf("%.0f s: %s, RMS error/σ %.3f (want within [1/3, 3])", tm, names[i], rz)
			}
		}
	}
}

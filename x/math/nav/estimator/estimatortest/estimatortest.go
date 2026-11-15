// Package estimatortest is the conformance suite every estimator brick
// passes, run from the brick's own tests: Attitude for an
// estimator.Attitude, Motion for an estimator.Motion. The samples are
// exact (the force of the attitude, the rate of the turn): what a brick may
// get wrong there is its own.
package estimatortest

import (
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nav/estimator"
)

const (
	rate = 100.0
	g    = 9.80665
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

// angle between the estimate and the truth, degrees (the full rotation:
// tilt and heading).
func angle(e [4]float32, q quat) float64 {
	d := quat{float64(e[0]), float64(e[1]), float64(e[2]), float64(e[3])}.conj().mul(q)
	v := math.Sqrt(d[1]*d[1] + d[2]*d[2] + d[3]*d[3])
	return 2 * math.Atan2(v, math.Abs(d[0])) * 180 / math.Pi
}

// sample: the force of attitude q at rest and the rate w.
func sample(q quat, w [3]float64) (acc, gyro [3]float32) {
	f := q.conj().rot([3]float64{0, 0, g})
	for i := range 3 {
		acc[i], gyro[i] = float32(f[i]), float32(w[i])
	}
	return
}

// upright: the sensor's y up (a badge held facing the user).
var upright = axisAngle([3]float64{1, 0, 0}, math.Pi/2)

// Attitude runs the attitude estimator's conformance tests on fresh
// estimators from ctor.
func Attitude(t *testing.T, ctor func(t *testing.T) estimator.Attitude) {
	t.Run("ReadyOnTheFirstSample", func(t *testing.T) { readyOnFirst(t, ctor) })
	t.Run("RefusesBadSamples", func(t *testing.T) { refuses(t, ctor, nil) })
	t.Run("UnitQuaternion", func(t *testing.T) { unitQuaternion(t, ctor) })
	t.Run("StillStaysStill", func(t *testing.T) { stillStays(t, ctor, nil) })
	t.Run("TracksARotation", func(t *testing.T) { tracksRotation(t, ctor) })
	t.Run("AttitudeAtRestAfterMotion", func(t *testing.T) { restAfterMotion(t, ctor) })
}

// Motion runs the motion estimator's conformance tests: the attitude's; a
// still body keeping its velocity (1 mm/s) and position (1 cm) after a
// minute; and a translation showing at its size on every axis.
func Motion(t *testing.T, ctor func(t *testing.T) estimator.Motion) {
	actor := func(t *testing.T) estimator.Attitude { return ctor(t) }
	Attitude(t, actor)
	t.Run("RefusesBadSamplesKeepsMotion", func(t *testing.T) {
		refuses(t, actor, func(e estimator.Attitude) [2][3]float32 {
			m := e.(estimator.Motion)
			return [2][3]float32{m.Velocity(), m.Position()}
		})
	})
	t.Run("TranslationResponse", func(t *testing.T) { translation(t, ctor) })
	t.Run("StillKeepsVelocityAndPosition", func(t *testing.T) {
		stillStays(t, actor, func(t *testing.T, e estimator.Attitude) {
			m := e.(estimator.Motion)
			v, p := m.Velocity(), m.Position()
			if n := norm(v); n > 0.001 {
				t.Errorf("speed %.4f m/s after 60 s still (want ≤ 0.001)", n)
			}
			if n := norm(p); n > 0.01 { // float32 rounding of the force and attitude, ~g·2⁻²⁴, gives ½·a·t² of a few mm
				t.Errorf("moved %.4f m in 60 s still (want ≤ 0.01)", n)
			}
		})
	})
}

func norm(v [3]float32) float64 {
	return math.Sqrt(float64(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
}

// Not Ready before any sample; Ready after the first, its attitude level
// with the force (to 0.01°, the heading free).
func readyOnFirst(t *testing.T, ctor func(t *testing.T) estimator.Attitude) {
	for _, pose := range []quat{upright, {1, 0, 0, 0}, axisAngle([3]float64{1, 1, 0}, 2.5)} {
		e := ctor(t)
		if e.Ready() {
			t.Fatal("Ready before any sample")
		}
		a, w := sample(pose, [3]float64{})
		if err := e.Update(1/rate, a, w); err != nil {
			t.Fatalf("first sample refused: %v", err)
		}
		if !e.Ready() {
			t.Fatal("not Ready after the first sample")
		}
		q := e.Attitude()
		up := quat{float64(q[0]), float64(q[1]), float64(q[2]), float64(q[3])}.conj().rot([3]float64{0, 0, 1})
		tu := pose.conj().rot([3]float64{0, 0, 1})
		c := [3]float64{up[1]*tu[2] - up[2]*tu[1], up[2]*tu[0] - up[0]*tu[2], up[0]*tu[1] - up[1]*tu[0]}
		if d := math.Atan2(math.Sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]), up[0]*tu[0]+up[1]*tu[1]+up[2]*tu[2]) * 180 / math.Pi; d > 0.01 {
			t.Errorf("pose %v: tilt %.4f° off after the first sample (want ≤ 0.01°)", pose, d)
		}
	}
}

// A sample with a NaN or an infinity, or a dt not above zero, is refused
// (ErrSample) before and after the first sample, the estimate untouched.
func refuses(t *testing.T, ctor func(t *testing.T) estimator.Attitude, more func(estimator.Attitude) [2][3]float32) {
	nan, inf := float32(math.NaN()), float32(math.Inf(1))
	a, w := sample(upright, [3]float64{})
	bad := []struct {
		dt   float32
		a, w [3]float32
	}{
		{1 / rate, [3]float32{nan, a[1], a[2]}, w},
		{1 / rate, a, [3]float32{0, inf, 0}},
		{1 / rate, [3]float32{a[0], -inf, a[2]}, w},
		{0, a, w}, {-1 / rate, a, w}, {nan, a, w}, {inf, a, w},
	}
	for _, started := range []bool{false, true} {
		e := ctor(t)
		if started {
			for range int(rate) {
				if err := e.Update(1/rate, a, w); err != nil {
					t.Fatal(err)
				}
			}
		}
		for i, b := range bad {
			q, r := e.Attitude(), e.Ready()
			var m [2][3]float32
			if more != nil {
				m = more(e)
			}
			if err := e.Update(b.dt, b.a, b.w); !errors.Is(err, estimator.ErrSample) {
				t.Errorf("started %v, bad sample %d: %v (want ErrSample)", started, i, err)
			}
			if e.Attitude() != q || e.Ready() != r || (more != nil && more(e) != m) {
				t.Errorf("started %v, bad sample %d changed the estimate", started, i)
			}
		}
	}
}

// The attitude stays a unit quaternion (within 1e-5) through a minute of
// turning about all axes at up to 3 rad/s.
func unitQuaternion(t *testing.T, ctor func(t *testing.T) estimator.Attitude) {
	e := ctor(t)
	q := upright
	worst := 0.0
	for i := range int(60 * rate) {
		tt := float64(i) / rate
		w := [3]float64{3 * math.Sin(1.3*tt), 2 * math.Sin(0.7*tt+1), 2.5 * math.Cos(0.9*tt)}
		q = q.mul(axisAngle(w, math.Sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])/rate))
		a, gy := sample(q, w)
		if err := e.Update(1/rate, a, gy); err != nil {
			t.Fatal(err)
		}
		e4 := e.Attitude()
		n := math.Sqrt(float64(e4[0]*e4[0] + e4[1]*e4[1] + e4[2]*e4[2] + e4[3]*e4[3]))
		worst = max(worst, math.Abs(n-1))
	}
	if worst > 1e-5 {
		t.Errorf("|q| off 1 by up to %.2e (want ≤ 1e-5)", worst)
	}
}

// A body held still (exact samples, two poses) keeps its attitude: within
// 0.01° after a minute, heading included. more checks a motion estimator's
// velocity and position after it.
func stillStays(t *testing.T, ctor func(t *testing.T) estimator.Attitude, more func(*testing.T, estimator.Attitude)) {
	for _, pose := range []quat{upright, axisAngle([3]float64{1, 1, 0}, 2.5)} {
		e := ctor(t)
		a, w := sample(pose, [3]float64{})
		for range int(60 * rate) {
			if err := e.Update(1/rate, a, w); err != nil {
				t.Fatal(err)
			}
		}
		if d := angle(e.Attitude(), pose); d > 0.01 {
			t.Errorf("pose %v: attitude moved %.4f° in 60 s still (want ≤ 0.01°)", pose, d)
		}
		if more != nil {
			more(t, e)
		}
	}
}

// A turn is tracked: 90° about each sensor axis in 2 s (a minimum-jerk
// profile), then 1 s held, leaves the attitude within 0.1° of the truth,
// heading included.
func tracksRotation(t *testing.T, ctor func(t *testing.T) estimator.Attitude) {
	for k := range 3 {
		var ax [3]float64
		ax[k] = 1
		e := ctor(t)
		a, w := sample(upright, [3]float64{})
		if err := e.Update(1/rate, a, w); err != nil {
			t.Fatal(err)
		}
		var q quat
		for i := 1; i <= int(3*rate); i++ {
			tt := math.Min(float64(i)/rate/2, 1)
			ang := math.Pi / 2 * (10*tt*tt*tt - 15*tt*tt*tt*tt + 6*tt*tt*tt*tt*tt)
			wv := math.Pi / 2 / 2 * (30*tt*tt - 60*tt*tt*tt + 30*tt*tt*tt*tt)
			q = upright.mul(axisAngle(ax, ang))
			a, w := sample(q, [3]float64{ax[0] * wv, ax[1] * wv, ax[2] * wv})
			if err := e.Update(1/rate, a, w); err != nil {
				t.Fatal(err)
			}
		}
		if d := angle(e.Attitude(), q); d > 0.1 {
			t.Errorf("90° about sensor axis %d: %.3f° off the truth (want ≤ 0.1°)", k, d)
		}
	}
}

// noisy is a deterministic source of the sensor's white noise (0.016 m/s²
// and 0.0006 rad/s a sample at 100 Hz): the same seed, the same samples.
type noisy uint64

func (r *noisy) gauss() float64 {
	*r = *r*6364136223846793005 + 1442695040888963407
	u1 := (float64(*r>>11) + 1) / (1 << 53)
	*r = *r*6364136223846793005 + 1442695040888963407
	u2 := float64(*r>>11) / (1 << 53)
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// moving: the force of attitude q with the world acceleration a, the rate
// w, and the sensor's noise.
func moving(q quat, a, w [3]float64, r *noisy) (acc, gyro [3]float32) {
	f := q.conj().rot([3]float64{a[0], a[1], a[2] + g})
	for i := range 3 {
		acc[i] = float32(f[i] + 0.016*r.gauss())
		gyro[i] = float32(w[i] + 0.0006*r.gauss())
	}
	return
}

// After motion, at rest, the attitude agrees with the force: 10 s of
// turning (up to 2 rad/s about every axis) and accelerating (±3 m/s²), then
// 10 s still; in every 1 s of the last 8 the estimate's up is within 0.5° of
// the true one. (A filter whose biases or attitude absorb the motion
// settles off the force at rest.)
func restAfterMotion(t *testing.T, ctor func(t *testing.T) estimator.Attitude) {
	e := ctor(t)
	r := noisy(11)
	q := upright
	for range int(2 * rate) {
		a, w := moving(q, [3]float64{}, [3]float64{}, &r)
		if err := e.Update(1/rate, a, w); err != nil {
			t.Fatal(err)
		}
	}
	for i := range int(10 * rate) {
		tt := float64(i) / rate
		w := [3]float64{2 * math.Sin(1.7*tt), 1.5 * math.Sin(2.3*tt+1), 1.8 * math.Cos(1.1*tt)}
		lin := [3]float64{3 * math.Sin(2*math.Pi*1.3*tt), 3 * math.Cos(2*math.Pi*0.9*tt), 2 * math.Sin(2*math.Pi*1.7*tt)}
		q = q.mul(axisAngle(w, math.Sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2])/rate))
		a, gy := moving(q, lin, w, &r)
		if err := e.Update(1/rate, a, gy); err != nil {
			t.Fatal(err)
		}
	}
	worst := 0.0
	for i := range int(10 * rate) {
		a, w := moving(q, [3]float64{}, [3]float64{}, &r)
		if err := e.Update(1/rate, a, w); err != nil {
			t.Fatal(err)
		}
		if i >= int(2*rate) && i%100 == 99 {
			est := e.Attitude()
			up := quat{float64(est[0]), float64(est[1]), float64(est[2]), float64(est[3])}.conj().rot([3]float64{0, 0, 1})
			tu := q.conj().rot([3]float64{0, 0, 1})
			c := [3]float64{up[1]*tu[2] - up[2]*tu[1], up[2]*tu[0] - up[0]*tu[2], up[0]*tu[1] - up[1]*tu[0]}
			worst = max(worst, math.Atan2(math.Sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2]), up[0]*tu[0]+up[1]*tu[1]+up[2]*tu[2])*180/math.Pi)
		}
	}
	if worst > 0.5 {
		t.Errorf("at rest after motion the tilt is up to %.2f° off (want ≤ 0.5°)", worst)
	}
}

// A translation shows at its size: the position change of a stroke run
// minus that of its no-stroke twin (the same noise, so the drift cancels)
// is within 20 % of the stroke on each world axis, 3 and 10 cm over 1 and
// 2 s (a minimum-jerk profile), after 13 s still. (A filter whose tilt
// takes a slow sideways force for a lean loses half of a stroke.)
func translation(t *testing.T, ctor func(t *testing.T) estimator.Motion) {
	for axis := range 3 {
		for _, c := range []struct{ d, T float64 }{{0.03, 1}, {0.10, 1}, {0.03, 2}, {0.10, 2}} {
			run := func(stroke bool) [3]float32 {
				e := ctor(t)
				r := noisy(7)
				for range int(13 * rate) {
					a, w := moving(upright, [3]float64{}, [3]float64{}, &r)
					if err := e.Update(1/rate, a, w); err != nil {
						t.Fatal(err)
					}
				}
				p0 := e.Position()
				for i := range int((c.T + 0.5) * rate) {
					var lin [3]float64
					if tt := float64(i) / rate / c.T; stroke && tt < 1 {
						lin[axis] = c.d / (c.T * c.T) * (60*tt - 180*tt*tt + 120*tt*tt*tt)
					}
					a, w := moving(upright, lin, [3]float64{}, &r)
					if err := e.Update(1/rate, a, w); err != nil {
						t.Fatal(err)
					}
				}
				p := e.Position()
				return [3]float32{p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]}
			}
			s, z := run(true), run(false)
			var err2 float64
			for k := range 3 {
				d := float64(s[k] - z[k])
				if k == axis {
					d -= c.d
				}
				err2 += d * d
			}
			if rel := math.Sqrt(err2) / c.d; rel > 0.2 {
				t.Errorf("%.0f cm along world %c in %.0f s: off by %.0f %% (want ≤ 20 %%)", c.d*100, "xyz"[axis], c.T, rel*100)
			}
		}
	}
}

package mahony

import (
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nav/estimator"
	"github.com/itohio/EasyRobot/x/math/nav/estimator/estimatortest"
)

func newFilter(t *testing.T, c Config) *Filter {
	t.Helper()
	var f Filter
	if err := Init(&f, c); err != nil {
		t.Fatal(err)
	}
	return &f
}

// The filter passes the attitude estimator's conformance suite.
func TestConformance(t *testing.T) {
	estimatortest.Attitude(t, func(t *testing.T) estimator.Attitude { return newFilter(t, DefaultConfig()) })
}

func TestInitRejectsConfig(t *testing.T) {
	for _, c := range []Config{{Kp: -1}, {Kp: 2, Ki: -0.1}, {Kp: float32(math.NaN())}, {Kp: float32(math.Inf(1))}} {
		f := newFilter(t, DefaultConfig())
		before := *f
		if err := Init(f, c); !errors.Is(err, ErrConfig) || *f != before {
			t.Errorf("%+v: %v, filter changed %v (want ErrConfig, false)", c, err, *f != before)
		}
	}
}

// tilt of the filter's up from the true up (sensor z up, at rest), degrees.
func tilt(q [4]float32) float64 {
	w, x, y, z := float64(q[0]), float64(q[1]), float64(q[2]), float64(q[3])
	up := [3]float64{2 * (x*z - w*y), 2 * (w*x + y*z), w*w - x*x - y*y + z*z} // the world's ẑ in the sensor frame
	return math.Acos(math.Min(1, up[2])) * 180 / math.Pi
}

// A constant gyro bias tilts a proportional filter by bias/Kp (0.02 rad/s
// at Kp 2: 0.57°); the integral learns it and the tilt returns to 0 (Ki
// 0.5: under 0.05° after a minute). Before the integral was per sample, so
// its gain depended on the sample rate.
func TestIntegralTakesOutAGyroBias(t *testing.T) {
	for _, c := range []struct {
		cfg    Config
		lo, hi float64
	}{{Config{Kp: 2}, 0.5, 0.65}, {Config{Kp: 2, Ki: 0.5}, 0, 0.05}} {
		f := newFilter(t, c.cfg)
		for range 6000 {
			if err := f.Update(0.01, [3]float32{0, 0, 9.80665}, [3]float32{0.02, 0, 0}); err != nil {
				t.Fatal(err)
			}
		}
		if e := tilt(f.Attitude()); e < c.lo || e > c.hi {
			t.Errorf("%+v: tilt %.3f° after 60 s with a 0.02 rad/s gyro bias (want %.2f–%.2f°)", c.cfg, e, c.lo, c.hi)
		}
	}
}

// Init and Update allocate nothing.
func TestDoesNotAllocate(t *testing.T) {
	f := newFilter(t, Config{Kp: 2, Ki: 0.1})
	c := DefaultConfig()
	if n := testing.AllocsPerRun(100, func() { _ = Init(f, c) }); n != 0 {
		t.Errorf("Init allocates %v times per call", n)
	}
	a, g := [3]float32{0.1, 0.2, 9.8}, [3]float32{0.01, -0.02, 0.03}
	if n := testing.AllocsPerRun(1000, func() { _ = f.Update(0.01, a, g) }); n != 0 {
		t.Errorf("Update allocates %v times per call", n)
	}
}

// ref is one step of the Mahony filter as published (x-io's MahonyAHRS),
// with the integral taken per second: from q and eInt, the 6-axis error
// (mag nil) or the 9-axis one.
func ref(q [4]float32, eInt [3]float32, acc, gyro [3]float32, mag *[3]float32, kp, ki, dt float32) ([4]float32, [3]float32) {
	q1, q2, q3, q4 := q[0], q[1], q[2], q[3]
	nrm := func(v [3]float32) [3]float32 {
		n := float32(math.Sqrt(float64(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])))
		return [3]float32{v[0] / n, v[1] / n, v[2] / n}
	}
	a := nrm(acc)
	v := [3]float32{2 * (q2*q4 - q1*q3), 2 * (q1*q2 + q3*q4), q1*q1 - q2*q2 - q3*q3 + q4*q4}
	e := [3]float32{a[1]*v[2] - a[2]*v[1], a[2]*v[0] - a[0]*v[2], a[0]*v[1] - a[1]*v[0]}
	if mag != nil {
		m := nrm(*mag)
		hx := 2*m[0]*(0.5-q3*q3-q4*q4) + 2*m[1]*(q2*q3-q1*q4) + 2*m[2]*(q2*q4+q1*q3)
		hy := 2*m[0]*(q2*q3+q1*q4) + 2*m[1]*(0.5-q2*q2-q4*q4) + 2*m[2]*(q3*q4-q1*q2)
		bx := float32(math.Hypot(float64(hx), float64(hy)))
		bz := 2*m[0]*(q2*q4-q1*q3) + 2*m[1]*(q3*q4+q1*q2) + 2*m[2]*(0.5-q2*q2-q3*q3)
		w := [3]float32{
			2*bx*(0.5-q3*q3-q4*q4) + 2*bz*(q2*q4-q1*q3),
			2*bx*(q2*q3-q1*q4) + 2*bz*(q1*q2+q3*q4),
			2*bx*(q1*q3+q2*q4) + 2*bz*(0.5-q2*q2-q3*q3),
		}
		e[0] += m[1]*w[2] - m[2]*w[1]
		e[1] += m[2]*w[0] - m[0]*w[2]
		e[2] += m[0]*w[1] - m[1]*w[0]
	}
	var g [3]float32
	for i := range 3 {
		if ki > 0 {
			eInt[i] += e[i] * dt
		}
		g[i] = gyro[i] + kp*e[i] + ki*eInt[i]
	}
	// the rotation exact (the published filter's step is first order)
	var gn float64
	for i := range 3 {
		gn += float64(g[i]) * float64(g[i])
	}
	gn = math.Sqrt(gn)
	c, s := 1.0, 0.5*float64(dt)
	if gn > 0 {
		c, s = math.Cos(0.5*gn*float64(dt)), math.Sin(0.5*gn*float64(dt))/gn
	}
	p := [4]float64{float64(q1), float64(q2), float64(q3), float64(q4)}
	r := [3]float64{float64(g[0]) * s, float64(g[1]) * s, float64(g[2]) * s}
	n := [4]float64{
		c*p[0] - p[1]*r[0] - p[2]*r[1] - p[3]*r[2],
		c*p[1] + p[0]*r[0] + p[2]*r[2] - p[3]*r[1],
		c*p[2] + p[0]*r[1] - p[1]*r[2] + p[3]*r[0],
		c*p[3] + p[0]*r[2] + p[1]*r[1] - p[2]*r[0],
	}
	k := 1 / math.Sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]+n[3]*n[3])
	return [4]float32{float32(n[0] * k), float32(n[1] * k), float32(n[2] * k), float32(n[3] * k)}, eInt
}

// One step from a set state matches the published filter (the integral per
// second, the rotation exact), with and without a magnetometer, at the
// step's rate: the mean of the last sample's and this one's.
func TestMatchesReference(t *testing.T) {
	for _, c := range []struct {
		q               [4]float32
		eInt            [3]float32
		acc, prev, gyro [3]float32
		mag             *[3]float32
		kp, ki, dt      float32
	}{
		{[4]float32{0.9659258, 0.2588190, 0, 0}, [3]float32{0.01, -0.02, 0.005}, [3]float32{0.3, 0.4, 0.8660254}, [3]float32{0.5, -0.2, 0.1}, [3]float32{0.02, -0.03, 0.04}, nil, 0.5, 0.1, 0.02},
		{[4]float32{0.9238795, 0.2, -0.2, 0.2}, [3]float32{-0.02, 0.015, -0.01}, [3]float32{0.25, -0.45, 0.86}, [3]float32{-0.3, 0.4, 0.2}, [3]float32{-0.015, 0.025, -0.035}, &[3]float32{0.48, 0.12, -0.32}, 0.52, 0.12, 0.01},
	} {
		f := newFilter(t, Config{Kp: c.kp, Ki: c.ki})
		k := float32(1 / math.Sqrt(float64(c.q[0]*c.q[0]+c.q[1]*c.q[1]+c.q[2]*c.q[2]+c.q[3]*c.q[3])))
		f.q = [4]float32{c.q[0] * k, c.q[1] * k, c.q[2] * k, c.q[3] * k}
		f.eInt, f.gPrev, f.ready = c.eInt, c.prev, true
		var mid [3]float32
		for i := range 3 {
			mid[i] = 0.5 * (c.prev[i] + c.gyro[i])
		}
		wq, we := ref(f.q, f.eInt, c.acc, mid, c.mag, c.kp, c.ki, c.dt)
		if c.mag == nil {
			f.stepFloat(c.acc, c.gyro, c.dt) // the float path in either build
		} else if err := f.UpdateMag(c.dt, c.acc, c.gyro, *c.mag); err != nil {
			t.Fatal(err)
		}
		if f.gPrev != c.gyro {
			t.Fatalf("mag %v: the last rate kept is %v (want this sample's %v)", c.mag != nil, f.gPrev, c.gyro)
		}
		for i := range 4 {
			if math.Abs(float64(f.q[i]-wq[i])) > 1e-6 {
				t.Fatalf("mag %v: q %v, reference %v", c.mag != nil, f.q, wq)
			}
		}
		for i := range 3 {
			if math.Abs(float64(f.eInt[i]-we[i])) > 1e-6 {
				t.Fatalf("mag %v: integral %v, reference %v", c.mag != nil, f.eInt, we)
			}
		}
	}
}

// A rate ramping linearly (a gyro reads it at each sample) turns the
// attitude by its exact integral: the step's rate is the mean of the two
// samples', not the newer one's, which would lead by half a sample.
func TestRampedRateTurnsExactly(t *testing.T) {
	const dt, alpha, n = 0.01, 1.0, 100 // rad/s², 1 s: ½·α·t² = 0.5 rad about z
	f := newFilter(t, Config{})         // no feedback: the rate alone turns it
	acc := [3]float32{0, 0, 9.81}
	for k := range n + 1 {
		if err := f.Update(dt, acc, [3]float32{0, 0, float32(alpha * dt * float64(k))}); err != nil {
			t.Fatal(err)
		}
	}
	_, _, yaw := euler(f.Attitude())
	if want := 0.5 * alpha * (n * dt) * (n * dt); math.Abs(yaw-want) > 1e-4 {
		t.Fatalf("yaw %.5f rad after %d ramped samples (want %.5f)", yaw, n, want)
	}
}

// A fast spin turns the attitude by its exact angle: 20 rad/s about z for
// 1 s ends 20 rad round, within 1e-4 rad (the first-order step falls
// θ³/12 short each sample: 0.067 rad here).
func TestFastSpinTurnsExactly(t *testing.T) {
	const dt, rate, n = 0.01, 20.0, 100
	f := newFilter(t, Config{}) // no feedback: the rate alone turns it
	acc := [3]float32{0, 0, 9.81}
	for range n + 1 { // the first sample levels
		if err := f.Update(dt, acc, [3]float32{0, 0, rate}); err != nil {
			t.Fatal(err)
		}
	}
	q := f.Attitude()
	half := 0.5 * rate * n * dt
	dot := math.Abs(float64(q[0])*math.Cos(half) + float64(q[3])*math.Sin(half))
	if e := 2 * math.Acos(min(dot, 1)); e > 1e-4 {
		t.Fatalf("%.4f rad off a %g rad turn after %d samples (want ≤ 1e-4)", e, rate*n*dt, n)
	}
}

// rot is the attitude from roll, pitch, yaw (radians, z-y-x).
func rot(roll, pitch, yaw float64) [4]float32 {
	cr, sr := math.Cos(roll/2), math.Sin(roll/2)
	cp, sp := math.Cos(pitch/2), math.Sin(pitch/2)
	cy, sy := math.Cos(yaw/2), math.Sin(yaw/2)
	return [4]float32{float32(cr*cp*cy + sr*sp*sy), float32(sr*cp*cy - cr*sp*sy), float32(cr*sp*cy + sr*cp*sy), float32(cr*cp*sy - sr*sp*cy)}
}

// sensor is the world vector v seen from attitude q.
func sensor(q [4]float32, v [3]float64) [3]float32 {
	w, x, y, z := float64(q[0]), float64(q[1]), float64(q[2]), float64(q[3])
	return [3]float32{
		float32((1-2*(y*y+z*z))*v[0] + 2*(x*y+w*z)*v[1] + 2*(x*z-w*y)*v[2]),
		float32(2*(x*y-w*z)*v[0] + (1-2*(x*x+z*z))*v[1] + 2*(y*z+w*x)*v[2]),
		float32(2*(x*z+w*y)*v[0] + 2*(y*z-w*x)*v[1] + (1-2*(x*x+y*y))*v[2]),
	}
}

// euler is roll, pitch, yaw of q (radians).
func euler(q [4]float32) (roll, pitch, yaw float64) {
	w, x, y, z := float64(q[0]), float64(q[1]), float64(q[2]), float64(q[3])
	return math.Atan2(2*(w*x+y*z), 1-2*(x*x+y*y)), math.Asin(math.Max(-1, math.Min(1, 2*(w*y-z*x)))), math.Atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
}

func run(f *Filter, n int, acc, gyro, mag [3]float32, withMag bool) {
	for range n {
		if withMag {
			_ = f.UpdateMag(0.01, acc, gyro, mag)
		} else {
			_ = f.Update(0.01, acc, gyro)
		}
	}
}

// From a level start (the first sample flat), a still sensor tilted by roll
// 30° and pitch −20° is followed to that tilt, not its mirror; with a
// magnetometer, to its heading too (40°).
func TestConvergesToAStaticTilt(t *testing.T) {
	const deg = math.Pi / 180
	truth := rot(30*deg, -20*deg, 40*deg)
	acc := sensor(truth, [3]float64{0, 0, 1})
	mag := sensor(truth, [3]float64{0.4, 0, -0.3})
	for _, withMag := range []bool{false, true} {
		f := newFilter(t, Config{Kp: 2})
		_ = f.Update(0.01, [3]float32{0, 0, 9.81}, [3]float32{})
		run(f, 4000, acc, [3]float32{}, mag, withMag) // 40 s
		g := sensor(f.Attitude(), [3]float64{0, 0, 1})
		for i := range 3 {
			if math.Abs(float64(g[i]-acc[i])) > 0.01 {
				t.Errorf("mag %v: gravity estimate %v, measured %v", withMag, g, acc)
				break
			}
		}
		if withMag {
			r, p, y := euler(f.Attitude())
			if math.Abs(r-30*deg) > 0.02 || math.Abs(p+20*deg) > 0.02 || math.Abs(y-40*deg) > 0.02 {
				t.Errorf("9-axis roll/pitch/yaw %.3f/%.3f/%.3f rad (want 0.524/−0.349/0.698)", r, p, y)
			}
		}
	}
}

// A still sensor whose gyro has a constant bias: the integral learns it
// (Ki·∫e dt → −bias) and the attitude stays put, over 10 minutes from the
// true attitude. 6-axis only the bias across gravity is observable; with a
// magnetometer all of it. (Ki 0.5/s² is the ahrs filter's 0.005 per sample
// at 100 Hz.)
func TestCancelsAGyroBias(t *testing.T) {
	const deg = math.Pi / 180
	truth := rot(20*deg, -10*deg, 30*deg)
	acc := sensor(truth, [3]float64{0, 0, 1})
	mag := sensor(truth, [3]float64{0.4, 0, -0.3})
	bias := [3]float32{0.12, -0.16, 0.1}
	for _, withMag := range []bool{false, true} {
		f := newFilter(t, Config{Kp: 2, Ki: 0.5})
		f.q, f.ready = truth, true
		run(f, 60000, acc, bias, mag, withMag)
		g := sensor(f.Attitude(), [3]float64{0, 0, 1})
		for i := range 3 {
			if math.Abs(float64(g[i]-acc[i])) > 0.002 {
				t.Errorf("mag %v: gravity estimate %v, measured %v", withMag, g, acc)
				break
			}
		}
		want := [3]float32{-bias[0], -bias[1], -bias[2]}
		if !withMag {
			along := bias[0]*acc[0] + bias[1]*acc[1] + bias[2]*acc[2]
			for i := range want {
				want[i] += along * acc[i]
			}
		}
		for i := range 3 {
			if l := f.cfg.Ki * f.eInt[i]; math.Abs(float64(l-want[i])) > 1e-3 {
				t.Errorf("mag %v: integral feedback %v, want %v", withMag, [3]float32{f.cfg.Ki * f.eInt[0], f.cfg.Ki * f.eInt[1], f.cfg.Ki * f.eInt[2]}, want)
				break
			}
		}
	}
}

// A zero accelerometer reading (free fall) gives no NaN: the gyro alone
// turns the attitude. Only the force's direction counts, not its scale.
func TestAccelEdgeCases(t *testing.T) {
	const deg = math.Pi / 180
	f := newFilter(t, Config{Kp: 2, Ki: 0.5})
	_ = f.Update(0.01, [3]float32{0, 0, 9.81}, [3]float32{})
	run(f, 100, [3]float32{}, [3]float32{0, 0, math.Pi / 2}, [3]float32{}, false)
	if _, _, y := euler(f.Attitude()); !(math.Abs(y-90*deg) <= 2*deg) {
		t.Fatalf("zero accelerometer: yaw %.1f° (want 90° from the gyro alone)", y/deg)
	}
	a1, a2 := newFilter(t, Config{Kp: 2}), newFilter(t, Config{Kp: 2})
	tilt := sensor(rot(25*deg, 10*deg, 0), [3]float64{0, 0, 1})
	_ = a1.Update(0.01, [3]float32{0, 0, 1}, [3]float32{})
	_ = a2.Update(0.01, [3]float32{0, 0, 9.81}, [3]float32{})
	run(a1, 500, tilt, [3]float32{}, [3]float32{}, false)
	run(a2, 500, [3]float32{9.81 * tilt[0], 9.81 * tilt[1], 9.81 * tilt[2]}, [3]float32{}, [3]float32{}, false)
	for i := range 4 {
		if d := a1.q[i] - a2.q[i]; !(d <= 1e-5 && d >= -1e-5) {
			t.Fatalf("|a| = 1 vs 9.81: %v vs %v", a1.q, a2.q)
		}
	}
}

// BenchmarkUpdate is the badge's call: the 6-axis update, Kp 1.
func BenchmarkUpdate(b *testing.B) {
	f, _ := New(Config{Kp: 1})
	acc, gyro := [3]float32{0.1, 0.4, 9.7}, [3]float32{0.4, 0.01, 0.05}
	_ = f.Update(0.01, acc, gyro)
	for b.Loop() {
		_ = f.Update(0.01, acc, gyro)
	}
}

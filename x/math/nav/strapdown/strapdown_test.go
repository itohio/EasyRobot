package strapdown

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/filter/ahrs"
	"github.com/itohio/EasyRobot/x/math/mat"
)

const rate = 100.0 // Hz

// quaternions in float64 for the synthetic truth: [w, x, y, z]
type quat [4]float64

func axisAngle(ax [3]float64, a float64) quat {
	n := math.Sqrt(ax[0]*ax[0] + ax[1]*ax[1] + ax[2]*ax[2])
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

// rot rotates v by q (sensor to world when q is the attitude).
func (p quat) rot(v [3]float64) [3]float64 {
	r := p.mul(quat{0, v[0], v[1], v[2]}).mul(p.conj())
	return [3]float64{r[1], r[2], r[3]}
}

func (p quat) f32() [4]float32 {
	return [4]float32{float32(p[0]), float32(p[1]), float32(p[2]), float32(p[3])}
}

func v32(v [3]float64) [3]float32 { return [3]float32{float32(v[0]), float32(v[1]), float32(v[2])} }

// sensorForce is what an ideal accelerometer reads with attitude q while
// the body accelerates at a (world, m/s²): R⁻¹·(a + g·ẑ).
func sensorForce(q quat, a [3]float64) [3]float64 {
	return q.conj().rot([3]float64{a[0], a[1], a[2] + StandardGravity})
}

func norm(v [3]float32) float64 {
	return math.Sqrt(float64(v[0])*float64(v[0]) + float64(v[1])*float64(v[1]) + float64(v[2])*float64(v[2]))
}

func newNav(t testing.TB, c Config) Nav {
	t.Helper()
	n, err := New(c)
	if err != nil {
		t.Fatal(err)
	}
	return n
}

// settled is a Nav that has spent 1 s at rest with attitude q (ideal
// sensor), so it has learned its bias and integrates from here on.
func settled(t testing.TB, c Config, q quat) Nav {
	t.Helper()
	n := newNav(t, c)
	for range int(rate) {
		n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{})), [3]float32{})
	}
	if !n.Still || !n.calibrated {
		t.Fatalf("not settled after 1 s at rest: still %v, calibrated %v", n.Still, n.calibrated)
	}
	return n
}

// lcg is a deterministic noise source (uniform, scaled to a standard
// deviation).
type lcg uint32

func (r *lcg) next(sd float64) float64 {
	*r = *r*1664525 + 1013904223
	return (float64(*r)/(1<<32) - 0.5) * sd * math.Sqrt(12)
}

// Acceptance: a pure rotation gives no translation — the badge spun about
// several axes at up to 3 rad/s with an exact attitude stays at the origin
// with no velocity (within float32 rounding).
func TestPureRotationGivesNoTranslation(t *testing.T) {
	for _, c := range []struct {
		axis  [3]float64
		speed float64 // rad/s
	}{{[3]float64{0, 0, 1}, 1}, {[3]float64{1, 0, 0}, 2}, {[3]float64{1, 2, 3}, 3}, {[3]float64{-1, 1, 0.2}, 1.5}} {
		q := axisAngle([3]float64{1, 0.5, 0}, 0.7) // tilted start
		n := settled(t, DefaultConfig(), q)
		step := axisAngle(c.axis, c.speed/rate)
		var worst float64
		for range int(4 * rate) {
			q = q.mul(step) // body-frame rotation: the gyro reads speed about axis
			ax := c.axis
			an := math.Sqrt(ax[0]*ax[0] + ax[1]*ax[1] + ax[2]*ax[2])
			gyro := [3]float64{ax[0] / an * c.speed, ax[1] / an * c.speed, ax[2] / an * c.speed}
			n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{})), v32(gyro))
			worst = max(worst, norm(n.P), norm(n.V))
		}
		t.Logf("axis %v at %.1f rad/s: worst |P| or |V| %.2e", c.axis, c.speed, worst)
		if worst > 1e-3 {
			t.Errorf("axis %v at %.1f rad/s: a pure rotation moved the badge (worst %.2e m or m/s)", c.axis, c.speed, worst)
		}
	}
}

// minJerk is the position (0 → 1) and acceleration (per T²) of a
// minimum-jerk move at s = t/T in [0, 1].
func minJerk(s float64) (x, a float64) {
	return 10*s*s*s - 15*s*s*s*s + 6*s*s*s*s*s, 60*s - 180*s*s + 120*s*s*s
}

// moveAndBack runs the badge, held at a tilt, through: rest 2 s, +10 cm
// along world x in 1 s, rest 1 s, back in 1 s, rest 1 s. The sensor has a
// bias, a 1 % scale error on z, noise and a 0.05° attitude error. It
// returns the Nav's position at the middle rest and at the end, and where
// a plain double integration of the same samples ends (no rest, no bias).
func moveAndBack(t *testing.T) (mid, end [3]float32, still bool, plain [3]float64) {
	n := newNav(t, DefaultConfig())
	q := axisAngle([3]float64{1, 0, 0}, 0.5).mul(axisAngle([3]float64{0, 0, 1}, 0.4))
	qerr := q.mul(axisAngle([3]float64{0, 1, 0}, 0.05*math.Pi/180)) // what the filter believes
	bias := [3]float64{0.05, -0.04, 0.08}
	var rnd lcg = 7
	var pv [3]float64
	const d = 0.1 // m
	for i := range int(6 * rate) {
		tt := float64(i) / rate
		var ax float64
		switch {
		case tt >= 2 && tt < 3:
			_, a := minJerk(tt - 2)
			ax = d * a
		case tt >= 4 && tt < 5:
			_, a := minJerk(tt - 4)
			ax = -d * a
		}
		f := sensorForce(q, [3]float64{ax, 0, 0})
		f[2] *= 1.01
		for k := range 3 {
			f[k] += bias[k] + rnd.next(0.02)
		}
		gyro := [3]float64{rnd.next(0.005), rnd.next(0.005), rnd.next(0.005)}
		n.Update(1/rate, qerr.f32(), v32(f), v32(gyro))
		if i == int(3.9*rate) {
			mid = n.P
		}
		a := qerr.rot(f)
		a[2] -= StandardGravity
		for k := range 3 {
			pv[k] += a[k] / rate
			plain[k] += pv[k] / rate
		}
	}
	return mid, n.P, n.Still, plain
}

// Acceptance: a 10 cm move and back returns near the start under ZUPT —
// with a biased, noisy, slightly miscalibrated accelerometer and a small
// attitude error. The middle rest is at +10 cm (the motion is tracked, not
// suppressed); a plain double integration of the same samples ends far
// away.
func TestMoveAndBackReturnsUnderZUPT(t *testing.T) {
	mid, end, still, plain := moveAndBack(t)
	t.Logf("ZUPT: middle (%.4f, %.4f, %.4f) m, end (%.4f, %.4f, %.4f) m; plain integration ends (%.3f, %.3f, %.3f) m",
		mid[0], mid[1], mid[2], end[0], end[1], end[2], plain[0], plain[1], plain[2])
	if d := norm([3]float32{mid[0] - 0.1, mid[1], mid[2]}); d > 0.01 {
		t.Errorf("middle rest %.4f m away from (0.1, 0, 0)", d)
	}
	if d := norm(end); d > 0.01 || !still {
		t.Errorf("end %.4f m from the start (still %v); want within 1 cm and still", d, still)
	}
	if p := math.Sqrt(plain[0]*plain[0] + plain[1]*plain[1] + plain[2]*plain[2]); p < 0.5 {
		t.Errorf("plain integration ends only %.3f m away: the sensor errors are too small to need ZUPT", p)
	}
}

// With a 6-axis Mahony filter supplying the attitude (as on the badge):
// after 10 s at rest (the filter settles, the Nav learns its bias),
// tilting the badge 45° about x over 1 s and back, held at one place, ends
// within 5 cm and never strays further than that; the filter's own tilt
// error during the turn is what leaks in.
func TestTiltAndBackWithMahony(t *testing.T) {
	n := newNav(t, DefaultConfig())
	f := ahrs.NewMahony(ahrs.WithKP(2), ahrs.WithKI(0))
	var in mat.Matrix3x3
	q := quat{1, 0, 0, 0}
	var worst float64
	for i := range int(14 * rate) {
		tt := float64(i)/rate - 10
		wx := 0.0
		switch {
		case tt >= 0 && tt < 1:
			wx = math.Pi / 4
		case tt >= 2 && tt < 3:
			wx = -math.Pi / 4
		}
		q = q.mul(axisAngle([3]float64{1, 0, 0}, wx/rate))
		in[0] = v32(sensorForce(q, [3]float64{}))
		in[1] = [3]float32{float32(wx), 0, 0}
		f.Update(1/rate, &in)
		n.Update(1/rate, [4]float32(f.Output()), in[0], in[1])
		worst = max(worst, norm(n.P))
	}
	t.Logf("tilt 45° and back with Mahony: end %.4f m, worst %.4f m, still %v", norm(n.P), worst, n.Still)
	if worst > 0.05 || !n.Still {
		t.Errorf("strayed %.4f m, ended %.4f m from where it was held (still %v)", worst, norm(n.P), n.Still)
	}
}

// stroke is a minimum-jerk move of d over T: position, speed and
// acceleration s into it (smooth: speed and acceleration are zero at both
// ends, as a hand's are).
func stroke(s, T, d float64) (z, v, a float64) {
	u := s / T
	x, acc := minJerk(u)
	return d * x, d / T * (30*u*u - 60*u*u*u + 30*u*u*u*u), d / (T * T) * acc
}

// pong runs the badge, held upright as for Pong (screen to the player:
// IMU +Y up, −Z out): six 10 cm strokes of 0.5 s, up and down without a
// pause (3 s of play), a 0.5 s pause at the bottom, then 10 cm up in 1 s,
// a 0.5 s pause, back down in 1 s, and 1.5 s at rest. The attitude is the
// badge's own 6-axis Mahony filter fed the same biased, noisy
// accelerometer (0.02 m/s² rms, about an MPU6050's at 44 Hz) and gyro.
// It returns the worst error of each stroke's vertical displacement
// (measured from where the stroke began), the worst vertical speed error,
// and where the badge ends.
func pong(t *testing.T, seed lcg) (worstD, worstV float64, end [3]float32, still bool) {
	n := newNav(t, DefaultConfig())
	f := ahrs.NewMahony(ahrs.WithKP(2), ahrs.WithKI(0))
	var in mat.Matrix3x3
	q := axisAngle([3]float64{1, 0, 0}, math.Pi/2) // IMU +Y → world +Z
	bias := [3]float64{0.04, -0.06, 0.05}
	rnd := seed
	const settle, d = 10.0, 0.1 // s at rest (filter and bias settle), stroke m
	var p0, z0 float64
	seg := -1
	for i := range int((settle + 8) * rate) {
		tt := float64(i)/rate - settle
		var z, v, a float64
		s := -1 // stroke number
		switch {
		case tt >= 0 && tt < 3:
			s = int(tt / 0.5)
			z, v, a = stroke(tt-0.5*float64(s), 0.5, d)
			if s%2 == 1 { // down
				z, v, a = d-z, -v, -a
			}
		case tt >= 3.5 && tt < 4.5:
			s = 6
			z, v, a = stroke(tt-3.5, 1, d)
		case tt >= 4.5 && tt < 5:
			z = d
		case tt >= 5 && tt < 6:
			s = 7
			z, v, a = stroke(tt-5, 1, d)
			z, v, a = d-z, -v, -a
		}
		fs := sensorForce(q, [3]float64{0, 0, a})
		for k := range 3 {
			fs[k] += bias[k] + rnd.next(0.02)
		}
		in[0] = v32(fs)
		in[1] = [3]float32{float32(rnd.next(0.005)), float32(rnd.next(0.005)), float32(rnd.next(0.005))}
		if s != seg && s >= 0 { // a stroke begins: measure from here
			p0, z0, seg = float64(n.P[2]), 0, s
			if s%2 == 1 && s < 6 || s == 7 { // a down stroke starts at the top
				z0 = d
			}
		}
		f.Update(1/rate, &in)
		n.Update(1/rate, [4]float32(f.Output()), in[0], in[1])
		if s >= 0 {
			worstD = max(worstD, math.Abs((float64(n.P[2])-p0)-(z-z0)))
			worstV = max(worstV, math.Abs(float64(n.V[2])-v))
		}
	}
	return worstD, worstV, n.P, n.Still
}

// Pong: moving the badge up and down moves the paddle with the same speed
// and motion. In every 0.5 s and 1 s stroke of 10 cm, the vertical
// displacement is within 1 cm of the truth (the noise alone gives a few
// millimetres; strokes without a pause carry the last one's speed error)
// and the vertical speed within 3 cm/s (of up to 37.5 cm/s), for three
// noise draws; the badge ends at rest within 3 cm of where it started (the
// noise's random walk over 6 s of strokes, which no rest can undo).
func TestVerticalMovesTrackedForPong(t *testing.T) {
	for _, seed := range []lcg{11, 12, 13} {
		worstD, worstV, end, still := pong(t, seed)
		t.Logf("pong seed %d: worst stroke error %.4f m, worst vz error %.4f m/s, end (%.4f, %.4f, %.4f) m, still %v", seed, worstD, worstV, end[0], end[1], end[2], still)
		if worstD > 0.01 || worstV > 0.03 {
			t.Errorf("seed %d: strokes off by up to %.4f m and %.4f m/s", seed, worstD, worstV)
		}
		if math.Abs(float64(end[2])) > 0.03 || !still {
			t.Errorf("seed %d: ended %.4f m off vertically (still %v)", seed, end[2], still)
		}
	}
}

// upStroke runs the badge upright as for Pong through the badge's own
// 6-axis Mahony filter: 10 s at rest, one minimum-jerk stroke of d up over
// T, then 1.5 s at rest. The accelerometer has bias ab and noise of sd
// (m/s², rms); the gyro has bias gb and noise of sd/4 (rad/s). It returns
// how far up the Nav moved over the stroke and whether it ends at rest.
func upStroke(t *testing.T, d, T, sd float64, ab, gb [3]float64, rnd lcg) (float64, bool) {
	n := newNav(t, DefaultConfig())
	f := ahrs.NewMahony(ahrs.WithKP(2), ahrs.WithKI(0))
	var in mat.Matrix3x3
	q := axisAngle([3]float64{1, 0, 0}, math.Pi/2) // IMU +Y → world +Z
	const settle = 10.0
	var p0 float32
	for i := range int((settle + T + 1.5) * rate) {
		tt := float64(i)/rate - settle
		var a float64
		if tt >= 0 && tt < T {
			_, _, a = stroke(tt, T, d)
		}
		if i == int(settle*rate) {
			p0 = n.P[2]
		}
		fs := sensorForce(q, [3]float64{0, 0, a})
		for k := range 3 {
			fs[k] += ab[k] + rnd.next(sd)
			in[1][k] = float32(gb[k] + rnd.next(sd/4))
		}
		in[0] = v32(fs)
		f.Update(1/rate, &in)
		n.Update(1/rate, [4]float32(f.Output()), in[0], in[1])
	}
	return float64(n.P[2] - p0), n.Still
}

// Pong: a gentle or slow stroke moves the paddle as far as the hand. Every
// stroke of 3–30 cm over 0.3–2 s (ideal sensor, the Mahony attitude) ends
// within 10 % of its length, at rest. A stroke that slow stays under
// StillAccel for most or all of its length (3 cm in 2 s peaks at 0.043
// m/s²), so only its speed tells it from rest: fpv-gate found the Nav
// declaring rest mid-stroke at up to 372 mm/s, and these strokes tracked
// 0 % to −24 %.
func TestGentleAndSlowStrokesTrack(t *testing.T) {
	for _, d := range []float64{0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3} {
		for _, T := range []float64{0.3, 0.5, 0.75, 1, 1.5, 2} {
			got, still := upStroke(t, d, T, 0, [3]float64{}, [3]float64{}, 1)
			if math.Abs(got-d) > 0.1*d || !still {
				t.Errorf("%.3f m in %.2f s: moved %.4f m (%.0f %%), still at the end %v", d, T, got, 100*got/d, still)
			}
		}
	}
}

// The same for gentle strokes with a real sensor's faults: accelerometer
// bias and 0.03 m/s² noise, gyro bias 0.02 rad/s (which the filter turns
// into a fixed tilt error) and noise, five draws each. 3 and 5 cm over 1
// and 1.2 s end within 10 % of their length plus 3 mm.
func TestGentleStrokesTrackWithNoise(t *testing.T) {
	ab, gb := [3]float64{0.04, -0.06, 0.05}, [3]float64{0.02, -0.01, 0.015}
	for _, d := range []float64{0.03, 0.05} {
		for _, T := range []float64{1, 1.2} {
			for seed := lcg(21); seed < 26; seed++ {
				got, still := upStroke(t, d, T, 0.03, ab, gb, seed)
				if math.Abs(got-d) > 0.1*d+0.003 || !still {
					t.Errorf("%.2f m in %.1f s, seed %d: moved %.4f m, still at the end %v", d, T, seed, got, still)
				}
			}
		}
	}
}

// After a push the body glides on along y with no acceleration. Speed is
// judged vertically only, so the glide counts as rest StillTime later;
// the velocity is zero and the position is where the push left it (the
// leftover velocity's distance over the quiet stretch is taken back, to
// 0.1 mm).
func TestZUPTAfterPush(t *testing.T) {
	c := DefaultConfig()
	q := quat{1, 0, 0, 0}
	n := settled(t, c, q)
	for range int(0.5 * rate) { // 1 m/s² along y for 0.5 s
		n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{0, 1, 0})), [3]float32{})
	}
	// samples are of a smooth motion: the step from rest reads as a ramp
	// over the first sample, so v = a·(t − dt/2) and y = a·(t − dt/2)²/2 +
	// a·dt²/24, which the double trapezoid gives exactly
	const dt, T = 1 / rate, 0.5
	wantV, wantY := T-dt/2, (T-dt/2)*(T-dt/2)/2+dt*dt/24
	if n.Still || math.Abs(float64(n.V[1])-wantV) > 1e-4 || math.Abs(float64(n.P[1])-wantY) > 1e-4 {
		t.Fatalf("after the push: still %v, vy %.5f, y %.5f; want moving, %.5f m/s, %.5f m", n.Still, n.V[1], n.P[1], wantV, wantY)
	}
	pEnd := n.P
	steps := 0
	for !n.Still && steps < int(rate) {
		n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{})), [3]float32{})
		steps++
	}
	t.Logf("still after %d samples; y %.4f at the end of the push, %.4f at rest", steps, pEnd[1], n.P[1])
	if !n.Still || n.V != [3]float32{} || steps > int(float64(c.StillTime)*rate)+1 || norm([3]float32{n.P[0] - pEnd[0], n.P[1] - pEnd[1], n.P[2] - pEnd[2]}) > 1e-4 {
		t.Errorf("still %v after %d samples, v %v, P %v; want still within StillTime, v 0, P %v", n.Still, steps, n.V, n.P, pEnd)
	}
}

// The first rest learns the bias as the plain mean of its samples: at the
// moment it counts as calibrated (StillTime + AvgTime of rest), a constant
// bias is known to within 1 %.
func TestFirstRestLearnsTheBias(t *testing.T) {
	n := newNav(t, DefaultConfig())
	bias := [3]float32{0.1, -0.2, 0.3}
	for !n.calibrated {
		n.Update(1/rate, [4]float32{1, 0, 0, 0}, [3]float32{bias[0], bias[1], StandardGravity + bias[2]}, [3]float32{})
	}
	for i := range 3 {
		if math.Abs(float64(n.Bias[i]-bias[i])) > 0.01*math.Abs(float64(bias[i])) {
			t.Errorf("bias %v when calibrated, want %v", n.Bias, bias)
			break
		}
	}
}

// The first samples of a motion arrive before the motion is detected and
// are averaged into the bias; once it is detected, the bias goes back to a
// value from before them. A slow ramp (0.8 m/s³) stays under StillAccel for
// 0.25 s of samples; afterwards the bias of an ideal sensor is still zero.
func TestBiasNotLearnedFromTheStartOfAMotion(t *testing.T) {
	q := quat{1, 0, 0, 0}
	n := settled(t, DefaultConfig(), q)
	for i := 1; n.Still && i < int(rate); i++ {
		a := 0.8 * float64(i) / rate
		n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{a, 0, 0})), [3]float32{})
	}
	if n.Still || norm(n.Bias) > 1e-4 {
		t.Errorf("after the ramp is detected: still %v, bias %v; want moving with bias 0", n.Still, n.Bias)
	}
}

// A hand easing into a move: a 1 s minimum-jerk move of 10 cm stays under
// StillAccel for its first 70 ms, while the Nav still counts as at rest.
// That part is not lost: the move ends 10 cm away to the millimetre.
func TestSlowStartIsNotLost(t *testing.T) {
	q := quat{1, 0, 0, 0}
	n := settled(t, DefaultConfig(), q)
	for i := range int(rate) {
		_, _, a := stroke(float64(i)/rate, 1, 0.1)
		n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{a, 0, 0})), [3]float32{})
	}
	if math.Abs(float64(n.P[0])-0.1) > 0.001 {
		t.Errorf("moved %.4f m, want 0.1000", n.P[0])
	}
}

// With an exact attitude (no filter in the way) slow strokes are tracked
// to 1 %, and the slowest, 3 cm in 2 s, to 5 %: it reaches the wake-up
// speed (StillSpeed/2) only a sixth of the way in. That takes the shadow's
// velocity and distance, integrated from a bias older than the motion.
func TestSlowStrokesWithExactAttitude(t *testing.T) {
	q := quat{1, 0, 0, 0}
	for _, c := range []struct{ d, T, tol float64 }{{0.03, 2, 0.05}, {0.05, 2, 0.01}, {0.03, 1.5, 0.01}, {0.1, 2, 0.01}} {
		n := settled(t, DefaultConfig(), q)
		p0 := n.P[2]
		for i := range int((c.T + 1.5) * rate) {
			var a float64
			if tt := float64(i) / rate; tt < c.T {
				_, _, a = stroke(tt, c.T, c.d)
			}
			n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{0, 0, a})), [3]float32{})
		}
		got := float64(n.P[2] - p0)
		t.Logf("%.2f m in %.1f s: moved %.5f m", c.d, c.T, got)
		if math.Abs(got-c.d) > c.tol*c.d || !n.Still {
			t.Errorf("%.2f m in %.1f s: moved %.5f m, still %v; want within %.0f %%", c.d, c.T, got, n.Still, 100*c.tol)
		}
	}
}

// A bias learned while the attitude was wrong (the filter still turning
// towards the truth) is learned again at the next rest: the force stays
// steady even though the bias says it is accelerating. Afterwards a push
// is measured right.
func TestStaleBiasIsRelearned(t *testing.T) {
	truth := axisAngle([3]float64{1, 0, 0}, 0.3)
	n := newNav(t, DefaultConfig())
	for range int(rate) { // first rest, believing it is level
		n.Update(1/rate, quat{1, 0, 0, 0}.f32(), v32(sensorForce(truth, [3]float64{})), [3]float32{})
	}
	for range int(3 * rate) { // the filter has found the tilt
		n.Update(1/rate, truth.f32(), v32(sensorForce(truth, [3]float64{})), [3]float32{})
	}
	if !n.Still || norm(n.Bias) > 0.01 {
		t.Fatalf("after 3 s at rest with the right attitude: still %v, bias %v; want still, bias ≈ 0", n.Still, n.Bias)
	}
	p0 := n.P
	for range int(0.5 * rate) {
		n.Update(1/rate, truth.f32(), v32(sensorForce(truth, [3]float64{0, 1, 0})), [3]float32{})
	}
	if d := float64(n.P[1] - p0[1]); math.Abs(d-0.1225) > 0.002 {
		t.Errorf("push measured %.4f m, want 0.1225", d)
	}
}

// A stale bias found while moving (a turn wakes the Nav just as the
// filter finds the true tilt): the strong steady acceleration that follows
// is a rest, so once the bias is learned again the position goes back to
// where that stretch began; without that it would be a metre away.
func TestStaleBiasFoundWhileMoving(t *testing.T) {
	truth := axisAngle([3]float64{1, 0, 0}, 0.3)
	n := newNav(t, DefaultConfig())
	for range int(rate) { // first rest, believing it is level
		n.Update(1/rate, quat{1, 0, 0, 0}.f32(), v32(sensorForce(truth, [3]float64{})), [3]float32{})
	}
	for range int(0.2 * rate) { // a turn: moving
		n.Update(1/rate, truth.f32(), v32(sensorForce(truth, [3]float64{})), [3]float32{0, 0, 0.5})
	}
	pTurn := n.P
	for range int(2 * rate) { // held still
		n.Update(1/rate, truth.f32(), v32(sensorForce(truth, [3]float64{})), [3]float32{})
	}
	d := [3]float32{n.P[0] - pTurn[0], n.P[1] - pTurn[1], n.P[2] - pTurn[2]}
	if !n.Still || norm(d) > 1e-4 || norm(n.Bias) > 0.01 {
		t.Errorf("after 2 s held still: still %v, %.4f m from where the turn ended, bias %v; want still, there, bias ≈ 0", n.Still, norm(d), n.Bias)
	}
}

// A bias drifting slowly during a long rest (warming up: 0.002 m/s² per
// second for 30 s) is followed: the learned bias lags by under 0.01 m/s².
// The average grows with the rest but stops at 4·AvgTime; a plain mean of
// the whole rest would lag by 0.03.
func TestBiasFollowsASlowDriftAtRest(t *testing.T) {
	q := quat{1, 0, 0, 0}
	n := settled(t, DefaultConfig(), q)
	var b float64
	for i := range int(30 * rate) {
		b = 0.002 * float64(i) / rate
		fs := sensorForce(q, [3]float64{})
		fs[0] += b
		n.Update(1/rate, q.f32(), v32(fs), [3]float32{})
	}
	if !n.Still || math.Abs(float64(n.Bias[0])-b) > 0.01 {
		t.Errorf("after 30 s: still %v, bias x %.4f, truth %.4f", n.Still, n.Bias[0], b)
	}
}

// Before its first rest a Nav has no bias and integrates nothing: a badge
// moved straight after New stays at the origin.
func TestHoldsOriginUntilFirstRest(t *testing.T) {
	n := newNav(t, DefaultConfig())
	q := quat{1, 0, 0, 0}
	for range int(0.5 * rate) {
		n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{1, 0, 0})), [3]float32{0.5, 0, 0})
	}
	if n.P != [3]float32{} || n.V != [3]float32{} {
		t.Errorf("moved before the first rest: P %v V %v", n.P, n.V)
	}
}

// A badge moved before it is first put down learns its bias from the rest
// only, not from the motion before it. (The rest starts once the force's
// average has settled, about 0.8 s after the motion, and lasts StillTime +
// AvgTime.)
func TestFirstBiasFromTheRestOnly(t *testing.T) {
	n := newNav(t, DefaultConfig())
	q := quat{1, 0, 0, 0}
	for range int(0.5 * rate) {
		n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{1, 0, 0})), [3]float32{0.5, 0, 0})
	}
	for range int(2 * rate) {
		n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{})), [3]float32{})
	}
	if !n.calibrated || norm(n.Bias) > 1e-4 {
		t.Errorf("after 2 s of first rest: calibrated %v, bias %v; want calibrated, bias 0", n.calibrated, n.Bias)
	}
}

// A vertical glide (5 cm/s up, no acceleration, gyro quiet) looks like a
// gentle stroke's middle, so it is not stopped while its speed is over
// StillSpeed; after MaxGlide of looking still it is (this bounds the drift
// of a wrong velocity), and its distance over the glide is taken back.
func TestLongGlideCountsAsRest(t *testing.T) {
	c := DefaultConfig()
	q := quat{1, 0, 0, 0}
	n := settled(t, c, q)
	p0 := n.P
	n.V, n.Still = [3]float32{0, 0, 0.05}, false
	steps := 0
	for !n.Still && steps < int(2*float64(c.MaxGlide)*rate) {
		n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{})), [3]float32{})
		steps++
	}
	glide := float64(steps) / rate
	t.Logf("still after %.2f s; z %.5f", glide, n.P[2])
	if !n.Still || math.Abs(glide-float64(c.MaxGlide)) > 1.5/rate || math.Abs(float64(n.P[2]-p0[2])) > 1e-3 {
		t.Errorf("still %v after %.2f s, z moved %.4f m; want still after MaxGlide %.2f s, back where the glide began", n.Still, glide, n.P[2]-p0[2], c.MaxGlide)
	}
}

// Damping: a glide (no acceleration, the badge turning so it never looks
// still) loses Damping·dt of its speed per sample; without damping it
// keeps it.
func TestDamping(t *testing.T) {
	for _, damp := range []float32{0, 2} {
		c := DefaultConfig()
		c.Damping = damp
		q := quat{1, 0, 0, 0}
		n := settled(t, c, q)
		n.V, n.Still = [3]float32{1, 0, 0}, false // already gliding
		step := axisAngle([3]float64{0, 0, 1}, 0.5/rate)
		for range int(rate) {
			q = q.mul(step)
			n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{})), [3]float32{0, 0, 0.5})
		}
		want := math.Pow(1-float64(damp)/rate, rate)
		if math.Abs(float64(n.V[0])-want) > 1e-3 || n.Still {
			t.Errorf("damping %v: vx %.4f after 1 s, want %.4f", damp, n.V[0], want)
		}
	}
}

func TestBadSamplesAreIgnored(t *testing.T) {
	n := newNav(t, DefaultConfig())
	q := [4]float32{1, 0, 0, 0}
	n.Update(0.01, q, [3]float32{0.3, 0, StandardGravity}, [3]float32{1, 0, 0})
	before := n
	inf := float32(math.Inf(1))
	nan := float32(math.NaN())
	n.Update(0, q, [3]float32{1, 0, StandardGravity}, [3]float32{})
	n.Update(-0.01, q, [3]float32{1, 0, StandardGravity}, [3]float32{})
	n.Update(nan, q, [3]float32{1, 0, StandardGravity}, [3]float32{})
	n.Update(0.01, [4]float32{nan, 0, 0, 0}, [3]float32{1, 0, StandardGravity}, [3]float32{})
	n.Update(0.01, q, [3]float32{inf, 0, StandardGravity}, [3]float32{})
	n.Update(0.01, q, [3]float32{1, 0, StandardGravity}, [3]float32{0, -inf, 0})
	if n != before {
		t.Errorf("a bad sample changed the state: %+v → %+v", before, n)
	}
}

func TestResetKeepsBias(t *testing.T) {
	n := newNav(t, DefaultConfig())
	q := [4]float32{1, 0, 0, 0}
	for range int(3 * rate) { // at rest with a 0.1 m/s² x bias
		n.Update(1/rate, q, [3]float32{0.1, 0, StandardGravity}, [3]float32{})
	}
	n.P = [3]float32{1, 2, 3}
	n.Reset()
	if n.P != [3]float32{} || n.V != [3]float32{} || n.Still || math.Abs(float64(n.Bias[0])-0.1) > 0.005 {
		t.Errorf("after Reset: P %v V %v still %v bias %v; want origin, at rest, bias 0.1 kept", n.P, n.V, n.Still, n.Bias)
	}
}

func TestNewRejectsConfig(t *testing.T) {
	bad := []func(*Config){
		func(c *Config) { c.G = 0 },
		func(c *Config) { c.StillGyro = 0 },
		func(c *Config) { c.StillAccel = -1 },
		func(c *Config) { c.StillTime = 0 },
		func(c *Config) { c.AvgTime = 0 },
		func(c *Config) { c.Damping = -0.1 },
		func(c *Config) { c.G = float32(math.NaN()) },
		func(c *Config) { c.Damping = float32(math.Inf(1)) },
		func(c *Config) { c.StillSpeed = 0 },
		func(c *Config) { c.MaxGlide = c.StillTime / 2 },
	}
	for i, f := range bad {
		c := DefaultConfig()
		f(&c)
		if _, err := New(c); err != ErrConfig {
			t.Errorf("case %d: err %v, want ErrConfig", i, err)
		}
	}
}

func TestUpdateDoesNotAllocate(t *testing.T) {
	n := newNav(t, DefaultConfig())
	q := [4]float32{0.9, 0.1, 0.3, 0.2}
	f := [3]float32{0.3, 1, 9.7}
	w := [3]float32{0.2, 0, 0.1}
	if a := testing.AllocsPerRun(1000, func() { n.Update(0.01, q, f, w) }); a != 0 {
		t.Errorf("%v allocations per Update", a)
	}
}

func BenchmarkUpdate(b *testing.B) {
	n, _ := New(DefaultConfig())
	q := [4]float32{0.9, 0.1, 0.3, 0.2}
	f := [3]float32{0.3, 1, 9.7}
	w := [3]float32{0.2, 0, 0.1}
	b.ReportAllocs()
	for range b.N {
		n.Update(0.01, q, f, w)
	}
}

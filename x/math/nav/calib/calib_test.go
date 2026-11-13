package calib

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/nav/strapdown"
)

const rate = 100.0 // Hz

// lcg is a deterministic noise source; gauss draws a standard normal from it.
type lcg uint32

func (r *lcg) gauss() float64 {
	*r = *r*1664525 + 1013904223
	u1 := (float64(*r) + 1) / (1 << 32)
	*r = *r*1664525 + 1013904223
	u2 := float64(*r) / (1 << 32)
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

// sensor is a still badge: up is the sensor axis that points up (0, 1, 2)
// and its sign, ab the accelerometer's offsets, gb the gyro's bias, with
// Gaussian noise of sa and sg; tilt leans the surface about the axis after
// up by tilt radians.
type sensor struct {
	up       int
	sign     float64
	ab, gb   [3]float64
	sa, sg   float64
	tilt     float64
	rnd      lcg
	nudge    float64 // s of a nudge still to come (a push and a twist)
	accelAdd [3]float64
	tiltRate float64 // rad/s the tilt grows by (a slow lean, gyro on the axis it turns about)
	turnRate float64 // rad/s about the up axis (a slow turn: the force is unchanged)
	ramp     float64 // rad/s² the gyro bias about the up axis grows by (a sensor warming up)
}

func (s *sensor) sample() (a, g [3]float32) {
	s.tilt += s.tiltRate / rate
	s.gb[s.up] += s.ramp / rate
	var f [3]float64
	f[s.up] = s.sign * StandardGravity * math.Cos(s.tilt)
	f[(s.up+1)%3] = StandardGravity * math.Sin(s.tilt)
	for i := range 3 {
		f[i] += s.ab[i] + s.accelAdd[i] + s.sa*s.rnd.gauss()
		g[i] = float32(s.gb[i] + s.sg*s.rnd.gauss())
		a[i] = float32(f[i])
	}
	g[(s.up+2)%3] += float32(s.tiltRate)
	g[s.up] += float32(s.turnRate)
	if s.nudge > 0 {
		s.nudge -= 1 / rate
		a[(s.up+1)%3] += 0.6
		g[s.up] += 0.4
	}
	return a, g
}

func newCal(t testing.TB) Calibrator {
	t.Helper()
	c, err := New(DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// untilDone is how long a still run takes from power-on with the default
// Config, and a little: the live statistics span Window, the sensor looks
// still for Window, then Measure.
var untilDone = float64(2*DefaultConfig().Window+DefaultConfig().Measure) + 0.2

// run feeds T seconds of samples.
func run(c *Calibrator, s *sensor, T float64) {
	for range int(T * rate) {
		a, g := s.sample()
		c.Update(1/rate, a, g)
	}
}

// mpu is an MPU6050 at 100 Hz: offsets up to 0.06 m/s², 0.03 m/s² of
// accelerometer noise, a gyro bias of a few mrad/s and 0.002 rad/s of noise.
func mpu(up int, sign float64, seed lcg) *sensor {
	return &sensor{up: up, sign: sign, ab: [3]float64{0.04, -0.06, 0.05}, gb: [3]float64{0.005, -0.003, 0.004}, sa: 0.03, sg: 0.002, rnd: seed}
}

// Acceptance (brief): the face that is down is found by itself, flat or on
// any side; the rest target is +G on the axis that points up (−G on one
// pointing down), 0 on the others; the run ends Done within
// untilDone.
func TestFindsTheFaceThatIsDown(t *testing.T) {
	for up := range 3 {
		for _, sign := range []float64{1, -1} {
			c := newCal(t)
			s := mpu(up, sign, lcg(7+up))
			run(&c, s, untilDone)
			axis, sg := c.Up()
			rest := c.AccelRest()
			if c.State() != Done || axis != up || float64(sg) != sign || math.Abs(float64(rest[up])-sign*StandardGravity) > 1e-5 || rest[(up+1)%3] != 0 || rest[(up+2)%3] != 0 {
				t.Errorf("up %d sign %+.0f: state %v, Up() = %d %+d, AccelRest %v", up, sign, c.State(), axis, sg, rest)
			}
		}
	}
}

// Acceptance (brief): from biased, noisy samples the gyro bias is recovered
// to within four standard deviations of a 3 s mean (0.002 rad/s of noise:
// 5e-4 rad/s), the accelerometer's readings at rest less the rest target
// to 0.01 m/s² (lying level, these are its offsets), and the noise to 15 %.
func TestRecoversBiasOffsetsAndNoise(t *testing.T) {
	for up := range 3 {
		c := newCal(t)
		s := mpu(up, 1, lcg(100+up))
		run(&c, s, untilDone)
		r, ok := c.Result()
		if !ok {
			t.Fatalf("up %d: no result after a still run (state %v)", up, c.State())
		}
		for i := range 3 {
			if d := math.Abs(float64(r.GyroBias[i]) - s.gb[i]); d > 5e-4 {
				t.Errorf("up %d: gyro bias %d %.5f, want %.5f", up, i, r.GyroBias[i], s.gb[i])
			}
			if d := math.Abs(float64(r.AccelOffset[i]) - s.ab[i]); d > 0.01 {
				t.Errorf("up %d: accel offset %d %.4f, want %.4f", up, i, r.AccelOffset[i], s.ab[i])
			}
			if d := math.Abs(float64(r.AccelNoise[i])/s.sa - 1); d > 0.15 {
				t.Errorf("up %d: accel noise %d %.4f, want %.4f", up, i, r.AccelNoise[i], s.sa)
			}
			if d := math.Abs(float64(r.GyroNoise[i])/s.sg - 1); d > 0.15 {
				t.Errorf("up %d: gyro noise %d %.5f, want %.5f", up, i, r.GyroNoise[i], s.sg)
			}
		}
	}
}

// One pose cannot tell an offset from the surface's tilt: on a table
// tilted by 1°, the horizontal reading off the rest target is the offset
// plus g·sin 1° (0.17 m/s²), and that is what Result reports.
func TestOnePoseReadsTheSurfaceTilt(t *testing.T) {
	c := newCal(t)
	s := mpu(2, 1, lcg(9))
	s.tilt = math.Pi / 180
	run(&c, s, untilDone)
	r, _ := c.Result()
	want := s.ab[0] + StandardGravity*math.Sin(s.tilt)
	if d := math.Abs(float64(r.AccelOffset[0]) - want); d > 0.01 {
		t.Errorf("x reading %.4f, want offset + tilt %.4f", r.AccelOffset[0], want)
	}
}

// Acceptance (brief): moving restarts the measurement. A nudge (a push and
// a twist for 0.1 s) halfway through measuring: the state turns Moved and
// Progress drops to 0; after MovedHold it waits, then measures again from
// 0 and ends Done with a result untouched by the nudge.
func TestMovingRestartsTheMeasurement(t *testing.T) {
	cfg := DefaultConfig()
	c := newCal(t)
	s := mpu(2, 1, lcg(11))
	run(&c, s, float64(2*cfg.Window)+float64(cfg.Measure)/2+0.1)
	if c.State() != Measuring || c.Progress() < 0.3 {
		t.Fatalf("before the nudge: state %v, progress %.2f; want measuring past 0.3", c.State(), c.Progress())
	}
	s.nudge = 0.1
	moved, sawWaiting := 0, false
	for range int(0.8 * rate) {
		a, g := s.sample()
		c.Update(1/rate, a, g)
		switch c.State() {
		case Moved:
			moved++
			if c.Progress() != 0 {
				t.Fatalf("moved with progress %.2f", c.Progress())
			}
		case Waiting:
			sawWaiting = moved > 0
		}
	}
	if math.Abs(float64(moved)/rate-float64(cfg.MovedHold)) > 1.5/rate || !sawWaiting {
		t.Fatalf("after a nudge: moved for %.2f s (want MovedHold %.2f s), then waiting %v", float64(moved)/rate, cfg.MovedHold, sawWaiting)
	}
	run(&c, s, 6) // the nudge leaves the live spread over the threshold for about 1.4 s
	r, ok := c.Result()
	if !ok {
		t.Fatalf("no result after the restart (state %v)", c.State())
	}
	for i := range 3 {
		if d := math.Abs(float64(r.GyroBias[i]) - s.gb[i]); d > 5e-4 {
			t.Errorf("gyro bias %d %.5f after the nudge, want %.5f", i, r.GyroBias[i], s.gb[i])
		}
	}
}

// A badge that never holds still never starts measuring: Progress stays
// 0. A hand's tremor shakes the accelerometer (0.1 m/s²); a motor's
// vibration on the desk jitters the gyro (0.03 rad/s) while the force
// looks quiet.
func TestWaitsForStillness(t *testing.T) {
	for _, tc := range []struct {
		name   string
		sa, sg float64
	}{{"tremor", 0.1, 0.002}, {"gyro jitter", 0.03, 0.03}} {
		c := newCal(t)
		s := mpu(2, 1, lcg(13))
		s.sa, s.sg = tc.sa, tc.sg
		run(&c, s, 5)
		if c.State() != Waiting || c.Progress() != 0 {
			t.Errorf("%s for 5 s: state %v, progress %.2f; want waiting at 0", tc.name, c.State(), c.Progress())
		}
	}
}

// Motions too slow to show in a window's spread still restart the
// measurement: a slow lean (0.5 °/s, which the gyro's 0.009 rad/s alone
// would pass) moves the force's mean, and a slow turn on a desk that
// swivels (0.015 rad/s about the up axis) moves the gyro's mean with the
// force unchanged. Left measuring, either would end with a wrong result:
// the lean's force, the turn's rate taken for gyro bias.
func TestSlowMotionsRestartTheMeasurement(t *testing.T) {
	cfg := DefaultConfig()
	for _, tc := range []struct {
		name       string
		tilt, turn float64
	}{{"lean", 0.5 * math.Pi / 180, 0}, {"turn", 0, 0.015}} {
		c := newCal(t)
		s := mpu(2, 1, lcg(25))
		run(&c, s, float64(2*cfg.Window)+0.2)
		if c.State() != Measuring {
			t.Fatalf("%s: not measuring before the motion: %v", tc.name, c.State())
		}
		s.tiltRate, s.turnRate = tc.tilt, tc.turn
		moved := false
		for range int(float64(cfg.Measure) * rate) {
			a, g := s.sample()
			c.Update(1/rate, a, g)
			moved = moved || c.State() == Moved
		}
		if !moved || c.State() == Done {
			t.Errorf("%s: moved %v, state %v after Measure; want a restart and no result", tc.name, moved, c.State())
		}
	}
}

// Progress rises steadily while measuring and reaches 1 when the run is
// Done, Measure after measuring began (within a sample); measuring begins
// 2·Window from power-on, no sooner.
func TestProgressRisesToDone(t *testing.T) {
	cfg := DefaultConfig()
	c := newCal(t)
	s := mpu(2, 1, lcg(15))
	prev, began, done := float32(0), -1.0, -1.0
	for i := range int(6 * rate) {
		a, g := s.sample()
		c.Update(1/rate, a, g)
		p := c.Progress()
		if p < prev {
			t.Fatalf("progress fell from %.3f to %.3f", prev, p)
		}
		if began < 0 && c.State() == Measuring {
			began = float64(i) / rate
		}
		if done < 0 && c.State() == Done {
			done = float64(i) / rate
			if p != 1 {
				t.Errorf("done with progress %.3f", p)
			}
		}
		prev = p
	}
	if math.Abs(began-float64(2*cfg.Window)) > 1.5/rate || math.Abs(done-began-float64(cfg.Measure)) > 1.5/rate {
		t.Errorf("measuring began at %.2f s, done at %.2f s; want 2·Window (%.2f s: the live statistics span Window, then still for Window) and Measure %.2f s apart", began, done, 2*cfg.Window, cfg.Measure)
	}
}

// The live stats are the strips' input: per axis, the mean and the
// standard deviation (not a range) of the last Window, in sensor units.
func TestStatsAreMeanAndStandardDeviation(t *testing.T) {
	c := newCal(t)
	s := mpu(2, 1, lcg(17))
	run(&c, s, 3)
	a, g := c.Accel(), c.Gyro()
	for i := range 3 {
		wantMean := s.ab[i]
		if i == 2 {
			wantMean += StandardGravity
		}
		if d := math.Abs(float64(a.Mean[i]) - wantMean); d > 0.02 {
			t.Errorf("accel mean %d %.4f, want %.4f", i, a.Mean[i], wantMean)
		}
		if d := math.Abs(float64(a.Spread[i])/s.sa - 1); d > 0.35 {
			t.Errorf("accel spread %d %.4f, want about %.4f", i, a.Spread[i], s.sa)
		}
		if d := math.Abs(float64(g.Spread[i])/s.sg - 1); d > 0.35 {
			t.Errorf("gyro spread %d %.5f, want about %.5f", i, g.Spread[i], s.sg)
		}
	}
}

// A recalibrates: Reset starts again from Waiting at 0, and the next run
// ends Done.
func TestResetStartsAgain(t *testing.T) {
	c := newCal(t)
	s := mpu(2, 1, lcg(19))
	run(&c, s, untilDone)
	if c.State() != Done {
		t.Fatalf("not done after a still run: %v", c.State())
	}
	c.Reset()
	if c.State() != Waiting || c.Progress() != 0 {
		t.Fatalf("after Reset: %v at %.2f", c.State(), c.Progress())
	}
	if _, ok := c.Result(); ok {
		t.Errorf("Result still reported after Reset")
	}
	run(&c, s, untilDone)
	if c.State() != Done {
		t.Errorf("not done again: %v", c.State())
	}
}

func TestBadSamplesAreIgnored(t *testing.T) {
	c := newCal(t)
	s := mpu(2, 1, lcg(21))
	for i := range int(6 * rate) {
		a, g := s.sample()
		if i%10 == 3 {
			a[1] = float32(math.NaN())
		}
		if i%10 == 7 {
			c.Update(0, a, g)
			c.Update(-1, a, g) // a clock stepping back
			continue
		}
		c.Update(1/rate, a, g)
	}
	r, ok := c.Result()
	if !ok || math.IsNaN(float64(r.AccelOffset[1])) {
		t.Errorf("with bad samples among good ones: done %v, result %v", ok, r)
	}
}

func TestNewRejectsConfig(t *testing.T) {
	for _, mod := range []func(*Config){
		func(c *Config) { c.G = 0 },
		func(c *Config) { c.Window = 0 },
		func(c *Config) { c.Measure = -1 },
		func(c *Config) { c.AccelStill = 0 },
		func(c *Config) { c.GyroStill = float32(math.NaN()) },
		func(c *Config) { c.MovedHold = -1 },
		func(c *Config) { c.MotionsPerMin = -1 },
		func(c *Config) { c.MotionTime = 0 },
	} {
		c := DefaultConfig()
		mod(&c)
		if _, err := New(c); err == nil {
			t.Errorf("New accepted %+v", c)
		}
	}
}

func TestUpdateDoesNotAllocate(t *testing.T) {
	c := newCal(t)
	s := mpu(2, 1, lcg(23))
	a, g := s.sample()
	if n := testing.AllocsPerRun(1000, func() { c.Update(1/rate, a, g); _, _ = c.Result() }); n != 0 {
		t.Errorf("%v allocations per Update", n)
	}
}

// BenchmarkUpdate: one sample of a still sensor, measuring and done.
func BenchmarkUpdate(b *testing.B) {
	c := newCal(b)
	s := mpu(2, 1, lcg(29))
	var a, g [1024][3]float32
	for i := range a {
		a[i], g[i] = s.sample()
	}
	b.ReportAllocs()
	for i := 0; b.Loop(); i++ {
		c.Update(1/rate, a[i%1024], g[i%1024])
	}
}

// The heading forecast against what calibrations leave: over many runs at
// an MPU6050's noise, the gyro bias left on the up axis (true minus
// measured) turns the heading by that much per minute; the forecast's
// systematic drift is 0.8–1.5 times its root mean square (it reads 1.2:
// half the change between the halves sometimes exceeds σ/√N on noise
// alone).
func TestHeadingForecastMatchesWhatCalibrationsLeave(t *testing.T) {
	var sum2, fc float64
	const runs = 40
	for seed := range lcg(runs) {
		c := newCal(t)
		s := mpu(2, 1, 300+seed)
		run(&c, s, untilDone)
		r, ok := c.Result()
		if !ok {
			t.Fatalf("seed %d: not done", seed)
		}
		left := (float64(r.GyroBias[2]) - s.gb[2]) * 60 * 180 / math.Pi
		sum2 += left * left
		fc += float64(r.Forecast.HeadingDrift)
	}
	rms, mean := math.Sqrt(sum2/runs), fc/runs
	t.Logf("heading left by the bias: %.3f °/min rms; forecast drift %.3f °/min", rms, mean)
	if mean < 0.8*rms || mean > 1.5*rms {
		t.Errorf("forecast heading drift %.3f °/min, the runs left %.3f °/min (rms)", mean, rms)
	}
}

// The position forecast against the strapdown path: a minute of play
// (MotionsPerMin strokes of MotionTime, 10 cm up and down, at rest
// between) with an exact attitude and the accelerometer's noise and
// offsets; where the badge ends against where it is, over seeds, is
// 0.75–1.5 times the forecast's total (drift and noise added in
// quadrature; it reads 1.05 here; strokes of 0.5–2 s read 1.2–1.75, the
// drift taken as adding up).
func TestPositionForecastMatchesTheStrapdownPath(t *testing.T) {
	cfg := DefaultConfig()
	c := newCal(t)
	s := mpu(2, 1, 400)
	run(&c, s, untilDone)
	r, ok := c.Result()
	if !ok {
		t.Fatal("not done")
	}
	fc := math.Hypot(float64(r.Forecast.PosDrift), float64(r.Forecast.PosNoise))
	M, T := int(cfg.MotionsPerMin), float64(cfg.MotionTime)
	rest := 60/float64(M) - T
	var sum2 float64
	const runs = 20
	for seed := range lcg(runs) {
		n, err := strapdown.New(strapdown.DefaultConfig())
		if err != nil {
			t.Fatal(err)
		}
		rnd := 500 + seed
		feed := func(az float64) {
			var f [3]float32
			for i := range 3 {
				v := s.ab[i] + s.sa*rnd.gauss()
				if i == 2 {
					v += StandardGravity + az
				}
				f[i] = float32(v)
			}
			n.Update(1/rate, [4]float32{1, 0, 0, 0}, f, [3]float32{})
		}
		for range int(2 * rate) {
			feed(0)
		}
		z0, dir := float64(n.P[2]), 1.0
		for range M {
			for i := range int(T * rate) {
				u := float64(i) / (T * rate)
				feed(dir * 0.1 / (T * T) * (60*u - 180*u*u + 120*u*u*u))
			}
			for range int(rest * rate) {
				feed(0)
			}
			dir = -dir
		}
		want := 0.0
		if M%2 == 1 {
			want = 0.1
		}
		got := float64(n.P[2]) - z0 - want
		x, y := float64(n.P[0]), float64(n.P[1])
		sum2 += (got*got + x*x + y*y) * 1e4
	}
	rms := math.Sqrt(sum2 / runs)
	t.Logf("strapdown after a minute of play: %.2f cm rms; forecast %.2f cm/min (drift %.2f, noise %.2f)", rms, fc, r.Forecast.PosDrift, r.Forecast.PosNoise)
	if fc < 0.75*rms || fc > 1.5*rms {
		t.Errorf("forecast %.2f cm/min, the path strayed %.2f cm (rms)", fc, rms)
	}
}

// A sensor warming up wanders: its gyro bias about the up axis growing by
// 1 mrad/s each second moves the halves' means apart by 1.5 mrad/s, and
// the forecast's heading drift goes over the acceptable line (it would
// show 0.4 °/min from the noise alone); a steady one stays under.
func TestHeadingForecastSeesAWanderingBias(t *testing.T) {
	for _, ramp := range []float64{0, 1e-3} {
		c := newCal(t)
		s := mpu(2, 1, lcg(27))
		s.ramp = ramp
		run(&c, s, untilDone)
		r, ok := c.Result()
		if !ok {
			t.Fatalf("ramp %g: not done (state %v)", ramp, c.State())
		}
		f := r.Forecast
		if over := f.HeadingDrift > f.HeadingOK; over != (ramp > 0) {
			t.Errorf("bias ramp %g rad/s²: heading drift %.2f °/min against the line %.1f", ramp, f.HeadingDrift, f.HeadingOK)
		}
	}
}

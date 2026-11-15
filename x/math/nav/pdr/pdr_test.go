package pdr

import (
	"errors"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/nav/mahony"
)

const g = 9.80665

// noise is a deterministic uniform noise source in [-a, a].
type noise struct{ s uint32 }

func (n *noise) next(a float64) float64 {
	n.s ^= n.s << 13
	n.s ^= n.s >> 17
	n.s ^= n.s << 5
	return a * (float64(n.s)/float64(^uint32(0))*2 - 1)
}

// walk feeds secs seconds of a body-worn accelerometer to d: gravity tilted
// by 25° plus a vertical bounce of amp m/s² at cadence steps/s, plus noise,
// all times the accelerometer's scale error scale; it returns the steps d
// reports.
func walk(t *testing.T, d *Detector, rate, secs, cadence, amp, noiseAmp, scale float64) int {
	t.Helper()
	n := noise{s: 2463534242}
	tilt := 25 * math.Pi / 180
	steps := 0
	for i := 0; i < int(secs*rate); i++ {
		tt := float64(i) / rate
		v := scale * (g + amp*math.Sin(2*math.Pi*cadence*tt) + n.next(noiseAmp))
		a := [3]float32{float32(v*math.Sin(tilt) + n.next(noiseAmp)), float32(n.next(noiseAmp)), float32(v * math.Cos(tilt))}
		if d.Update(a) {
			steps++
		}
	}
	return steps
}

func newDetector(t *testing.T) Detector {
	t.Helper()
	d, err := NewDetector(DefaultDetectorConfig(100))
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// Walking at 1.2, 2 and 2.5 steps/s for 20 s: one step per bounce (±1 for
// the ends of the record).
func TestDetectorCountsSteps(t *testing.T) {
	for _, c := range []struct{ cadence, amp float64 }{{1.2, 1.8}, {2, 2.5}, {2.5, 3}} {
		d := newDetector(t)
		want := int(c.cadence * 20)
		if got := walk(t, &d, 100, 20, c.cadence, c.amp, 0.4, 1); got < want-1 || got > want+1 {
			t.Errorf("cadence %v/s: %d steps, want %d±1", c.cadence, got, want)
		}
		if d.Steps() < want-1 {
			t.Errorf("Steps() %d", d.Steps())
		}
	}
}

// Standing still with sensor noise, and a hand tremor/vibration at 15 Hz:
// no steps.
func TestDetectorIgnoresRestAndVibration(t *testing.T) {
	d := newDetector(t)
	if got := walk(t, &d, 100, 30, 0, 0, 0.3, 1); got != 0 {
		t.Errorf("at rest: %d steps", got)
	}
	d = newDetector(t)
	if got := walk(t, &d, 100, 10, 15, 3, 0.2, 1); got != 0 {
		t.Errorf("15 Hz vibration: %d steps", got)
	}
}

// An uncalibrated accelerometer reading 0.9 g or 1.12 g at rest: the
// running baseline follows it, so rest still gives no steps and walking the
// right count.
func TestDetectorFollowsAScaleError(t *testing.T) {
	for _, scale := range []float64{0.9, 1.12} {
		d := newDetector(t)
		if got := walk(t, &d, 100, 20, 0, 0, 0.3, scale); got != 0 {
			t.Errorf("scale %v at rest: %d steps", scale, got)
		}
		d = newDetector(t)
		if got := walk(t, &d, 100, 20, 2, 2, 0.3, scale); got < 39 || got > 41 {
			t.Errorf("scale %v walking 2/s: %d steps, want 40±1", scale, got)
		}
	}
}

// Steps closer than MinInterval count once: a 5 steps/s bounce gives at
// most 1/MinInterval steps a second.
func TestDetectorMinInterval(t *testing.T) {
	cfg := DefaultDetectorConfig(100)
	cfg.Cutoff = 10 // let the fast bounce through the low-pass
	d, err := NewDetector(cfg)
	if err != nil {
		t.Fatal(err)
	}
	got := walk(t, &d, 100, 10, 5, 4, 0, 1)
	if max := int(10/cfg.MinInterval) + 1; got > max || got < 10 {
		t.Errorf("%d steps in 10 s at 5/s, want between 10 and %d", got, max)
	}
}

// With MinInterval 0 only the hysteresis (arm above Threshold, fire on the
// fall through the baseline) keeps a noisy peak to one step: a bounce that
// barely clears Threshold, with noise, still counts 40±1 in 20 s at 2/s
// (firing on the fall through Threshold instead counts 44–57).
func TestDetectorHysteresis(t *testing.T) {
	for _, c := range []struct{ amp, noise float64 }{{1.4, 0.6}, {1.4, 1}, {1.8, 1}} {
		cfg := DefaultDetectorConfig(100)
		cfg.MinInterval = 0
		d, err := NewDetector(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if got := walk(t, &d, 100, 20, 2, c.amp, c.noise, 1); got < 39 || got > 41 {
			t.Errorf("bounce %v m/s², noise %v: %d steps, want 40±1", c.amp, c.noise, got)
		}
	}
}

func TestNewDetectorRefusesBadConfig(t *testing.T) {
	for _, c := range []DetectorConfig{
		{},
		{SampleRate: 100, Cutoff: 3, Threshold: 0, MinInterval: 0.3, Baseline: 2},
		{SampleRate: 100, Cutoff: 60, Threshold: 1, MinInterval: 0.3, Baseline: 2}, // above Nyquist
		{SampleRate: 100, Cutoff: 3, Threshold: 1, MinInterval: -1, Baseline: 2},
		{SampleRate: 100, Cutoff: 3, Threshold: 1, MinInterval: 0.3, Baseline: 0},
		{SampleRate: float32(math.Inf(1)), Cutoff: 3, Threshold: 1, MinInterval: 0.3, Baseline: 2},
		{SampleRate: 100, Cutoff: 3, Threshold: float32(math.Inf(1)), MinInterval: 0.3, Baseline: 2},
		{SampleRate: 100, Cutoff: 3, Threshold: 1, MinInterval: float32(math.NaN()), Baseline: 2},
		{SampleRate: 1e6, Cutoff: 3, Threshold: 1, MinInterval: 1e4, Baseline: 2}, // gap beyond int32
	} {
		if _, err := NewDetector(c); !errors.Is(err, ErrConfig) {
			t.Errorf("%+v: err %v, want ErrConfig", c, err)
		}
	}
}

// Only |a| counts: the same walk with gravity along x, y, z or a diagonal of
// the sensor gives the same steps.
func TestDetectorAnyOrientation(t *testing.T) {
	count := func(axis [3]float64) int {
		d := newDetector(t)
		n := math.Sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
		nz := noise{s: 99}
		steps := 0
		for i := 0; i < 2000; i++ {
			v := g + 2*math.Sin(2*math.Pi*2*float64(i)/100) + nz.next(0.3)
			if d.Update([3]float32{float32(v * axis[0] / n), float32(v * axis[1] / n), float32(v * axis[2] / n)}) {
				steps++
			}
		}
		return steps
	}
	want := count([3]float64{0, 0, 1})
	if want < 39 || want > 41 {
		t.Fatalf("z: %d steps in 20 s at 2/s", want)
	}
	for _, axis := range [][3]float64{{1, 0, 0}, {0, -1, 0}, {1, 1, 1}, {-0.3, 0.8, 0.5}} {
		if got := count(axis); got != want {
			t.Errorf("gravity along %v: %d steps, want %d as along z", axis, got, want)
		}
	}
}

// The running baseline follows a lasting change of |a| (a sensor scale or
// mounting change) with the configured time constant: 63 % of a step after
// Baseline seconds, and the change does not count as steps.
func TestDetectorBaselineTimeConstant(t *testing.T) {
	d := newDetector(t) // Baseline 2 s, 100 Hz
	for i := 0; i < 1000; i++ {
		d.Update([3]float32{0, 0, g})
	}
	for i := 0; i < 200; i++ { // 2 s after a +0.5 m/s² step
		d.Update([3]float32{0, 0, g + 0.5})
	}
	if got, want := float64(d.base)-g, 0.5*(1-math.Exp(-1)); math.Abs(got-want) > 0.05*want {
		t.Fatalf("baseline rose %.4f m/s² after 2 s, want %.4f (63 %% of 0.5) ±5 %%", got, want)
	}
	if d.Steps() != 0 {
		t.Fatalf("%d steps from a level change", d.Steps())
	}
}

// A sensor that reads zero before it wakes does not seed the filter: no
// step when real samples start. A NaN or infinite sample is ignored.
func TestDetectorPrimingAndBadSamples(t *testing.T) {
	d := newDetector(t)
	for i := 0; i < 50; i++ {
		d.Update([3]float32{})
	}
	nz := noise{s: 7}
	for i := 0; i < 3000; i++ { // 30 s standing, sensor noise ±0.3 m/s²
		d.Update([3]float32{float32(nz.next(0.3)), float32(nz.next(0.3)), float32(g + nz.next(0.3))})
	}
	if d.Steps() != 0 {
		t.Fatalf("zero samples before waking: %d steps at rest", d.Steps())
	}
	d = newDetector(t)
	nan := float32(math.NaN())
	steps := 0
	for i := 0; i < 2000; i++ {
		a := [3]float32{0, 0, float32(g + 2.5*math.Sin(2*math.Pi*2*float64(i)/100))}
		if i == 700 {
			a[1] = nan
		}
		if i == 1100 {
			a[0] = float32(math.Inf(1))
		}
		if d.Update(a) {
			steps++
		}
	}
	if steps < 39 || steps > 41 {
		t.Fatalf("a NaN and an infinite sample mid-walk: %d steps in 20 s at 2/s, want 40±1", steps)
	}
}

// quatFromColumns is the [w,x,y,z] quaternion of the rotation whose columns
// are the world images of the sensor's x, y and z axes.
func quatFromColumns(cx, cy, cz [3]float64) [4]float32 {
	m00, m11, m22 := cx[0], cy[1], cz[2]
	m21, m12 := cy[2], cz[1]
	m02, m20 := cz[0], cx[2]
	m10, m01 := cx[1], cy[0]
	tr := m00 + m11 + m22
	var w, x, y, z float64
	switch {
	case tr > 0:
		s := math.Sqrt(tr+1) * 2
		w, x, y, z = s/4, (m21-m12)/s, (m02-m20)/s, (m10-m01)/s
	case m00 > m11 && m00 > m22:
		s := math.Sqrt(1+m00-m11-m22) * 2
		w, x, y, z = (m21-m12)/s, s/4, (m01+m10)/s, (m02+m20)/s
	case m11 > m22:
		s := math.Sqrt(1+m11-m00-m22) * 2
		w, x, y, z = (m02-m20)/s, (m01+m10)/s, s/4, (m12+m21)/s
	default:
		s := math.Sqrt(1+m22-m00-m11) * 2
		w, x, y, z = (m10-m01)/s, (m02+m20)/s, (m12+m21)/s, s/4
	}
	return [4]float32{float32(w), float32(x), float32(y), float32(z)}
}

// yawed turns q by psi about the world's vertical: rotZ(psi) ⊗ q.
func yawed(q [4]float32, psi float64) [4]float32 {
	c, s := float32(math.Cos(psi/2)), float32(math.Sin(psi/2))
	return [4]float32{c*q[0] - s*q[3], c*q[1] - s*q[2], c*q[2] + s*q[1], c*q[3] + s*q[0]}
}

// The badge (screen right = IMU −X, up = +Y, out = −Z) walks along its
// screen-out axis −Z. Heading of that axis is the turn angle, in portrait
// and with the IMU's x axis vertical (where Euler yaw has no meaning), and
// is refused for an axis pointing straight up.
func TestHeading(t *testing.T) {
	out := [3]float32{0, 0, -1}
	poses := map[string][4]float32{
		// portrait: X → −y (screen right = wearer's left), Y → up, Z → −x (out = forward)
		"portrait": quatFromColumns([3]float64{0, -1, 0}, [3]float64{0, 0, 1}, [3]float64{-1, 0, 0}),
		// on its side, IMU x vertical: X → up, Y → +y, Z → −x
		"x vertical": quatFromColumns([3]float64{0, 0, 1}, [3]float64{0, 1, 0}, [3]float64{-1, 0, 0}),
	}
	for name, q := range poses {
		for _, psi := range []float64{0, 90 * math.Pi / 180, -135 * math.Pi / 180, 170 * math.Pi / 180} {
			h, ok := Heading(yawed(q, psi), out)
			diff := math.Remainder(float64(h)-psi, 2*math.Pi)
			if !ok || !(math.Abs(diff) <= 1e-4) {
				t.Errorf("%s turned %.0f°: heading %.4f ok %v, want %.4f", name, psi*180/math.Pi, h, ok, psi)
			}
		}
	}
	if _, ok := Heading(poses["portrait"], [3]float32{0, 1, 0}); ok {
		t.Error("the vertical axis (IMU +Y in portrait) has no heading")
	}
	if h, ok := Heading([4]float32{1, 0, 0, 0}, [3]float32{0, 2, 0}); !ok || math.Abs(float64(h)-math.Pi/2) > 1e-6 {
		t.Errorf("identity, axis y: %v %v, want π/2", h, ok)
	}
	// The refusal cone is about 6° (horizontal part 0.1 of the unit axis),
	// whatever the axis length: an axis 5.5° from vertical is refused, one
	// 6° from vertical points along +x; a zero axis has no direction.
	for _, c := range []struct {
		deg float64
		ok  bool
	}{{5.5, false}, {6, true}} {
		th := c.deg * math.Pi / 180
		h, ok := Heading([4]float32{1, 0, 0, 0}, [3]float32{float32(10 * math.Sin(th)), 0, float32(10 * math.Cos(th))})
		if ok != c.ok || (ok && math.Abs(float64(h)) > 1e-6) {
			t.Errorf("axis %.1f° from vertical (length 10): %v %v, want ok %v (heading 0)", c.deg, h, ok, c.ok)
		}
	}
	if _, ok := Heading([4]float32{1, 0, 0, 0}, [3]float32{}); ok {
		t.Error("a zero axis has no heading")
	}
}

// A badge worn on the chest (screen out, IMU +Y up) walks 11 steps (5.5 s at
// 2 steps/s), turns 90° left standing, walks 11 more: the second leg is 90°
// counter-clockwise from the first, each about 11 step lengths. Heading is
// the screen-out axis −Z through a 6-axis Mahony filter (no magnetometer)
// fed the gyro of the turn about the IMU's y axis.
func TestWalkWithMahonyHeading(t *testing.T) {
	const rate, L = 100.0, 0.7
	d := newDetector(t)
	tr := Track{StepLength: L}
	f := newMahony(2, 0)
	var in mat.Matrix3x3
	const settle = 10.0 // standing still while the filter finds the badge's tilt
	phase := 0.0
	var leg1X, leg1Y float32
	for i := 0; i < int((settle+12)*rate); i++ {
		tt := float64(i)/rate - settle
		yawRate := 0.0
		if tt >= 5.5 && tt < 6.5 {
			yawRate = math.Pi / 2
		}
		bounce := 0.0
		if tt >= 0 && (tt < 5.5 || tt >= 6.5) {
			bounce = 2.5 * math.Sin(phase)
			phase += 2 * math.Pi * 2 / rate
		}
		if tt >= 5.5 && leg1X == 0 && leg1Y == 0 {
			leg1X, leg1Y = tr.X, tr.Y
		}
		in[0] = [3]float32{0, float32(g + bounce), 0}
		in[1] = [3]float32{0, float32(yawRate), 0}
		_ = f.Update(1/rate, in[0], in[1])
		if d.Update(in[0]) {
			q := f.Attitude()
			h, ok := Heading([4]float32(q), [3]float32{0, 0, -1})
			if !ok {
				t.Fatalf("no heading at t = %.2f s", tt)
			}
			tr.Step(h)
		}
	}
	if tr.Steps < 21 || tr.Steps > 23 {
		t.Fatalf("%d steps, want 22±1", tr.Steps)
	}
	l2x, l2y := tr.X-leg1X, tr.Y-leg1Y
	n1 := math.Hypot(float64(leg1X), float64(leg1Y))
	n2 := math.Hypot(float64(l2x), float64(l2y))
	turn := math.Atan2(float64(leg1X*l2y-leg1Y*l2x), float64(leg1X*l2x+leg1Y*l2y))
	if math.Abs(n1-11*L) > 1.5*L || math.Abs(n2-11*L) > 1.5*L || math.Abs(turn-math.Pi/2) > 10*math.Pi/180 {
		t.Fatalf("legs %.2f m and %.2f m, turn %.1f°; want ≈ %.1f m each and +90°", n1, n2, turn*180/math.Pi, 11*L)
	}
}

func TestTrackStep(t *testing.T) {
	tr := Track{StepLength: 0.5}
	tr.Step(0)
	tr.Step(math.Pi / 2)
	tr.Step(math.Pi)
	if math.Abs(float64(tr.X)) > 1e-6 || math.Abs(float64(tr.Y)-0.5) > 1e-6 || tr.Steps != 3 {
		t.Fatalf("track %+v, want (0, 0.5) after 3 steps", tr)
	}
}

func TestDetectorDoesNotAllocate(t *testing.T) {
	d := newDetector(t)
	tr := Track{StepLength: 0.7}
	a := [3]float32{0, 0, g}
	if n := testing.AllocsPerRun(1000, func() {
		if d.Update(a) {
			tr.Step(0.3)
		}
	}); n != 0 {
		t.Fatalf("%v allocations per sample", n)
	}
}

func BenchmarkDetectorUpdate(b *testing.B) {
	d, _ := NewDetector(DefaultDetectorConfig(100))
	a := [3]float32{1, 2, g}
	i := 0
	for b.Loop() {
		a[2] = g + float32(i&15)/8
		d.Update(a)
		i++
	}
}

// newMahony is the badge's attitude filter, the Mahony brick, with gains kp
// and ki (1/s, 1/s²).
func newMahony(kp, ki float32) *mahony.Filter {
	f, err := mahony.New(mahony.Config{Kp: kp, Ki: ki})
	if err != nil {
		panic(err)
	}
	return &f
}

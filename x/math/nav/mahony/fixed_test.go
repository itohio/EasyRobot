//go:build fixedmotion

package mahony

import (
	"math"
	"testing"
)

// angle is the rotation between two attitudes, radians.
func angle(a, b [4]float32) float64 {
	d := math.Abs(float64(a[0])*float64(b[0]) + float64(a[1])*float64(b[1]) + float64(a[2])*float64(b[2]) + float64(a[3])*float64(b[3]))
	na := math.Sqrt(float64(a[0])*float64(a[0]) + float64(a[1])*float64(a[1]) + float64(a[2])*float64(a[2]) + float64(a[3])*float64(a[3]))
	nb := math.Sqrt(float64(b[0])*float64(b[0]) + float64(b[1])*float64(b[1]) + float64(b[2])*float64(b[2]) + float64(b[3])*float64(b[3]))
	return 2 * math.Acos(min(d/(na*nb), 1))
}

type imuScene struct {
	name   string
	kp, ki float32
	steps  int
	sample func(i int, noise func(float32) float32) (acc, gyro [3]float32)
}

func scenes() []imuScene {
	const g = 9.81
	tilt := func(deg float64) func(int, func(float32) float32) ([3]float32, [3]float32) {
		r := deg * math.Pi / 180
		return func(_ int, n func(float32) float32) ([3]float32, [3]float32) {
			return [3]float32{n(0.05), float32(g*math.Sin(r)) + n(0.05), float32(g*math.Cos(r)) + n(0.05)}, [3]float32{n(0.003), n(0.003), n(0.003)}
		}
	}
	spin := func(w float32) func(int, func(float32) float32) ([3]float32, [3]float32) {
		return func(i int, n func(float32) float32) ([3]float32, [3]float32) {
			// gravity seen by a sensor spinning about z while tilted 20° about x
			r := 20 * math.Pi / 180
			return [3]float32{n(0.05), float32(g*math.Sin(r)) + n(0.05), float32(g*math.Cos(r)) + n(0.05)}, [3]float32{n(0.003), n(0.003), w + n(0.003)}
		}
	}
	hand := func(i int, n func(float32) float32) ([3]float32, [3]float32) {
		t := float64(i) / 100
		a := 0.6 * math.Sin(2*math.Pi*0.7*t) // tilt about x, rad
		w := 0.6 * 2 * math.Pi * 0.7 * math.Cos(2*math.Pi*0.7*t)
		push := 3 * math.Sin(2*math.Pi*2.1*t) // m/s² along y
		return [3]float32{n(0.06), float32(g*math.Sin(a)+push) + n(0.06), float32(g*math.Cos(a)) + n(0.06)},
			[3]float32{float32(w) + n(0.004), float32(0.3*math.Sin(2*math.Pi*0.3*t)) + n(0.004), float32(0.2*math.Cos(2*math.Pi*0.5*t)) + n(0.004)}
	}
	freefall := func(i int, n func(float32) float32) ([3]float32, [3]float32) {
		if i%50 < 3 { // a zero accelerometer sample: gyro only
			return [3]float32{}, [3]float32{0.5, 0, 0}
		}
		return [3]float32{n(0.05), 0.3 + n(0.05), g + n(0.05)}, [3]float32{0.5, n(0.003), n(0.003)}
	}
	var out []imuScene
	for _, kp := range []float32{2, 10} { // the badge demo's gains (settled, settling)
		for _, deg := range []float64{0, 30, 89, 135, 180} {
			out = append(out, imuScene{"tilt", kp, 0, 6000, tilt(deg)})
		}
		for _, w := range []float32{0.5, 2, 6, -6} {
			out = append(out, imuScene{"spin", kp, 0, 1000, spin(w)})
		}
		out = append(out, imuScene{"hand", kp, 0, 6000, hand}, imuScene{"zero-accel", kp, 0, 1000, freefall})
	}
	out = append(out, imuScene{"hand-ki", 2, 10, 6000, hand}, imuScene{"tilt-ki", 0.5, 5, 6000, tilt(45)})
	// Ki 0.5/s² (0.005 per sample at 100 Hz, the strapdown tests' setting) from
	// the start, with a large initial error: the integral grows large, so a
	// gain off by a fraction of a percent shows (in Q16 it did, by 2.5e-4 rad)
	for _, kp := range []float32{2, 10} {
		for _, deg := range []float64{90, 180} {
			out = append(out, imuScene{"tilt-ki5m", kp, 0.5, 6000, tilt(deg)})
		}
		out = append(out, imuScene{"spin-ki5m", kp, 0.5, 3000, spin(6)}, imuScene{"hand-ki5m", kp, 0.5, 6000, hand})
	}
	return out
}

// The fixed-point 6-axis update (fixedmotion) follows the float one within
// 1e-4 rad of attitude at every sample: rest in several tilts, spins up to
// 6 rad/s, hand motion, zero accelerometer samples and integral gains (the
// ahrs filter's per-sample gains times 100 Hz), each from a level first
// sample, so a large initial error (up to upside down) is among them; the
// badge's gains (2 and 10).
func TestFixedFollowsFloat(t *testing.T) {
	if !fixedPath {
		t.Fatal("built without the fixed-point path")
	}
	const tol = 1e-4
	for _, sc := range scenes() {
		fx, fl := newFilter(t, Config{Kp: sc.kp, Ki: sc.ki}), newFilter(t, Config{Kp: sc.kp, Ki: sc.ki})
		_ = fx.Update(0.01, [3]float32{0, 0, 9.81}, [3]float32{})
		_ = fl.Update(0.01, [3]float32{0, 0, 9.81}, [3]float32{})
		x := uint32(2463534242)
		noise := func(s float32) float32 {
			x ^= x << 13
			x ^= x >> 17
			x ^= x << 5
			return s * (float32(x>>8)/(1<<24) - 0.5)
		}
		worst := 0.0
		for i := range sc.steps {
			acc, gyro := sc.sample(i, noise)
			_ = fx.Update(0.01, acc, gyro)
			fl.stepFloat(acc, gyro, 0.01) // the float path
			e := angle(fx.Attitude(), fl.Attitude())
			worst = max(worst, e)
			if e > tol {
				t.Fatalf("%s kp %g ki %g: sample %d, %.3g rad from the float path", sc.name, sc.kp, sc.ki, i, e)
			}
		}
		t.Logf("%-10s kp %4g ki %4g: worst %.2e rad over %d samples", sc.name, sc.kp, sc.ki, worst, sc.steps)
	}
}

// Init forgets the fixed-point state too: after it a level, still sensor
// levels at the identity.
func TestFixedInitForgets(t *testing.T) {
	f := newFilter(t, Config{Kp: 2})
	for range 200 {
		_ = f.Update(0.01, [3]float32{0, 9.81, 0}, [3]float32{0, 0, 3})
	}
	if err := Init(f, Config{Kp: 2}); err != nil {
		t.Fatal(err)
	}
	for range 2 {
		_ = f.Update(0.01, [3]float32{0, 0, 9.81}, [3]float32{})
	}
	if e := angle(f.Attitude(), [4]float32{1, 0, 0, 0}); e > 1e-6 {
		t.Fatalf("level and still after Init: %.3g rad from identity", e)
	}
}

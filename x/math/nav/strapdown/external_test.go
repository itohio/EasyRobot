package strapdown

import (
	"math"
	"testing"
)

// With ExternalAttitude the Nav's attitude is the one supplied, on every
// sample from power-on: through the first rest and a long rest after it,
// where the Nav otherwise levels its attitude by the horizontal bias it
// learned (here the accelerometer's offsets put 0.1 m/s² there), and
// through a turn the gyro reads 0.05 rad/s off the true one. Its attitude
// is not levelled, so that horizontal bias stays in Bias, and the rest's
// linear acceleration is nil; levelling (the default) moves it into the
// attitude instead.
func TestExternalAttitudeIsTheSuppliedOne(t *testing.T) {
	for _, ext := range []bool{false, true} {
		c := DefaultConfig()
		c.ExternalAttitude = ext
		n := newNav(t, c)
		q := axisAngle([3]float64{1, 0, 0}, math.Pi/2)
		off := [3]float32{0.1, -0.08, 0.05}
		differs := false
		step := func(w [3]float64) {
			if wn := math.Sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2]); wn > 0 {
				q = q.mul(axisAngle(w, wn/rate))
			}
			f := v32(sensorForce(q, [3]float64{}))
			for k := range 3 {
				f[k] += off[k]
			}
			gyro := [3]float32{float32(w[0]), float32(w[1]) + 0.05, float32(w[2])}
			n.Update(1/rate, q.f32(), f, gyro)
			if n.Q != q.f32() {
				differs = true
			}
		}
		for range int(rate) { // power-on and the first rest
			step([3]float64{})
		}
		if !n.calibrated {
			t.Fatalf("ExternalAttitude %v: not calibrated after 1 s at rest", ext)
		}
		for range int(2 * rate) { // a rest long enough for its levelling
			step([3]float64{})
		}
		var o [3]float64
		for k := range 3 {
			o[k] = float64(off[k])
		}
		ow := q.rot(o) // the offsets as the Nav sees them, in the world
		bh := math.Hypot(float64(n.Bias[0])-ow[0], float64(n.Bias[1])-ow[1])
		if ext && (bh > 0.01 || norm(n.A) > 0.01) {
			t.Errorf("ExternalAttitude, 2 s into a rest: horizontal Bias (%.3f, %.3f) m/s², want the offsets (%.3f, %.3f) ± 0.01; |A| %.3f m/s², want ≤ 0.01", n.Bias[0], n.Bias[1], ow[0], ow[1], norm(n.A))
		}
		if !ext && bh < 0.05 {
			t.Errorf("default, 2 s into a rest: horizontal Bias (%.3f, %.3f) m/s² still holds the offsets; the rest's levelling should have moved them into the attitude", n.Bias[0], n.Bias[1])
		}
		for range int(rate) { // a turn
			step([3]float64{0, 0.3, 0})
		}
		if ext && differs {
			t.Errorf("ExternalAttitude: Q left the supplied attitude")
		}
		if !ext && !differs {
			t.Errorf("default: Q stayed the supplied attitude through the rest's levelling and a biased turn (the option would test nothing)")
		}
	}
}

// With ExternalAttitude a stroke still tracks: 10 cm up in 1 s with the
// true attitude supplied reads 10 cm, and the Nav comes to rest after it.
func TestExternalAttitudeStrokeTracks(t *testing.T) {
	c := DefaultConfig()
	c.ExternalAttitude = true
	q := axisAngle([3]float64{1, 0, 0}, math.Pi/2)
	n := settled(t, c, q)
	const T, d = 1.0, 0.10
	for i := range int(2 * T * rate) {
		a := 0.0
		if i < int(T*rate) { // the stroke, then still for T
			_, _, a = stroke(float64(i)/rate, T, d)
		}
		n.Update(1/rate, q.f32(), v32(sensorForce(q, [3]float64{0, 0, a})), [3]float32{})
	}
	if got := float64(n.P[2]); math.Abs(got-d) > 0.01 || !n.Still {
		t.Errorf("10 cm up in 1 s: P z %.3f m, still %v (want %.2f ± 0.01 m and still)", got, n.Still, d)
	}
}

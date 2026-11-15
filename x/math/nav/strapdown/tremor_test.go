package strapdown

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/filter/ahrs"
	"github.com/itohio/EasyRobot/x/math/mat"
)

// shakyHand is a model of a shaky hand holding the IMU, fitted to a
// recorded hand-held MPU6050 (its first, shakiest 10 s): the gyro's tremor is a
// real rotation θ (θ' = w − θ/s, w correlated sample to sample, lag-1 0.92,
// sd 0.027 rad/s) which the accelerometer sees, and the hand's translation
// a damped 8 Hz oscillation (its acceleration sd 0.09 m/s², position
// bounded); white noise 0.016 m/s² and 0.0006 rad/s, the accelerometer
// offset and the gyro biased.
type shakyHand struct {
	rnd    lcg
	tg, th [3]float64
	ox, ov [3]float64
	unit   float64 // the oscillator's acceleration sd for a unit drive
}

const (
	handRho  = 0.92
	handLam  = 1.0 // 1/s
	handW0   = 2 * math.Pi * 8
	handZeta = 0.3
	handA    = 0.09  // m/s²
	handG    = 0.027 // rad/s
)

func newShakyHand(seed int) *shakyHand {
	h := &shakyHand{rnd: lcg(101 + 37*seed)}
	r := lcg(7)
	var x, v, sq float64
	for i := range 100000 {
		a := -handW0*handW0*x - 2*handZeta*handW0*v
		v += (a + gauss(&r)) / rate
		x += v / rate
		if i > 1000 {
			sq += a * a
		}
	}
	h.unit = math.Sqrt(sq / (100000 - 1001))
	return h
}

// sample is one reading of an upright badge (up: the true attitude without
// the tremor) and the tremor's true attitude.
func (h *shakyHand) sample(up quat) (f, w [3]float32, q quat) {
	var rate3, ta [3]float64
	for k := range 3 {
		h.tg[k] = handRho*h.tg[k] + math.Sqrt(1-handRho*handRho)*gauss(&h.rnd)*handG
		rate3[k] = h.tg[k] - handLam*h.th[k]
		h.th[k] += rate3[k] / rate
		a := -handW0*handW0*h.ox[k] - 2*handZeta*handW0*h.ov[k]
		h.ov[k] += (a + gauss(&h.rnd)*handA/h.unit) / rate
		h.ox[k] += h.ov[k] / rate
		ta[k] = a
	}
	th := math.Sqrt(h.th[0]*h.th[0] + h.th[1]*h.th[1] + h.th[2]*h.th[2])
	q = up
	if th > 0 {
		q = up.mul(axisAngle(h.th, th))
	}
	fs := sensorForce(q, ta)
	off := [3]float64{0.04, -0.06, 0.05}
	gb := [3]float64{0.01, -0.006, 0.008}
	for k := range 3 {
		f[k] = float32(fs[k] + off[k] + gauss(&h.rnd)*0.016)
		w[k] = float32(gb[k] + rate3[k] + gauss(&h.rnd)*0.0006)
	}
	return f, w, q
}

// Held in a shaky hand, the Nav must reach its first rest; until then it
// holds the origin and nothing moves. The first rest asks for a steady
// force and a settled attitude. The tremor's 8 Hz part took the force out
// of StillAccel of its average on about 30 % of samples, and with it the
// 0.8 s of unbroken rest the first rest needs. With steadiness judged on
// the force with the tremor averaged out (over AvgTime/8), every seed rests
// within 10 s (attitude from a 6-axis Mahony, KP 2, fed from power-on).
func TestFirstRestInAShakyHand(t *testing.T) {
	up := axisAngle([3]float64{1, 0, 0}, math.Pi/2)
	for seed := range 8 {
		n := newNav(t, DefaultConfig())
		flt := ahrs.NewMahony(ahrs.WithKP(2), ahrs.WithKI(0))
		var in mat.Matrix3x3
		h := newShakyHand(seed)
		at := -1.0
		for i := range int(10 * rate) {
			f, w, _ := h.sample(up)
			in[0], in[1] = f, w
			flt.Update(1/rate, &in)
			n.Update(1/rate, [4]float32(flt.Output()), f, w)
			if n.calibrated {
				at = float64(i) / rate
				break
			}
		}
		if at < 0 {
			t.Errorf("seed %d: no first rest within 10 s in a shaky hand", seed)
		}
	}
}

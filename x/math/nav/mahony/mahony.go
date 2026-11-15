// Package mahony is a Mahony attitude filter (complementary filter on the
// rotation group) for a 6-axis IMU: the gyro carries the attitude and the
// accelerometer's direction pulls its tilt towards gravity, proportionally
// (Kp) and, optionally, through an integral that learns a gyro bias (Ki).
// It is an estimator.Attitude brick, the same shape as eskf's motion
// estimator without the velocity and position. UpdateMag adds a
// magnetometer, which pulls the heading as gravity pulls the tilt. With
// -tags fixedmotion the 6-axis update runs in fixed point (for cores without
// an FPU, the RP2040's Cortex-M0+), within 1e-4 rad of the float one.
//
// A Filter is a value: Init and the updates allocate nothing.
package mahony

import (
	"errors"

	"github.com/chewxy/math32"

	"github.com/itohio/EasyRobot/x/math/nav/estimator"
)

// ErrConfig reports an invalid Config.
var ErrConfig = errors.New("mahony: invalid configuration")

// Config holds the gains.
type Config struct {
	Kp float32 // 1/s: the tilt's pull towards the accelerometer's gravity (2: an error halves in about 0.35 s)
	Ki float32 // 1/s²: the integral's, which learns a gyro bias (0: none)
}

// DefaultConfig is the badge's: Kp 2, no integral.
func DefaultConfig() Config { return Config{Kp: 2} }

func (c *Config) valid() bool {
	return c.Kp >= 0 && c.Ki >= 0 && !math32.IsInf(c.Kp+c.Ki, 0)
}

// Filter is a Mahony filter; it implements estimator.Attitude. Rejected
// counts the samples Update refused.
type Filter struct {
	Rejected uint32

	cfg   Config
	q     [4]float32 // sensor to world, z up
	eInt  [3]float32 // ∫e dt, the integral of the error (rad·s)
	gPrev [3]float32 // the last accepted sample's rate
	ready bool
	fx    fixed6 // the fixed-point 6-axis state (-tags fixedmotion; empty otherwise)
}

var _ estimator.Attitude = (*Filter)(nil)

// Init checks c and sets *f, in place, to a filter that starts at its first
// accepted sample; an invalid c is refused (ErrConfig) and leaves *f as it
// was. It allocates nothing and restarts a used filter.
func Init(f *Filter, c Config) error {
	if !c.valid() {
		return ErrConfig
	}
	*f = Filter{}
	f.cfg = c
	f.q = [4]float32{1, 0, 0, 0}
	return nil
}

// New is Init into a Filter returned by value.
func New(c Config) (Filter, error) {
	var f Filter
	err := Init(&f, c)
	return f, err
}

// Reset forgets the attitude and the integral; the next accepted sample
// starts the filter again.
func (f *Filter) Reset() { _ = Init(f, f.cfg) }

// Attitude is the quaternion [w,x,y,z] from the sensor frame to the world.
func (f *Filter) Attitude() [4]float32 { return f.q }

// Ready is true once a sample has set the attitude.
func (f *Filter) Ready() bool { return f.ready }

// Update takes one sample (estimator.Attitude): the first accepted one
// levels the attitude from its force (heading 0); each later one turns it
// by the rate over the step, the mean of the two samples' rates (a gyro
// reads the rate at the sample, the convention eskf's prediction uses),
// corrected by the tilt error. A sample with a NaN or an infinity, or a dt
// not above zero, is refused (estimator.ErrSample, Rejected) and changes
// nothing.
func (f *Filter) Update(dt float32, acc, gyro [3]float32) error {
	if !finite(dt, acc, gyro) {
		f.Rejected++
		return estimator.ErrSample
	}
	if !f.ready {
		f.level(acc, gyro)
		return nil
	}
	f.step(acc, gyro, dt)
	return nil
}

// UpdateMag is Update with a magnetometer's reading mag (any unit: only its
// direction counts): the error adds the measured field's direction against
// the field the attitude predicts (its horizontal part and dip taken from
// mag itself), so the heading is corrected too. The first accepted sample
// levels the attitude from acc (heading 0); the magnetometer then pulls the
// heading. Refused like Update.
func (f *Filter) UpdateMag(dt float32, acc, gyro, mag [3]float32) error {
	if !finite(dt, acc, gyro) || mag[0]-mag[0] != 0 || mag[1]-mag[1] != 0 || mag[2]-mag[2] != 0 {
		f.Rejected++
		return estimator.ErrSample
	}
	if !f.ready {
		f.level(acc, gyro)
		return nil
	}
	f.turn(f.correctedMag(acc, f.mid(gyro), mag, dt), dt)
	f.gPrev = gyro
	return nil
}

// level starts the filter at the attitude acc levels (heading 0); a zero
// force leaves it waiting.
func (f *Filter) level(acc, gyro [3]float32) {
	if q, ok := estimator.Level(acc); ok {
		f.q, f.gPrev, f.ready = q, gyro, true
	}
}

// mid is the rate over the step: the mean of the last sample's rate and
// gyro's.
func (f *Filter) mid(gyro [3]float32) [3]float32 {
	return [3]float32{0.5 * (gyro[0] + f.gPrev[0]), 0.5 * (gyro[1] + f.gPrev[1]), 0.5 * (gyro[2] + f.gPrev[2])}
}

// stepFloat is the 6-axis update in float32, the reference the fixed-point
// one follows: the attitude turns by the step's corrected rate.
func (f *Filter) stepFloat(acc, gyro [3]float32, dt float32) {
	f.turn(f.corrected(acc, f.mid(gyro), dt), dt)
	f.gPrev = gyro
}

// corrected is the rate with the feedback: e = a × v, the measured
// gravity's direction a against the estimated one v (the world's ẑ seen in
// the sensor frame), Kp·e + Ki·∫e dt added to the gyro's rate. A zero force
// gives no correction.
func (f *Filter) corrected(acc, gyro [3]float32, dt float32) [3]float32 {
	n := math32.Sqrt(acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2])
	if !(n > 0) {
		return gyro
	}
	ax, ay, az := acc[0]/n, acc[1]/n, acc[2]/n
	w, x, y, z := f.q[0], f.q[1], f.q[2], f.q[3]
	v := [3]float32{2 * (x*z - w*y), 2 * (w*x + y*z), w*w - x*x - y*y + z*z}
	e := [3]float32{ay*v[2] - az*v[1], az*v[0] - ax*v[2], ax*v[1] - ay*v[0]}
	var r [3]float32
	for i := range 3 {
		if f.cfg.Ki > 0 {
			f.eInt[i] += e[i] * dt
		}
		r[i] = gyro[i] + f.cfg.Kp*e[i] + f.cfg.Ki*f.eInt[i]
	}
	return r
}

// correctedMag is corrected with the magnetometer's error added: the
// measured field's direction m against the one the attitude predicts, w,
// from the field's horizontal part bx and its vertical bz in the world.
func (f *Filter) correctedMag(acc, gyro, mag [3]float32, dt float32) [3]float32 {
	an := math32.Sqrt(acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2])
	mn := math32.Sqrt(mag[0]*mag[0] + mag[1]*mag[1] + mag[2]*mag[2])
	if !(an > 0) || !(mn > 0) {
		return f.corrected(acc, gyro, dt)
	}
	ax, ay, az := acc[0]/an, acc[1]/an, acc[2]/an
	mx, my, mz := mag[0]/mn, mag[1]/mn, mag[2]/mn
	q1, q2, q3, q4 := f.q[0], f.q[1], f.q[2], f.q[3]
	hx := 2*mx*(0.5-q3*q3-q4*q4) + 2*my*(q2*q3-q1*q4) + 2*mz*(q2*q4+q1*q3)
	hy := 2*mx*(q2*q3+q1*q4) + 2*my*(0.5-q2*q2-q4*q4) + 2*mz*(q3*q4-q1*q2)
	bx := math32.Sqrt(hx*hx + hy*hy)
	bz := 2*mx*(q2*q4-q1*q3) + 2*my*(q3*q4+q1*q2) + 2*mz*(0.5-q2*q2-q3*q3)
	v := [3]float32{2 * (q2*q4 - q1*q3), 2 * (q1*q2 + q3*q4), q1*q1 - q2*q2 - q3*q3 + q4*q4}
	w := [3]float32{
		2*bx*(0.5-q3*q3-q4*q4) + 2*bz*(q2*q4-q1*q3),
		2*bx*(q2*q3-q1*q4) + 2*bz*(q1*q2+q3*q4),
		2*bx*(q1*q3+q2*q4) + 2*bz*(0.5-q2*q2-q3*q3),
	}
	e := [3]float32{
		(ay*v[2] - az*v[1]) + (my*w[2] - mz*w[1]),
		(az*v[0] - ax*v[2]) + (mz*w[0] - mx*w[2]),
		(ax*v[1] - ay*v[0]) + (mx*w[1] - my*w[0]),
	}
	var r [3]float32
	for i := range 3 {
		if f.cfg.Ki > 0 {
			f.eInt[i] += e[i] * dt
		}
		r[i] = gyro[i] + f.cfg.Kp*e[i] + f.cfg.Ki*f.eInt[i]
	}
	return r
}

// turn steps the attitude by the rate w over dt, an exact rotation by the
// angle θ = |w|·dt about w, as eskf's: q ← q ⊗ [cos(θ/2), sin(θ/2)·w/|w|],
// normalised against rounding. (The published filter's first-order step
// q ⊗ [1, w·dt/2] turns by 2·atan(θ/2), short by θ³/12: 0.75 % of a
// 30 rad/s spin at 100 Hz.)
func (f *Filter) turn(w [3]float32, dt float32) {
	a := 0.5 * dt * math32.Sqrt(w[0]*w[0]+w[1]*w[1]+w[2]*w[2]) // θ/2
	c, sa := 1-a*a/2, 1-a*a/6                                  // cos(θ/2), sin(θ/2)/(θ/2), to a⁴ below 1e-3
	if a >= 1e-3 {
		var sn float32
		sn, c = math32.Sincos(a)
		sa = sn / a
	}
	h := 0.5 * dt * sa
	q := f.q
	f.q = [4]float32{
		c*q[0] - h*(q[1]*w[0]+q[2]*w[1]+q[3]*w[2]),
		c*q[1] + h*(q[0]*w[0]+q[2]*w[2]-q[3]*w[1]),
		c*q[2] + h*(q[0]*w[1]-q[1]*w[2]+q[3]*w[0]),
		c*q[3] + h*(q[0]*w[2]+q[1]*w[1]-q[2]*w[0]),
	}
	k := 1 / math32.Sqrt(f.q[0]*f.q[0]+f.q[1]*f.q[1]+f.q[2]*f.q[2]+f.q[3]*f.q[3])
	for i := range 4 {
		f.q[i] *= k
	}
}

func finite(dt float32, acc, gyro [3]float32) bool {
	if !(dt > 0) || math32.IsInf(dt, 0) {
		return false
	}
	for i := range 3 {
		if acc[i]-acc[i] != 0 || gyro[i]-gyro[i] != 0 {
			return false
		}
	}
	return true
}

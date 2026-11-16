// Package eskf is an error-state (indirect) Kalman filter for a body's
// motion from a 6-axis IMU: a motion estimator and nothing else. It
// estimates the attitude, velocity and position in the world (z up)
// together with the gyro's and the accelerometer's biases and their
// covariance. It decides nothing: no rest, turn or glide is judged inside,
// and nothing outside is fed back into it.
//
// Its parts each do one thing: State is the nominal state and its
// mechanisation (state.go), covariance the error state's covariance and
// its scalar update (covariance.go), tilt the one measurement model
// (tilt.go), inputs the samples' validity and clip limits (input.go).
// Filter sequences them and is an estimator.Motion: the same shape as an
// attitude brick (mahony), with a velocity and a position.
//
// The nominal state is propagated from every sample: the attitude by the
// rate over the step, the force turned into the world with the attitude at
// the step's end. The error state (attitude δθ in the world, δv, δp, δb_g,
// δb_a: 15) is propagated as Φ·P·Φᵀ + Q·dt. Every TiltEvery samples the
// force, turned into the world, is measured against ẑ: gravity's
// direction, with a noise that grows with the force's distance from g;
// it corrects the attitude and the biases, never the velocity or the
// position. The first accepted sample sets roll and pitch from its force.
//
// Position has no measurement: an acceleration error δa (a tilt error δθ
// gives g·δθ) moves it by ½·δa·t², smoothly, without bound.
//
// A Filter is a value with every array in it: Update allocates nothing.
package eskf

import (
	"errors"

	"github.com/chewxy/math32"

	"github.com/itohio/EasyRobot/x/math/nav/calib"
	"github.com/itohio/EasyRobot/x/math/nav/estimator"
)

// StandardGravity in m/s².
const StandardGravity = 9.80665

// n is the error state's size.
const n = 15

// offsets in the error state
const (
	iTh = 0
	iV  = 3
	iP  = 6
	iBg = 9
	iBa = 12
)

// ErrConfig reports an invalid Config.
var ErrConfig = errors.New("eskf: invalid configuration")

// Config holds the noise model, the tilt schedule and the full scales.
type Config struct {
	Gravity       float32 // m/s²: local gravity as the calibrated accelerometer reads it, the mean corrected |f| at rest a calibration measures (StandardGravity without one)
	AccelNoise    float32 // accelerometer white noise density, m/s²/√Hz
	GyroNoise     float32 // gyro white noise density, rad/s/√Hz
	AccelBiasWalk float32 // accelerometer bias random walk, m/s² per √s
	GyroBiasWalk  float32 // gyro bias random walk, rad/s per √s
	TiltNoise     float32 // the tilt measurement's sd when the force's magnitude is g, rad (0.014: a slow stroke stays a translation with the bias's tilt in the covariance; at 0.01 a 5 cm stroke 3 s after the start read up to 16 mm off)
	LinAccelTime  float32 // s: the body's own acceleration as tilt noise: its power (the force's variance over the last half second above the sensor's white noise) times this is a white-equivalent spectral density; it sets how long the tilt waits while the body moves (0: none)
	TiltEvery     int     // samples between tilt updates
	AccelRange    float32 // m/s²: the accelerometer's full scale; a reading at it is clipped (Update) (0: no check)
	GyroRange     float32 // rad/s: the gyro's full scale, likewise
}

// DefaultConfig is the badge's MPU6050 at 100 Hz (white noise 0.016 m/s²
// and 0.0006 rad/s per sample; gyro bias wander 0.0007 rad/s between still
// stretches in a hand-held capture), at ±2 g and ±2000 °/s, the ranges
// of the recorded shaking capture (the MPU6050 powers up at ±2 g and
// ±250 °/s); a sensor set to another range sets its own (SetRange).
func DefaultConfig() Config {
	return Config{
		Gravity: StandardGravity, AccelNoise: 0.0016, GyroNoise: 0.00006,
		AccelBiasWalk: 0.0005, GyroBiasWalk: 0.0002,
		TiltNoise: 0.014, LinAccelTime: 96, TiltEvery: 4,
		AccelRange: 2 * StandardGravity, GyroRange: 2000 * math32.Pi / 180,
	}
}

func (c *Config) valid() bool {
	pos := c.Gravity > 0 && c.AccelNoise > 0 && c.GyroNoise > 0 && c.AccelBiasWalk > 0 && c.GyroBiasWalk > 0 &&
		c.TiltNoise > 0 && c.LinAccelTime >= 0 && c.AccelRange >= 0 && c.GyroRange >= 0
	sum := c.Gravity + c.AccelNoise + c.GyroNoise + c.AccelBiasWalk + c.GyroBiasWalk + c.TiltNoise + c.LinAccelTime + c.AccelRange + c.GyroRange
	return pos && c.TiltEvery >= 1 && !math32.IsInf(sum, 0) && !math32.IsNaN(sum)
}

// Filter is the estimate after each Update: the embedded State (Q the
// attitude [w,x,y,z], sensor to world, z up; V and P the world velocity
// (m/s) and position (m); GyroBias and AccelBias in the sensor frame);
// Rejected and Clipped count the samples Update refused (not finite, or dt
// not above zero) and those with a reading at a full scale. It implements
// estimator.Motion (Attitude, Velocity, Position, Ready).
type Filter struct {
	State
	Rejected uint32
	Clipped  uint32

	cfg   Config
	in    inputs
	cov   covariance
	dx    [n]float32  // error accumulated by the updates, injected after
	tilts int         // samples since the last tilt update
	span  float32     // s since the last tilt update
	power motionPower // the force's recent variance: the body's acceleration for the tilt's noise
	gPrev [3]float32  // the last accepted sample's rate
	gMid  [3]float32  // the rate over the step: the mean of that and this sample's (a gyro reads the rate at the sample)
	gb0   [3]float32  // the gyro bias's prior (Prior; zero without one), to which Reset returns
	gbP   [3]float32  // the gyro bias's prior variance
	ready bool        // the first accepted sample has set the tilt
	baP   float32     // the accelerometer bias's prior variance: 0.1 m/s² without a multi-pose calibration (an MPU6050's zero-g offset reaches 0.3), or that calibration's
}

// Init checks c and sets *f, in place, to a filter that starts at its
// first accepted sample; an invalid c is refused (ErrConfig) and leaves
// *f as it was. It allocates nothing and builds no second Filter: on a
// microcontroller a Filter (2.9 KB) returned by value is a second block
// for the copy. Init also restarts a used filter from scratch, its prior
// and ranges included.
func Init(f *Filter, c Config) error {
	if !c.valid() {
		return ErrConfig
	}
	*f = Filter{}
	f.cfg = c
	f.gbP = [3]float32{4e-4, 4e-4, 4e-4}
	f.baP = 0.01
	f.State.reset([3]float32{})
	f.in.fit([3]float32{}, [3]float32{1, 1, 1})
	f.in.ranges(c.AccelRange, c.GyroRange)
	return nil
}

var _ estimator.Motion = (*Filter)(nil)

// New is Init into a Filter returned by value, for a caller that can
// afford the copy.
func New(c Config) (Filter, error) {
	var f Filter
	err := Init(&f, c)
	return f, err
}

// Prior seeds the biases from a multi-pose calibration (calib.Solver) whose
// Apply the caller runs on the accelerometer's samples: the gyro bias from
// its still poses, known to 2 mrad/s; the accelerometer's residual bias at
// zero, known to σ, the largest of accelBiasSigma, the fit's RMS and
// 0.01 m/s². accelBiasSigma (m/s²) is how far the caller expects the bias
// to have moved since the fit (between sessions, with temperature): an
// exactly determined fit (six poses, six unknowns) has an RMS of 0 and
// says nothing of that; 0, a negative, NaN or infinite value adds nothing,
// and likewise a NaN or infinite RMS (a saved fit can carry anything).
// A non-finite offset in the fit counts as 0 for the clip limits.
// The accelerometer's full scale is then taken through the fit too: a
// reading at the sensor's full scale arrives as (±AccelRange −
// offset)/scale, which differs per axis and sign. Call it before the first
// Update (or after Reset).
func (f *Filter) Prior(fit *calib.Fit, accelBiasSigma float32) {
	f.GyroBias, f.gb0 = fit.GyroBias, fit.GyroBias
	f.gbP = [3]float32{4e-6, 4e-6, 4e-6}
	s := float32(0.01)
	for _, v := range [2]float32{fit.RMS, accelBiasSigma} {
		if v > s && !math32.IsInf(v, 0) {
			s = v
		}
	}
	f.baP = s * s
	f.AccelBias = [3]float32{}
	f.in.fit(fit.Offset, fit.Scale)
	f.in.ranges(f.cfg.AccelRange, f.cfg.GyroRange)
}

// SetRange changes the full scales (Config.AccelRange, GyroRange) in place,
// for a sensor whose range is switched: a reading at the new full scale
// (through the Prior's fit, if any) is clipped from the next sample on.
// The ranges are the sensor's own, as for New, in either order with Prior:
// a range already taken through the fit would be taken through it twice.
// A negative, NaN or infinite range is refused (ErrConfig) and changes
// nothing; 0 turns the check off.
func (f *Filter) SetRange(accel, gyro float32) error {
	c := f.cfg
	c.AccelRange, c.GyroRange = accel, gyro
	if !c.valid() {
		return ErrConfig
	}
	f.cfg = c
	f.in.ranges(accel, gyro)
	return nil
}

// Reset forgets the motion (attitude, velocity, position) and returns the
// biases to the prior (Prior's, or zero); the next accepted sample starts
// it again. Keeping them would keep a bias corrupted by a glitch (a gyro
// spike inside its range).
func (f *Filter) Reset() {
	f.State.reset(f.gb0)
	f.ready, f.tilts, f.span = false, 0, 0
	f.dx = [n]float32{}
}

// Attitude is Q (estimator.Attitude).
func (f *Filter) Attitude() [4]float32 { return f.Q }

// Velocity is V (estimator.Motion).
func (f *Filter) Velocity() [3]float32 { return f.V }

// Position is P (estimator.Motion).
func (f *Filter) Position() [3]float32 { return f.P }

// Ready is true once the first accepted sample has set the tilt.
func (f *Filter) Ready() bool { return f.ready }

// Update takes one sample: dt (s), acc (m/s²) and gyro (rad/s) in the
// sensor frame (estimator.Attitude). A sample with a NaN or an infinity, or
// a dt not above zero, is refused (estimator.ErrSample, Rejected) without
// touching the state. One with a reading at a
// full scale is clipped (Clipped): that reading's true value is beyond the
// scale, by an unknown amount. A clipped gyro loses the step's rotation:
// the sample is skipped and the attitude's variance grows by the full
// scale's angle over the step, (GyroRange·dt)², on each axis. A clipped
// accelerometer alone keeps the gyro's rotation: the step is propagated
// with the clipped force (a bound), the velocity's variance grows by
// (AccelRange·dt)², and the force isn't taken as a tilt. (Skipping such a
// sample whole lost the rotation with the covariance unaware: after a
// shake the attitude was 80° off at σ 0.2°, and the put-down's tilt went
// into the accelerometer bias.)
func (f *Filter) Update(dt float32, acc, gyro [3]float32) error {
	ac, gc, ok := f.validate(dt, acc, gyro)
	switch {
	case !ok && !(ac || gc):
		return estimator.ErrSample
	case !ok:
	case !f.ready:
		f.start(acc, gyro)
	case gc:
		f.cov.grow(iTh, f.cfg.GyroRange*dt) // the step's rotation is lost
	default:
		f.power.add(acc, dt)
		f.predict(dt, acc, gyro)
		if ac {
			f.cov.grow(iV, f.cfg.AccelRange*dt) // the force is only a bound
		} else if f.tiltDue(dt) && f.tiltUpdate(acc) {
			f.inject()
		}
	}
	return nil
}

// validate counts and classifies a sample: ok false for one refused (a NaN
// or an infinity, dt not above zero: Rejected) and for one clipped before
// the first accepted sample; ac and gc whether the accelerometer's and the
// gyro's readings are at a full scale (Clipped).
func (f *Filter) validate(dt float32, acc, gyro [3]float32) (ac, gc, ok bool) {
	if !finiteSample(dt, acc, gyro) {
		f.Rejected++
		return false, false, false
	}
	ac, gc = f.in.clipped(acc, gyro)
	if ac || gc {
		f.Clipped++
	}
	return ac, gc, f.ready || !(ac || gc)
}

// tiltDue counts the samples to the next tilt update: every TiltEvery.
func (f *Filter) tiltDue(dt float32) bool {
	f.tilts++
	f.span += dt
	if f.tilts < f.cfg.TiltEvery {
		return false
	}
	f.tilts = 0
	return true
}

// start sets the tilt from the first accepted sample's force (yaw 0) and
// the covariance to its initial values: the attitude to the force's own
// uncertainty as gravity's direction (TiltNoise, and how far its magnitude
// is from g), the yaw unknown, the velocity to 0.1 m/s (a hand's), the
// position to the origin.
func (f *Filter) start(acc, gyro [3]float32) {
	var m [3]float32
	for i := range 3 {
		m[i] = acc[i] - f.AccelBias[i]
	}
	q, ok := estimator.Level(m)
	if !ok {
		return
	}
	g := math32.Sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2])
	f.Q = q
	f.refresh()
	f.V, f.P = [3]float32{}, [3]float32{}
	e := (g - f.cfg.Gravity) / f.cfg.Gravity
	att := f.cfg.TiltNoise*f.cfg.TiltNoise + e*e
	var d [n]float32
	for i := range 3 {
		d[iTh+i], d[iV+i], d[iP+i], d[iBg+i], d[iBa+i] = att, 1e-2, 1e-6, f.gbP[i], f.baP
	}
	d[iTh+2] = 1
	f.cov.init(&d)
	f.gPrev, f.tilts, f.span = gyro, 0, 0
	f.power.reset(acc)
	f.ready = true
}

// predict propagates the nominal state by one sample and the covariance
// with it: the attitude first, by the rate over the step (gMid: the mean of
// the last two samples' rates; a gyro reads the rate at the sample); the
// sample's force is the body's at the sample, the step's end (rotated with
// the attitude at the step's start, a turn leaked g·ω·dt of gravity into
// the world acceleration: a turn in place with exact samples moved 2–17 cm).
func (f *Filter) predict(dt float32, acc, gyro [3]float32) {
	var w, a [3]float32
	for i := range 3 {
		f.gMid[i] = 0.5 * (gyro[i] + f.gPrev[i])
		w[i] = f.gMid[i] - f.GyroBias[i]
		a[i] = acc[i] - f.AccelBias[i]
	}
	f.gPrev = gyro
	f.turn(w, dt)
	aw := f.move(a, f.cfg.Gravity, dt)
	f.cov.propagate(&f.r, aw, &f.cfg, dt) // with the attitude at the step's end (r), as the force
}

// tiltUpdate applies the tilt measurement model's two rows to the
// accumulated error; false for a zero force (nothing measured).
func (f *Filter) tiltUpdate(acc [3]float32) bool {
	lin := f.power.linear(f.cfg.AccelNoise, f.span/float32(f.cfg.TiltEvery))
	rows, ys, rv, ok := tiltModel(&f.r, f.AccelBias, acc, &f.cfg, f.span, lin)
	f.span = 0
	if !ok {
		return false
	}
	for k := range 2 {
		f.cov.update(&rows[k], ys[k], rv, tiltConsider, &f.dx)
	}
	return true
}

// inject moves the accumulated error into the nominal state and keeps the
// covariance usable.
func (f *Filter) inject() {
	f.correct(&f.dx)
	f.dx = [n]float32{}
	f.cov.guard()
}

// Sigma returns the standard deviations of the errors: the attitude's about
// the world axes (rad: att[0] and att[1] the tilt about world x and y, att[2]
// the yaw, which nothing observes), the velocity's (m/s) and the position's
// (m), world x, y, z.
func (f *Filter) Sigma() (att, vel, pos [3]float32) {
	for i := range 3 {
		att[i] = f.cov.sd(iTh + i)
		vel[i] = f.cov.sd(iV + i)
		pos[i] = f.cov.sd(iP + i)
	}
	return
}

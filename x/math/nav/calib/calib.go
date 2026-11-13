// Package calib calibrates a motion sensor (a 3-axis accelerometer and gyro)
// at rest, for the badge's startup page: fed one sample per call, it finds
// the face that is down (lying flat or standing on any side), waits until
// the sensor is still, averages, and starts again when moved. It reports
// the gyro bias, the accelerometer's readings at rest, the noise of both,
// a forecast of the drift to expect in motion, and live per-axis statistics
// for the page to draw while it runs.
//
// A run: Waiting until every axis has looked still for Window (its spread
// under the sensor's threshold and its mean within the threshold of where
// it was when it began to look still); then Measuring, for Measure, while a
// plain mean and variance accumulate (Progress rises from 0 to 1); any axis
// failing that test turns it Moved for MovedHold, then Waiting, from 0.
// Then Done: Result holds the run's figures until Reset (the live
// statistics keep running).
//
// The live statistics (Accel, Gyro) are an exponentially weighted mean and
// standard deviation with time constant Window, in sensor units (m/s²,
// rad/s): what the page's strips draw. The face down is the sensor axis
// with the largest mean force when measuring begins, and AccelRest is +G
// on it (−G if it points down), 0 on the others.
//
// One pose cannot tell an accelerometer's offset from the surface's tilt:
// lying on a table tilted by 1°, a horizontal axis reads its offset plus
// g·sin 1° (0.17 m/s², more than an MPU6050's typical offset), and the up
// axis reads its offset plus the sensor's scale error at 1 g (an MPU6050's
// ±3 % is 0.29 m/s²). Result.AccelOffset reports these readings as they
// are; they are for display, not for correcting samples. The gyro bias, a
// rate at rest, has no such ambiguity; strapdown.Nav.Seed takes it, and
// the Nav learns the accelerometer's offsets across turns by itself. (The
// badge does not seed yet: fpv-gate found seeding widens strokes made
// right after a turn; strapdown r11 looks into it.) The sensor's hardware
// offset registers are not used: the result applies to any IMU, needs no
// bus writes, is testable on the host, and the registers would hold the
// same tilt-confounded figures.
//
// The forecast (Result.Forecast), the model in full:
//
//   - Heading: the gyro bias left about the up axis turns the heading at a
//     constant rate. The mean of N samples is off by σg/√N, and the bias
//     may wander: half the change between the measurement's two halves
//     shows it, but holds that same noise, so it counts only where larger.
//     HeadingDrift = max(σg/√N, |m₂ − m₁|/2)·60 s, in °/min (added, the
//     two counted the noise twice: 1.8 times the bias the runs left).
//     HeadingNoise is the angle random walk after a minute, σg·√(dt·60 s).
//   - Position: strapdown holds the position at rest and integrates only
//     while the badge moves. A minute of play is taken as MotionsPerMin
//     motions of MotionTime each, at rest between for long enough that
//     strapdown takes the rest (it waits up to MaxGlide, 2.5 s, after a
//     stroke whose speed left over is above StillSpeed; with 2 s rests
//     between 1 s strokes at an MPU6050's noise it found none after the
//     second and strayed by metres). Strapdown integrates each motion and
//     the StillTime (0.3 s) it takes to see the rest after it:
//     T = MotionTime + 0.3 s. Each integrates
//     the accelerometer's noise twice, σa·√(dt·T³/3) per axis, and these
//     add as a random walk over the motions and the three axes:
//     PosNoise = √(3·M)·σa·√(dt·T³/3). Each also starts from a bias learned
//     at the rest before it, off by σa/√n over the n samples of a rest
//     (strapdown averages over up to 4·0.5 s), which builds ½·e·T² of
//     position; these are taken as adding up over the motions:
//     PosDrift = M·½·(σa/√n)·T². Both in cm/min.
//   - The acceptable lines: 2 °/min of heading (tinygui's scale runs to 6)
//     and 5 cm/min of position (to 10): four and 1.5 times what an MPU6050
//     at 100 Hz gives (0.48 °/min and 3.3 cm/min).
//
// float32, no allocation per sample, no I/O; Calibrator is a small value.
package calib

import (
	"errors"

	"github.com/chewxy/math32"
)

// StandardGravity in m/s².
const StandardGravity = 9.80665

// seeRest is strapdown's StillTime: the s after a motion that it goes on
// integrating before it sees the rest (the forecast's T is the motion's and
// this; without it, the forecast read 0.64 of the path's stray).
const seeRest = 0.3

// ErrConfig is returned by New for an invalid Config.
var ErrConfig = errors.New("calib: invalid configuration")

// State of a calibration run.
type State uint8

const (
	Waiting   State = iota // not still yet (also after New and Reset)
	Measuring              // still: averaging, Progress rising
	Moved                  // moved while measuring: back to 0, for MovedHold, then Waiting
	Done                   // Result is valid
)

func (s State) String() string {
	switch s {
	case Waiting:
		return "waiting"
	case Measuring:
		return "measuring"
	case Moved:
		return "moved"
	case Done:
		return "done"
	}
	return "unknown"
}

// Config tunes a run. DefaultConfig suits an MPU6050 at 100 Hz on the badge.
type Config struct {
	G             float32 // gravity, m/s² (> 0)
	Window        float32 // s: the live statistics' time constant, and how long it must look still before measuring (> 0)
	Measure       float32 // s of averaging once still (> 0)
	AccelStill    float32 // m/s²: an axis is still while its spread is under this and its mean stays within it (> 0)
	GyroStill     float32 // rad/s: the same for the gyro (> 0)
	MovedHold     float32 // s the Moved state shows before Waiting (≥ 0)
	MotionsPerMin float32 // the forecast's motions per minute of play (> 0)
	MotionTime    float32 // s, the forecast's length of one motion (> 0)
}

// DefaultConfig: standard gravity; 0.5 s windows; 3 s of averaging; still
// under 0.06 m/s² and 0.01 rad/s (twice and five times an MPU6050's noise at
// 100 Hz; a hand holding it shakes by 0.1–0.3 m/s²); Moved shown for 0.5 s;
// the forecast's play is 15 motions of 1 s a minute, 3 s at rest between.
func DefaultConfig() Config {
	return Config{G: StandardGravity, Window: 0.5, Measure: 3, AccelStill: 0.06, GyroStill: 0.01, MovedHold: 0.5, MotionsPerMin: 15, MotionTime: 1}
}

// Stats are per-axis live statistics: the mean and the standard deviation,
// exponentially weighted over Config.Window.
type Stats struct{ Mean, Spread [3]float32 }

// Forecast is the drift to expect in motion after this calibration (the
// model is in the package doc): heading in °/min, position in cm/min, each
// as systematic drift and noise random walk, with the acceptable line.
type Forecast struct {
	HeadingDrift, HeadingNoise, HeadingOK float32
	PosDrift, PosNoise, PosOK             float32
}

// Result of a finished run.
type Result struct {
	GyroBias    [3]float32 // rad/s: the mean rate at rest; subtract it from gyro samples
	AccelOffset [3]float32 // m/s²: the mean force less AccelRest (see the package doc: display only)
	AccelNoise  [3]float32 // m/s²: standard deviation per axis at rest
	GyroNoise   [3]float32 // rad/s
	Forecast    Forecast
}

// Calibrator runs one calibration after another (Reset starts again).
type Calibrator struct {
	cfg        Config
	state      State
	live       float32    // samples in the live statistics, counted to the first Window
	warm       float32    // s of live statistics, counted to Window: no stillness before
	am, av     [3]float32 // live accel mean and variance
	gm, gv     [3]float32 // live gyro mean and variance
	aRef, gRef [3]float32 // live means when it began to look still
	stillFor   float32    // s it has looked still
	heldFor    float32    // s in Moved
	measFor    float32    // s measured
	n          float32    // samples measured
	aMean, aM2 [3]float32 // Welford over the measurement
	gMean, gM2 [3]float32
	gHalf      [3]float32 // gyro mean of the measurement's first half
	nHalf      float32
	up         int
	sign       int8
}

// New checks c and returns a Calibrator, Waiting.
func New(c Config) (Calibrator, error) {
	if !(c.G > 0) || !(c.Window > 0) || !(c.Measure > 0) || !(c.AccelStill > 0) || !(c.GyroStill > 0) ||
		!(c.MovedHold >= 0) || !(c.MotionsPerMin > 0) || !(c.MotionTime > 0) ||
		!finite(c.G+c.Window+c.Measure+c.AccelStill+c.GyroStill+c.MovedHold+c.MotionsPerMin+c.MotionTime) {
		return Calibrator{}, ErrConfig
	}
	return Calibrator{cfg: c, up: 2, sign: 1}, nil
}

// Reset starts a new run (A on the page: recalibrate); the live statistics
// carry on.
func (c *Calibrator) Reset() {
	c.state, c.stillFor, c.heldFor = Waiting, 0, 0
	c.restart()
}

// State is the run's state.
func (c *Calibrator) State() State { return c.state }

// Progress is how far measuring has got: 0 to 1 while Measuring, 1 when
// Done, 0 otherwise.
func (c *Calibrator) Progress() float32 {
	switch c.state {
	case Measuring:
		return min(c.measFor/c.cfg.Measure, 1)
	case Done:
		return 1
	}
	return 0
}

// Accel is the accelerometer's live statistics (m/s²).
func (c *Calibrator) Accel() Stats { return Stats{c.am, sqrt3(c.av)} }

// Gyro is the gyro's live statistics (rad/s).
func (c *Calibrator) Gyro() Stats { return Stats{c.gm, sqrt3(c.gv)} }

// Thresholds are the stillness thresholds, accelerometer (m/s²) and gyro
// (rad/s).
func (c *Calibrator) Thresholds() (accel, gyro float32) { return c.cfg.AccelStill, c.cfg.GyroStill }

// Up is the sensor axis that points up (0 x, 1 y, 2 z) and its sign: the
// face that is down, as found when measuring last began (z, + before).
func (c *Calibrator) Up() (axis int, sign int8) { return c.up, c.sign }

// AccelRest is the accelerometer's rest target per axis: ±G on the axis
// that points up, 0 on the others (the gyro's is 0).
func (c *Calibrator) AccelRest() [3]float32 {
	var r [3]float32
	r[c.up] = float32(c.sign) * c.cfg.G
	return r
}

// Result is the last finished run's, and whether there is one.
func (c *Calibrator) Result() (Result, bool) {
	if c.state != Done {
		return Result{}, false
	}
	return c.result(), true
}

// Update takes one sample: dt in s, the accelerometer's specific force in
// m/s² (+g up at rest) and the gyro rate in rad/s, both in the sensor
// frame. A sample with a non-positive or non-finite dt, or any non-finite
// value, is ignored.
func (c *Calibrator) Update(dt float32, accel, gyro [3]float32) {
	if !(dt > 0) || !finite(dt+accel[0]+accel[1]+accel[2]+gyro[0]+gyro[1]+gyro[2]) {
		return
	}
	cfg := &c.cfg
	// a plain running mean and variance of what has come until they span
	// Window, then exponentially weighted (from the first sample alone, the
	// mean was off by its noise and settled through the first measurement)
	if c.warm < cfg.Window {
		c.live++
		c.warm += dt
	}
	k := max(dt/cfg.Window, 1/c.live)
	for i := range 3 {
		d := accel[i] - c.am[i]
		c.am[i] += k * d
		c.av[i] = (1 - k) * (c.av[i] + k*d*d)
		d = gyro[i] - c.gm[i]
		c.gm[i] += k * d
		c.gv[i] = (1 - k) * (c.gv[i] + k*d*d)
	}
	still := c.still()

	switch c.state {
	case Waiting:
		if !still {
			c.stillFor, c.aRef, c.gRef = 0, c.am, c.gm
			return
		}
		c.stillFor += dt
		if c.stillFor >= cfg.Window {
			c.begin()
		}
	case Measuring:
		if !still {
			c.state, c.heldFor = Moved, 0 // Progress reads 0; begin clears the measurement
			return
		}
		c.n++
		for i := range 3 {
			d := accel[i] - c.aMean[i]
			c.aMean[i] += d / c.n
			c.aM2[i] += d * (accel[i] - c.aMean[i])
			d = gyro[i] - c.gMean[i]
			c.gMean[i] += d / c.n
			c.gM2[i] += d * (gyro[i] - c.gMean[i])
		}
		c.measFor += dt
		if c.nHalf == 0 && c.measFor >= cfg.Measure/2 {
			c.gHalf, c.nHalf = c.gMean, c.n
		}
		if c.measFor >= cfg.Measure-dt/2 {
			c.state = Done // the sums stay: Result reads them
		}
	case Moved:
		c.heldFor += dt
		if c.heldFor >= cfg.MovedHold {
			c.state, c.stillFor = Waiting, 0 // the first sample not still takes a new reference
		}
	}
}

// still: every axis's spread under its threshold and its mean within the
// threshold of where it was when it began to look still; never before the
// live statistics span Window.
func (c *Calibrator) still() bool {
	if c.warm < c.cfg.Window {
		return false
	}
	ta, tg := c.cfg.AccelStill, c.cfg.GyroStill
	for i := range 3 {
		if c.av[i] >= ta*ta || c.gv[i] >= tg*tg || abs(c.am[i]-c.aRef[i]) >= ta || abs(c.gm[i]-c.gRef[i]) >= tg {
			return false
		}
	}
	return true
}

// begin starts measuring, with the face down taken from the mean force.
func (c *Calibrator) begin() {
	c.state = Measuring
	c.restart()
	c.up = 0
	for i := 1; i < 3; i++ {
		if abs(c.am[i]) > abs(c.am[c.up]) {
			c.up = i
		}
	}
	c.sign = 1
	if c.am[c.up] < 0 {
		c.sign = -1
	}
}

// restart clears the measurement.
func (c *Calibrator) restart() {
	c.measFor, c.n, c.nHalf = 0, 0, 0
	c.aMean, c.aM2, c.gMean, c.gM2, c.gHalf = [3]float32{}, [3]float32{}, [3]float32{}, [3]float32{}, [3]float32{}
}

// result turns the finished measurement (its sums stay as they were at
// Done) into the Result and the forecast.
func (c *Calibrator) result() Result {
	cfg := &c.cfg
	rest := c.AccelRest()
	r := Result{GyroBias: c.gMean}
	var sa2 float32
	for i := range 3 {
		r.AccelOffset[i] = c.aMean[i] - rest[i]
		r.AccelNoise[i] = math32.Sqrt(c.aM2[i] / max(c.n-1, 1))
		r.GyroNoise[i] = math32.Sqrt(c.gM2[i] / max(c.n-1, 1))
		sa2 += r.AccelNoise[i] * r.AccelNoise[i]
	}
	dt := c.measFor / c.n
	const deg = 180 / math32.Pi
	sg := r.GyroNoise[c.up]
	// the second half's mean from the whole and the first half's
	m2 := (c.gMean[c.up]*c.n - c.gHalf[c.up]*c.nHalf) / max(c.n-c.nHalf, 1)
	r.Forecast.HeadingDrift = max(sg/math32.Sqrt(c.n), abs(m2-c.gHalf[c.up])/2) * 60 * deg
	r.Forecast.HeadingNoise = sg * math32.Sqrt(dt*60) * deg
	sa := math32.Sqrt(sa2 / 3)
	M, T := cfg.MotionsPerMin, cfg.MotionTime+seeRest
	nRest := 4 * 0.5 / dt // strapdown's longest rest average, 4·AvgTime
	r.Forecast.PosNoise = math32.Sqrt(3*M*dt*T*T*T/3) * sa * 100
	r.Forecast.PosDrift = M * 0.5 * sa / math32.Sqrt(nRest) * T * T * 100
	r.Forecast.HeadingOK, r.Forecast.PosOK = 2, 5
	return r
}

func sqrt3(v [3]float32) [3]float32 {
	return [3]float32{math32.Sqrt(v[0]), math32.Sqrt(v[1]), math32.Sqrt(v[2])}
}

func abs(x float32) float32 { return max(x, -x) }

// finite reports whether x is neither NaN nor ±Inf.
func finite(x float32) bool { return x-x == 0 }

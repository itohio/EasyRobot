// Package eskf is an error-state (indirect) Kalman filter for a body's
// motion from a 6-axis IMU. It estimates the attitude, velocity
// and position in the world (z up) together with the gyro's and the
// accelerometer's biases and their covariance, instead of learning each by
// a heuristic that another's signal can pass.
//
// The nominal state is propagated from every sample. The error state
// (attitude δθ in the world, δv, δp, δb_g, δb_a: 15) has a first-order,
// sparse covariance step. The updates are sequential scalar ones, each
// gated on its innovation (χ², one degree of freedom):
//   - tilt, every TiltEvery samples: the specific force turned into the
//     world points up when the body isn't accelerating. Right after a
//     rest the attitude is known to a fraction of a degree, so a stroke's
//     horizontal acceleration is an outlier and is refused, and the gyro
//     carries the attitude through it;
//   - at rest, zero velocity (ZUPT) and zero rate (ZARU: the gyro reads its
//     bias). The rate's gate keeps a slow turn, which can look still, out
//     of the bias: once the bias is known to 0.3 mrad/s, a 40 mrad/s turn
//     is refused.
//
// Zero velocity begins when the body looks still, except in a stroke's
// middle: after a translation, at a speed over MoveSpeed. A slow stroke's
// acceleration can stay under the calm limit throughout, so at rest a
// shadow velocity (the body's acceleration over the last second, gravity
// taken off by a gyro-turned reference, untouched by zero velocity) ends
// the rest when it passes StillSpeed, and gives back its velocity and
// displacement. It decides only while the body is quieter than that: a
// hand's tremor moves it as fast as a 3 cm stroke. After such a wake the
// body glides: rest returns once its speed has stayed under StillSpeed for
// StillTime, or after MaxGlide looking still.
//
// Rest: the gyro quiet (low-passed over 0.1 s) and the force steady: its
// averages over AvgTime/8 and over 0.15 s agree (a hand's tremor averaged
// out; a stroke's end shows within 0.15 s), both for StillTime. The first rest sets the tilt from the force's mean; until
// then the filter holds still (Ready false).
//
// Position has no fix: it stays bounded only through the rests. An
// acceleration error δa (a tilt error δθ gives g·δθ) moves it by ½·δa·t²
// over t of motion without a rest.
//
// A Filter is a value with every array in it: Update allocates nothing.
package eskf

import (
	"errors"

	"github.com/chewxy/math32"

	"github.com/itohio/EasyRobot/x/math/nav/calib"
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

// Config holds the noise model, the update schedule and the rest test.
type Config struct {
	G             float32 // gravity as the (calibrated) accelerometer reads it, m/s²
	AccelNoise    float32 // accelerometer white noise density, m/s²/√Hz
	GyroNoise     float32 // gyro white noise density, rad/s/√Hz
	AccelBiasWalk float32 // accelerometer bias random walk, m/s² per √s
	GyroBiasWalk  float32 // gyro bias random walk, rad/s per √s
	TiltNoise     float32 // the tilt measurement's sd, rad
	ZUPTNoise     float32 // zero velocity's sd, m/s (a hand's pause is near-still)
	Gate          float32 // χ² (1 dof) above which an innovation is refused
	TiltEvery     int     // samples between tilt updates
	StillGyro     float32 // rad/s: the low-passed rate under this is quiet
	StillForce    float32 // m/s²: the tremor-averaged force within this of its 0.15 s average is steady
	StillAccel    float32 // m/s²: the world acceleration (gravity removed) under this is calm
	StillTime     float32 // s: quiet and steady for this long is rest
	CalmTime      float32 // s: the world acceleration under StillAccel for this long is calm
	RestTime      float32 // s: quiet with the force on its gyro-turned reference this long is rest even when not calm (a tilt error can keep the world acceleration up)
	AvgTime       float32 // s: the force's running average
	StillSpeed    float32 // m/s: at rest, a shadow velocity over this is motion; gliding, a speed under it for StillTime is rest
	MoveSpeed     float32 // m/s: moving faster than this, looking still is a stroke's middle, not rest (a pause's velocity error stays under it)
	MaxGlide      float32 // s: moving while looking still this long is rest whatever the speed
}

// DefaultConfig is the badge's MPU6050 at 100 Hz (white noise 0.016 m/s²
// and 0.0006 rad/s per sample; gyro bias wander 0.0007 rad/s between still
// stretches in a hand-held capture).
func DefaultConfig() Config {
	return Config{
		G: StandardGravity, AccelNoise: 0.0016, GyroNoise: 0.00006,
		AccelBiasWalk: 0.0005, GyroBiasWalk: 0.0002,
		TiltNoise: 0.01, ZUPTNoise: 0.02, Gate: 9, TiltEvery: 4,
		StillGyro: 0.05, StillForce: 0.1, StillAccel: 0.05, StillTime: 0.3, CalmTime: 0.15, RestTime: 1, AvgTime: 0.5,
		StillSpeed: 0.02, MoveSpeed: 0.05, MaxGlide: 2.5,
	}
}

func (c *Config) valid() bool {
	pos := c.G > 0 && c.AccelNoise > 0 && c.GyroNoise > 0 && c.AccelBiasWalk > 0 && c.GyroBiasWalk > 0 &&
		c.TiltNoise > 0 && c.ZUPTNoise > 0 && c.Gate > 0 && c.StillGyro > 0 && c.StillForce > 0 && c.StillAccel > 0 &&
		c.StillTime > 0 && c.CalmTime > 0 && c.RestTime > 0 && c.AvgTime > 0 && c.StillSpeed > 0 && c.MoveSpeed > 0 && c.MaxGlide > 0
	sum := c.G + c.AccelNoise + c.GyroNoise + c.AccelBiasWalk + c.GyroBiasWalk + c.TiltNoise + c.ZUPTNoise +
		c.Gate + c.StillGyro + c.StillForce + c.StillAccel + c.StillTime + c.CalmTime + c.RestTime + c.AvgTime +
		c.StillSpeed + c.MoveSpeed + c.MaxGlide
	return pos && c.TiltEvery >= 1 && !math32.IsInf(sum, 0) && !math32.IsNaN(sum)
}

// Filter is the estimate after each Update: Q the attitude [w,x,y,z]
// (sensor to world, z up), V and P the world velocity (m/s) and position
// (m), GyroBias and AccelBias in the sensor frame, Still the rest test,
// Ready once the first rest has set the tilt.
type Filter struct {
	cfg       Config
	Q         [4]float32
	V, P      [3]float32
	GyroBias  [3]float32
	AccelBias [3]float32
	Still     bool
	Ready     bool

	r        [3][3]float32 // Q as a matrix
	aw       [3]float32    // the last world acceleration, gravity removed
	cov      [n][n]float32
	ap       [n][n]float32 // scratch: the nonzero rows of A·P
	dx       [n]float32    // error accumulated by the updates, injected after
	tilts    int
	wLow     [3]float32 // the rate low-passed (0.1 s)
	fHeld    [3]float32 // the force low-passed over AvgTime/8
	fMid     [3]float32 // the force low-passed over 0.15 s
	fTurn    [3]float32 // the force turned with the gyro, followed over 0.15 s (no lag in a turn, little memory of a stroke)
	wSum     [3]float32 // the rate summed over the zero-rate window
	wSq      [3]float32 // and its square
	wN       int
	refusedT float32    // rest time with every zero-rate update refused
	awLow    [3]float32 // the world acceleration low-passed over 0.15 s
	calmT    float32    // time the body hasn't translated (|awLow| under StillAccel)
	zRefT    float32    // time zero velocity was refused while calm
	Calm     bool       // not translating (rotating or not): zero velocity applies
	fRef     [3]float32 // the force the body had, turned with the gyro (a fixed world vector), followed over RestTime
	heldT    float32    // time the force has matched fRef
	fSum     [3]float32 // the first rest's force, summed
	fN       int
	stillT   float32
	primed   bool
	turned   bool           // spun since the force last left its gyro-turned reference (a translation)
	moved    bool           // the force left its gyro-turned reference since the last rest began (a translation, not an offset's leak)
	gliding  bool           // moving since the shadow woke a rest: a slow motion that looks still
	slowT    float32        // moving: time looking still at a speed under StillSpeed (a glide's way back to rest)
	glideT   float32        // moving: time looking still (MaxGlide of it is rest whatever the speed)
	zeroing  bool           // zero velocity applied at the last sample
	sh       [10][3]float32 // the shadow: the body's acceleration (gravity off, turned into the world) integrated per 0.1 s slot, the last second
	shCur    [3]float32     // and over the slot being filled
	shT      float32        // that slot's time
	shK      int            // the oldest slot
	shN      int            // slots filled since the rest began
	vS, pS   [3]float32     // the shadow's velocity and displacement over the window (from rest at its start)
	shV, shP [3]float32     // the filled slots' share of them, summed at each slot's end
	vSq      float32        // the shadow's mean square at rest over 1 s, a second late (starts at StillSpeed²: undecided until learned)
	sq       [10]float32    // the shadow's square at each slot of the last second
	gRef     [3]float32     // the force at rest seen from the body (gravity and the accelerometer's offset), turned with the gyro
	gbSlow   [3]float32     // the gyro bias over 10 s, which turns gRef (the estimate wanders ±0.001 rad/s a second as a hand's tremor passes into it)
	gbP      [3]float32     // the gyro bias's prior variance (Prior), applied at the first rest
	baP      float32        // the accelerometer bias's prior variance: 0.1 m/s² without a multi-pose calibration (an MPU6050's zero-g offset reaches 0.3; a wider prior lets the first rest's tilt wander), or that calibration's residual
}

// New checks c and returns a filter that waits for its first rest.
func New(c Config) (Filter, error) {
	if !c.valid() {
		return Filter{}, ErrConfig
	}
	f := Filter{cfg: c, Q: [4]float32{1, 0, 0, 0}, gbP: [3]float32{4e-4, 4e-4, 4e-4}, baP: 0.01}
	f.matrix()
	return f, nil
}

// Prior seeds the biases from a multi-pose calibration (calib.Solver) whose
// Apply the caller runs on the accelerometer's samples: the gyro bias from
// its still poses, known to 2 mrad/s; the accelerometer's residual bias at
// zero, known to the fit's RMS (at least 0.01 m/s²). Call it before the
// first rest.
func (f *Filter) Prior(fit *calib.Fit) {
	f.GyroBias = fit.GyroBias
	for i := range 3 {
		f.gbP[i] = 4e-6
	}
	s := max(fit.RMS, 0.01)
	f.baP = s * s
	f.AccelBias = [3]float32{}
}

// Reset forgets the motion (attitude, velocity, position) and waits for a
// rest again; the biases are kept.
func (f *Filter) Reset() {
	f.Q, f.V, f.P = [4]float32{1, 0, 0, 0}, [3]float32{}, [3]float32{}
	f.Ready, f.Still, f.primed = false, false, false
	f.fSum, f.fN, f.stillT, f.tilts = [3]float32{}, 0, 0, 0
	f.dx = [n]float32{}
	f.matrix()
}

// Update takes one sample: dt (s), acc (m/s²) and gyro (rad/s) in the
// sensor frame.
func (f *Filter) Update(dt float32, acc, gyro [3]float32) {
	f.rest(dt, acc, gyro)
	if !f.Ready {
		if f.Still {
			for i := range 3 {
				f.fSum[i] += acc[i]
			}
			f.fN++
			if float32(f.fN)*dt >= f.cfg.AvgTime {
				f.start()
			}
		} else {
			f.fSum, f.fN = [3]float32{}, 0
		}
		return
	}
	f.predict(dt, acc, gyro)
	f.calm(dt)
	f.tilts++
	if f.tilts >= f.cfg.TiltEvery {
		f.tilts = 0
		f.tilt(acc)
	}
	f.shadow(dt, acc, gyro)
	c := &f.cfg
	looks := f.Calm || f.heldT >= c.RestTime
	if f.zeroing {
		// at rest: the shadow starts at the rest, so it is the speed since
		// (zero velocity holds V). It decides only while the body is
		// quieter than the strokes it must see (3× its rms at rest under
		// StillSpeed: a sensor on a desk, not a hand's tremor, which moves
		// it as fast as a 3 cm stroke does).
		s2 := f.vS[0]*f.vS[0] + f.vS[1]*f.vS[1] + f.vS[2]*f.vS[2]
		ss := c.StillSpeed * c.StillSpeed
		quiet := 9*f.vSq < ss
		moved := quiet && s2 >= ss
		if looks && !moved {
			f.zupt(dt)
		} else {
			f.wake()
			f.zeroing, f.zRefT, f.gliding, f.slowT, f.glideT = false, 0, looks, 0, 0
		}
	} else {
		// moving. A real acceleration (no longer looking still) ends a
		// glide. Gliding, the speed is the filter's own (the shadow gave
		// it, nothing holds it): a rest begins once it has stayed under
		// StillSpeed for StillTime, a stroke's slowing end past. Otherwise
		// a rest begins as soon as the body looks still (a corner's
		// pause), unless after a translation it is faster than MoveSpeed (a
		// stroke's middle). After a turn a rest doesn't ask the speed (the
		// offset's leak runs it up), nor after MaxGlide looking still (no
		// speed error keeps zero velocity out for good).
		s2 := f.V[0]*f.V[0] + f.V[1]*f.V[1] + f.V[2]*f.V[2]
		if !looks {
			f.gliding, f.glideT = false, 0
		} else {
			f.glideT += dt
		}
		if looks && s2 < c.StillSpeed*c.StillSpeed {
			f.slowT += dt
		} else {
			f.slowT = 0
		}
		// a translation's speed is real; an offset's leak (no translation:
		// the force stayed on its reference) is error, and rest is due
		calm := !f.moved || s2 < c.MoveSpeed*c.MoveSpeed
		if f.gliding {
			calm = f.slowT >= c.StillTime
		}
		if looks && (calm || f.turned || f.glideT >= c.MaxGlide) {
			f.clearShadow()
			f.seedRef()
			f.moved = false
			f.zupt(dt)
			f.zeroing = true
		}
	}
	if f.Still {
		f.zaru(dt, gyro)
	} else {
		f.wSum, f.wSq, f.wN = [3]float32{}, [3]float32{}, 0
	}
}

// calm: the body isn't translating while the world acceleration (gravity
// removed, low-passed over 0.1 s: a steady hand's 8 Hz tremor and the
// noise stay near 0.01 m/s²) stays under StillAccel for CalmTime (a 0.5 s
// pause shows); a rotation in place passes. A horizontal acceleration
// hardly changes |f| (a²/2g: 0.013 m/s² for 0.5 m/s²), and a smooth
// stroke passes the force's steadiness test, so only the world
// acceleration can tell a stroke from rest, and a slow one (3 cm over 2 s
// peaks at 0.043 m/s²) passes it too: the shadow velocity decides.
func (f *Filter) calm(dt float32) {
	c := &f.cfg
	km := min(dt/0.1, 1)
	var a2 float32
	for i := range 3 {
		f.awLow[i] += km * (f.aw[i] - f.awLow[i])
		a2 += f.awLow[i] * f.awLow[i]
	}
	if a2 < c.StillAccel*c.StillAccel {
		f.calmT += dt
	} else {
		f.calmT = 0
	}
	f.Calm = f.calmT >= c.CalmTime
}

// rest: the gyro quiet and the force steady (attitude-free, so it holds
// before the first rest and while the bias is still unknown): the zero
// rate. A stroke without rotation can pass it; the gyro then reads only its
// bias, so that is harmless. A slow turn is kept out by the quiet test and
// the zero rate's gate.
func (f *Filter) rest(dt float32, acc, gyro [3]float32) {
	c := &f.cfg
	if !f.primed {
		f.fHeld, f.fMid, f.fRef, f.fTurn, f.primed = acc, acc, acc, acc, true
		for i := range 3 {
			f.wLow[i] = gyro[i] - f.GyroBias[i]
		}
	}
	kw := min(dt/0.1, 1)
	kh := min(8*dt/c.AvgTime, 1)
	km := min(dt/0.15, 1)
	var w2, h2 float32
	for i := range 3 {
		f.wLow[i] += kw * (gyro[i] - f.GyroBias[i] - f.wLow[i])
		w2 += f.wLow[i] * f.wLow[i]
		f.fHeld[i] += kh * (acc[i] - f.fHeld[i])
		f.fMid[i] += km * (acc[i] - f.fMid[i])
		d := f.fHeld[i] - f.fMid[i]
		h2 += d * d
	}
	if w2 < c.StillGyro*c.StillGyro && h2 < c.StillForce*c.StillForce {
		f.stillT += dt
	} else {
		f.stillT = 0
	}
	if w2 >= c.StillGyro*c.StillGyro {
		f.turned = true
	}
	// fRef: a fixed world vector seen from the turning body changes as
	// −ω×v; followed towards the force over RestTime. A rotation in place
	// keeps the force on it; a translation pulls the force off it (a
	// smooth stroke passes the steadiness above, not this)
	wx, wy, wz := gyro[0]-f.GyroBias[0], gyro[1]-f.GyroBias[1], gyro[2]-f.GyroBias[2]
	r0, r1, r2 := f.fRef[0], f.fRef[1], f.fRef[2]
	f.fRef[0] -= (wy*r2 - wz*r1) * dt
	f.fRef[1] -= (wz*r0 - wx*r2) * dt
	f.fRef[2] -= (wx*r1 - wy*r0) * dt
	t0, t1, t2 := f.fTurn[0], f.fTurn[1], f.fTurn[2]
	f.fTurn[0] -= (wy*t2 - wz*t1) * dt
	f.fTurn[1] -= (wz*t0 - wx*t2) * dt
	f.fTurn[2] -= (wx*t1 - wy*t0) * dt
	kr := min(dt/c.RestTime, 1)
	var d2 float32
	for i := range 3 {
		f.fRef[i] += kr * (acc[i] - f.fRef[i])
		f.fTurn[i] += km * (acc[i] - f.fTurn[i])
		d := f.fHeld[i] - f.fRef[i]
		d2 += d * d
	}
	if d2 < c.StillForce*c.StillForce && w2 < c.StillGyro*c.StillGyro {
		f.heldT += dt
	} else {
		f.heldT = 0
	}
	if d2 >= c.StillForce*c.StillForce && w2 < c.StillGyro*c.StillGyro {
		f.turned, f.moved = false, true // a translation: the force left its turned reference
	}
	f.Still = f.stillT >= c.StillTime
}

// shadow integrates the body's acceleration over the last second in 0.1 s
// slots: vS = v(t) − v(t − 1 s), from rest at the window's start, and pS
// the displacement that velocity made. Gravity is taken off in the body
// frame (gRef, turned with the gyro and followed over 2 s at rest), not
// in the world: there a 0.1° error in the attitude leaves g·δθ, 0.017
// m/s², as much as a 3 cm stroke's acceleration, and a hand's attitude
// wanders that much. The attitude then only turns the small remainder into
// the world. A stroke shows within its first second, and once it ends the
// window empties in a second.
func (f *Filter) shadow(dt float32, acc, gyro [3]float32) {
	// over 10 s, but snapped to the estimate when it moves more than twice
	// a tremor's wander (2 mrad/s) away: while the bias is being learned a
	// lag turns gRef by the whole unlearned part (0.01 rad/s is 0.1 m/s² a
	// second)
	k10 := min(dt/10, 1)
	for i := range 3 {
		if e := f.GyroBias[i] - f.gbSlow[i]; e > 0.002 || e < -0.002 {
			f.gbSlow[i] = f.GyroBias[i]
		} else {
			f.gbSlow[i] += k10 * e
		}
	}
	wx, wy, wz := gyro[0]-f.gbSlow[0], gyro[1]-f.gbSlow[1], gyro[2]-f.gbSlow[2]
	r0, r1, r2 := f.gRef[0], f.gRef[1], f.gRef[2]
	f.gRef[0] -= (wy*r2 - wz*r1) * dt
	f.gRef[1] -= (wz*r0 - wx*r2) * dt
	f.gRef[2] -= (wx*r1 - wy*r0) * dt
	var d [3]float32
	k := min(dt/2, 1)
	for i := range 3 {
		a := acc[i] // gRef holds the offset: the bias estimate trades with the tilt at rest
		if f.zeroing {
			f.gRef[i] += k * (a - f.gRef[i])
		}
		d[i] = a - f.gRef[i]
	}
	for i := range 3 {
		f.shCur[i] += (f.r[i][0]*d[0] + f.r[i][1]*d[1] + f.r[i][2]*d[2]) * dt
	}
	f.shT += dt
	if f.shT >= 0.1 {
		// the floor learns the shadow of a second ago, so a stroke can't
		// raise it before it has woken the rest
		if f.zeroing && f.shN == len(f.sh) {
			f.vSq += 0.1 * (f.sq[f.shK] - f.vSq)
		}
		f.sq[f.shK] = f.vS[0]*f.vS[0] + f.vS[1]*f.vS[1] + f.vS[2]*f.vS[2]
		f.sh[f.shK], f.shCur, f.shT = f.shCur, [3]float32{}, 0
		f.shK = (f.shK + 1) % len(f.sh)
		f.shN = min(f.shN+1, len(f.sh))
		f.shV, f.shP = [3]float32{}, [3]float32{}
		for j := len(f.sh) - f.shN; j < len(f.sh); j++ {
			s := &f.sh[(f.shK+j)%len(f.sh)]
			for i := range 3 {
				f.shP[i] += (f.shV[i] + 0.5*s[i]) * 0.1
				f.shV[i] += s[i]
			}
		}
	}
	for i := range 3 {
		f.pS[i] = f.shP[i] + (f.shV[i]+0.5*f.shCur[i])*f.shT
		f.vS[i] = f.shV[i] + f.shCur[i]
	}
}

// clearShadow starts the shadow at a rest.
func (f *Filter) clearShadow() {
	f.sh, f.shCur, f.shT, f.shN, f.vS, f.pS = [10][3]float32{}, [3]float32{}, 0, 0, [3]float32{}, [3]float32{}
	f.shV, f.shP = [3]float32{}, [3]float32{}
}

// seedRef sets gravity seen from the body to the force's gyro-turned 0.15 s
// average (the body looks still). A plain average lags once a hand's
// tremor turns the body (0.016 rad/s turns g by 0.16 m/s² a second: the
// first rest's mean is 0.014 m/s² off, a plain 0.15 s average 0.01), and
// a longer one still holds a stroke that ended a moment ago; either error
// the 2 s follower would carry into the shadow for seconds.
func (f *Filter) seedRef() {
	for i := range 3 {
		f.gRef[i] = f.fTurn[i]
	}
}

// wake: zero velocity has stopped: the body was at rest and has started
// to move. The shadow is its velocity since (V = vS), and P gains the
// shadow's displacement; the velocity's variance opens to the zero
// velocity's own size.
func (f *Filter) wake() {
	rv := f.cfg.ZUPTNoise * f.cfg.ZUPTNoise
	for i := range 3 {
		f.V[i] = f.vS[i]
		f.P[i] += f.pS[i]
		f.cov[iV+i][iV+i] = max(f.cov[iV+i][iV+i], rv)
	}
}

// start sets the tilt from the first rest's mean force (yaw 0) and the
// covariance to its initial values.
func (f *Filter) start() {
	var m [3]float32
	k := 1 / float32(f.fN)
	for i := range 3 {
		m[i] = f.fSum[i]*k - f.AccelBias[i]
	}
	g := math32.Sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2])
	u := [3]float32{m[0] / g, m[1] / g, m[2] / g} // world up in the sensor frame
	// the rotation taking u to ẑ: axis u×ẑ = (u_y, −u_x, 0), half-angle form
	w := 1 + u[2]
	if w < 1e-6 { // upside down: turn about x by π
		f.Q = [4]float32{0, 1, 0, 0}
	} else {
		s := 1 / math32.Sqrt(2*w)
		f.Q = [4]float32{w * s, u[1] * s, -u[0] * s, 0}
	}
	f.matrix()
	f.V, f.P = [3]float32{}, [3]float32{}
	f.cov = [n][n]float32{}
	set := func(o int, v float32) {
		for i := range 3 {
			f.cov[o+i][o+i] = v
		}
	}
	set(iTh, 1e-4) // 0.6°: the mean of a rest's force, with the accelerometer's bias unknown
	f.cov[iTh+2][iTh+2] = 1
	set(iV, 1e-4)
	set(iP, 1e-6)
	for i := range 3 {
		f.cov[iBg+i][iBg+i] = f.gbP[i]
	}
	set(iBa, f.baP)
	f.awLow, f.calmT, f.zRefT = [3]float32{}, f.cfg.CalmTime, 0
	f.Calm, f.zeroing = true, false
	f.clearShadow()
	f.seedRef()
	f.gbSlow = f.GyroBias
	f.vSq = f.cfg.StillSpeed * f.cfg.StillSpeed
	f.Ready = true
}

// matrix refreshes r from Q.
func (f *Filter) matrix() {
	w, x, y, z := f.Q[0], f.Q[1], f.Q[2], f.Q[3]
	f.r = [3][3]float32{
		{1 - 2*(y*y+z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)},
		{2 * (x*y + w*z), 1 - 2*(x*x+z*z), 2 * (y*z - w*x)},
		{2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2*(x*x+y*y)},
	}
}

func (f *Filter) normalise() {
	s := 1 / math32.Sqrt(f.Q[0]*f.Q[0]+f.Q[1]*f.Q[1]+f.Q[2]*f.Q[2]+f.Q[3]*f.Q[3])
	for i := range 4 {
		f.Q[i] *= s
	}
	f.matrix()
}

// predict propagates the nominal state by one sample and the covariance
// with it.
func (f *Filter) predict(dt float32, acc, gyro [3]float32) {
	var w, a [3]float32
	for i := range 3 {
		w[i] = gyro[i] - f.GyroBias[i]
		a[i] = acc[i] - f.AccelBias[i]
	}
	for i := range 3 {
		f.aw[i] = f.r[i][0]*a[0] + f.r[i][1]*a[1] + f.r[i][2]*a[2]
	}
	f.aw[2] -= f.cfg.G
	for i := range 3 {
		f.P[i] += f.V[i]*dt + 0.5*f.aw[i]*dt*dt
		f.V[i] += f.aw[i] * dt
	}
	h := 0.5 * dt
	q := f.Q
	f.Q = [4]float32{
		q[0] - h*(q[1]*w[0]+q[2]*w[1]+q[3]*w[2]),
		q[1] + h*(q[0]*w[0]+q[2]*w[2]-q[3]*w[1]),
		q[2] + h*(q[0]*w[1]-q[1]*w[2]+q[3]*w[0]),
		q[3] + h*(q[0]*w[2]+q[1]*w[1]-q[2]*w[0]),
	}
	f.propagate(dt) // with the attitude at the step's start (r)
	f.normalise()
}

// propagate: P += dt(AP + PAᵀ) + Q dt, A's nonzero blocks A[θ,bg] = −R,
// A[v,θ] = −[a_w + g ẑ]×, A[v,ba] = −R, A[p,v] = I.
func (f *Filter) propagate(dt float32) {
	r := &f.r
	c := &f.cov
	ap := &f.ap
	ax, ay, az := f.aw[0], f.aw[1], f.aw[2]+f.cfg.G
	for j := range n {
		b0, b1, b2 := c[iBg][j], c[iBg+1][j], c[iBg+2][j]
		t0, t1, t2 := c[iTh][j], c[iTh+1][j], c[iTh+2][j]
		e0, e1, e2 := c[iBa][j], c[iBa+1][j], c[iBa+2][j]
		for i := range 3 {
			ap[iTh+i][j] = -(r[i][0]*b0 + r[i][1]*b1 + r[i][2]*b2)
		}
		ap[iV][j] = az*t1 - ay*t2 - (r[0][0]*e0 + r[0][1]*e1 + r[0][2]*e2)
		ap[iV+1][j] = -az*t0 + ax*t2 - (r[1][0]*e0 + r[1][1]*e1 + r[1][2]*e2)
		ap[iV+2][j] = ay*t0 - ax*t1 - (r[2][0]*e0 + r[2][1]*e1 + r[2][2]*e2)
		ap[iP][j], ap[iP+1][j], ap[iP+2][j] = c[iV][j], c[iV+1][j], c[iV+2][j]
	}
	// AP's rows are θ, v, p (0–8); add dt·AP and its transpose
	for i := range iBg {
		for j := range n {
			d := dt * ap[i][j]
			c[i][j] += d
			c[j][i] += d
		}
	}
	cg, ca := f.cfg.GyroNoise, f.cfg.AccelNoise
	wg, wa := f.cfg.GyroBiasWalk, f.cfg.AccelBiasWalk
	for i := range 3 {
		c[iTh+i][iTh+i] += cg * cg * dt
		c[iV+i][iV+i] += ca * ca * dt
		c[iBg+i][iBg+i] += wg * wg * dt
		c[iBa+i][iBa+i] += wa * wa * dt
	}
}

// scalar is one measurement y = h·δx + noise of variance rv, h with m ≤ 4
// nonzeros at idx: it accumulates into dx and downdates the covariance,
// unless gated and the innovation's χ² is above the gate (false).
func (f *Filter) scalar(idx *[4]int, val *[4]float32, m int, y, rv float32, gated bool) bool {
	c := &f.cov
	var ph [n]float32
	for i := range n {
		var s float32
		for k := range m {
			s += c[i][idx[k]] * val[k]
		}
		ph[i] = s
	}
	s := rv
	var hd float32
	for k := range m {
		s += val[k] * ph[idx[k]]
		hd += val[k] * f.dx[idx[k]]
	}
	e := y - hd
	if gated && e*e > f.cfg.Gate*s {
		return false
	}
	inv := 1 / s
	for i := range n {
		k := ph[i] * inv
		f.dx[i] += k * e
		for j := i; j < n; j++ {
			v := c[i][j] - k*ph[j]
			c[i][j] = v
			c[j][i] = v
		}
	}
	return true
}

// guard keeps the covariance a covariance in float32 (after each batch of
// scalar updates): a positive diagonal
// and correlations within ±1 (P − k·phᵀ cancels to rounding when a
// measurement is far surer than the state, and a negative variance would
// turn the next gain around).
func (f *Filter) guard() {
	c := &f.cov
	var sd [n]float32
	for i := range n {
		c[i][i] = max(c[i][i], 1e-12)
		sd[i] = math32.Sqrt(c[i][i])
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			lim := sd[i] * sd[j]
			v := min(max(c[i][j], -lim), lim)
			c[i][j], c[j][i] = v, v
		}
	}
}

// inject moves the accumulated error into the nominal state.
func (f *Filter) inject() {
	d := &f.dx
	h0, h1, h2 := 0.5*d[iTh], 0.5*d[iTh+1], 0.5*d[iTh+2]
	q := f.Q // q ← [1, δθ/2] ⊗ q (the error is in the world)
	f.Q = [4]float32{
		q[0] - h0*q[1] - h1*q[2] - h2*q[3],
		q[1] + h0*q[0] + h1*q[3] - h2*q[2],
		q[2] - h0*q[3] + h1*q[0] + h2*q[1],
		q[3] + h0*q[2] - h1*q[1] + h2*q[0],
	}
	for i := range 3 {
		f.V[i] += d[iV+i]
		f.P[i] += d[iP+i]
		f.GyroBias[i] += d[iBg+i]
		f.AccelBias[i] += d[iBa+i]
	}
	f.dx = [n]float32{}
	f.guard()
	f.normalise()
}

// tilt: the force turned into the world, u = R(f − b_a)/|·|, against ẑ;
// gated unless calm (an acceleration is an outlier). Calm, the body isn't
// translating, turning or not, so the force is gravity (a hand's turn
// adds no centripetal force to speak of); a gate there could lock a grown
// error out for good.
// To first order u_x = −δθ_y and u_y = δθ_x, and the bias error adds
// −R δb_a/g.
func (f *Filter) tilt(acc [3]float32) {
	var a, u [3]float32
	for i := range 3 {
		a[i] = acc[i] - f.AccelBias[i]
	}
	for i := range 3 {
		u[i] = f.r[i][0]*a[0] + f.r[i][1]*a[1] + f.r[i][2]*a[2]
	}
	g := math32.Sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2])
	if !(g > 0) {
		return
	}
	ig := 1 / g
	rv := f.cfg.TiltNoise * f.cfg.TiltNoise
	idx := [4]int{iTh + 1, iBa, iBa + 1, iBa + 2}
	val := [4]float32{-1, -f.r[0][0] * ig, -f.r[0][1] * ig, -f.r[0][2] * ig}
	f.scalar(&idx, &val, 4, u[0]*ig, rv, !f.Calm)
	idx = [4]int{iTh, iBa, iBa + 1, iBa + 2}
	val = [4]float32{1, -f.r[1][0] * ig, -f.r[1][1] * ig, -f.r[1][2] * ig}
	f.scalar(&idx, &val, 4, u[1]*ig, rv, !f.Calm)
	f.inject()
}

// zupt: zero velocity at rest (Update decides when). Gated, so a velocity
// the covariance says is real is kept, unless the body is still (quiet
// gyro, steady force: rest by its own evidence; gated there, a velocity
// grown from an unseen offset's leak is refused, the leak runs on, and
// the offset is never learnt). Refused for half a second, it is taken
// anyway, with the velocity's variance reopened to its own size: the
// velocity was error the covariance didn't know of, and a gate must not
// lock it out.
func (f *Filter) zupt(dt float32) {
	rv := f.cfg.ZUPTNoise * f.cfg.ZUPTNoise
	force := f.zRefT >= 0.5
	if force {
		// the covariance was too sure: the velocity is error of its own size
		for i := range 3 {
			f.cov[iV+i][iV+i] = max(f.cov[iV+i][iV+i], f.V[i]*f.V[i])
		}
	}
	refused := false
	for i := range 3 {
		idx := [4]int{iV + i}
		val := [4]float32{1}
		if !f.scalar(&idx, &val, 1, -f.V[i], rv, !force && !f.Still) {
			refused = true
		}
	}
	f.inject()
	if refused {
		f.zRefT += dt
	} else {
		f.zRefT = 0
	}
}

// zaru: zero rate over a window of 0.1 s at rest: the gyro's mean reads
// its bias. A hand at rest still turns (0.016 rad/s of tremor, correlated
// sample to sample), so the noise is the window's own spread, at least the
// white noise's mean; gated, so a slow turn that looks still stays out of
// the bias. An axis refused through a second of rest reopens the bias's
// uncertainty to the size of what it refused (it was too sure to see a
// real change: a warming gyro's; about the vertical, only the zero rate
// sees it).
func (f *Filter) zaru(dt float32, gyro [3]float32) {
	for i := range 3 {
		f.wSum[i] += gyro[i]
		f.wSq[i] += gyro[i] * gyro[i]
	}
	f.wN++
	if float32(f.wN)*dt < 0.1 {
		return
	}
	k := 1 / float32(f.wN)
	white := f.cfg.GyroNoise * f.cfg.GyroNoise / dt * k
	refused := false
	var miss [3]float32 // each refused axis's innovation
	for i := range 3 {
		m := f.wSum[i] * k
		v := max(f.wSq[i]*k-m*m, 0)
		idx := [4]int{iBg + i}
		val := [4]float32{1}
		if e := m - f.GyroBias[i]; !f.scalar(&idx, &val, 1, e, max(v, white), true) {
			refused, miss[i] = true, e
		}
	}
	f.wSum, f.wSq, f.wN = [3]float32{}, [3]float32{}, 0
	f.inject()
	if !refused {
		f.refusedT = 0
		return
	}
	f.refusedT += 0.1
	if f.refusedT >= 1 {
		for i := range 3 {
			f.cov[iBg+i][iBg+i] = max(f.cov[iBg+i][iBg+i], miss[i]*miss[i])
		}
		f.refusedT = 0
	}
}

// Sigma returns the standard deviations of the attitude (rad, world x, y,
// z), velocity (m/s) and position (m) errors.
func (f *Filter) Sigma() (att, vel, pos [3]float32) {
	for i := range 3 {
		att[i] = math32.Sqrt(f.cov[iTh+i][iTh+i])
		vel[i] = math32.Sqrt(f.cov[iV+i][iV+i])
		pos[i] = math32.Sqrt(f.cov[iP+i][iP+i])
	}
	return
}

// Package strapdown dead-reckons the translation of a rigid body from an
// IMU: the accelerometer's specific force is rotated into the world by an
// attitude, gravity is removed, and the rest is integrated twice into
// velocity and position. Until its first rest the Nav takes the attitude
// of a filter (e.g. ahrs Mahony, [w,x,y,z], sensor to world, z up); from
// then on it carries its own on the gyro, less a gyro bias it learns at
// rest, and levels it to gravity only at rest. A filter leans its attitude
// into any acceleration it sees, so a slow horizontal stroke reads as tilt
// and cancels itself; the gyro does not. After every sample the Nav is the
// body's whole motion state: its attitude Q, position P and velocity V in
// the world and PB, VB in the body frame (a turn in place moves neither),
// linear acceleration A (world, every sample) and specific force F (sensor
// frame: gravity plus the body's own acceleration), for every consumer at
// once.
//
// Integrating an accelerometer drifts fast (a constant error e gives
// e·t²/2 of position), so the integrator leans on rest:
//
//   - Rest: the gyro is quiet, no linear acceleration is left (world
//     acceleration minus the learned bias) and the integrated speed is
//     under StillSpeed, all at once and without a break for StillTime. Low
//     acceleration alone is not rest: a gentle stroke (3 cm in 2 s peaks
//     at 0.04 m/s²) never shows more, only speed tells it apart. Vertical
//     speed always counts; horizontal speed over the first MaxGlide of a
//     motion (past that, over a long motion, the horizontal velocity
//     drifts by as much as a gentle stroke's speed). After a turn the
//     rules change (see Turns).
//   - Zero-velocity update (ZUPT): at rest, velocity is zero and the
//     position holds; the distance the velocity left over built while the
//     body looked still is taken back.
//   - Leaving rest: on a rotation, on a strong acceleration that is not
//     steady, or when the motion integrated during the rest (a shadow
//     velocity, from a bias snapshot StillTime to 2·StillTime old) passes
//     StillSpeed/2 vertically or StillSpeed horizontally, or five standard
//     deviations of what the noise (measured in the first rest) and the
//     bias's own error can give it, if that is more. The shadow's velocity
//     and distance are taken up, so a slow start is not lost.
//   - Rotation in place: a motion that begins with the gyro alone is a
//     rotation, and moves nothing while it lasts: a shadow integrates it
//     instead. If it shows a translation (|f| moving off its value at rest
//     by more than StillAccel plus what unlearned offsets can make of the
//     turn, an unsteady acceleration with the gyro quiet, or a speed no
//     turned offset could give), the motion is taken up from the shadow,
//     so a stroke made while turning is kept from its start. If not, its
//     rest decides: what the shadow holds, less the turn's error (see
//     Turns), is taken up if it stands out of what the noise and offsets
//     the rest cannot read could give it, and never under 1 cm. A tilt in
//     place moves nothing; a gentle stroke while the wrist turns shows at
//     the rest after it.
//   - Turns: the accelerometer's offsets turn with the sensor, and an
//     error δb in them puts (R − R0)·δb into the world acceleration from
//     the start of the motion (R0, R the attitudes then and now). The Nav
//     keeps ∫(R − R0)dt and ∫∫(R − R0)dt² over each motion. A motion that
//     turned (by 3° or more) rests on a steady force even past StillAccel,
//     without asking its speed, once StillTime + AvgTime/2 holds only
//     noise about a constant (its halves agree, its spread is the noise's)
//     that also explains the velocity at the motion's end; that constant
//     is (R1 − R0)·δb, which gives δb across the turn's axis. Its error is
//     then taken out of P, PB and the velocity over the whole motion, and
//     δb learned into the sensor-frame offsets (turns of 20° or more).
//   - Bias learning: at rest, whatever world acceleration remains is the
//     accelerometer's bias plus any error in G (the plain mean over the
//     first rest, then a running average). Its horizontal part is a tilt
//     error at rest and moves into the attitude over 4·AvgTime; the gyro
//     rate at rest is its bias. A strong acceleration held steady for
//     StillTime + AvgTime is not a hand but a stale bias (after a turn its
//     own rest comes first): the attitude goes back to the filter's and
//     the bias is learned again from scratch.
//   - First rest: the gyro quiet, the specific force steady at its running
//     average and the attitude settled (the world acceleration not
//     drifting while the filter still turns towards the true tilt) for
//     StillTime, then AvgTime of learning. Nothing is integrated before
//     it: the badge holds the origin until the filter has settled, so it
//     may be fed from power-on.
//   - Glides: a body that has looked still (quiet gyro, no acceleration,
//     steady force) for MaxGlide is at rest whatever its integrated speed;
//     this bounds the drift after a wrong velocity. What is left in its
//     acceleration over the glide built part of that velocity there, and
//     is allowed for in the distance taken back. A real glide that long is
//     stopped too, in any direction.
//   - Damping (optional): velocity decays at Damping per second while
//     moving, trading a short motion's accuracy for bounded drift in a long
//     one.
//
// Strokes without a pause have no rest to correct them: the velocity error
// accumulates. A paddle waved without a pause (10 cm strokes of 0.5 s,
// 0.02 m/s² of noise, five draws) is off by up to 4 mm after 2 s, 2 cm
// after 4 s and 11 cm after 9.5 s.
//
// Slow strokes are tracked in every direction. With an exact attitude,
// horizontal strokes of 5 cm in 2 s and faster come out within 1 %; the
// slowest, 3 cm in 2 s, at 88 % (it wakes the Nav at StillSpeed, a third
// of the way in). Through a 6-axis Mahony filter with a real sensor's bias
// and noise, 41 of 42 horizontal and 36 of 42 vertical strokes (3–30 cm,
// 0.3–2 s) end within 10 % and at rest.
//
// Tilting a badge in place, fpv-gate's 60 cases (15–90°, over 1, 0.5 and
// 0.2 s, upright and flat, about four axes) with an MPU6050's noise
// (accelerometer 0.03 m/s², gyro 0.002 rad/s) and offsets of 0.09 m/s²,
// an uncalibrated MPU6050's 0.7 m/s² and 1.05 m/s² (past the datasheet's
// 1.04): the path and the paddle stray 0.0 mm in all 37,800 runs (30
// seeds, six ranges, gyro bias 0.005 and 0.02 rad/s) and the badge rests
// after. Strokes of 5 cm or more, and 3 cm strokes of 1 s or less,
// register (fpv-gate ER-067 §r10, 144 runs). Strokes made while the wrist
// wobbles (3–20 cm over 0.5–1.5 s at 0.3 and 1 rad/s) are tracked within
// 15 and 30 % and on average 1 % short; with an MPU6050's noise and offsets
// they read 72–140 % of their length (240 runs, ten seeds). A stroke right
// after a turn, before any rest, is tracked within 1 % with an exact
// attitude (3–20 cm in 0.5–2 s); with the noise, 5–20 cm read 59–153 %
// (180 runs), every one at rest after, and the badge then holds still (no
// wake in 30 s; a turn and a 10 cm stroke peak at 21.6 cm before the rest
// corrects them to 9.9 cm). Strokes that never show themselves while they
// move (3–5 cm over 1–1.5 s while turning, 3–10 cm over 1.5–2 s right
// after a turn) jump into place at the rest after them.
//
// Known failures (fpv-gate ER-067 §r10; the next revision's targets):
//   - 3 cm over 1.5 s: of fpv-gate's 144 stroke runs, 5 do not register
//     (one while the wrist wobbles at 0.1 rad/s, three at 1 rad/s, one
//     right after a 45° turn) and 3
//     read the wrong way (−109 % wobbling at 0.1 rad/s; −736 % and −781 %
//     right after a 45° turn). fpv-gate measured these as mostly design
//     (the registration threshold and how a turn's offset error is
//     attributed), not the sensor: the physical part is about 23 %
//     (standard deviation) at an MPU6050's noise. What a misread rest after
//     such a stroke leaves drifts up to 20 cm over the next 10 s before it
//     rests for good.
//   - 3 cm in 2 s with no turn: one draw reads −67 % at the stroke's end,
//     −204 % a second later and −39 % at rest; fpv-gate's noisy horizontal
//     grid reads −302 % 1.5 s after, still gliding (an MPU6050's noise
//     leaves up to 2 cm/s, which holds rest off until MaxGlide).
//   - Slow 2 s strokes can jump back at their late rest, about 1 s after
//     the stroke (one of three draws each): 10 cm in 2 s reads 106–110 %
//     live, then 48 %; 5 cm, 95 % then 54 %.
//
// Between rests the attitude is the gyro's, so its noise random-walks the
// attitude; the filter's integral term no longer matters after the first
// rest.
//
// float32 throughout (the badge's filters are float32; per sample this is
// about a hundred operations, cheap even in software floating point).
// Update does not allocate; Nav is a 736-byte value with fixed state
// (TinyGo puts a value over 256 bytes on the heap, so New allocates it
// once).
package strapdown

import (
	"errors"

	"github.com/chewxy/math32"
)

// ErrConfig is returned by New for an invalid Config.
var ErrConfig = errors.New("strapdown: invalid configuration")

// StandardGravity in m/s².
const StandardGravity = 9.80665

// offsetMax is the largest accelerometer offset assumed, m/s²: an MPU6050's
// zero-g tolerance (±50 mg on x and y, ±80 mg on z: 1.04 m/s² as a vector).
// It sets how much |f| a turn may change before the Nav takes it for a
// translation, and what offsets a rest cannot read (along the turn's axis,
// or all of them after a turn that came back to where it began) may have
// put into a motion (see update, register).
const offsetMax = 1.0

// chordTurn is how far gravity's direction in the sensor frame must move
// over a motion (|ĝ − ĝrest|, about 3°) for the motion to count as turned:
// its rest then reads what is left as the offsets' error (see update).
const chordTurn = 0.05

// Config sets the integrator. DefaultConfig suits a hand-held badge.
type Config struct {
	G          float32 // gravity removed from the world z axis, m/s² (> 0)
	StillGyro  float32 // rad/s: a faster rotation is motion (> 0)
	StillAccel float32 // m/s²: more linear acceleration than this is motion (> 0)
	StillSpeed float32 // m/s: rest needs the integrated vertical speed under this (a rotation in place has none); half of it gained during a rest ends the rest (> 0)
	StillTime  float32 // s the body must look still before velocity is zeroed (> 0)
	MaxGlide   float32 // s of looking still after which the body is at rest whatever its speed (≥ StillTime)
	AvgTime    float32 // s, time constant of the force average, of bias learning after the first rest, and the first rest's learning time (> 0)
	Damping    float32 // 1/s velocity decay while moving; 0 = none
}

// DefaultConfig: standard gravity; still below 0.1 rad/s (6°/s), 0.2 m/s²
// of linear acceleration and 1.5 cm/s for 0.3 s, or for 2.5 s whatever the
// speed; 0.5 s averaging; no damping.
func DefaultConfig() Config {
	return Config{G: StandardGravity, StillGyro: 0.1, StillAccel: 0.2, StillSpeed: 0.015, StillTime: 0.3, MaxGlide: 2.5, AvgTime: 0.5}
}

// Nav is the badge's motion state, refreshed by every Update: position P
// (m) and velocity V (m/s) in the world frame of the attitude (z up); the
// same motion in the body (sensor) frame, PB and VB (see Update); the
// linear acceleration A (m/s², world, gravity and the learned bias
// removed, on every sample, at rest too); the specific force F (m/s², the
// sensor frame: gravity plus the body's own acceleration, as the
// accelerometer read it); the attitude Q ([w, x, y, z]: the filter's
// until the first rest, then the Nav's own, see the package doc); the
// learned world bias Bias (m/s²); and whether the body is Still (velocity
// held at zero). One Nav can feed every consumer: the path (P), the
// paddle (PB along the screen's up axis), and gravity-driven simulations
// (F).
type Nav struct {
	P, V, A, Bias [3]float32
	PB, VB        [3]float32
	F             [3]float32
	Q             [4]float32
	Still         bool

	cfg        Config
	att        [4]float32 // the Nav's attitude: the filter's until calibrated, then the gyro's, levelled at rest
	gBias      [3]float32 // gyro bias learned at rest, rad/s
	gbNew      [3]float32 // gBias at the last snapshot, and at the one before
	gbOld      [3]float32 // (as biasNew, biasOld: a slow turn's first samples are learned at rest)
	aInt       [3]float32 // the acceleration the integrator took last sample
	sb         [3]float32 // accelerometer bias in the sensor frame, learned across turns (see learnTurn)
	attRest    [4]float32 // att when the body last left rest
	rot        bool       // the motion so far is a rotation in place: nothing is integrated
	fRest      float32    // |f − sb| at rest when the motion began
	sRot       shadow     // what the rotation would have integrated, taken up if it turns out to translate
	gRest      [3]float32 // gravity's direction in the sensor frame when the motion began
	chordMax   float32    // the largest |ĝ − ĝrest| of the motion so far
	rotFor     float32    // s since the motion began
	r0         [9]float32 // the attitude when the motion began (its rest), as a rotation matrix
	m1, m2     [9]float32 // ∫(R − R0)dt and ∫∫(R − R0)dt² over the motion: an offset error δb puts (R − R0)·δb into the world acceleration
	endP       [3]float32 // a rotation's shadow P when its quiet stretch began: the motion's end
	endM2      [9]float32 // m2 then
	endT       float32    // rotFor then
	endV       [3]float32 // the velocity (a rotation's shadow's) and m1 then
	endM1      [9]float32
	noise2     float32    // variance of the world acceleration at rest, per axis (m/s²)²
	bN         float32    // samples the first rest learned the bias from (how well a young rest's bias is known)
	movingFor  float32    // s since the body last left rest
	calmFor    float32    // s quiet and calm (the body looks still, whatever its speed)
	quietSum   [3]float32 // ∫ (r − Bias) dt over the quietFor stretch
	quietLate  [3]float32 // the same over its part after half the rest's wait
	quietSq    float32    // ∫ |r − Bias|² dt over the quietFor stretch
	lateFor    float32    // s of its part after half the rest's wait
	avg        [3]float32 // low-pass of the specific force, sensor frame
	rFast      [3]float32 // low-passes of the world acceleration over AvgTime/2
	rSlow      [3]float32 // and AvgTime: they part while the attitude still turns
	primed     bool       // avg, rFast and rSlow hold a sample
	calibrated bool       // Bias has been learned over one full rest
	seeded     bool       // gBias comes from a calibration (Seed): the first rest keeps it
	stillFor   float32    // s of the first rest so far
	learned    float32    // samples averaged into Bias during the first rest
	quietFor   float32    // s moving with a quiet gyro, no acceleration and speed under StillSpeed
	glideFor   float32    // s moving with a quiet gyro, no acceleration and a steady force
	staleFor   float32    // s with a quiet gyro and a strong steady acceleration
	glideSum   [3]float32 // ∫ (r − Bias) dt over the glideFor stretch
	pStale     [3]float32 // P when that began
	biasNew    [3]float32 // Bias at the last snapshot (see atRest)
	biasOld    [3]float32 // Bias at the snapshot before
	snapFor    float32    // s of rest since the last snapshot
	restFor    float32    // s of the current rest
	sNew, sOld shadow     // motion integrated during the rest from biasNew and biasOld
}

// shadow is the motion integrated during a rest from one bias snapshot.
type shadow struct{ V, A, P [3]float32 }

// New checks c and returns a Nav at rest at the origin.
func New(c Config) (Nav, error) {
	if !(c.G > 0) || !(c.StillGyro > 0) || !(c.StillAccel > 0) || !(c.StillSpeed > 0) || !(c.StillTime > 0) ||
		!(c.MaxGlide >= c.StillTime) || !(c.AvgTime > 0) || !(c.Damping >= 0) ||
		!finite(c.G+c.StillGyro+c.StillAccel+c.StillSpeed+c.StillTime+c.MaxGlide+c.AvgTime+c.Damping) {
		return Nav{}, ErrConfig
	}
	return Nav{cfg: c}, nil
}

// Reset puts the body back at the origin, moving until it looks still, and
// forgets the average force; the learned biases and the attitude are kept
// (they belong to the sensor).
func (n *Nav) Reset() {
	n.P, n.V, n.A, n.aInt = [3]float32{}, [3]float32{}, [3]float32{}, [3]float32{}
	n.PB, n.VB = [3]float32{}, [3]float32{}
	n.Still, n.primed, n.stillFor = false, false, 0
	n.quietFor, n.glideFor, n.staleFor, n.movingFor, n.calmFor, n.glideSum = 0, 0, 0, 0, 0, [3]float32{}
	n.quietSum, n.quietLate, n.quietSq, n.lateFor, n.attRest, n.rot = [3]float32{}, [3]float32{}, 0, 0, n.att, false
	n.sNew, n.sOld, n.sRot, n.chordMax = shadow{}, shadow{}, shadow{}, 0
}

// Seed gives the Nav a calibration's gyro bias (rad/s), as a calibration at
// rest measures it over seconds (x/math/nav/calib): the first rest keeps it
// rather than learning it from its own shorter stretch, and later rests
// refine it from there as from a long rest. Call it before the first rest
// ends. (The accelerometer's noise is not seeded: the first rest's estimate
// of it is within 10 %, and seeding it changed nothing measurable.)
func (n *Nav) Seed(gyroBias [3]float32) {
	n.gBias, n.gbOld, n.gbNew = gyroBias, gyroBias, gyroBias
	n.seeded = true
}

// Update advances by dt seconds with the attitude q ([w, x, y, z], sensor
// to world, unit length), the accelerometer's specific force f (sensor
// frame, m/s²; +g up at rest) and the gyro rate w (sensor frame, rad/s).
// A sample with a non-positive or non-finite dt, or any non-finite value,
// is ignored.
//
// The body-frame motion: VB is V in the sensor frame now; PB adds up each
// sample's world displacement rotated into the sensor frame of that
// sample. A turn in place adds nothing to PB, where P rotated by the
// attitude now would swing with every tilt (a paddle that jumps when the
// badge tilts).
func (n *Nav) Update(dt float32, q [4]float32, f, w [3]float32) {
	if !(dt > 0) || !finite(dt+q[0]+q[1]+q[2]+q[3]+f[0]+f[1]+f[2]+w[0]+w[1]+w[2]) {
		return
	}
	p := n.P
	n.update(dt, q, f, w)
	qc := [4]float32{n.Q[0], -n.Q[1], -n.Q[2], -n.Q[3]}
	d := rotate(qc, [3]float32{n.P[0] - p[0], n.P[1] - p[1], n.P[2] - p[2]})
	for i := range 3 {
		n.PB[i] += d[i]
	}
	n.VB = rotate(qc, n.V)
}

// update is one valid sample of world-frame motion.
func (n *Nav) update(dt float32, q [4]float32, f, w [3]float32) {
	c := &n.cfg

	spin2 := w[0]*w[0] + w[1]*w[1] + w[2]*w[2]
	// the attitude: the filter's until the bias is learned; then the gyro's
	// (less its bias), levelled to gravity only at rest (see atRest). The
	// filter leans its attitude into any acceleration it sees (a slow
	// stroke reads as tilt and cancels itself) and lags a turn; the gyro
	// does neither.
	if n.calibrated {
		n.att = turn(n.att, [3]float32{w[0] - n.gBias[0], w[1] - n.gBias[1], w[2] - n.gBias[2]}, dt)
	} else {
		n.att = q
	}
	// world acceleration: rotate f, less the sensor-frame bias, by the
	// attitude, remove gravity
	r := rotate(n.att, [3]float32{f[0] - n.sb[0], f[1] - n.sb[1], f[2] - n.sb[2]})
	r[2] -= c.G
	n.F, n.Q = f, n.att
	// at rest the running bias is taking in the first samples of a motion
	// not seen yet; the older snapshot has none of them (the first rest
	// has no snapshot: A follows its mean, below)
	b := &n.Bias
	if n.Still {
		b = &n.biasOld
	}
	for i := range 3 {
		n.A[i] = r[i] - b[i]
	}

	// the measures: a quiet gyro; a specific force steady at its running
	// average (needs no bias); no linear acceleration left after the bias;
	// and a settled attitude: the world acceleration not drifting. An
	// attitude filter still turning towards the true tilt after power-on
	// moves it while the force stays steady; two averages of it then part
	// by about half the error left, which would otherwise be learned as
	// bias. They restart together after motion, so only an ongoing drift
	// parts them.
	if !n.primed {
		n.avg, n.rFast, n.rSlow, n.primed = f, r, r, true
	}
	if spin2 >= c.StillGyro*c.StillGyro {
		// turning: the force follows the attitude, and steadiness counts
		// from when the gyro goes quiet (a 40° turn moves it 6.7 m/s²,
		// which the average would take 1.75 s to forget)
		n.avg = f
	}
	k := min(dt/c.AvgTime, 1)
	k2 := min(2*dt/c.AvgTime, 1)
	var steady2, left2 float32
	for i := range 3 {
		d := f[i] - n.avg[i]
		n.avg[i] += k * d
		steady2 += d * d
		e := r[i] - n.Bias[i]
		left2 += e * e
	}
	lim2 := c.StillAccel * c.StillAccel
	quiet := spin2 < c.StillGyro*c.StillGyro
	steady, calm := steady2 < lim2, left2 < lim2
	var trend2 float32
	if !quiet || !steady {
		n.rFast, n.rSlow = r, r
	} else {
		for i := range 3 {
			n.rFast[i] += k2 * (r[i] - n.rFast[i])
			n.rSlow[i] += k * (r[i] - n.rSlow[i])
			t := n.rFast[i] - n.rSlow[i]
			trend2 += t * t
		}
	}
	settled := trend2 < lim2/64

	if !n.calibrated {
		// first rest: no bias to judge acceleration by, so a steady force
		// and a settled attitude; the bias is the plain mean of the rest's
		// samples
		if quiet && steady && settled {
			n.stillFor += dt
		} else {
			n.stillFor = 0
		}
		n.Still = n.stillFor >= c.StillTime
		if !n.Still {
			return // no bias yet: hold the origin
		}
		n.learned++
		n.bN = n.learned
		var e2 float32
		for i := range 3 {
			e := r[i] - n.Bias[i]
			n.Bias[i] += (r[i] - n.Bias[i]) / n.learned
			n.A[i] = r[i] - n.Bias[i]
			if !n.seeded {
				n.gBias[i] += (w[i] - n.gBias[i]) / n.learned
			}
			e2 += e * e
		}
		// the noise too, so the first rest after it judges a gentle start
		// by it (left to warm up from zero, it read 0.6 of the noise 0.6 s
		// later and let the noise wake a still badge)
		n.noise2 += (e2/3 - n.noise2) / n.learned
		if n.stillFor >= c.StillTime+c.AvgTime {
			n.calibrated = true
			n.tilt(n.Bias)
			n.rest()
			if n.seeded {
				// the seed is as good as a long rest: the running averages
				// start at their longest (a 0.5 s one soon replaced a 3 s
				// calibration's gyro bias with its own noise)
				n.restFor = 3 * c.AvgTime
			}
		}
		return
	}

	// a strong acceleration held steady is no hand: the bias is stale, the
	// stretch was a rest, and what was integrated over it is error. After a
	// turn the turned rest, 0.55 s, comes before this at 0.8 s (with the
	// halves timed exactly; a sample off made the rest after a 90° turn
	// with uncalibrated offsets miss, and this then threw the attitude
	// away)
	if quiet && steady && !calm {
		if n.staleFor == 0 {
			n.pStale = n.P
		}
		n.staleFor += dt
	} else {
		n.staleFor = 0
	}
	if n.staleFor >= c.StillTime+c.AvgTime {
		// the attitude goes back to the filter's (a gyro glitch, or the
		// filter still settling); its averages start again from there
		n.calibrated, n.learned, n.staleFor, n.primed = false, 0, 0, false
		n.stillFor, n.Still = c.StillTime, true
		n.P, n.V, n.aInt = n.pStale, [3]float32{}, [3]float32{}
		return
	}

	if n.Still {
		wake := !quiet || !calm && !steady
		if calm { // a gentle start: no acceleration to see, only speed
			// vertically at StillSpeed/2; horizontally at StillSpeed (two
			// axes of noise: 0.03 m/s² over a 0.6 s shadow is 2.3 mm/s
			// each, and at half the speed a still badge kept waking). Both
			// rise to what the noise measured at rest can give the shadow
			// (5 standard deviations, and the same tail for two axes): it integrates the noise over
			// its age T, and the bias it runs against (biasOld, the mean of
			// N samples) is off by noise/√N, which T multiplies; so
			// σv² = σ²·(dt·T + T²/N). At 0.03 m/s² that is 11 mm/s at a
			// long rest, 16 mm/s right after a turn (its bias is 0.3 s old).
			T := n.snapFor + c.StillTime
			sv2 := n.noise2 * (dt*T + T*T/max(n.bN, 1))
			vz := max(c.StillSpeed*c.StillSpeed/4, 25*sv2)
			vh := max(c.StillSpeed*c.StillSpeed, 30*sv2)
			s := n.sOld.V
			wake = wake || s[2]*s[2] >= vz || s[0]*s[0]+s[1]*s[1] >= vh
		}
		if !wake {
			n.atRest(r, w, dt)
			return
		}
		if quiet || !calm && !steady {
			// moving since before the older snapshot at the latest: take
			// up what was integrated from there, against the bias of then
			n.Bias = n.biasOld
			n.V, n.aInt = n.sOld.V, n.sOld.A
			for i := range 3 {
				n.P[i] += n.sOld.P[i]
			}
		}
		// woken by the gyro alone (the block above skipped): a turn shows
		// from its first sample, so none of it was learned and the
		// accelerometer bias now is the freshest.
		// the gyro bias goes back to the older snapshot whatever woke the
		// body: a slow turn's first samples, under StillGyro, were learned
		// into it at rest
		n.gBias = n.gbOld
		n.Still, n.quietFor, n.glideFor, n.movingFor, n.calmFor, n.glideSum = false, 0, 0, 0, 0, [3]float32{}
		// woken by the gyro alone: a rotation, until something shows a
		// translation (see below); the force's magnitude now is the
		// reference a turn in place keeps
		n.rot = !(quiet || !calm && !steady)
		a := [3]float32{n.avg[0] - n.sb[0], n.avg[1] - n.sb[1], n.avg[2] - n.sb[2]}
		n.fRest = math32.Sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
		n.sRot, n.chordMax, n.rotFor = shadow{}, 0, 0
	}

	// A rotation in place moves nothing, and integrating it only adds the
	// errors a turn brings: the accelerometer's offsets turned with the
	// sensor (0.09 m/s² over 60°), the attitude's own. So a motion that
	// began as a turn integrates nothing while |f| stays near what it was
	// at rest (turning does not change it; a vertical stroke does) and,
	// with the gyro quiet, no unsteady acceleration shows. Near: within
	// StillAccel, plus what unlearned offsets can make of the turn so far,
	// |b|·|ĝ − ĝrest| with ĝ gravity's direction in the sensor frame and
	// |b| up to offsetMax (an uncalibrated MPU6050's 0.32/−0.48/0.40 m/s²
	// moved |f| 0.36 m/s² over 30° and read as a stroke; with that much
	// allowed, 0.03 m/s² of noise never crosses). Then integration starts,
	// from what the shadow integrated since the turn began. A translation
	// that never shows itself (made while turning by a large angle that
	// leaves |f| alone, a gentle stroke) is taken up at the rest after it
	// (see register).
	g := rotate([4]float32{n.att[0], -n.att[1], -n.att[2], -n.att[3]}, [3]float32{0, 0, 1})
	gx, gy, gz := g[0]-n.gRest[0], g[1]-n.gRest[1], g[2]-n.gRest[2]
	chord := math32.Sqrt(gx*gx + gy*gy + gz*gz)
	n.chordMax, n.rotFor = max(n.chordMax, chord), n.rotFor+dt
	if n.rot {
		fx, fy, fz := f[0]-n.sb[0], f[1]-n.sb[1], f[2]-n.sb[2]
		d := fx*fx + fy*fy + fz*fz - n.fRest*n.fRest
		tol := c.StillAccel + offsetMax*chord
		if abs(d) > 2*n.fRest*tol || quiet && !calm && !steady {
			// a translation after all: take up what a shadow integrated
			// since the turn began, so a stroke made while turning is
			// kept from its start
			n.rot = false
			n.V, n.aInt = n.sRot.V, n.sRot.A
			for i := range 3 {
				n.P[i] += n.sRot.P[i]
			}
		}
	}

	keep := float32(1)
	if c.Damping > 0 {
		keep = max(1-c.Damping*dt, 0)
	}
	for i := range 3 {
		// trapezoid on both integrals: the samples are of a smooth motion,
		// and taking each acceleration for the whole step before it would
		// add half a step of it to the velocity (1 cm/s for a 2 m/s² stroke
		// at 100 Hz, a centimetre a second of drift)
		a := r[i] - n.Bias[i]
		if n.rot { // the shadow takes it; the body holds still
			s := &n.sRot
			v := s.V[i] + (s.A[i]+a)*dt/2
			s.P[i] += (s.V[i] + v) * dt / 2
			s.V[i], s.A[i] = v, a
			a = 0
		}
		v := (n.V[i] + (n.aInt[i]+a)*dt/2) * keep
		n.P[i] += (n.V[i] + v) * dt / 2
		n.V[i], n.aInt[i] = v, a
	}
	// the turn's error terms: an error δb in the sensor-frame offsets adds
	// (R − R0)·δb to the world acceleration from the motion's start, so
	// m1·δb to the velocity and m2·δb to P (see turnErr)
	R := matrix(n.att)
	for k := range 9 {
		n.m1[k] += (R[k] - n.r0[k]) * dt
		n.m2[k] += n.m1[k] * dt
	}

	// rest: everything quiet at once for StillTime, the speed included; or
	// looking still for MaxGlide whatever the speed (bounds the drift).
	// Vertical speed always counts. Horizontal speed counts over the first
	// MaxGlide of a motion: a slow stroke (30 cm in 2 s) stays under
	// StillAccel for 0.36 s around its peak speed, and a gentle one
	// throughout; past that, over a long motion, the horizontal velocity
	// drifts by as much as a gentle stroke's speed (4 cm/s after an 8 s
	// Pong rally) and would keep the body from ever resting. A rotation in
	// place integrates nothing, so its speed is zero; what is left in its
	// acceleration over its rest is the bias's error at the new attitude
	// (0.07 m/s² for a 0.05 m/s² sensor bias over 60°), and the rest starts
	// from the biases corrected by it (see learnTurn), or its first moments
	// would read the error as a gentle start.
	n.movingFor += dt
	h2 := float32(0)
	if n.movingFor < c.MaxGlide {
		h2 = n.V[0]*n.V[0] + n.V[1]*n.V[1]
	}
	// A turned motion (a rotation in place, or one that turned by more
	// than chordTurn) rests on a steady force: what the turn left is the
	// sensor's bias turned with it (0.3 m/s² for an MPU6050's 0.5 m/s²
	// offset over 40°), past StillAccel, and only a rest at the new
	// attitude can learn it (see learnTurn). Its speed is not asked: that
	// leftover built it (a stroke right after a 45° turn with 0.09 m/s² of
	// offsets kept 0.07 m/s² of it, never rested and drifted 1.3 m in
	// 30 s); the rest takes it back with the leftover's own error (see
	// turnErr). A hand pushing steadily and constantly just after a turn
	// would be taken for bias, and a glide after a turn rests at once
	// (its distance is lost). The force average restarts with each turn.
	// The rest needs StillTime + AvgTime/2, and what it learns must be the
	// turn's leftover, constant, not a gentle stroke right after the turn
	// (its acceleration lies within a fifth of its peak for 0.3 of the
	// stroke, under the 0.55 s window for strokes up to 1.8 s; a 0.3 s
	// rest on the plateau of a 5 cm stroke over 1.5 s put 0.22 m/s² into
	// the offsets). Longer, the rest after a turn with uncalibrated
	// offsets no longer fitted a 1 s hold.
	turned := n.rot || n.chordMax >= chordTurn
	need := c.StillTime
	if turned {
		need += c.AvgTime / 2
	}
	if quiet && (calm || turned && steady) && (turned || h2+n.V[2]*n.V[2] < c.StillSpeed*c.StillSpeed) {
		if n.quietFor == 0 && turned {
			// the motion ends here, if this stretch becomes its rest
			n.endP, n.endM2, n.endT = n.sRot.P, n.m2, n.rotFor
			n.endV, n.endM1 = n.V, n.m1
			if n.rot {
				n.endV = n.sRot.V
			}
		}
		n.quietFor += dt
		late := n.quietFor > need/2
		if late {
			n.lateFor += dt
		}
		for i := range 3 {
			e := (r[i] - n.Bias[i]) * dt
			n.quietSum[i] += e
			n.quietSq += e * e / dt
			if late {
				n.quietLate[i] += e
			}
		}
	} else {
		n.quietFor, n.quietSum, n.quietLate, n.quietSq, n.lateFor = 0, [3]float32{}, [3]float32{}, 0, 0
	}
	if quiet && calm {
		n.calmFor += dt
	} else {
		n.calmFor = 0
	}
	if quiet && calm && steady {
		n.glideFor += dt
		for i := range 3 {
			n.glideSum[i] += (r[i] - n.Bias[i]) * dt
		}
	} else {
		n.glideFor, n.glideSum = 0, [3]float32{}
	}
	if turned && n.quietFor >= need {
		// and constant it must be: if the stretch's two halves differ by
		// more than its noise (and by StillAccel/16: a sensor that shows
		// none would otherwise never rest after a turn), or its samples
		// spread about their mean by more than the noise (four standard
		// deviations of a variance of N samples, and the same floor), a
		// stroke is under way, so no rest yet. The halves catch a slope;
		// the spread catches a stroke's hump, whose halves agree (the
		// first 0.55 s of 3 cm in 1 s right after a turn peaks at 0.17 m/s²
		// and was learned as 0.13 m/s² of offsets)
		// (each half by its own time: a sample off between them biased the
		// means 3.6 % of the leftover apart, and after a 90° turn with an
		// uncalibrated MPU6050's offsets the rest missed a 1 s hold)
		tA, tB := n.quietFor-n.lateFor, n.lateFor
		var d2, m2 float32
		for i := range 3 {
			d := n.quietLate[i]/tB - (n.quietSum[i]-n.quietLate[i])/tA
			d2 += d * d
			m := n.quietSum[i] / n.quietFor
			m2 += m * m
		}
		floor := c.StillAccel * c.StillAccel / 256
		spread := n.quietSq/n.quietFor - m2
		bad := d2 > max(floor, 16*n.noise2*dt*(1/tA+1/tB)) || spread > max(floor, 3*n.noise2*(1+4*math32.Sqrt(2*dt/(3*n.quietFor))))
		if !bad {
			// and it must explain the motion's end: an offsets' error δb
			// built m1·δb of velocity by the time the stretch began, and
			// no stroke's speed may be left there (2·StillSpeed, or four
			// standard deviations of the noise integrated over the motion:
			// 0.03 m/s² of noise leaves up to 2 cm/s after a 1 s stroke),
			// beyond what offsets the rest cannot read could build
			// (offsetMax through m1 along the turn's axis, or all of m1 for
			// a turn that came back to where it began: a tilt there and
			// back otherwise never rested). A stroke still slowing down
			// looks like an offset over 0.55 s (3 cm in 1.5 s, 0.83 s in:
			// −0.05 to −0.02 m/s²) but still moves at 3.7 cm/s; so does a
			// glide.
			e := [3]float32{n.quietSum[0] / n.quietFor, n.quietSum[1] / n.quietFor, n.quietSum[2] / n.quietFor}
			db, s := n.turnErr(e)
			v := mul(n.endM1, db)
			var v2, u2 float32
			for i := range 3 {
				x := n.endV[i] - v[i]
				v2 += x * x
			}
			if s > sMin {
				m := mul(n.endM1, n.turnAxis())
				u2 = m[0]*m[0] + m[1]*m[1] + m[2]*m[2]
			} else {
				for k := range 9 {
					u2 += n.endM1[k] * n.endM1[k]
				}
			}
			lim := max(2*c.StillSpeed, 4*math32.Sqrt(n.noise2*dt*n.endT)) + offsetMax*math32.Sqrt(u2)
			bad = v2 > lim*lim
		}
		if bad {
			n.quietFor, n.quietSum, n.quietLate, n.quietSq, n.lateFor = 0, [3]float32{}, [3]float32{}, 0, 0
		}
	}
	// (a glide's rest waits while a turned motion's quiet stretch is under
	// way: the stroke after a turn ends, its stretch starts, and the glide
	// that counted through the stroke would take a rest that reads nothing)
	if n.quietFor >= need || n.glideFor >= c.MaxGlide && !(turned && n.quietFor > 0) {
		// ZUPT. The acceleration was nil over the whole quiet stretch, so
		// the velocity left now was there all along and is error (a real
		// stroke's tail has slowed to nothing): take back its distance.
		// Horizontally, once its speed no longer holds rest off, the
		// velocity is drift that was there for as long as the body has
		// looked still: take back that stretch's distance.
		win := n.glideFor
		if n.quietFor >= need {
			win = n.quietFor
		}
		winH := win
		if n.movingFor >= c.MaxGlide {
			winH = max(win, n.calmFor)
		}
		v := n.V
		if turned && n.quietFor >= need {
			// after a turn, what is left is the offsets' error δb; it was
			// in the motion from its start: out of P, PB and the velocity
			// with it. A rotation in place built nothing: what its shadow
			// holds, less that error, is a translation if it stands out of
			// what the noise and unlearned offsets can make (see register)
			e := [3]float32{n.quietSum[0] / n.quietFor, n.quietSum[1] / n.quietFor, n.quietSum[2] / n.quietFor}
			db, s := n.turnErr(e)
			if n.rot {
				n.register(db, s, dt)
			} else {
				p, vc := mul(n.m2, db), mul(n.m1, db)
				for i := range 3 {
					n.P[i] -= p[i]
					v[i] -= vc[i]
				}
			}
			n.learnTurn(e, db, s)
		}
		// a glide's rest: what is left in its acceleration (a bias gone
		// wrong, 0.006 m/s² of slow noise after a stroke, or a stroke's
		// start learned as a turn's leftover) built the velocity up over
		// the glide, so not all of it was there throughout: take back
		// V·T − ē·T²/2 (the plain V·T sent a 3 cm stroke 11 cm on)
		var eg [3]float32
		if n.quietFor < need {
			for i := range 3 {
				eg[i] = n.glideSum[i] / n.glideFor
			}
		}
		n.P[0] -= v[0]*winH - eg[0]*winH*winH/2
		n.P[1] -= v[1]*winH - eg[1]*winH*winH/2
		n.P[2] -= v[2]*win - eg[2]*win*win/2
		n.Still = true
		n.V, n.aInt = [3]float32{}, [3]float32{}
		n.rest()
	}
}

// register decides at the rest after a rotation in place whether it
// carried a translation that never showed (a gentle stroke while the wrist
// turns): what the shadow integrated up to the motion's end (where the
// quiet stretch began), less the offsets' error over it (m2·δb), is taken
// up into P if it stands out of what the noise can give it (four standard
// deviations per axis of the noise integrated twice over the motion,
// σ²·dt·T³/3) and of what offsets the rest cannot read can give it (all
// of m2 through offsetMax after a turn under sMin; along a larger turn's
// axis m2 holds nothing), and never under 1 cm. A tilt in place then moves
// nothing; a gentle stroke while the wrist wobbles shows at its rest.
// (Weighing δb's own noise as well, through m2·(R1 − R0)ᵀ/s, lost more
// strokes and kept no tilt any better.)
func (n *Nav) register(db [3]float32, s float32, dt float32) {
	p := mul(n.endM2, db)
	d := [3]float32{n.endP[0] - p[0], n.endP[1] - p[1], n.endP[2] - p[2]}
	T := n.endT
	var2 := n.noise2 * dt * T * T * T / 3
	var u2 float32
	if s <= sMin {
		for k := range 9 {
			u2 += n.endM2[k] * n.endM2[k]
		}
	}
	thr := max(0.01, 4*math32.Sqrt(var2)) + offsetMax*math32.Sqrt(u2)
	if d[0]*d[0]+d[1]*d[1]+d[2]*d[2] <= thr*thr {
		return
	}
	for i := range 3 {
		n.P[i] += d[i]
	}
}

// sMin is the smallest 2 − 2·cos φ (a turn of 0.6°) from which turnErr
// reads the offsets' error for the motion itself.
const sMin = 1e-4

// turnErr reads e, the acceleration left at the first rest after a turn,
// as an error in the offsets. The attitude's error is fixed in the world
// (the gyro carries it so) and at the rest before the turn was cancelled
// with the part of the accelerometer's bias it could not tell from tilt;
// but that bias turns with the sensor. So e = (R1 − R0)·δb, R0 and R1 the
// attitudes before and after, δb the error in the sensor-frame bias sb;
// and (R1 − R0)ᵀ(R1 − R0) = (2 − 2·cos φ)(I − uuᵀ) for a turn by φ about
// u, so δb = (R1 − R0)ᵀ·e / (2 − 2·cos φ) across the axis (along it
// nothing shows). It returns δb (zero under sMin) and s = 2 − 2·cos φ.
func (n *Nav) turnErr(e [3]float32) (db [3]float32, s float32) {
	q0, q1 := n.attRest, n.att
	d := q0[0]*q1[0] + q0[1]*q1[1] + q0[2]*q1[2] + q0[3]*q1[3]
	s = 4 * (1 - d*d) // 2 − 2·cos φ
	if !n.calibrated || s <= sMin {
		return db, s
	}
	a := rotate([4]float32{q1[0], -q1[1], -q1[2], -q1[3]}, e)
	b := rotate([4]float32{q0[0], -q0[1], -q0[2], -q0[3]}, e)
	return [3]float32{(a[0] - b[0]) / s, (a[1] - b[1]) / s, (a[2] - b[2]) / s}, s
}

// turnAxis is the axis of the turn since the motion began, in the sensor
// frame: the vector part of attRest⁻¹·att, along which turnErr reads
// nothing.
func (n *Nav) turnAxis() [3]float32 {
	a, b := n.attRest, n.att
	x := a[0]*b[1] - a[1]*b[0] - a[2]*b[3] + a[3]*b[2]
	y := a[0]*b[2] + a[1]*b[3] - a[2]*b[0] - a[3]*b[1]
	z := a[0]*b[3] - a[1]*b[2] + a[2]*b[1] - a[3]*b[0]
	l := math32.Sqrt(x*x + y*y + z*z)
	if !(l > 0) {
		return [3]float32{}
	}
	return [3]float32{x / l, y / l, z / l}
}

// learnTurn takes e and its reading db (turnErr) into the biases: sb takes
// δb and Bias the rest of e, which cancels e now; Bias's horizontal part,
// the tilt mistaken for bias before, levels back into the attitude (see
// tilt). A turn under 20° (s under 0.12) gives Bias e whole: the division
// mostly amplifies the noise, and sb would keep it (a 0.1 rad turn the
// gyro saw and the body did not make put its leak into sb as 3 m/s²);
// even a small turn's change matters: left to the rest's average, a 15°
// tilt held still woke the body.
func (n *Nav) learnTurn(e, db [3]float32, s float32) {
	if s < 0.12 {
		db = [3]float32{}
	}
	w := rotate(n.att, db)
	for i := range 3 {
		n.sb[i] += db[i]
		n.Bias[i] += e[i] - w[i]
	}
}

// matrix is the rotation matrix of the unit quaternion q (row-major).
func matrix(q [4]float32) [9]float32 {
	w, x, y, z := q[0], q[1], q[2], q[3]
	return [9]float32{
		1 - 2*(y*y+z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
		2 * (x*y + w*z), 1 - 2*(x*x+z*z), 2 * (y*z - w*x),
		2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2*(x*x+y*y),
	}
}

// mul is the 3×3 matrix m (row-major) times v.
func mul(m [9]float32, v [3]float32) [3]float32 {
	return [3]float32{
		m[0]*v[0] + m[1]*v[1] + m[2]*v[2],
		m[3]*v[0] + m[4]*v[1] + m[5]*v[2],
		m[6]*v[0] + m[7]*v[1] + m[8]*v[2],
	}
}

// tilt moves the horizontal part of m (an acceleration seen at rest,
// already in Bias) out of the bias and into the attitude: at rest a
// horizontal force is a tilt error, and the gyro carries the attitude
// from then on. The force at rest stands g = G + Bias[2] tall (a z bias of
// a few tenths is common) and leans by θ, tan θ = |m|/g; turned upright
// it is F = √(g² + |m|²) tall, so the bias loses m and gains F − g
// vertically. lean turns by 2·atan(|d|/2), and tan(θ/2) = |m|/(F + g).
// Both snapshots move with it, so a motion that restores one finds it
// consistent with the attitude.
func (n *Nav) tilt(m [3]float32) {
	g := n.cfg.G + n.Bias[2]
	F := math32.Sqrt(g*g + m[0]*m[0] + m[1]*m[1])
	k := 2 / (F + g)
	n.att = lean(n.att, [3]float32{m[1] * k, -m[0] * k, 0})
	d := [3]float32{-m[0], -m[1], F - g}
	for i := range 3 {
		n.Bias[i] += d[i]
		n.biasNew[i] += d[i]
		n.biasOld[i] += d[i]
	}
}

// rest starts a rest: both bias snapshots are the bias now, both shadows
// empty, and the motion's references (its attitude, gravity's direction,
// the turn's error terms) start from here, so a motion is measured from
// its rest however it begins.
func (n *Nav) rest() {
	n.biasOld, n.biasNew, n.snapFor, n.restFor, n.rot, n.chordMax = n.Bias, n.Bias, 0, 0, false, 0
	n.sRot = shadow{}
	n.gbOld, n.gbNew = n.gBias, n.gBias
	n.sOld, n.sNew = shadow{}, shadow{}
	n.attRest, n.r0, n.m1, n.m2 = n.att, matrix(n.att), [9]float32{}, [9]float32{}
	n.gRest = rotate([4]float32{n.att[0], -n.att[1], -n.att[2], -n.att[3]}, [3]float32{0, 0, 1})
}

// atRest is one sample at rest: velocity zero, the bias learned, and the
// shadows integrate what is left after their snapshot's bias.
//
// The bias is a running average whose time constant grows with the rest,
// from AvgTime to 4·AvgTime: a strokes-without-pause rally integrates
// whatever error it has for seconds (0.002 m/s² of noise in a 0.5 s average
// is 6 mm over 3 s), and a long rest can afford a longer average.
//
// The first samples of a motion arrive before it is detected and are
// learned into the bias; a snapshot every StillTime keeps one from at least
// StillTime ago, so the motion starts from a bias and a shadow older than
// it.
func (n *Nav) atRest(r, w [3]float32, dt float32) {
	c := &n.cfg
	n.V, n.aInt = [3]float32{}, [3]float32{}
	n.restFor += dt
	k := min(dt/min(c.AvgTime+n.restFor, 4*c.AvgTime), 1)
	for i := range 3 {
		n.gBias[i] += k * (w[i] - n.gBias[i])
		n.Bias[i] += k * (r[i] - n.Bias[i])
	}
	n.sOld.step(r, n.biasOld, dt)
	n.sNew.step(r, n.biasNew, dt)
	n.snapFor += dt
	if n.snapFor >= c.StillTime {
		// level the attitude: move the older snapshot's horizontal bias
		// (a tilt error, at rest) into the attitude over 4·AvgTime; the
		// older snapshot has none of a motion's first samples
		kl := min(n.snapFor/(4*c.AvgTime), 1)
		n.tilt([3]float32{kl * n.biasOld[0], kl * n.biasOld[1], 0})
		n.biasOld, n.biasNew, n.snapFor = n.biasNew, n.Bias, 0
		n.gbOld, n.gbNew = n.gbNew, n.gBias
		n.sOld, n.sNew = n.sNew, shadow{}
	}
}

// step integrates one sample of world acceleration r less bias b
// (trapezoid, as for the motion itself).
func (s *shadow) step(r, b [3]float32, dt float32) {
	for i := range 3 {
		a := r[i] - b[i]
		v := s.V[i] + (s.A[i]+a)*dt/2
		s.P[i] += (s.V[i] + v) * dt / 2
		s.V[i], s.A[i] = v, a
	}
}

// rotate returns v rotated by the unit quaternion q = [w, x, y, z]:
// v' = v + w·t + u × t with t = 2·(u × v), u = (x, y, z).
func rotate(q [4]float32, v [3]float32) [3]float32 {
	w, x, y, z := q[0], q[1], q[2], q[3]
	tx, ty, tz := 2*(y*v[2]-z*v[1]), 2*(z*v[0]-x*v[2]), 2*(x*v[1]-y*v[0])
	return [3]float32{
		v[0] + w*tx + y*tz - z*ty,
		v[1] + w*ty + z*tx - x*tz,
		v[2] + w*tz + x*ty - y*tx,
	}
}

// finite reports whether x is neither NaN nor ±Inf (a sum of values is
// finite only if every value is, barring overflow to Inf, which is also
// rejected).
func finite(x float32) bool { return x-x == 0 }

func abs(x float32) float32 { return max(x, -x) }

// turn is q advanced by the body-frame rate w over dt (first order, then
// brought back to unit length by one Newton step: no square root).
func turn(q [4]float32, w [3]float32, dt float32) [4]float32 {
	hx, hy, hz := w[0]*dt/2, w[1]*dt/2, w[2]*dt/2
	return unit([4]float32{
		q[0] - q[1]*hx - q[2]*hy - q[3]*hz,
		q[1] + q[0]*hx + q[2]*hz - q[3]*hy,
		q[2] + q[0]*hy - q[1]*hz + q[3]*hx,
		q[3] + q[0]*hz + q[1]*hy - q[2]*hx,
	})
}

// lean is q turned by the small world-frame rotation vector d.
func lean(q [4]float32, d [3]float32) [4]float32 {
	hx, hy, hz := d[0]/2, d[1]/2, d[2]/2
	return unit([4]float32{
		q[0] - hx*q[1] - hy*q[2] - hz*q[3],
		q[1] + hx*q[0] + hy*q[3] - hz*q[2],
		q[2] - hx*q[3] + hy*q[0] + hz*q[1],
		q[3] + hx*q[2] - hy*q[1] + hz*q[0],
	})
}

// unit brings a nearly unit quaternion back to unit length.
func unit(q [4]float32) [4]float32 {
	s := (3 - (q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])) / 2
	return [4]float32{q[0] * s, q[1] * s, q[2] * s, q[3] * s}
}

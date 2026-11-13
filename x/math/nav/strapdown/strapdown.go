// Package strapdown dead-reckons the translation of a rigid body from an
// IMU: the accelerometer's specific force is rotated into the world by an
// attitude filter's quaternion (e.g. ahrs Mahony, [w,x,y,z], sensor to
// world, z up), gravity is removed, and the rest is integrated twice into
// velocity and position. Rotation comes from the filter; this package adds
// the translation, so together they give the body's path in 3D.
//
// Integrating an accelerometer drifts fast (a constant error e gives
// e·t²/2 of position), so the integrator leans on rest:
//
//   - Rest: the gyro is quiet, no linear acceleration is left (world
//     acceleration minus the learned bias) and the integrated speed is
//     under StillSpeed, all at once and without a break for StillTime.
//     Low acceleration alone is not rest: a gentle stroke (3 cm in 2 s
//     peaks at 0.04 m/s²) never shows more, only speed tells it apart.
//   - Zero-velocity update (ZUPT): at rest, velocity is zero and the
//     position holds.
//   - Leaving rest: on a rotation, on a strong acceleration that is not
//     steady, or when the motion integrated during the rest (a shadow
//     velocity, from a bias snapshot StillTime to 2·StillTime old) passes
//     StillSpeed/2. The shadow's velocity and distance are taken up, so a
//     slow start is not lost.
//   - Bias learning: at rest, whatever world acceleration remains is the
//     accelerometer's bias plus any error in G (the plain mean over the
//     first rest, then a running average); it is subtracted once the body
//     moves. A strong acceleration held steady for StillTime + AvgTime is
//     not a hand but a stale bias (the attitude filter still converging at
//     power-on): the bias is learned again from scratch.
//   - First rest: the gyro quiet and the specific force steady at its
//     running average for StillTime, then AvgTime of learning. Nothing is
//     integrated before it: start the badge at rest, as the attitude filter
//     needs anyway.
//   - Glides: a body that has looked still (quiet gyro, no acceleration,
//     steady force) for MaxGlide is at rest whatever its integrated speed;
//     this bounds the drift after a wrong velocity. A real glide that long
//     is stopped too.
//   - Damping (optional): velocity decays at Damping per second while
//     moving, trading a short motion's accuracy for bounded drift in a long
//     one.
//
// Strokes without a pause have no rest to correct them: the velocity error
// accumulates. A paddle waved without a pause (10 cm strokes of 0.5 s,
// 0.02 m/s² of noise, five draws) is off by up to 7 mm after 2 s, 2 cm
// after 4 s and 8 cm after 10 s.
//
// An attitude error θ leaks g·sin θ into the horizontal acceleration
// (0.17 m/s² per degree), which bias learning removes only while the
// attitude error stays what it was at rest: a 6-axis Mahony filter (no
// integral term) with a gyro bias of 0.02 rad/s, turned 90° and back,
// strays up to 0.6 m horizontally (fpv-gate); vertical motion of an upright
// badge stays within a millimetre.
//
// float32 throughout (the badge's filters are float32; per sample this is
// a few dozen operations, cheap even in software floating point). No
// allocation; Nav is a value with fixed state.
package strapdown

import "errors"

// ErrConfig is returned by New for an invalid Config.
var ErrConfig = errors.New("strapdown: invalid configuration")

// StandardGravity in m/s².
const StandardGravity = 9.80665

// Config sets the integrator. DefaultConfig suits a hand-held badge.
type Config struct {
	G          float32 // gravity removed from the world z axis, m/s² (> 0)
	StillGyro  float32 // rad/s: a faster rotation is motion (> 0)
	StillAccel float32 // m/s²: more linear acceleration than this is motion (> 0)
	StillSpeed float32 // m/s: rest needs the integrated speed under this; half of it gained during a rest ends the rest (> 0)
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

// Nav is the integrator's state: position P (m) and velocity V (m/s) in the
// world frame of the attitude (z up), the last linear acceleration A (m/s²,
// bias removed), the learned world bias Bias (m/s²), and whether the body
// is Still (velocity held at zero).
type Nav struct {
	P, V, A, Bias [3]float32
	Still         bool

	cfg        Config
	avg        [3]float32 // low-pass of the specific force, sensor frame
	primed     bool       // avg holds a sample
	calibrated bool       // Bias has been learned over one full rest
	stillFor   float32    // s of the first rest so far
	learned    float32    // samples averaged into Bias during the first rest
	quietFor   float32    // s moving with a quiet gyro, no acceleration and speed under StillSpeed
	glideFor   float32    // s moving with a quiet gyro, no acceleration and a steady force
	staleFor   float32    // s with a quiet gyro and a strong steady acceleration
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
// forgets the average force; the learned bias is kept (it belongs to the
// sensor).
func (n *Nav) Reset() {
	n.P, n.V, n.A = [3]float32{}, [3]float32{}, [3]float32{}
	n.Still, n.primed, n.stillFor = false, false, 0
	n.quietFor, n.glideFor, n.staleFor = 0, 0, 0
	n.sNew, n.sOld = shadow{}, shadow{}
}

// Update advances by dt seconds with the attitude q ([w, x, y, z], sensor
// to world, unit length), the accelerometer's specific force f (sensor
// frame, m/s²; +g up at rest) and the gyro rate w (sensor frame, rad/s).
// A sample with a non-positive or non-finite dt, or any non-finite value,
// is ignored.
func (n *Nav) Update(dt float32, q [4]float32, f, w [3]float32) {
	if !(dt > 0) || !finite(dt+q[0]+q[1]+q[2]+q[3]+f[0]+f[1]+f[2]+w[0]+w[1]+w[2]) {
		return
	}
	c := &n.cfg

	// world acceleration: rotate f by q, remove gravity
	r := rotate(q, f)
	r[2] -= c.G

	// the measures: a quiet gyro; a specific force steady at its running
	// average (needs no bias); no linear acceleration left after the bias
	if !n.primed {
		n.avg, n.primed = f, true
	}
	k := min(dt/c.AvgTime, 1)
	var steady2, left2 float32
	for i := range 3 {
		d := f[i] - n.avg[i]
		n.avg[i] += k * d
		steady2 += d * d
		e := r[i] - n.Bias[i]
		left2 += e * e
	}
	lim2 := c.StillAccel * c.StillAccel
	quiet := w[0]*w[0]+w[1]*w[1]+w[2]*w[2] < c.StillGyro*c.StillGyro
	steady, calm := steady2 < lim2, left2 < lim2

	if !n.calibrated {
		// first rest: no bias to judge acceleration by, so a steady force;
		// the bias is the plain mean of the rest's samples
		if quiet && steady {
			n.stillFor += dt
		} else {
			n.stillFor = 0
		}
		n.Still = n.stillFor >= c.StillTime
		if !n.Still {
			return // no bias yet: hold the origin
		}
		n.learned++
		for i := range 3 {
			n.Bias[i] += (r[i] - n.Bias[i]) / n.learned
		}
		if n.stillFor >= c.StillTime+c.AvgTime {
			n.calibrated = true
			n.rest()
		}
		return
	}

	// a strong acceleration held steady is no hand: the bias is stale, the
	// stretch was a rest, and what was integrated over it is error
	if quiet && steady && !calm {
		if n.staleFor == 0 {
			n.pStale = n.P
		}
		n.staleFor += dt
	} else {
		n.staleFor = 0
	}
	if n.staleFor >= c.StillTime+c.AvgTime {
		n.calibrated, n.learned, n.staleFor = false, 0, 0
		n.stillFor, n.Still = c.StillTime, true
		n.P, n.V, n.A = n.pStale, [3]float32{}, [3]float32{}
		return
	}

	if n.Still {
		wake := !quiet || !calm && !steady
		if calm { // a gentle start: no acceleration to see, only speed
			wake = wake || 2*abs(n.sOld.V[2]) >= c.StillSpeed
		}
		if !wake {
			n.atRest(r, dt)
			return
		}
		// moving since before the older snapshot at the latest: take up what
		// was integrated from there, against the bias of then
		n.Bias = n.biasOld
		n.V, n.A = n.sOld.V, n.sOld.A
		for i := range 3 {
			n.P[i] += n.sOld.P[i]
		}
		n.Still, n.quietFor, n.glideFor = false, 0, 0
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
		v := (n.V[i] + (n.A[i]+a)*dt/2) * keep
		n.P[i] += (n.V[i] + v) * dt / 2
		n.V[i], n.A[i] = v, a
	}

	// rest: everything quiet at once for StillTime, the speed included; or
	// looking still for MaxGlide whatever the speed (bounds the drift).
	// Speed is judged vertically only: an attitude error θ puts g·sin θ
	// into the horizontal acceleration (0.17 m/s² per degree, as much as a
	// gentle stroke's) but only g·(1 − cos θ) into the vertical.
	if quiet && calm && abs(n.V[2]) < c.StillSpeed {
		n.quietFor += dt
	} else {
		n.quietFor = 0
	}
	if quiet && calm && steady {
		n.glideFor += dt
	} else {
		n.glideFor = 0
	}
	if n.quietFor >= c.StillTime || n.glideFor >= c.MaxGlide {
		// ZUPT. The acceleration was nil over the whole quiet stretch, so
		// the velocity left now was there all along and is error (a real
		// stroke's tail has slowed to nothing): take back its distance.
		win := n.glideFor
		if n.quietFor >= c.StillTime {
			win = n.quietFor
		}
		for i := range 3 {
			n.P[i] -= n.V[i] * win
		}
		n.Still = true
		n.V, n.A = [3]float32{}, [3]float32{}
		n.rest()
	}
}

// rest starts a rest: both bias snapshots are the bias now, both shadows
// empty.
func (n *Nav) rest() {
	n.biasOld, n.biasNew, n.snapFor, n.restFor = n.Bias, n.Bias, 0, 0
	n.sOld, n.sNew = shadow{}, shadow{}
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
func (n *Nav) atRest(r [3]float32, dt float32) {
	c := &n.cfg
	n.V, n.A = [3]float32{}, [3]float32{}
	n.restFor += dt
	k := min(dt/min(c.AvgTime+n.restFor, 4*c.AvgTime), 1)
	for i := range 3 {
		n.Bias[i] += k * (r[i] - n.Bias[i])
	}
	n.sOld.step(r, n.biasOld, dt)
	n.sNew.step(r, n.biasNew, dt)
	n.snapFor += dt
	if n.snapFor >= c.StillTime {
		n.biasOld, n.biasNew, n.snapFor = n.biasNew, n.Bias, 0
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

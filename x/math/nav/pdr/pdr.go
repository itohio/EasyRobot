// Package pdr is pedestrian dead reckoning for a body-worn IMU: Detector
// finds steps in the accelerometer magnitude, Heading gives the walking
// direction from an attitude filter, and Track advances a 2D position by one
// step length along that heading for each step.
//
// Heading takes the filter's orientation (e.g. mahony.Output(), [w,x,y,z],
// sensor to world, z up) and the body axis that points where the wearer
// walks, in sensor coordinates, and returns that axis's direction projected
// onto the horizontal plane. It works for any mounting (a badge hangs with
// its screen out: the walking axis is the IMU's −Z), unlike the yaw of an
// Euler decomposition, which is undefined when the sensor's x axis is near
// vertical. Without a magnetometer (an MPU6050 has none) the heading is
// relative to the start and drifts with the gyro bias about the vertical,
// so the track turns slowly even when walking straight; with a magnetometer
// it is absolute. Step length is a constant here; it varies with the walker
// and the pace.
//
// All types are values with fixed state: no allocation per sample.
package pdr

import (
	"errors"

	"github.com/chewxy/math32"
)

// ErrConfig is returned for an invalid DetectorConfig.
var ErrConfig = errors.New("pdr: invalid detector configuration")

// DetectorConfig sets the step detector.
type DetectorConfig struct {
	SampleRate  float32 // Hz of the accelerometer samples
	Cutoff      float32 // Hz, first-order low-pass on |a|; under SampleRate/2
	Threshold   float32 // m/s² the filtered |a| must rise above its baseline
	MinInterval float32 // s between the starts of two steps (arming is blocked this long)
	Baseline    float32 // s, time constant of the running baseline (≈ gravity)
}

// DefaultDetectorConfig suits a badge or phone carried on the body: steps of
// 0.3–2.5 a second with a bounce of at least about 1.25–1.5 m/s² peak (a
// smaller bounce, e.g. a loosely hung badge, needs a lower Threshold).
// Running at 3 steps a second or faster is counted at about half the rate
// (MinInterval 0.3 s).
func DefaultDetectorConfig(sampleRate float32) DetectorConfig {
	return DetectorConfig{SampleRate: sampleRate, Cutoff: 3, Threshold: 1, MinInterval: 0.3, Baseline: 2}
}

// Detector counts steps: |a| is low-pass filtered and compared with a slow
// running baseline (gravity and the sensor's scale error); a rise above the
// baseline by Threshold followed by the fall back through it is one step,
// at least MinInterval after the previous one. Like any threshold detector it
// also counts a single jolt big enough to cross the threshold.
type Detector struct {
	alpha, beta   float32 // low-pass and baseline smoothing per sample
	threshold     float32
	minGap        int32 // samples
	y, base       float32
	since         int32 // samples since the last step
	armed, primed bool
	steps         int
}

// maxGap bounds MinInterval·SampleRate so the sample count fits an int32.
const maxGap = 1 << 30

// NewDetector checks c and returns a detector.
func NewDetector(c DetectorConfig) (Detector, error) {
	finite := func(v float32) bool { return v-v == 0 } // false for ±Inf and NaN
	if !(finite(c.SampleRate) && finite(c.Cutoff) && finite(c.Threshold) && finite(c.MinInterval) && finite(c.Baseline)) ||
		!(c.SampleRate > 0 && c.Cutoff > 0 && c.Cutoff < c.SampleRate/2 && c.Threshold > 0 && c.MinInterval >= 0 && c.Baseline > 0) ||
		c.MinInterval*c.SampleRate > maxGap {
		return Detector{}, ErrConfig
	}
	gap := int32(c.MinInterval*c.SampleRate + 0.5)
	return Detector{
		alpha:     1 - math32.Exp(-2*math32.Pi*c.Cutoff/c.SampleRate),
		beta:      1 - math32.Exp(-1/(c.Baseline*c.SampleRate)),
		threshold: c.Threshold,
		minGap:    gap,
		since:     gap,
	}, nil
}

// Update takes one accelerometer sample (m/s², any orientation: only |a| is
// used) and reports whether it completes a step. A sample with a NaN or
// infinite component is ignored. The filter starts from the first sample
// with a non-zero magnitude (a sensor not yet awake reads zero).
func (d *Detector) Update(a [3]float32) bool {
	m := math32.Sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])
	if m-m != 0 {
		return false // NaN or Inf: it would poison the filter state
	}
	if !d.primed {
		if m == 0 {
			return false
		}
		d.y, d.base, d.primed = m, m, true
	}
	d.y += d.alpha * (m - d.y)
	d.base += d.beta * (d.y - d.base)
	if d.since < d.minGap {
		d.since++
	}
	diff := d.y - d.base
	if !d.armed {
		if diff > d.threshold && d.since >= d.minGap {
			d.armed = true
		}
		return false
	}
	if diff > 0 {
		return false
	}
	d.armed = false
	d.since = 0
	d.steps++
	return true
}

// Steps is the number of steps found so far.
func (d *Detector) Steps() int { return d.steps }

// minHorizontal is the smallest horizontal length (of a unit axis) Heading
// accepts: the axis must be at least about 6° from vertical.
const minHorizontal = 0.1

// Heading returns the direction, in radians counter-clockwise from the
// world's x axis, of the body axis (sensor coordinates, any length) rotated
// into the world by q = [w, x, y, z] (sensor to world, z up) and projected
// onto the horizontal plane. ok is false when the axis is within about 6° of
// vertical (no direction to project), or zero.
func Heading(q [4]float32, axis [3]float32) (heading float32, ok bool) {
	n := math32.Sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2])
	if n == 0 {
		return 0, false
	}
	w, x, y, z := q[0], q[1], q[2], q[3]
	vx, vy, vz := axis[0]/n, axis[1]/n, axis[2]/n
	// v' = v + w·t + u × t with t = 2·(u × v), u = (x, y, z)
	tx, ty, tz := 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
	hx := vx + w*tx + y*tz - z*ty
	hy := vy + w*ty + z*tx - x*tz
	if hx*hx+hy*hy < minHorizontal*minHorizontal {
		return 0, false
	}
	return math32.Atan2(hy, hx), true
}

// Track is a dead-reckoned 2D position in the heading's frame (x along
// heading 0, y along heading +π/2), metres.
type Track struct {
	X, Y       float32
	Steps      int
	StepLength float32 // m
}

// Step moves the position one StepLength along heading (radians,
// counter-clockwise from x).
func (t *Track) Step(heading float32) {
	t.X += t.StepLength * math32.Cos(heading)
	t.Y += t.StepLength * math32.Sin(heading)
	t.Steps++
}

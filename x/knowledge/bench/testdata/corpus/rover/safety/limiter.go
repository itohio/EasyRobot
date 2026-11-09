// Package safety holds the rover's runtime protections.
package safety

import (
	"math"

	"example.com/fixture/rover/imu"
	"example.com/fixture/rover/motor"
)

// AttitudeLimiter scales throttle down as pitch or roll leave the envelope.
type AttitudeLimiter struct {
	MaxPitch, MaxRoll float64 // radians where scaling starts
	Stop              float64 // radians where throttle is cut
	Speed             *motor.Controller
}

// Scale returns the throttle factor in [0, 1] for the given attitude.
func (l AttitudeLimiter) Scale(a imu.Attitude) float64 {
	worst := math.Max(math.Abs(a.Pitch)/l.MaxPitch, math.Abs(a.Roll)/l.MaxRoll)
	if worst <= 1 {
		return 1
	}
	stop := l.Stop / math.Min(l.MaxPitch, l.MaxRoll)
	if worst >= stop {
		return 0
	}
	return (stop - worst) / (stop - 1)
}

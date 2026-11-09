// Package odometry integrates wheel encoder ticks into a planar pose.
package odometry

import (
	"math"

	"example.com/fixture/rover/imu"
)

// WheelConfig describes the drive geometry.
type WheelConfig struct {
	TicksPerRev int
	Radius      float64 // m
	Track       float64 // m between wheel centres
}

// Pose is a planar position and heading.
type Pose struct {
	X, Y, Theta float64
}

// Integrate advances p by the given encoder tick deltas (differential drive).
func Integrate(p Pose, ticksLeft, ticksRight int, cfg WheelConfig) Pose {
	perTick := 2 * math.Pi * cfg.Radius / float64(cfg.TicksPerRev)
	dl := float64(ticksLeft) * perTick
	dr := float64(ticksRight) * perTick
	d := (dl + dr) / 2
	dtheta := (dr - dl) / cfg.Track
	p.X += d * math.Cos(p.Theta+dtheta/2)
	p.Y += d * math.Sin(p.Theta+dtheta/2)
	p.Theta += dtheta
	return p
}

// DetectSlip reports wheel slip when the encoder yaw rate disagrees with the
// gyro yaw rate by more than threshold rad/s.
func DetectSlip(encoderYawRate float64, s imu.Sample, threshold float64) bool {
	return math.Abs(encoderYawRate-s.Gyro[2]) > threshold
}

// Package imu fuses gyroscope and accelerometer samples into an attitude.
package imu

import "math"

// Sample is one raw IMU reading.
type Sample struct {
	Gyro  [3]float64 // rad/s
	Accel [3]float64 // m/s²
}

// Attitude is roll and pitch in radians.
type Attitude struct {
	Roll, Pitch float64
}

// Fusion is a complementary filter.
type Fusion struct {
	Alpha float64
	Bias  [3]float64
	att   Attitude
}

// Update blends the integrated gyro rate with the accelerometer gravity
// direction and returns the new attitude.
func (f *Fusion) Update(s Sample, dt float64) Attitude {
	gx := s.Gyro[0] - f.Bias[0]
	gy := s.Gyro[1] - f.Bias[1]
	accRoll := math.Atan2(s.Accel[1], s.Accel[2])
	accPitch := math.Atan2(-s.Accel[0], math.Hypot(s.Accel[1], s.Accel[2]))
	f.att.Roll = f.Alpha*(f.att.Roll+gx*dt) + (1-f.Alpha)*accRoll
	f.att.Pitch = f.Alpha*(f.att.Pitch+gy*dt) + (1-f.Alpha)*accPitch
	return f.att
}

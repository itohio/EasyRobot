package imu

// Calibrate averages samples taken while the robot is motionless and returns
// the gyro bias to store in Fusion.Bias.
func Calibrate(samples []Sample) [3]float64 {
	var bias [3]float64
	if len(samples) == 0 {
		return bias
	}
	for _, s := range samples {
		for i := range bias {
			bias[i] += s.Gyro[i]
		}
	}
	for i := range bias {
		bias[i] /= float64(len(samples))
	}
	return bias
}

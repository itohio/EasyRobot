package ahrs

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
)

// BenchmarkMahony6 is the badge demo's call: the 6-axis path (no
// magnetometer, KI 0), one Update per 100 Hz IMU sample with the
// accelerometer and gyro in a Matrix3x3 (KERNELS-1).
func BenchmarkMahony6(b *testing.B) {
	m := NewMahony(WithKP(1), WithKI(0), WithMagnetometer(false))
	var acc, gyro [64][3]float32
	for i := range acc {
		s := float32(i%16) / 16
		acc[i] = [3]float32{0.1 * s, 0.4 + s, 9.7}
		gyro[i] = [3]float32{0.4, 0.01 * s, 0.05}
	}
	var in mat.Matrix3x3
	b.ReportAllocs()
	for i := 0; b.Loop(); i++ {
		in = mat.Matrix3x3{acc[i&63], gyro[i&63], {}}
		m.Update(0.01, &in)
	}
}

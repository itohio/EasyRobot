package ahrs

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// The filters' quaternion is [w, x, y, z] and rotates sensor-frame vectors
// into the world frame (z up; with a magnetometer x points along the
// horizontal magnetic field).

const deg = math.Pi / 180

// fromEuler is the [w,x,y,z] quaternion of yaw ψ about z, then pitch θ about
// y, then roll φ about x (Z-Y-X).
func fromEuler(roll, pitch, yaw float64) vec.Quaternion {
	cr, sr := math.Cos(roll/2), math.Sin(roll/2)
	cp, sp := math.Cos(pitch/2), math.Sin(pitch/2)
	cy, sy := math.Cos(yaw/2), math.Sin(yaw/2)
	return vec.Quaternion{
		float32(cr*cp*cy + sr*sp*sy),
		float32(sr*cp*cy - cr*sp*sy),
		float32(cr*sp*cy + sr*cp*sy),
		float32(cr*cp*sy - sr*sp*cy),
	}
}

// toSensor rotates a world vector into the sensor frame of orientation q
// (q* v q).
func toSensor(q vec.Quaternion, v [3]float64) [3]float32 {
	w, x, y, z := float64(q[0]), -float64(q[1]), -float64(q[2]), -float64(q[3]) // conjugate
	// t = 2 (u × v); v' = v + w t + u × t
	tx := 2 * (y*v[2] - z*v[1])
	ty := 2 * (z*v[0] - x*v[2])
	tz := 2 * (x*v[1] - y*v[0])
	return [3]float32{
		float32(v[0] + w*tx + y*tz - z*ty),
		float32(v[1] + w*ty + z*tx - x*tz),
		float32(v[2] + w*tz + x*ty - y*tx),
	}
}

// gravityEstimate is the world's up direction in the sensor frame as the
// filter's quaternion has it (what the accelerometer should read at rest).
func gravityEstimate(q vec.Quaternion) [3]float32 {
	return toSensor(q, [3]float64{0, 0, 1})
}

func near3(t *testing.T, what string, got, want [3]float32, tol float32) {
	t.Helper()
	for i := range got {
		if d := got[i] - want[i]; !(d <= tol && d >= -tol) { // NaN fails too
			t.Fatalf("%s: %v, want %v (±%v)", what, got, want, tol)
		}
	}
}

func run(m *MahonyAHRS, n int, dt float32, accel, gyro, magn [3]float32) {
	in := mat.Matrix3x3{accel, gyro, magn}
	for i := 0; i < n; i++ {
		m.Update(dt, &in)
	}
}

// A still sensor tilted by roll 30° and pitch −20°: both paths converge to
// that tilt (gravity estimate equals the measured direction), not to its
// mirror image.
func TestMahonyConvergesToAStaticTilt(t *testing.T) {
	truth := fromEuler(30*deg, -20*deg, 40*deg)
	accel := toSensor(truth, [3]float64{0, 0, 1})
	magn := toSensor(truth, [3]float64{0.4, 0, -0.3})
	for _, mag := range []bool{false, true} {
		m := NewMahony(WithMagnetometer(mag), WithKP(2), WithKI(0))
		run(m, 4000, 0.01, accel, [3]float32{}, magn) // 40 s at 100 Hz
		near3(t, "gravity estimate", gravityEstimate(m.Output()), accel, 0.01)
		if mag {
			roll, pitch, yaw := Euler(m.Output())
			near3(t, "9-DOF roll/pitch/yaw", [3]float32{roll, pitch, yaw}, [3]float32{30 * deg, -20 * deg, 40 * deg}, 0.02)
		}
	}
}

// A still sensor whose gyro has a constant bias: the integral feedback
// learns it (GainI·eInt → −bias) and the attitude stays exact. Without the
// integral the proportional term alone leaves a tilt of |bias|/Kp ≈ 0.1 rad
// here, far outside the tolerance. 6-axis: only the bias across gravity is
// observable (the part about the gravity axis is a yaw drift the filter
// cannot see); 9-axis: all of it.
func TestMahonyCancelsAGyroBias(t *testing.T) {
	truth := fromEuler(20*deg, -10*deg, 30*deg)
	accel := toSensor(truth, [3]float64{0, 0, 1})
	magn := toSensor(truth, [3]float64{0.4, 0, -0.3})
	bias := [3]float32{0.12, -0.16, 0.1}
	for _, mag := range []bool{false, true} {
		m := NewMahony(WithMagnetometer(mag), WithKP(2), WithKI(0.005))
		m.q = truth                            // start on the true attitude: only the bias pulls it away
		run(m, 60000, 0.01, accel, bias, magn) // 10 min at 100 Hz
		near3(t, "gravity estimate", gravityEstimate(m.Output()), accel, 0.002)
		want := [3]float32{-bias[0], -bias[1], -bias[2]}
		if !mag { // −(bias minus its component along gravity)
			along := bias[0]*accel[0] + bias[1]*accel[1] + bias[2]*accel[2]
			for i := range want {
				want[i] += along * accel[i]
			}
		}
		learned := [3]float32{m.GainI * m.eInt[0], m.GainI * m.eInt[1], m.GainI * m.eInt[2]}
		near3(t, "integral feedback", learned, want, 1e-3)
	}
}

// No accelerometer reading (free fall, or a zero sample): no NaN, the gyro
// alone integrates. The accelerometer's scale does not matter, only its
// direction.
func TestMahonyAccelEdgeCases(t *testing.T) {
	m := NewMahony(WithKP(2), WithKI(0.005))
	run(m, 100, 0.01, [3]float32{}, [3]float32{0, 0, math.Pi / 2}, [3]float32{})
	if _, _, yaw := Euler(m.Output()); !(math.Abs(float64(yaw)-90*deg) <= 2*deg) {
		t.Fatalf("zero accel: yaw %v, want 90° from the gyro alone", yaw/deg)
	}
	a1, a2 := NewMahony(WithKP(2)), NewMahony(WithKP(2))
	tilt := toSensor(fromEuler(25*deg, 10*deg, 0), [3]float64{0, 0, 1})
	run(a1, 500, 0.01, tilt, [3]float32{}, [3]float32{})
	run(a2, 500, 0.01, [3]float32{9.81 * tilt[0], 9.81 * tilt[1], 9.81 * tilt[2]}, [3]float32{}, [3]float32{})
	for i := range a1.q {
		if d := a1.q[i] - a2.q[i]; !(d <= 1e-5 && d >= -1e-5) {
			t.Fatalf("|a| = 1 vs 9.81: %v vs %v", a1.q, a2.q)
		}
	}
}

// Turning at +90°/s about the vertical for 1 s turns the heading by +90°
// (counter-clockwise seen from above).
func TestMahonyIntegratesTheYawRate(t *testing.T) {
	m := NewMahony(WithKP(2), WithKI(0))
	run(m, 100, 0.01, [3]float32{0, 0, 9.81}, [3]float32{0, 0, math.Pi / 2}, [3]float32{})
	if _, _, yaw := Euler(m.Output()); math.Abs(float64(yaw)-90*deg) > 2*deg {
		t.Fatalf("yaw %.1f°, want 90°", yaw/deg)
	}
}

func TestEulerOfKnownOrientations(t *testing.T) {
	for _, c := range [][3]float64{{0, 0, 0}, {30, -20, 40}, {-170, 60, -120}, {10, 0, 179}, {0, 89, 0}} {
		roll, pitch, yaw := Euler(fromEuler(c[0]*deg, c[1]*deg, c[2]*deg))
		near3(t, "Euler", [3]float32{roll, pitch, yaw}, [3]float32{float32(c[0] * deg), float32(c[1] * deg), float32(c[2] * deg)}, 2e-3)
	}
	// Pitch +90° with sqrt(½) rounded up in float32: 2(wy − zx) is just over
	// 1, which must still give 90°, not NaN.
	if _, pitch, _ := Euler(vec.Quaternion{0.70710683, 0, 0.70710683, 0}); !(math.Abs(float64(pitch)-90*deg) <= 1e-3) {
		t.Fatalf("pitch at gimbal lock %v, want π/2", pitch)
	}
	if _, pitch, _ := Euler(vec.Quaternion{0.70710683, 0, -0.70710683, 0}); !(math.Abs(float64(pitch)+90*deg) <= 1e-3) {
		t.Fatalf("pitch at negative gimbal lock %v, want −π/2", pitch)
	}
}

// Update with a *mat.Matrix3x3 allocates nothing (the badge calls it at
// 100 Hz under TinyGo).
func TestMahonyUpdateDoesNotAllocate(t *testing.T) {
	m := NewMahony()
	in := mat.Matrix3x3{{0, 0, 9.81}, {0.01, 0, 0}, {}}
	if n := testing.AllocsPerRun(100, func() { m.Update(0.01, &in) }); n != 0 {
		t.Fatalf("%v allocations per update", n)
	}
}

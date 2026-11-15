package ahrs

import (
	"math"
	"testing"

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

func near3(t *testing.T, what string, got, want [3]float32, tol float32) {
	t.Helper()
	for i := range got {
		if d := got[i] - want[i]; !(d <= tol && d >= -tol) { // NaN fails too
			t.Fatalf("%s: %v, want %v (±%v)", what, got, want, tol)
		}
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

// At and near pitch ±90° roll and yaw are not separable (only roll ∓ yaw
// is), but the angles Euler returns must still describe q: rebuilt into a
// quaternion they are within 0.1° of it (up to 179.5° off, e.g.
// roll 60°, pitch 90° came back as roll −90°, yaw 90°).
func TestEulerAtThePitchPoles(t *testing.T) {
	s := uint32(2463534242)
	rnd := func() float64 { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return float64(s) / float64(^uint32(0)) }
	worst := 0.0
	for i := 0; i < 20000; i++ {
		pitch := 90 - 0.01*rnd() // within 0.01° of the pole
		if i%4 >= 2 {
			pitch = 90 - 0.1*rnd() // both sides of the switch at ≈ 89.97°
		}
		if i%2 == 1 {
			pitch = -pitch
		}
		if i%5 == 0 {
			pitch = math.Copysign(90, pitch) // exactly at the pole
		}
		q := fromEuler((360*rnd()-180)*deg, pitch*deg, (360*rnd()-180)*deg)
		roll, p, yaw := Euler(q)
		back := fromEuler(float64(roll), float64(p), float64(yaw))
		dot := math.Abs(float64(q[0])*float64(back[0]) + float64(q[1])*float64(back[1]) + float64(q[2])*float64(back[2]) + float64(q[3])*float64(back[3]))
		angle := 2 * math.Acos(math.Min(1, dot)) / deg
		if angle > worst {
			worst = angle
		}
		if !(angle <= 0.1) {
			t.Fatalf("case %d (pitch %.6f°): Euler gives (%.3f°, %.3f°, %.3f°), %.3f° from q", i, pitch, roll/deg, p/deg, yaw/deg, angle)
		}
	}
	t.Logf("worst rebuilt error %.4f°", worst)
}

// Package estimator is the shape every navigation estimator brick has: an
// Attitude estimator turns IMU samples into an attitude (mahony); a Motion
// estimator also into a velocity and a position (eskf). A caller holds the
// brick it needs by the smallest interface that serves it, and bricks of
// one shape replace each other.
//
// The interfaces return values, not pointers, and a brick behind them
// allocates nothing per sample (a pointer in an interface value doesn't
// allocate).
package estimator

import (
	"errors"

	"github.com/chewxy/math32"
)

// ErrSample reports a sample an estimator refused: a NaN or an infinity in
// it, or a dt not above zero. The estimator's state is untouched.
var ErrSample = errors.New("estimator: sample refused (not finite, or dt not above zero)")

// Attitude estimates the attitude from a 6-axis IMU.
type Attitude interface {
	// Update takes one sample: dt (s), acc (m/s², the specific force, +g
	// up at rest) and gyro (rad/s) in the sensor frame; ErrSample when it
	// is refused.
	Update(dt float32, acc, gyro [3]float32) error
	// Attitude is the quaternion [w,x,y,z] from the sensor frame to the
	// world (z up).
	Attitude() [4]float32
	// Ready is true once a sample has set the attitude.
	Ready() bool
}

// Motion estimates the attitude, the velocity and the position.
type Motion interface {
	Attitude
	// Velocity in the world, m/s.
	Velocity() [3]float32
	// Position in the world, m, from where the estimator started.
	Position() [3]float32
}

// Level is the attitude whose up is the force f (sensor frame), yaw 0: the
// rotation taking f's direction to ẑ, the first attitude a brick takes
// from its first sample; ok false for a zero force.
func Level(f [3]float32) (q [4]float32, ok bool) {
	h2 := f[0]*f[0] + f[1]*f[1]
	g := math32.Sqrt(h2 + f[2]*f[2])
	if !(g > 0) {
		return q, false
	}
	// the rotation about f×ẑ = (f_y, −f_x, 0): q ∝ [g + f_z, f_y, −f_x, 0],
	// normalised by its own norm (not by √(2g(g+f_z)), which holds only for
	// an exact |f| and, upside down, amplified float32's rounding of it by
	// 1/(g+f_z)); facing down, g + f_z cancels, so it is h²/(g − f_z)
	w := g + f[2]
	if f[2] < 0 {
		w = h2 / (g - f[2])
	}
	n2 := w*w + h2
	if !(n2 > 1e-24*g*g) { // upside down (within 1e-12 rad): π about x
		return [4]float32{0, 1, 0, 0}, true
	}
	s := 1 / math32.Sqrt(n2)
	return [4]float32{w * s, f[1] * s, -f[0] * s, 0}, true
}

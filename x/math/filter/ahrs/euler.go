package ahrs

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// Euler returns roll, pitch and yaw in radians (Z-Y-X: yaw about the world's
// z, then pitch about y, then roll about x) of a filter output q = [w, x, y,
// z], the rotation from the sensor frame to the world frame (z up). Positive
// pitch dips the sensor's x axis below the horizon; positive roll lifts its
// y axis. Yaw is the heading of the sensor's x axis, counter-clockwise seen
// from above; it is relative to the start without a magnetometer, and drifts
// with the gyro's z bias.
//
// Madgwick's and the Mahony brick's (x/math/nav/mahony) outputs put w first; vec.Quaternion's own
// Roll/Pitch/Yaw read w from index 3, so they do not apply to them.
func Euler(q vec.Quaternion) (roll, pitch, yaw float32) {
	w, x, y, z := q[0], q[1], q[2], q[3]
	s := 2 * (w*y - z*x)
	if s > 1 {
		s = 1
	} else if s < -1 {
		s = -1
	}
	pitch = math32.Asin(s)
	if s >= poleSin || s <= -poleSin {
		// At the poles only roll ∓ yaw is defined, and the general formulas
		// below divide rounding residue by rounding residue. Report roll 0
		// and put the whole rotation about the vertical into yaw:
		// atan2(−R01, R11) = yaw − roll at +90°, yaw + roll at −90°.
		return 0, pitch, math32.Atan2(2*(w*z-x*y), 1-2*(x*x+z*z))
	}
	roll = math32.Atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
	yaw = math32.Atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
	return roll, pitch, yaw
}

// poleSin is the |sin(pitch)| from which Euler treats the pose as at a pole
// (within about 0.03° of ±90°): one float32 step below 1.
const poleSin = 0.99999988

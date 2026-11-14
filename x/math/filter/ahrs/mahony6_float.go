//go:build !fixedmotion

package ahrs

// fixedPath reports whether the 6-axis update runs in fixed point
// (-tags fixedmotion, see mahony6_fixed.go). By default it is float32.
const fixedPath = false

// fixed6 holds the fixed-point 6-axis state; empty in the float build.
type fixed6 struct{}

// update6 is the 6-axis update: float32 in the default build.
func (m *MahonyAHRS) update6() { m.calculateWOMag() }

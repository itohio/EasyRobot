//go:build !fixedmotion

package mahony

// fixedPath reports whether the 6-axis update runs in fixed point
// (-tags fixedmotion); in this build it doesn't.
const fixedPath = false

// fixed6 holds the fixed-point 6-axis state; empty in the float build.
type fixed6 struct{}

// step is the 6-axis update: float32 in the default build.
func (f *Filter) step(acc, gyro [3]float32, dt float32) { f.stepFloat(acc, gyro, dt) }

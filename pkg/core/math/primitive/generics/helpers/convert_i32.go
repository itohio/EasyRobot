//go:build 386 || arm || mips || mipsle

package helpers

import "math"

// ClampableToInt types that need clamping when converting to int (32-bit platform).
// On 32-bit platforms, int64 also needs clamping when converting to int.
type ClampableToInt interface {
	~float64 | ~int64 | ~float32
}

// ClampToIntValue clamps a single value to int range (32-bit platform).
// On 32-bit platforms, int64 also needs clamping when converting to int.
func ClampToIntValue[U ClampableToInt](v U) int {
	if v > U(math.MaxInt) {
		return math.MaxInt
	}
	if v < U(math.MinInt) {
		return math.MinInt
	}
	return int(v)
}

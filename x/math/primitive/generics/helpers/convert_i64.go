//go:build amd64 || arm64 || ppc64 || ppc64le || mips64 || mips64le || riscv64 || s390x

package helpers

import "math"

// ClampableToInt types that need clamping when converting to int (64-bit platform).
// On 64-bit platforms, only float32/float64 need clamping when converting to int.
type ClampableToInt interface {
	~float64 | ~float32
}

// ClampToIntValue clamps a single value to int range (64-bit platform).
// On 64-bit platforms, only float32/float64 need clamping when converting to int.
func ClampToIntValue[U ClampableToInt](v U) int {
	if v > U(math.MaxInt) {
		return math.MaxInt
	}
	if v < U(math.MinInt) {
		return math.MinInt
	}
	return int(v)
}

// ClampToIntComplex clamps a complex value's real part to int range (32-bit platform).
// Only the real component is considered for clamping.
// On 32-bit platforms, int64 and float32/float64 real parts need clamping.
func ClampToIntComplex(v complex64) int {
	realPart := real(v)
	if realPart > float32(math.MaxInt) {
		return math.MaxInt
	}
	if realPart < float32(math.MinInt) {
		return math.MinInt
	}
	return int(realPart)
}

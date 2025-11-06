//go:build amd64 || arm64 || ppc64 || ppc64le || mips64 || mips64le || riscv64 || s390x

package generics

// valueConvertToInt handles conversion to int with clamping (64-bit platform).
// On 64-bit platforms, only float32/float64 need clamping when converting to int.
func valueConvertToInt[T, U Numeric](value T) (zeroU U) {
	switch v := any(value).(type) {
	case float32:
		return U(clampToIntValue(v))
	case float64:
		return U(clampToIntValue(v))
	default:
		return U(value)
	}
}


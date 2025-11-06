//go:build 386 || arm || mips || mipsle

package helpers

// ValueConvertToInt handles conversion to int with clamping (32-bit platform).
// On 32-bit platforms, int64 also needs clamping when converting to int.
func ValueConvertToInt[T, U Numeric](value T) (zeroU U) {
	switch v := any(value).(type) {
	case float32:
		return U(ClampToIntValue(v))
	case float64:
		return U(ClampToIntValue(v))
	case int64:
		return U(ClampToIntValue(v))
	default:
		return U(value)
	}
}

//go:build 386 || arm || mips || mipsle

package types

// ValueConvertToInt handles conversion to int with clamping (32-bit platform).
// On 32-bit platforms, int64 also needs clamping when converting to int.
// T can be any Numeric (including complex64), but U must be NumericReal (int).
func ValueConvertToInt[T Number, U NumberReal](value T) (zeroU U) {
	switch v := any(value).(type) {
	case float32:
		return U(ClampToIntValue(v))
	case float64:
		return U(ClampToIntValue(v))
	case int64:
		return U(ClampToIntValue(v))
	case complex64:
		// Extract real part and reuse ClampToIntValue
		return U(ClampToIntValue(float32(real(v))))
	case complex128:
		// Extract real part and reuse ClampToIntValue (complex128 not officially supported, but handle gracefully)
		return U(ClampToIntValue(float64(real(v))))
	case int:
		return U(v)
	case int32:
		return U(int(v))
	case int16:
		return U(int(v))
	case int8:
		return U(int(v))
	default:
		// Should not reach here - all supported numeric types are handled above
		// Return zero value as fallback
		return zeroU
	}
}

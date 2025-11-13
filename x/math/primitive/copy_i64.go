//go:build amd64 || arm64 || ppc64 || ppc64le || mips64 || mips64le || riscv64 || s390x

package primitive

// Types that need clamping when converting to int (sorted by type size)
type clampableToInt interface {
	~float64 | ~float32
}

// copyConvertToInt handles conversions to int with clamping.
// Handles: float32/float64 -> int (needs clamping)
func copyConvertToInt(dst []int, src any) []int {
	switch s := src.(type) {
	case []float32:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt(dst, s, n)
		}
	case []float64:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt(dst, s, n)
		}

	}
	return dst
}

// copyConvertNumeric dispatches to specialized functions only for down-conversions that need clamping.
// For up-conversions and same-type, uses generic function with direct T(src) conversion.
func copyConvertNumeric[T, U numeric](dst []T, src []U) any {
	// Check if this is a down-conversion that needs clamping
	// Down-conversions: float32/float64 -> int64/int32/int16/int8, int64 -> int32/int16/int8, int32 -> int16/int8, int16 -> int8
	switch d := any(dst).(type) {
	case []int64:
		// Check if source is larger (float32/float64) - needs clamping
		switch any(src).(type) {
		case []float32, []float64:
			return copyConvertToInt64(d, src)
		default:
			// int64->int64 or smaller->int64: direct conversion, no clamping
			return copyGeneric(dst, src)
		}
	case []int:
		// Check if source is larger (float32/float64/int64) - needs clamping
		switch any(src).(type) {
		case []float32, []float64:
			return copyConvertToInt(d, src)
		default:
			// int->int or smaller->int: direct conversion, no clamping
			return copyGeneric(dst, src)
		}
	case []int32:
		// Check if source is larger (float32/float64/int64) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int:
			return copyConvertToInt32(d, src)
		default:
			// int32->int32 or smaller->int32: direct conversion, no clamping
			return copyGeneric(dst, src)
		}
	case []int16:
		// Check if source is larger (float32/float64/int64/int32) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int32, []int:
			return copyConvertToInt16(d, src)
		default:
			// int16->int16 or int8->int16: direct conversion, no clamping
			return copyGeneric(dst, src)
		}
	case []int8:
		// Check if source is larger (float32/float64/int64/int32/int16) - needs clamping
		switch any(src).(type) {
		case []float32, []float64, []int64, []int32, []int16, []int:
			return copyConvertToInt8(d, src)
		default:
			// int8->int8: direct conversion, no clamping
			return copyGeneric(dst, src)
		}
	default:
		// For float32/float64: always direct conversion (up-conversion or same-type)
		return copyGeneric(dst, src)
	}
}

// copyConvertToInt32 handles conversions to int32 with clamping.
// Handles: float32/float64/int64 -> int32 (needs clamping)
// int32 -> int32 (same type, fast path)
// int16/int8 -> int32 (up-conversion, no clamping)
func copyConvertToInt32(dst []int32, src any) []int32 {
	switch s := src.(type) {
	case []float32:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt32(dst, s, n)
		}
	case []float64:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt32(dst, s, n)
		}
	case []int64:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt32(dst, s, n)
		}
	case []int:
		n := len(dst)
		if len(s) < n {
			n = len(s)
		}
		if n > 0 {
			clampToInt32(dst, s, n)
		}
	}
	return dst
}

// ConvertValue converts a value from type T to type U with appropriate clamping.
// This is a naive implementation for single value conversion.
// Handles all conversions including clamping for down-conversions (e.g., float64 -> int8).
func ConvertValue[T, U numeric](value T) (zeroU U) {
	switch any(zeroU).(type) {
	case float32, float64:
		return U(value)
	case int64:
		// Need to clamp if converting from larger types
		switch v := any(value).(type) {
		case float32:
			return U(clampToInt64Value(v))
		case float64:
			return U(clampToInt64Value(v))
		default:
			return U(value)
		}
	case int:
		// Need to clamp if converting from larger types
		switch v := any(value).(type) {
		case float32:
			return U(clampToIntValue(v))
		case float64:
			return U(clampToIntValue(v))
		default:
			return U(value)
		}
	case int32:
		// Need to clamp if converting from larger types
		switch v := any(value).(type) {
		case float32:
			return U(clampToInt32Value(v))
		case float64:
			return U(clampToInt32Value(v))
		case int64:
			return U(clampToInt32Value(v))
		case int:
			return U(clampToInt32Value(v))
		default:
			return U(value)
		}
	case int16:
		// Need to clamp if converting from larger types
		switch v := any(value).(type) {
		case float32:
			return U(clampToInt16Value(v))
		case float64:
			return U(clampToInt16Value(v))
		case int64:
			return U(clampToInt16Value(v))
		case int:
			return U(clampToInt16Value(v))
		case int32:
			return U(clampToInt16Value(v))
		default:
			return U(value)
		}
	case int8:
		// Need to clamp if converting from larger types
		switch v := any(value).(type) {
		case float32:
			return U(clampToInt8Value(v))
		case float64:
			return U(clampToInt8Value(v))
		case int64:
			return U(clampToInt8Value(v))
		case int:
			return U(clampToInt8Value(v))
		case int32:
			return U(clampToInt8Value(v))
		case int16:
			return U(clampToInt8Value(v))
		default:
			return U(value)
		}
	default:
		return U(value)
	}
}

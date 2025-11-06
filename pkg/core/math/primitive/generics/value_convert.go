package generics

// ValueConvert converts a value from type T to type U with appropriate clamping.
// This is a naive implementation for single value conversion.
// Handles all conversions including clamping for down-conversions (e.g., float64 -> int8).
func ValueConvert[T, U Numeric](value T) (zeroU U) {
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
		// Need to clamp if converting from larger types (platform-specific)
		// Platform-specific handling is in value_convert_i32.go and value_convert_i64.go
		return valueConvertToInt[T, U](value)
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

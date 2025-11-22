package generics

// valueConvertReal handles conversions between real numeric types (non-complex).
func valueConvertReal[T, U NumericReal](value T) (zeroU U) {
	switch any(zeroU).(type) {
	case float32, float64:
		return U(value)
	case int64:
		// Need to clamp if converting from larger types
		switch v := any(value).(type) {
		case float32:
			return U(ClampToInt64Value(v))
		case float64:
			return U(ClampToInt64Value(v))
		default:
			return U(value)
		}
	case int:
		// Need to clamp if converting from larger types (platform-specific)
		result := ValueConvertToInt[T, int](value)
		return any(result).(U)
	case int32:
		// Need to clamp if converting from larger types
		switch v := any(value).(type) {
		case float32:
			return U(ClampToInt32Value(v))
		case float64:
			return U(ClampToInt32Value(v))
		case int64:
			return U(ClampToInt32Value(v))
		case int:
			return U(ClampToInt32Value(v))
		default:
			return U(value)
		}
	case int16:
		// Need to clamp if converting from larger types
		switch v := any(value).(type) {
		case float32:
			return U(ClampToInt16Value(v))
		case float64:
			return U(ClampToInt16Value(v))
		case int64:
			return U(ClampToInt16Value(v))
		case int:
			return U(ClampToInt16Value(v))
		case int32:
			return U(ClampToInt16Value(v))
		default:
			return U(value)
		}
	case int8:
		// Need to clamp if converting from larger types
		switch v := any(value).(type) {
		case float32:
			return U(ClampToInt8Value(v))
		case float64:
			return U(ClampToInt8Value(v))
		case int64:
			return U(ClampToInt8Value(v))
		case int:
			return U(ClampToInt8Value(v))
		case int32:
			return U(ClampToInt8Value(v))
		case int16:
			return U(ClampToInt8Value(v))
		default:
			return U(value)
		}
	default:
		return U(value)
	}
}

// realConvertComplex converts a real numeric value to complex64.
func realConvertComplex[T NumericReal](value T) complex64 {
	// Convert to float32 first, then create complex64 with real part
	realPart := valueConvertReal[T, float32](value)
	return complex64(complex(realPart, 0))
}

// complexConvertReal converts a complex64 value to a real numeric type.
func complexConvertReal[U NumericReal](value complex64) (zeroU U) {
	// Extract real part and convert to target type
	realPart := float32(real(value))
	return valueConvertReal[float32, U](realPart)
}

// ValueConvert converts a value from type T to type U with appropriate clamping.
// This is a naive implementation for single value conversion.
// Handles all conversions including clamping for down-conversions (e.g., float64 -> int8).
func ValueConvert[T, U Numeric](value T) (zeroU U) {
	// Use type assertions on both dst and src to determine conversion path
	_, dstIsComplex := any(zeroU).(complex64)
	srcComplex, srcIsComplex := any(value).(complex64)

	switch {
	case srcIsComplex && dstIsComplex:
		// Complex to Complex: just return the complex value
		return any(srcComplex).(U)
	case srcIsComplex && !dstIsComplex:
		// Complex to Real: extract real part and convert
		// Use type switch to extract concrete destination type
		switch any(zeroU).(type) {
		case float32:
			return any(complexConvertReal[float32](srcComplex)).(U)
		case float64:
			return any(complexConvertReal[float64](srcComplex)).(U)
		case int64:
			return any(complexConvertReal[int64](srcComplex)).(U)
		case int:
			return any(complexConvertReal[int](srcComplex)).(U)
		case int32:
			return any(complexConvertReal[int32](srcComplex)).(U)
		case int16:
			return any(complexConvertReal[int16](srcComplex)).(U)
		case int8:
			return any(complexConvertReal[int8](srcComplex)).(U)
		}
	case !srcIsComplex && dstIsComplex:
		// Real to Complex: convert to float32, then create complex64
		// Use type assertion to extract concrete source type and call helper
		switch src := any(value).(type) {
		case float32:
			return any(realConvertComplex[float32](src)).(U)
		case float64:
			return any(realConvertComplex[float64](src)).(U)
		case int64:
			return any(realConvertComplex[int64](src)).(U)
		case int:
			return any(realConvertComplex[int](src)).(U)
		case int32:
			return any(realConvertComplex[int32](src)).(U)
		case int16:
			return any(realConvertComplex[int16](src)).(U)
		case int8:
			return any(realConvertComplex[int8](src)).(U)
		}
	case !srcIsComplex && !dstIsComplex:
		// Real to Real: use existing conversion logic
		// Use type assertions to extract concrete types and dispatch
		switch any(zeroU).(type) {
		case float32:
			var result float32
			switch src := any(value).(type) {
			case float32:
				result = valueConvertReal[float32, float32](src)
			case float64:
				result = valueConvertReal[float64, float32](src)
			case int64:
				result = valueConvertReal[int64, float32](src)
			case int:
				result = valueConvertReal[int, float32](src)
			case int32:
				result = valueConvertReal[int32, float32](src)
			case int16:
				result = valueConvertReal[int16, float32](src)
			case int8:
				result = valueConvertReal[int8, float32](src)
			}
			return any(result).(U)
		case float64:
			var result float64
			switch src := any(value).(type) {
			case float32:
				result = valueConvertReal[float32, float64](src)
			case float64:
				result = valueConvertReal[float64, float64](src)
			case int64:
				result = valueConvertReal[int64, float64](src)
			case int:
				result = valueConvertReal[int, float64](src)
			case int32:
				result = valueConvertReal[int32, float64](src)
			case int16:
				result = valueConvertReal[int16, float64](src)
			case int8:
				result = valueConvertReal[int8, float64](src)
			}
			return any(result).(U)
		case int64:
			var result int64
			switch src := any(value).(type) {
			case float32:
				result = valueConvertReal[float32, int64](src)
			case float64:
				result = valueConvertReal[float64, int64](src)
			case int64:
				result = valueConvertReal[int64, int64](src)
			case int:
				result = valueConvertReal[int, int64](src)
			case int32:
				result = valueConvertReal[int32, int64](src)
			case int16:
				result = valueConvertReal[int16, int64](src)
			case int8:
				result = valueConvertReal[int8, int64](src)
			}
			return any(result).(U)
		case int:
			var result int
			switch src := any(value).(type) {
			case float32:
				result = valueConvertReal[float32, int](src)
			case float64:
				result = valueConvertReal[float64, int](src)
			case int64:
				result = valueConvertReal[int64, int](src)
			case int:
				result = valueConvertReal[int, int](src)
			case int32:
				result = valueConvertReal[int32, int](src)
			case int16:
				result = valueConvertReal[int16, int](src)
			case int8:
				result = valueConvertReal[int8, int](src)
			}
			return any(result).(U)
		case int32:
			var result int32
			switch src := any(value).(type) {
			case float32:
				result = valueConvertReal[float32, int32](src)
			case float64:
				result = valueConvertReal[float64, int32](src)
			case int64:
				result = valueConvertReal[int64, int32](src)
			case int:
				result = valueConvertReal[int, int32](src)
			case int32:
				result = valueConvertReal[int32, int32](src)
			case int16:
				result = valueConvertReal[int16, int32](src)
			case int8:
				result = valueConvertReal[int8, int32](src)
			}
			return any(result).(U)
		case int16:
			var result int16
			switch src := any(value).(type) {
			case float32:
				result = valueConvertReal[float32, int16](src)
			case float64:
				result = valueConvertReal[float64, int16](src)
			case int64:
				result = valueConvertReal[int64, int16](src)
			case int:
				result = valueConvertReal[int, int16](src)
			case int32:
				result = valueConvertReal[int32, int16](src)
			case int16:
				result = valueConvertReal[int16, int16](src)
			case int8:
				result = valueConvertReal[int8, int16](src)
			}
			return any(result).(U)
		case int8:
			var result int8
			switch src := any(value).(type) {
			case float32:
				result = valueConvertReal[float32, int8](src)
			case float64:
				result = valueConvertReal[float64, int8](src)
			case int64:
				result = valueConvertReal[int64, int8](src)
			case int:
				result = valueConvertReal[int, int8](src)
			case int32:
				result = valueConvertReal[int32, int8](src)
			case int16:
				result = valueConvertReal[int16, int8](src)
			case int8:
				result = valueConvertReal[int8, int8](src)
			}
			return any(result).(U)
		}
	}
	return zeroU
}

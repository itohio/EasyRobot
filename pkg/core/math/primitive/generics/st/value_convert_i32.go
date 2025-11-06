//go:build 386 || arm || mips || mipsle

package st

import . "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

// valueConvertToInt handles conversion to int with clamping (32-bit platform).
// On 32-bit platforms, int64 also needs clamping when converting to int.
func valueConvertToInt[T, U Numeric](value T) (zeroU U) {
	switch v := any(value).(type) {
	case float32:
		return U(clampToIntValue(v))
	case float64:
		return U(clampToIntValue(v))
	case int64:
		return U(clampToIntValue(v))
	default:
		return U(value)
	}
}


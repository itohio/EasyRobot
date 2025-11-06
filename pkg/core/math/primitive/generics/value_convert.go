//go:build !use_mt

package generics

import (
	. "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
	st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"
)

// ValueConvert converts a value from type T to type U with appropriate clamping.
// This is a naive implementation for single value conversion.
// Handles all conversions including clamping for down-conversions (e.g., float64 -> int8).
func ValueConvert[T, U Numeric](value T) (zeroU U) {
	return st.ValueConvert[T, U](value)
}

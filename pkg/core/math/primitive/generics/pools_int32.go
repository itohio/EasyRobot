//go:build 386 || arm || mips || mipsle

package generics

import helpers "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

type ClampableToInt = helpers.ClampableToInt

func ClampToIntValue[U ClampableToInt](v U) int {
	return helpers.ClampToIntValue(v)
}

func ValueConvertToInt[T, U Numeric](value T) (zeroU U) {
	return helpers.ValueConvertToInt[T, U](value)
}

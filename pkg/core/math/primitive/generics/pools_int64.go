//go:build amd64 || arm64 || ppc64 || ppc64le || mips64 || mips64le || riscv64 || s390x

package generics

import helpers "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

type ClampableToInt = helpers.ClampableToInt

func ClampToIntValue[U ClampableToInt](v U) int {
	return helpers.ClampToIntValue(v)
}

func ValueConvertToInt[T, U Numeric](value T) (zeroU U) {
	return helpers.ValueConvertToInt[T, U](value)
}

//go:build !use_mt

package types

import (
	st "github.com/itohio/EasyRobot/x/types/st"
)

// Re-export conversion functions from single-threaded implementation
func CastClamped[T, U Number](dst []T, src []U, n int) {
	st.CastClamped(dst, src, n)
}

func CastClampedStrided[T, U Number](dst []T, src []U, shape []int, stridesDst, stridesSrc []int) {
	st.CastClampedStrided(dst, src, shape, stridesDst, stridesSrc)
}

//go:build !use_mt

package types

import (
	st "github.com/itohio/EasyRobot/x/types/st"
)

// Re-export vector/matrix conversion functions from single-threaded implementation
func CastVecClampedStrided[T, U Number](dst []T, src []U, n int, strideDst, strideSrc int) {
	st.CastVecClampedStrided(dst, src, n, strideDst, strideSrc)
}

func CastMatClampedStrided[T, U Number](dst []T, src []U, rows, cols int, ldDst, ldSrc int) {
	st.CastMatClampedStrided(dst, src, rows, cols, ldDst, ldSrc)
}

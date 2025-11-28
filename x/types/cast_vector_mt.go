//go:build use_mt

package types

import (
	mt "github.com/itohio/EasyRobot/x/convert/mt"
)

// Re-export vector/matrix conversion functions from single-threaded implementation
func CastVecClampedStrided[T, U Number](dst []T, src []U, n int, strideDst, strideSrc int) {
	mt.CastVecClampedStrided(dst, src, n, strideDst, strideSrc)
}

func CastMatClampedStrided[T, U Number](dst []T, src []U, rows, cols int, ldDst, ldSrc int) {
	mt.CastMatClampedStrided(dst, src, rows, cols, ldDst, ldSrc)
}

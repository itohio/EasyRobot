//go:build use_mt

package types

import (
	// TODO: Implement multi-threaded conversion functions that use the st implementation under the hood
	mt "github.com/itohio/EasyRobot/x/iter/mt"
)

// Re-export conversion functions from single-threaded implementation
func CastClamped[T, U Number](dst []T, src []U, n int) {
	mt.CastClamped(dst, src, n)
}

func CastClampedStrided[T, U Number](dst []T, src []U, shape []int, stridesDst, stridesSrc []int) {
	mt.CastClampedStrided(dst, src, shape, stridesDst, stridesSrc)
}

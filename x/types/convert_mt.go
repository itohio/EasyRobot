//go:build use_mt

package types

import st "github.com/itohio/EasyRobot/x/types/st"

// TODO: Implement multi-threaded conversion functions that use the st implementation under the hood

func Convert[T, U Number](dst []T, src []U, n int) {
	st.Convert(dst, src, n)
}

func ConvertStrided[T, U Number](dst []T, src []U, shape []int, stridesDst, stridesSrc []int) {
	st.ConvertStrided(dst, src, shape, stridesDst, stridesSrc)
}

//go:build !use_mt

package generics

import st "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/st"

// Re-export iterator functions from single-threaded implementation
func Elements(shape []int) func(func([]int) bool) {
	return st.Elements(shape)
}

func ElementsStrided(shape []int, strides []int) func(func([]int) bool) {
	return st.ElementsStrided(shape, strides)
}

func ElementsVec(n int) func(func(int) bool) {
	return st.ElementsVec(n)
}

func ElementsVecStrided(n int, stride int) func(func(int) bool) {
	return st.ElementsVecStrided(n, stride)
}

func ElementsMat(rows, cols int) func(func([2]int) bool) {
	return st.ElementsMat(rows, cols)
}

func ElementsMatStrided(rows, cols int, ld int) func(func([2]int) bool) {
	return st.ElementsMatStrided(rows, cols, ld)
}

func ElementsWindow(windowOffset, windowShape, parentShape []int) func(func([]int, bool) bool) {
	return st.ElementsWindow(windowOffset, windowShape, parentShape)
}

func ElementsWindows(outputShape, kernelShape, inputShape []int, stride, padding []int) func(func([]int, []int, bool) bool) {
	return st.ElementsWindows(outputShape, kernelShape, inputShape, stride, padding)
}

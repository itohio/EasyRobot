//go:build use_mt

package generics

import mt "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/mt"

// Re-export iterator functions from multi-threaded implementation
func Elements(shape []int) func(func([]int) bool) {
	return mt.Elements(shape)
}

func ElementsStrided(shape []int, strides []int) func(func([]int) bool) {
	return mt.ElementsStrided(shape, strides)
}

func ElementsVec(n int) func(func(int) bool) {
	return mt.ElementsVec(n)
}

func ElementsVecStrided(n int, stride int) func(func(int) bool) {
	return mt.ElementsVecStrided(n, stride)
}

func ElementsMat(rows, cols int) func(func([2]int) bool) {
	return mt.ElementsMat(rows, cols)
}

func ElementsMatStrided(rows, cols int, ld int) func(func([2]int) bool) {
	return mt.ElementsMatStrided(rows, cols, ld)
}

func ElementsWindow(windowOffset, windowShape, parentShape []int) func(func([]int, bool) bool) {
	return mt.ElementsWindow(windowOffset, windowShape, parentShape)
}

func ElementsWindows(outputShape, kernelShape, inputShape []int, stride, padding []int) func(func([]int, []int, bool) bool) {
	return mt.ElementsWindows(outputShape, kernelShape, inputShape, stride, padding)
}

func ElementsIndices(shape []int, dims ...int) func(func([]int) bool) {
	return mt.ElementsIndices(shape, dims...)
}

func ElementsIndicesStrided(shape []int, strides []int, dims ...int) func(func([]int) bool) {
	return mt.ElementsIndicesStrided(shape, strides, dims...)
}

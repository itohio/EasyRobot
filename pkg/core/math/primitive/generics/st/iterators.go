package st

import . "github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"

// Elements returns an iterator over multi-dimensional indices for the given shape.
// Yields []int representing the indices for each element.
//
// **IMPORTANT**: The indices slice is reused across iterations and must not be modified.
// If you need to store the indices, copy them: `indicesCopy := make([]int, len(indices)); copy(indicesCopy, indices)`
//
// Usage: for indices := range Elements(shape) { ... }
func Elements(shape []int) func(func([]int) bool) {
	if len(shape) == 0 {
		return func(yield func([]int) bool) {
			// Empty shape - yield empty indices once
			// Use nil to avoid allocation
			var empty []int
			yield(empty)
		}
	}

	size := SizeFromShape(shape)
	if size == 0 {
		return func(yield func([]int) bool) {
			// Empty tensor
		}
	}

	return func(yield func([]int) bool) {
		indices := make([]int, len(shape))
		for {
			// Yield current indices directly (reused slice - caller must not modify)
			// This avoids allocations at the cost of requiring caller to copy if needed
			if !yield(indices) {
				return
			}

			// Advance indices in row-major order (last dimension changes fastest)
			advanced := false
			for i := len(indices) - 1; i >= 0; i-- {
				indices[i]++
				if indices[i] < shape[i] {
					advanced = true
					break
				}
				indices[i] = 0
			}

			if !advanced {
				// All combinations exhausted
				break
			}
		}
	}
}

// ElementsStrided returns an iterator over multi-dimensional indices for the given shape with stride support.
// Yields []int representing the indices for each element.
//
// **IMPORTANT**: The indices slice is reused across iterations and must not be modified.
// If you need to store the indices, copy them: `indicesCopy := make([]int, len(indices)); copy(indicesCopy, indices)`
//
// Usage: for indices := range ElementsStrided(shape, strides) { ... }
func ElementsStrided(shape []int, strides []int) func(func([]int) bool) {
	if len(shape) == 0 {
		return func(yield func([]int) bool) {
			// Empty shape - yield empty indices once
			// Use nil to avoid allocation
			var empty []int
			yield(empty)
		}
	}

	size := SizeFromShape(shape)
	if size == 0 {
		return func(yield func([]int) bool) {
			// Empty tensor
		}
	}

	// Ensure strides are valid
	strides = EnsureStrides(strides, shape)

	return func(yield func([]int) bool) {
		indices := make([]int, len(shape))
		for {
			// Yield current indices directly (reused slice - caller must not modify)
			// This avoids allocations at the cost of requiring caller to copy if needed
			if !yield(indices) {
				return
			}

			// Advance indices in row-major order (last dimension changes fastest)
			advanced := false
			for i := len(indices) - 1; i >= 0; i-- {
				indices[i]++
				if indices[i] < shape[i] {
					advanced = true
					break
				}
				indices[i] = 0
			}

			if !advanced {
				// All combinations exhausted
				break
			}
		}
	}
}

// ElementsVec returns an iterator over vector indices (scalar index).
// Yields int representing the linear index for each element.
// Usage: for idx := range ElementsVec(n) { ... }
func ElementsVec(n int) func(func(int) bool) {
	return func(yield func(int) bool) {
		for i := 0; i < n; i++ {
			if !yield(i) {
				return
			}
		}
	}
}

// ElementsVecStrided returns an iterator over vector indices with stride support.
// Yields int representing the linear index for each element.
// Usage: for idx := range ElementsVecStrided(n, stride) { ... }
func ElementsVecStrided(n int, stride int) func(func(int) bool) {
	return func(yield func(int) bool) {
		idx := 0
		for i := 0; i < n; i++ {
			if !yield(idx) {
				return
			}
			idx += stride
		}
	}
}

// ElementsMat returns an iterator over matrix indices (row, col tuple).
// Yields [2]int representing (row, col) for each element.
// Usage: for idx := range ElementsMat(rows, cols) { ... }
func ElementsMat(rows, cols int) func(func([2]int) bool) {
	return func(yield func([2]int) bool) {
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				if !yield([2]int{i, j}) {
					return
				}
			}
		}
	}
}

// ElementsMatStrided returns an iterator over matrix indices with leading dimension support.
// Yields [2]int representing (row, col) for each element.
// Usage: for idx := range ElementsMatStrided(rows, cols, ld) { ... }
func ElementsMatStrided(rows, cols int, ld int) func(func([2]int) bool) {
	return func(yield func([2]int) bool) {
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				if !yield([2]int{i, j}) {
					return
				}
			}
		}
	}
}

// ElementsWindow iterates over positions in a window within a parent tensor.
// windowOffset: starting position of window [offsetH, offsetW, ...]
// windowShape: size of window [kernelH, kernelW, ...]
// parentShape: shape of parent tensor [height, width, ...]
// Yields: (absoluteIndices, isValid) where absoluteIndices are positions in parent tensor,
// and isValid indicates if the position is within bounds.
//
// **IMPORTANT**: The indices slice is reused across iterations and must not be modified.
// If you need to store the indices, copy them.
//
// Usage: for absIndices, isValid := range ElementsWindow(windowOffset, windowShape, parentShape) { ... }
func ElementsWindow(
	windowOffset, windowShape, parentShape []int,
) func(func([]int, bool) bool) {
	if len(windowShape) == 0 || len(parentShape) == 0 {
		return func(yield func([]int, bool) bool) {
			// Empty window or parent
		}
	}

	if len(windowOffset) != len(parentShape) || len(windowShape) != len(parentShape) {
		return func(yield func([]int, bool) bool) {
			// Dimension mismatch
		}
	}

	windowSize := SizeFromShape(windowShape)
	if windowSize == 0 {
		return func(yield func([]int, bool) bool) {
			// Empty window
		}
	}

	return func(yield func([]int, bool) bool) {
		// Iterate over window positions
		for windowIndices := range Elements(windowShape) {
			// Calculate absolute position in parent
			absIndices := make([]int, len(parentShape))
			isValid := true

			for i := range parentShape {
				absPos := windowOffset[i] + windowIndices[i]
				absIndices[i] = absPos

				// Check bounds
				if absPos < 0 || absPos >= parentShape[i] {
					isValid = false
					// Continue setting all indices even if invalid
				}
			}

			if !yield(absIndices, isValid) {
				return
			}
		}
	}
}

// ElementsWindows iterates over all windows in a tensor (for convolution operations).
// outputShape: shape of output positions [outH, outW, ...]
// kernelShape: shape of kernel [kernelH, kernelW, ...]
// inputShape: shape of input [inH, inW, ...]
// stride: stride for each dimension [strideH, strideW, ...]
// padding: padding for each dimension [padH, padW, ...] (applied before: inputPos = outputPos * stride + kernelPos - padding)
// Yields: (outputIndices, inputIndices, isValid) where:
//   - outputIndices: position in output tensor
//   - inputIndices: position in input tensor
//   - isValid: whether input position is within bounds
//
// **IMPORTANT**: The indices slices are reused across iterations and must not be modified.
// If you need to store the indices, copy them.
//
// Usage: for outIdx, inIdx, isValid := range ElementsWindows(outputShape, kernelShape, inputShape, stride, padding) { ... }
func ElementsWindows(
	outputShape, kernelShape, inputShape []int,
	stride, padding []int,
) func(func([]int, []int, bool) bool) {
	if len(outputShape) == 0 || len(kernelShape) == 0 || len(inputShape) == 0 {
		return func(yield func([]int, []int, bool) bool) {
			// Empty shapes
		}
	}

	if len(outputShape) != len(inputShape) || len(kernelShape) != len(inputShape) {
		return func(yield func([]int, []int, bool) bool) {
			// Dimension mismatch
		}
	}

	if len(stride) != len(inputShape) || len(padding) != len(inputShape) {
		return func(yield func([]int, []int, bool) bool) {
			// Stride/padding dimension mismatch
		}
	}

	outputSize := SizeFromShape(outputShape)
	if outputSize == 0 {
		return func(yield func([]int, []int, bool) bool) {
			// Empty output
		}
	}

	kernelSize := SizeFromShape(kernelShape)
	if kernelSize == 0 {
		return func(yield func([]int, []int, bool) bool) {
			// Empty kernel
		}
	}

	return func(yield func([]int, []int, bool) bool) {
		// Iterate over output positions
		for outIndices := range Elements(outputShape) {
			// Calculate window offset for this output position
			windowOffset := make([]int, len(inputShape))
			for i := range inputShape {
				windowOffset[i] = outIndices[i]*stride[i] - padding[i]
			}

			// Iterate over window positions
			for absIndices, isValid := range ElementsWindow(
				windowOffset, kernelShape, inputShape,
			) {
				// Yield output indices, input indices, and validity
				if !yield(outIndices, absIndices, isValid) {
					return
				}
			}
		}
	}
}

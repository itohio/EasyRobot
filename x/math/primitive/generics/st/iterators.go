package st

import (
	. "github.com/itohio/EasyRobot/x/math/primitive/generics/helpers"

	"fmt"
)

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
		// Use stack-allocated array for indices
		var indicesStatic [MAX_DIMS]int
		indices := indicesStatic[:len(shape)]
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
	// Use stack-allocated array for stride computation
	var stridesStatic [MAX_DIMS]int
	strides = EnsureStrides(stridesStatic[:len(shape)], strides, shape)

	return func(yield func([]int) bool) {
		// Use stack-allocated array for indices
		var indicesStatic [MAX_DIMS]int
		indices := indicesStatic[:len(shape)]
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

// ElementsIndices returns an iterator that fixes specified dimensions and iterates over the remaining ones.
// Returns a function that can be used in Go 1.22+ range loops.
// fixedAxisValuePairs are pairs of axis index and fixed value: axis1, value1, axis2, value2, ...
// The iterator yields complete indices for all dimensions.
// If fixedAxisValuePairs is empty, iterates over all dimensions (equivalent to Elements).
//
// **IMPORTANT**: The indices slice is reused across iterations and must not be modified.
// If you need to store the indices, copy them: `indicesCopy := make([]int, len(indices)); copy(indicesCopy, indices)`
//
// Usage:
//   - for indices := range ElementsIndices(shape) { ... } // iterates over all dimensions
//   - for indices := range ElementsIndices(shape, 0, 1) { ... } // fixes dimension 0 at value 1, iterates over remaining dimensions
//   - for indices := range ElementsIndices(shape, 0, 1, 2, 3) { ... } // fixes dimension 0 at 1, dimension 2 at 3
func ElementsIndices(shape []int, fixedAxisValuePairs ...int) func(func([]int) bool) {
	// Validate early, before closure creation
	if len(fixedAxisValuePairs)%2 != 0 {
		panic(fmt.Sprintf("tensor: Iterator requires even number of arguments (axis-value pairs), got %d", len(fixedAxisValuePairs)))
	}
	if len(shape) > MAX_DIMS {
		panic(fmt.Sprintf("tensor: Iterator supports up to %d dimensions, got %d", MAX_DIMS, len(shape)))
	}

	rank := len(shape)
	// Check for duplicate axes first (before value validation)
	if len(fixedAxisValuePairs) > 0 {
		seen := make(map[int]bool, len(fixedAxisValuePairs)/2)
		for i := 0; i < len(fixedAxisValuePairs); i += 2 {
			axis := fixedAxisValuePairs[i]
			if seen[axis] {
				panic(fmt.Sprintf("tensor: duplicate axis %d in Iterator arguments", axis))
			}
			seen[axis] = true
		}
	}

	// Validate all inputs early, before closure creation
	for i := 0; i < len(fixedAxisValuePairs); i += 2 {
		axis := fixedAxisValuePairs[i]
		value := fixedAxisValuePairs[i+1]

		if axis < 0 || axis >= rank {
			panic(fmt.Sprintf("tensor: fixed axis %d out of range for rank %d", axis, rank))
		}
		if value < 0 || value >= shape[axis] {
			panic(fmt.Sprintf("tensor: fixed value %d out of range for axis %d (size %d)", value, axis, shape[axis]))
		}
	}

	// Capture only what's absolutely necessary
	shapeVals := shape
	fixedPairs := fixedAxisValuePairs

	return func(yield func([]int) bool) {
		// ALL arrays declared inside closure - nothing escapes from outer function

		if len(shapeVals) == 0 {
			// Empty shape
			var emptyIndices [0]int
			yield(emptyIndices[:])
			return
		}

		numFixedDims := len(fixedPairs) / 2

		// Fast path: no fixed dimensions (most common case)
		if numFixedDims == 0 {
			var indicesArr [MAX_DIMS]int
			indices := indicesArr[:rank]

			// Iterate over all combinations without fixed dimensions
			for {
				// Yield the current indices
				if !yield(indices) {
					return
				}

				// Advance indices in row-major order (last dimension changes fastest)
				advanced := false
				for i := rank - 1; i >= 0; i-- {
					indices[i]++
					if indices[i] < shapeVals[i] {
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
			return
		}

		// Build fixed dimensions array inside closure
		var fixedDimsArr [MAX_DIMS]int
		for i := 0; i < rank; i++ {
			fixedDimsArr[i] = -1 // -1 means not fixed
		}

		// Build fixed dimensions array from pairs (validation already done above)
		for i := 0; i < len(fixedPairs); i += 2 {
			axis := fixedPairs[i]
			value := fixedPairs[i+1]
			fixedDimsArr[axis] = value
		}

		// Count remaining dimensions
		remainingCount := 0
		for i := 0; i < rank; i++ {
			if fixedDimsArr[i] == -1 {
				remainingCount++
			}
		}

		// Build full indices array
		var fullIndicesArr [MAX_DIMS]int
		fullIndices := fullIndicesArr[:rank]

		// Set fixed dimensions
		for i := 0; i < rank; i++ {
			if fixedDimsArr[i] != -1 {
				fullIndices[i] = fixedDimsArr[i]
			}
		}

		if remainingCount == 0 {
			// All dimensions fixed - yield once
			yield(fullIndices)
			return
		}

		// Build list of remaining dimensions
		var remainingArr [MAX_DIMS]int
		remaining := remainingArr[:0]
		for i := 0; i < rank; i++ {
			if fixedDimsArr[i] == -1 {
				remaining = append(remaining, i)
			}
		}

		// Initialize indices for remaining dimensions to all zeros
		var indicesArr [MAX_DIMS]int
		indices := indicesArr[:remainingCount]

		// Iterate over all combinations
		for {
			// Set remaining dimensions in full indices
			for i, dim := range remaining {
				fullIndices[dim] = indices[i]
			}

			// Yield the current full indices
			if !yield(fullIndices) {
				return
			}

			// Advance indices in row-major order (last dimension changes fastest)
			advanced := false
			for i := remainingCount - 1; i >= 0; i-- {
				dim := remaining[i]
				indices[i]++
				if indices[i] < shapeVals[dim] {
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

// ElementsIndicesStrided returns an iterator over multi-dimensional indices for selected dimensions of the given shape with stride support.
// Yields []int representing the indices for the selected dimensions only.
// If dims is empty or nil, iterates over all dimensions (equivalent to ElementsStrided).
//
// **IMPORTANT**: The indices slice is reused across iterations and must not be modified.
// If you need to store the indices, copy them: `indicesCopy := make([]int, len(indices)); copy(indicesCopy, indices)`
//
// Usage:
//   - for indices := range ElementsIndicesStrided(shape, strides) { ... } // iterates over all dimensions
//   - for indices := range ElementsIndicesStrided(shape, strides, 0, 2) { ... } // iterates over dimensions 0 and 2 only
func ElementsIndicesStrided(shape []int, strides []int, dims ...int) func(func([]int) bool) {
	rank := len(shape)
	if rank == 0 {
		return func(yield func([]int) bool) {
			// Empty shape - yield empty indices once
			var empty []int
			yield(empty)
		}
	}

	// Validate rank
	if rank > MAX_DIMS {
		return func(yield func([]int) bool) {
			// Rank exceeds MAX_DIMS, return early without iterating
		}
	}

	// Ensure strides are valid
	// Use stack-allocated array for stride computation
	var stridesStatic [MAX_DIMS]int
	strides = EnsureStrides(stridesStatic[:len(shape)], strides, shape)

	// If no dims specified, use all dimensions (0, 1, 2, ..., rank-1)
	var dimsArr [MAX_DIMS]int
	numDims := len(dims)
	if numDims == 0 {
		numDims = rank
		for i := 0; i < rank; i++ {
			dimsArr[i] = i
		}
	} else {
		// Validate number of dimensions
		if numDims > MAX_DIMS {
			return func(yield func([]int) bool) {
				// Too many dimensions selected
			}
		}

		// Check for duplicates and validate dimension indices
		for i, dim := range dims {
			if dim < 0 || dim >= rank {
				return func(yield func([]int) bool) {
					// Invalid dimension index
				}
			}
			// Check for duplicates in already processed dimensions
			for j := 0; j < i; j++ {
				if dimsArr[j] == dim {
					return func(yield func([]int) bool) {
						// Duplicate dimension
					}
				}
			}
			dimsArr[i] = dim
		}
	}

	// Build selected shape and validate it's not empty
	var selectedShapeArr [MAX_DIMS]int
	selectedShape := selectedShapeArr[:numDims]
	selectedSize := 1
	for i := 0; i < numDims; i++ {
		dimSize := shape[dimsArr[i]]
		if dimSize <= 0 {
			return func(yield func([]int) bool) {
				// Empty dimension
			}
		}
		selectedShape[i] = dimSize
		selectedSize *= dimSize
	}

	if selectedSize == 0 {
		return func(yield func([]int) bool) {
			// Empty selected tensor
		}
	}

	// Capture shape and dims for closure
	shapeVals := shape
	selectedDims := dimsArr[:numDims]

	return func(yield func([]int) bool) {
		// All arrays declared inside closure - nothing escapes from outer function
		var indicesArr [MAX_DIMS]int
		indices := indicesArr[:numDims]

		for {
			if !yield(indices) {
				return
			}

			// Advance indices in row-major order (last selected dimension changes fastest)
			advanced := false
			for i := numDims - 1; i >= 0; i-- {
				dim := selectedDims[i]
				indices[i]++
				if indices[i] < shapeVals[dim] {
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

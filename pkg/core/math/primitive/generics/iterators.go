package generics

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

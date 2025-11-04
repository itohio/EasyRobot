package types

import (
	"fmt"
	"sort"
)

// Shape represents tensor dimensions.
type Shape []int

// NewShape returns a copy of dims as a Shape.
func NewShape(dims ...int) Shape {
	if len(dims) == 0 {
		return nil
	}
	s := make(Shape, len(dims))
	copy(s, dims)
	return s
}

// Rank returns the number of dimensions.
func (s Shape) Rank() int {
	return len(s)
}

// Size returns total number of elements represented by the shape.
// Scalars (len=0) report size 1.
func (s Shape) Size() int {
	if len(s) == 0 {
		return 1
	}
	size := 1
	for _, d := range s {
		if d <= 0 {
			return 0
		}
		size *= d
	}
	return size
}

// Equal checks if two shapes are equal.
func (s Shape) Equal(other Shape) bool {
	if len(s) != len(other) {
		return false
	}
	for i := range s {
		if s[i] != other[i] {
			return false
		}
	}
	return true
}

// Strides computes row-major strides for the shape.
func (s Shape) Strides() []int {
	if len(s) == 0 {
		return nil
	}
	strides := make([]int, len(s))
	stride := 1
	for i := len(s) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= s[i]
	}
	return strides
}

// IsContiguous reports whether the given strides describe a dense row-major layout.
func (s Shape) IsContiguous(strides []int) bool {
	if len(s) == 0 {
		return true
	}
	canonical := s.Strides()
	if len(strides) != len(canonical) {
		return false
	}
	for i := range canonical {
		if strides[i] != canonical[i] {
			return false
		}
	}
	return true
}

// ValidateAxes ensures axes are in range and unique. It sorts axes in-place.
func (s Shape) ValidateAxes(axes []int) error {
	if len(s) == 0 {
		return fmt.Errorf("tensor: empty shape")
	}
	if len(axes) == 0 {
		return nil
	}
	max := len(s)
	seen := make(map[int]struct{}, len(axes))
	for _, axis := range axes {
		if axis < 0 || axis >= max {
			return fmt.Errorf("tensor: axis %d out of range for rank %d", axis, max)
		}
		if _, ok := seen[axis]; ok {
			return fmt.Errorf("tensor: duplicate axis %d", axis)
		}
		seen[axis] = struct{}{}
	}
	sort.Ints(axes)
	return nil
}

// ToSlice returns a copy of the shape as []int.
func (s Shape) ToSlice() []int {
	if len(s) == 0 {
		return nil
	}
	out := make([]int, len(s))
	copy(out, s)
	return out
}

func (s Shape) Clone() Shape {
	if s == nil {
		return nil
	}
	return NewShape(s...)
}

// Iterator creates an iterator that fixes specified dimensions and iterates over the remaining ones.
// Returns a function that can be used in Go 1.22+ range loops.
// fixedAxisValuePairs are pairs of axis index and fixed value: axis1, value1, axis2, value2, ...
// The iterator yields complete indices for all dimensions that can be used with At or SetAt.
// Example: For shape [2, 3, 4] with Iterator(0, 1), the iterator will iterate over dimensions 1 and 2.
//
// Usage:
//
//	for indices := range shape.Iterator(0, 1) {
//	    value := tensor.At(indices...)
//	    tensor.SetAt(indices, value)
//	}
//
//	for indices := range shape.Iterator(0, 1, 2, 3) {
//	    // Fixes dimension 0 at 1, dimension 2 at 3
//	}
func (s Shape) Iterator(fixedAxisValuePairs ...int) func(func([]int) bool) {
	if len(s) == 0 {
		return func(yield func([]int) bool) {
			// Empty shape - yield empty indices once
			yield([]int{})
		}
	}

	// Validate that we have an even number of arguments (pairs)
	if len(fixedAxisValuePairs)%2 != 0 {
		panic(fmt.Sprintf("tensor: Iterator requires even number of arguments (axis-value pairs), got %d", len(fixedAxisValuePairs)))
	}

	// Build fixed dimensions map from pairs
	fixedDims := make(map[int]int, len(fixedAxisValuePairs)/2)
	for i := 0; i < len(fixedAxisValuePairs); i += 2 {
		axis := fixedAxisValuePairs[i]
		value := fixedAxisValuePairs[i+1]

		if axis < 0 || axis >= len(s) {
			panic(fmt.Sprintf("tensor: fixed axis %d out of range for rank %d", axis, len(s)))
		}
		if _, exists := fixedDims[axis]; exists {
			panic(fmt.Sprintf("tensor: duplicate axis %d in Iterator arguments", axis))
		}
		if value < 0 || value >= s[axis] {
			panic(fmt.Sprintf("tensor: fixed value %d out of range for axis %d (size %d)", value, axis, s[axis]))
		}
		fixedDims[axis] = value
	}

	// Build list of remaining dimensions (not fixed)
	remaining := make([]int, 0, len(s))
	for i := range s {
		if _, isFixed := fixedDims[i]; !isFixed {
			remaining = append(remaining, i)
		}
	}

	// Build full indices array (reused for each iteration)
	fullIndices := make([]int, len(s))

	// Set fixed dimensions
	for dim, val := range fixedDims {
		fullIndices[dim] = val
	}

	return func(yield func([]int) bool) {
		if len(remaining) == 0 {
			// All dimensions fixed - yield once
			indicesCopy := make([]int, len(fullIndices))
			copy(indicesCopy, fullIndices)
			yield(indicesCopy)
			return
		}

		// Initialize indices for remaining dimensions to all zeros
		indices := make([]int, len(remaining))

		// Iterate over all combinations
		for {
			// Set remaining dimensions in full indices
			for i, dim := range remaining {
				fullIndices[dim] = indices[i]
			}

			// Yield the current full indices
			// Create a copy to avoid issues if the caller modifies the slice
			indicesCopy := make([]int, len(fullIndices))
			copy(indicesCopy, fullIndices)
			if !yield(indicesCopy) {
				return
			}

			// Advance indices in row-major order (last dimension changes fastest)
			advanced := false
			for i := len(indices) - 1; i >= 0; i-- {
				dim := remaining[i]
				indices[i]++
				if indices[i] < s[dim] {
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

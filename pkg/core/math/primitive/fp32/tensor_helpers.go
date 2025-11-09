package fp32

import (
	"errors"
	"fmt"
	"sort"
)

// ComputeStrides returns the canonical row-major strides for the given shape.
// Example: shape [2,3,4] -> strides [12,4,1].
func ComputeStrides(shape []int) []int {
	if len(shape) == 0 {
		return nil
	}

	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}

	return strides
}

// SizeFromShape computes the total number of elements described by the shape.
func SizeFromShape(shape []int) int {
	size := 1
	if len(shape) == 0 {
		return 0
	}
	for _, dim := range shape {
		if dim <= 0 {
			return 0
		}
		size *= dim
	}
	return size
}

// EnsureStrides returns the provided strides if they match the shape; otherwise it falls back to canonical row-major strides.
func EnsureStrides(strides []int, shape []int) []int {
	if len(shape) == 0 {
		return nil
	}
	if len(strides) != len(shape) {
		return ComputeStrides(shape)
	}
	return strides
}

// IsContiguous reports whether the strides describe a dense row-major layout for the shape.
func IsContiguous(strides []int, shape []int) bool {
	if len(shape) == 0 {
		return true
	}
	canonical := ComputeStrides(shape)
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

// IndexLinear converts multi-dimensional indices to a linear offset using the supplied strides.
func IndexLinear(indices []int, strides []int) int {
	if len(indices) != len(strides) {
		return 0
	}
	offset := 0
	for i := range indices {
		offset += indices[i] * strides[i]
	}
	return offset
}

// ValidateAxes ensures the provided axes are within range and unique.
// Returns an error describing the problem instead of panicking so callers can surface domain-specific messages.
func ValidateAxes(shape []int, axes []int) error {
	if len(shape) == 0 {
		return errors.New("fp32: empty shape provided to ValidateAxes")
	}
	if len(axes) == 0 {
		return nil
	}

	maxDim := len(shape)
	seen := make(map[int]struct{}, len(axes))
	for _, axis := range axes {
		if axis < 0 || axis >= maxDim {
			return fmt.Errorf("fp32: axis %d out of range for rank %d", axis, maxDim)
		}
		if _, ok := seen[axis]; ok {
			return fmt.Errorf("fp32: duplicate axis %d", axis)
		}
		seen[axis] = struct{}{}
	}

	// Normalise order so downstream code can assume ascending axes.
	sort.Ints(axes)
	return nil
}

// advanceOffsets advances the multi-dimensional indices/offsets tuple.
// Returns true if the iteration should continue, false when the final
// element has been processed.
func advanceOffsets(shape []int, indices []int, offsets []int, strides [][]int) bool {
	if len(shape) == 0 {
		return false
	}

	for dim := len(shape) - 1; dim >= 0; dim-- {
		indices[dim]++
		for buf := range offsets {
			offsets[buf] += strides[buf][dim]
		}

		if indices[dim] < shape[dim] {
			return true
		}

		for buf := range offsets {
			offsets[buf] -= strides[buf][dim] * shape[dim]
		}
		indices[dim] = 0
	}

	return false
}

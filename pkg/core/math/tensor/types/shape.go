package types

import (
	"fmt"
	"sort"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/generics/helpers"
)

const MAX_DIMS = helpers.MAX_DIMS

// Shape represents tensor dimensions.
type Shape []int

// NewShape returns a copy of dims as a Shape.
func NewShape(dims ...int) Shape {
	return dims
}

// Rank returns the number of dimensions.
func (s Shape) Rank() int {
	return len(s)
}

// Size returns total number of elements represented by the shape.
// Scalars (len=0) report size 1.
// Uses optimized helper function for non-empty shapes.
func (s Shape) Size() int {
	if len(s) == 0 {
		return 1
	}
	return helpers.SizeFromShape(s)
}

// Equal checks if two shapes are equal.
func (s Shape) Equal(other Shape) bool {
	if s.Rank() != other.Rank() {
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
// Uses optimized helper function with stack allocation when dst is nil.
func (s Shape) Strides(dst []int) []int {
	return helpers.ComputeStrides(dst, s)
}

// IsContiguous reports whether the given strides describe a dense row-major layout.
// Uses optimized helper function with stack allocation.
func (s Shape) IsContiguous(strides []int) bool {
	return helpers.IsContiguous(strides, s)
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

	return []int(s)
}

func (s Shape) Clone() Shape {
	if s == nil {
		return nil
	}
	var static [helpers.MAX_DIMS]int
	out := static[:len(s)]
	copy(out[:], s)
	return out
}

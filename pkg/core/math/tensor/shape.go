package tensor

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

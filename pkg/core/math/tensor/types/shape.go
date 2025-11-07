package types

import (
	"fmt"
	"sort"
)

const MAX_DIMS = 16

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
func (s Shape) Strides(dst []int) []int {
	if len(s) == 0 {
		return nil
	}
	if dst == nil {
		dst = make([]int, len(s))
	}
	stride := 1
	for i := len(s) - 1; i >= 0; i-- {
		dst[i] = stride
		stride *= s[i]
	}
	return dst
}

// IsContiguous reports whether the given strides describe a dense row-major layout.
func (s Shape) IsContiguous(strides []int) bool {
	if len(s) == 0 {
		return true
	}
	static := [MAX_DIMS]int{}
	canonical := s.Strides(static[:len(s)])
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

	return []int(s)
}

func (s Shape) Clone() Shape {
	if s == nil {
		return nil
	}
	out := make([]int, len(s))
	copy(out, s)
	return out
}

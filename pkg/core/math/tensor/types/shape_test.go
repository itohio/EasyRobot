package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShapeIterator(t *testing.T) {
	t.Run("no fixed dimensions - iterate all", func(t *testing.T) {
		shape := NewShape(2, 3)

		expected := [][]int{
			{0, 0}, {0, 1}, {0, 2},
			{1, 0}, {1, 1}, {1, 2},
		}

		var got [][]int
		for indices := range shape.Iterator() {
			got = append(got, indices)
		}

		assert.Equal(t, expected, got)
	})

	t.Run("fix first dimension", func(t *testing.T) {
		shape := NewShape(2, 3, 4)

		// Should iterate over dimensions 1 and 2 (12 combinations)
		expectedCount := 3 * 4
		count := 0

		for indices := range shape.Iterator(0, 1) {
			assert.Equal(t, 3, len(indices)) // All dimensions
			assert.Equal(t, 1, indices[0])   // Fixed dimension 0
			count++
		}

		assert.Equal(t, expectedCount, count)

		// Verify first and last indices
		var first, last []int
		for indices := range shape.Iterator(0, 1) {
			if first == nil {
				first = indices
			}
			last = indices
		}
		assert.Equal(t, []int{1, 0, 0}, first) // Fixed dim 0 = 1, remaining dims start at 0
		assert.Equal(t, []int{1, 2, 3}, last)  // Last combination
	})

	t.Run("fix middle dimension", func(t *testing.T) {
		shape := NewShape(2, 3, 4)

		// Should iterate over dimensions 0 and 2 (8 combinations)
		expectedCount := 2 * 4
		count := 0

		for indices := range shape.Iterator(1, 2) {
			assert.Equal(t, 3, len(indices)) // All dimensions
			assert.Equal(t, 2, indices[1])   // Fixed dimension 1
			count++
		}

		assert.Equal(t, expectedCount, count)

		// Verify first indices
		var first []int
		for indices := range shape.Iterator(1, 2) {
			if first == nil {
				first = indices
			}
			break
		}
		assert.Equal(t, []int{0, 2, 0}, first) // Fixed dim 1 = 2, remaining dims start at 0
	})

	t.Run("fix multiple dimensions", func(t *testing.T) {
		shape := NewShape(2, 3, 4, 5)

		// Should iterate over dimensions 1 and 3 (15 combinations)
		expectedCount := 3 * 5
		count := 0

		for indices := range shape.Iterator(0, 1, 2, 3) {
			assert.Equal(t, 4, len(indices)) // All dimensions
			assert.Equal(t, 1, indices[0])   // Fixed dimension 0
			assert.Equal(t, 3, indices[2])   // Fixed dimension 2
			count++
		}

		assert.Equal(t, expectedCount, count)

		// Verify first indices
		var first []int
		for indices := range shape.Iterator(0, 1, 2, 3) {
			if first == nil {
				first = indices
			}
			break
		}
		assert.Equal(t, []int{1, 0, 3, 0}, first)
	})

	t.Run("fix all dimensions", func(t *testing.T) {
		shape := NewShape(2, 3, 4)

		// Should iterate once (single combination)
		count := 0
		var indices []int

		for idx := range shape.Iterator(0, 1, 1, 2, 2, 3) {
			indices = idx
			count++
		}

		assert.Equal(t, 1, count)
		assert.Equal(t, []int{1, 2, 3}, indices)
	})

	t.Run("empty shape", func(t *testing.T) {
		shape := NewShape()

		count := 0
		var indices []int
		for idx := range shape.Iterator() {
			indices = idx
			count++
		}

		assert.Equal(t, 1, count) // Should yield once with empty indices
		assert.Equal(t, []int{}, indices)
	})

	t.Run("single dimension", func(t *testing.T) {
		shape := NewShape(3)

		expected := [][]int{{0}, {1}, {2}}
		var got [][]int

		for indices := range shape.Iterator() {
			got = append(got, indices)
		}

		assert.Equal(t, expected, got)
	})

	t.Run("row-major order", func(t *testing.T) {
		shape := NewShape(2, 3)

		// Row-major: last dimension changes fastest
		expected := [][]int{
			{0, 0}, {0, 1}, {0, 2}, // First row
			{1, 0}, {1, 1}, {1, 2}, // Second row
		}

		var got [][]int
		for indices := range shape.Iterator() {
			got = append(got, indices)
		}

		assert.Equal(t, expected, got)
	})
}

func TestShapeIteratorPanics(t *testing.T) {
	t.Run("odd number of arguments", func(t *testing.T) {
		shape := NewShape(2, 3)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "even number")
		}()
		shape.Iterator(0, 1, 2) // Odd number of arguments
	})

	t.Run("invalid dimension index", func(t *testing.T) {
		shape := NewShape(2, 3)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "out of range")
		}()
		shape.Iterator(5, 0) // Dimension 5 doesn't exist
	})

	t.Run("invalid fixed value", func(t *testing.T) {
		shape := NewShape(2, 3)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "out of range")
		}()
		shape.Iterator(0, 5) // Value 5 out of range for dimension 0 (size 2)
	})

	t.Run("negative dimension index", func(t *testing.T) {
		shape := NewShape(2, 3)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "out of range")
		}()
		shape.Iterator(-1, 0)
	})

	t.Run("negative fixed value", func(t *testing.T) {
		shape := NewShape(2, 3)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "out of range")
		}()
		shape.Iterator(0, -1)
	})

	t.Run("duplicate axis", func(t *testing.T) {
		shape := NewShape(2, 3, 4)
		defer func() {
			r := recover()
			assert.NotNil(t, r)
			assert.Contains(t, r.(string), "duplicate axis")
		}()
		shape.Iterator(0, 1, 0, 2) // Duplicate axis 0
	})
}

func TestNewShape(t *testing.T) {
	tests := []struct {
		name     string
		dims     []int
		expected Shape
	}{
		{
			name:     "empty shape",
			dims:     []int{},
			expected: nil,
		},
		{
			name:     "single dimension",
			dims:     []int{5},
			expected: Shape{5},
		},
		{
			name:     "2D shape",
			dims:     []int{2, 3},
			expected: Shape{2, 3},
		},
		{
			name:     "3D shape",
			dims:     []int{2, 3, 4},
			expected: Shape{2, 3, 4},
		},
		{
			name:     "4D shape",
			dims:     []int{1, 2, 3, 4},
			expected: Shape{1, 2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NewShape(tt.dims...)
			assert.Equal(t, tt.expected, result)
			// Verify it's a copy (modifying input shouldn't affect result)
			if len(tt.dims) > 0 {
				tt.dims[0] = 999
				assert.NotEqual(t, tt.dims[0], result[0])
			}
		})
	}
}

func TestShapeRank(t *testing.T) {
	tests := []struct {
		name     string
		shape    Shape
		expected int
	}{
		{
			name:     "empty shape",
			shape:    NewShape(),
			expected: 0,
		},
		{
			name:     "scalar-like 1D",
			shape:    NewShape(1),
			expected: 1,
		},
		{
			name:     "2D shape",
			shape:    NewShape(2, 3),
			expected: 2,
		},
		{
			name:     "3D shape",
			shape:    NewShape(2, 3, 4),
			expected: 3,
		},
		{
			name:     "4D shape",
			shape:    NewShape(1, 2, 3, 4),
			expected: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.shape.Rank())
		})
	}
}

func TestShapeSize(t *testing.T) {
	tests := []struct {
		name     string
		shape    Shape
		expected int
	}{
		{
			name:     "empty shape (scalar)",
			shape:    NewShape(),
			expected: 1,
		},
		{
			name:     "1D shape",
			shape:    NewShape(5),
			expected: 5,
		},
		{
			name:     "2D shape",
			shape:    NewShape(2, 3),
			expected: 6,
		},
		{
			name:     "3D shape",
			shape:    NewShape(2, 3, 4),
			expected: 24,
		},
		{
			name:     "4D shape",
			shape:    NewShape(2, 3, 4, 5),
			expected: 120,
		},
		{
			name:     "shape with zero dimension",
			shape:    NewShape(2, 0, 4),
			expected: 0,
		},
		{
			name:     "shape with negative dimension",
			shape:    NewShape(2, -1, 4),
			expected: 0,
		},
		{
			name:     "single element",
			shape:    NewShape(1, 1),
			expected: 1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.shape.Size())
		})
	}
}

func TestShapeEqual(t *testing.T) {
	tests := []struct {
		name     string
		s1       Shape
		s2       Shape
		expected bool
	}{
		{
			name:     "both empty",
			s1:       NewShape(),
			s2:       NewShape(),
			expected: true,
		},
		{
			name:     "both nil",
			s1:       nil,
			s2:       nil,
			expected: true,
		},
		{
			name:     "empty and nil",
			s1:       NewShape(),
			s2:       nil,
			expected: true,
		},
		{
			name:     "same 1D shape",
			s1:       NewShape(5),
			s2:       NewShape(5),
			expected: true,
		},
		{
			name:     "same 2D shape",
			s1:       NewShape(2, 3),
			s2:       NewShape(2, 3),
			expected: true,
		},
		{
			name:     "same 3D shape",
			s1:       NewShape(2, 3, 4),
			s2:       NewShape(2, 3, 4),
			expected: true,
		},
		{
			name:     "different lengths",
			s1:       NewShape(2, 3),
			s2:       NewShape(2, 3, 4),
			expected: false,
		},
		{
			name:     "different values same length",
			s1:       NewShape(2, 3),
			s2:       NewShape(3, 2),
			expected: false,
		},
		{
			name:     "different first dimension",
			s1:       NewShape(2, 3),
			s2:       NewShape(5, 3),
			expected: false,
		},
		{
			name:     "different last dimension",
			s1:       NewShape(2, 3),
			s2:       NewShape(2, 5),
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.s1.Equal(tt.s2))
			// Should be symmetric
			assert.Equal(t, tt.expected, tt.s2.Equal(tt.s1))
		})
	}
}

func TestShapeStrides(t *testing.T) {
	tests := []struct {
		name     string
		shape    Shape
		expected []int
	}{
		{
			name:     "empty shape",
			shape:    NewShape(),
			expected: nil,
		},
		{
			name:     "1D shape",
			shape:    NewShape(5),
			expected: []int{1},
		},
		{
			name:     "2D shape",
			shape:    NewShape(2, 3),
			expected: []int{3, 1},
		},
		{
			name:     "3D shape",
			shape:    NewShape(2, 3, 4),
			expected: []int{12, 4, 1},
		},
		{
			name:     "4D shape",
			shape:    NewShape(2, 3, 4, 5),
			expected: []int{60, 20, 5, 1},
		},
		{
			name:     "large dimensions",
			shape:    NewShape(10, 20, 30),
			expected: []int{600, 30, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.shape.Strides()
			assert.Equal(t, tt.expected, result)
			// Verify row-major order: last dimension has stride 1
			if len(result) > 0 {
				assert.Equal(t, 1, result[len(result)-1])
			}
		})
	}
}

func TestShapeIsContiguous(t *testing.T) {
	tests := []struct {
		name    string
		shape   Shape
		strides []int
		want    bool
	}{
		{
			name:    "empty shape",
			shape:   NewShape(),
			strides: nil,
			want:    true,
		},
		{
			name:    "contiguous 1D",
			shape:   NewShape(5),
			strides: []int{1},
			want:    true,
		},
		{
			name:    "contiguous 2D",
			shape:   NewShape(2, 3),
			strides: []int{3, 1},
			want:    true,
		},
		{
			name:    "contiguous 3D",
			shape:   NewShape(2, 3, 4),
			strides: []int{12, 4, 1},
			want:    true,
		},
		{
			name:    "non-contiguous wrong strides",
			shape:   NewShape(2, 3),
			strides: []int{4, 1},
			want:    false,
		},
		{
			name:    "non-contiguous wrong length",
			shape:   NewShape(2, 3),
			strides: []int{3, 1, 1},
			want:    false,
		},
		{
			name:    "non-contiguous transposed",
			shape:   NewShape(2, 3),
			strides: []int{1, 2},
			want:    false,
		},
		{
			name:    "non-contiguous missing dimension",
			shape:   NewShape(2, 3),
			strides: []int{1},
			want:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.shape.IsContiguous(tt.strides))
		})
	}
}

func TestShapeValidateAxes(t *testing.T) {
	tests := []struct {
		name      string
		shape     Shape
		axes      []int
		wantError bool
		errorMsg  string
	}{
		{
			name:      "empty shape",
			shape:     NewShape(),
			axes:      []int{0},
			wantError: true,
			errorMsg:  "empty shape",
		},
		{
			name:      "no axes",
			shape:     NewShape(2, 3),
			axes:      []int{},
			wantError: false,
		},
		{
			name:      "valid single axis",
			shape:     NewShape(2, 3),
			axes:      []int{0},
			wantError: false,
		},
		{
			name:      "valid multiple axes",
			shape:     NewShape(2, 3, 4),
			axes:      []int{0, 2},
			wantError: false,
		},
		{
			name:      "axis out of range (too high)",
			shape:     NewShape(2, 3),
			axes:      []int{2},
			wantError: true,
			errorMsg:  "out of range",
		},
		{
			name:      "axis out of range (negative)",
			shape:     NewShape(2, 3),
			axes:      []int{-1},
			wantError: true,
			errorMsg:  "out of range",
		},
		{
			name:      "duplicate axis",
			shape:     NewShape(2, 3, 4),
			axes:      []int{0, 1, 0},
			wantError: true,
			errorMsg:  "duplicate axis",
		},
		{
			name:      "valid axes get sorted",
			shape:     NewShape(2, 3, 4),
			axes:      []int{2, 0},
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			axesCopy := make([]int, len(tt.axes))
			copy(axesCopy, tt.axes)
			err := tt.shape.ValidateAxes(axesCopy)
			if tt.wantError {
				assert.Error(t, err)
				if tt.errorMsg != "" {
					assert.Contains(t, err.Error(), tt.errorMsg)
				}
			} else {
				assert.NoError(t, err)
				// Verify axes are sorted
				if len(axesCopy) > 0 {
					for i := 1; i < len(axesCopy); i++ {
						assert.LessOrEqual(t, axesCopy[i-1], axesCopy[i])
					}
				}
			}
		})
	}
}

func TestShapeToSlice(t *testing.T) {
	tests := []struct {
		name     string
		shape    Shape
		expected []int
	}{
		{
			name:     "empty shape",
			shape:    NewShape(),
			expected: nil,
		},
		{
			name:     "1D shape",
			shape:    NewShape(5),
			expected: []int{5},
		},
		{
			name:     "2D shape",
			shape:    NewShape(2, 3),
			expected: []int{2, 3},
		},
		{
			name:     "3D shape",
			shape:    NewShape(2, 3, 4),
			expected: []int{2, 3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.shape.ToSlice()
			assert.Equal(t, tt.expected, result)
			// Verify it's a copy (modifying result shouldn't affect original)
			if len(result) > 0 {
				result[0] = 999
				if len(tt.shape) > 0 {
					assert.NotEqual(t, 999, tt.shape[0])
				}
			}
		})
	}
}

func TestShapeClone(t *testing.T) {
	tests := []struct {
		name     string
		shape    Shape
		expected Shape
	}{
		{
			name:     "nil shape",
			shape:    nil,
			expected: nil,
		},
		{
			name:     "empty shape",
			shape:    NewShape(),
			expected: NewShape(),
		},
		{
			name:     "1D shape",
			shape:    NewShape(5),
			expected: NewShape(5),
		},
		{
			name:     "2D shape",
			shape:    NewShape(2, 3),
			expected: NewShape(2, 3),
		},
		{
			name:     "3D shape",
			shape:    NewShape(2, 3, 4),
			expected: NewShape(2, 3, 4),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.shape.Clone()
			assert.Equal(t, tt.expected, result)
			// Verify it's a deep copy (modifying original shouldn't affect clone)
			if len(tt.shape) > 0 {
				original := tt.shape[0]
				tt.shape[0] = 999
				assert.NotEqual(t, 999, result[0])
				assert.Equal(t, original, result[0])
				// Restore for cleanup
				tt.shape[0] = original
			}
		})
	}
}

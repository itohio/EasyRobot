package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewShape(t *testing.T) {
	tests := []struct {
		name     string
		dims     []int
		expected Shape
	}{
		{
			name:     "empty shape",
			dims:     []int{},
			expected: Shape{},
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
			// Make a copy of dims to avoid modifying the test data
			dimsCopy := make([]int, len(tt.dims))
			copy(dimsCopy, tt.dims)
			result := NewShape(dimsCopy...)
			assert.Equal(t, tt.expected, result)
			// Verify it's the same slice (modifying input affects result)
			if len(dimsCopy) > 0 && len(result) > 0 {
				originalValue := result[0]
				dimsCopy[0] = 999
				assert.Equal(t, 999, result[0], "NewShape should return the same slice")
				// Restore for cleanup
				dimsCopy[0] = originalValue
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
			result := tt.shape.Strides(nil)
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
			// Verify it's the same slice (modifying result affects original)
			if len(result) > 0 && len(tt.shape) > 0 {
				originalValue := tt.shape[0]
				result[0] = 999
				assert.Equal(t, 999, tt.shape[0], "ToSlice should return the same backing array")
				// Restore for cleanup
				tt.shape[0] = originalValue
				result[0] = originalValue
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

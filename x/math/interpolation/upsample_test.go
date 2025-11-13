package interpolation

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestLinearUpsample(t *testing.T) {
	tests := []struct {
		name     string
		src      vec.Vector
		dstSize  int
		expected vec.Vector
	}{
		{
			name:     "2 to 5",
			src:      vec.Vector{0, 4},
			dstSize:  5,
			expected: vec.Vector{0, 1, 2, 3, 4},
		},
		{
			name:    "3 to 10",
			src:     vec.Vector{0, 5, 10},
			dstSize: 10,
		},
		{
			name:    "single value",
			src:     vec.Vector{42},
			dstSize: 5,
		},
		{
			name:    "same size",
			src:     vec.Vector{1, 2, 3},
			dstSize: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := vec.New(tt.dstSize)
			result := LinearUpsample(tt.src, dst)

			if result == nil {
				t.Fatal("LinearUpsample returned nil")
			}

			// Check endpoints match
			if result[0] != tt.src[0] {
				t.Errorf("Start point: got %v, want %v", result[0], tt.src[0])
			}
			if result[len(result)-1] != tt.src[len(tt.src)-1] {
				t.Errorf("End point: got %v, want %v", result[len(result)-1], tt.src[len(tt.src)-1])
			}

			// Check expected values if provided
			if tt.expected != nil {
				for i := range tt.expected {
					if abs(result[i]-tt.expected[i]) > 0.001 {
						t.Errorf("result[%d] = %v, want %v", i, result[i], tt.expected[i])
					}
				}
			}
		})
	}
}

func TestLinearUpsample_Invalid(t *testing.T) {
	// Test with empty source
	dst := vec.New(10)
	result := LinearUpsample(nil, dst)
	if result != nil {
		t.Error("Expected nil for empty source")
	}

	// Test with empty destination
	src := vec.Vector{1, 2, 3}
	result = LinearUpsample(src, nil)
	if result != nil {
		t.Error("Expected nil for empty destination")
	}
}

func TestCubicUpsample(t *testing.T) {
	tests := []struct {
		name     string
		src      vec.Vector
		dstSize  int
		validate func(t *testing.T, result vec.Vector)
	}{
		{
			name:    "linear data",
			src:     vec.Vector{0, 2, 4, 6},
			dstSize: 8,
			validate: func(t *testing.T, result vec.Vector) {
				// Cubic should interpolate smoothly through linear data
				if result[0] != 0 {
					t.Errorf("Start point: got %v, want 0", result[0])
				}
				if result[len(result)-1] != 6 {
					t.Errorf("End point: got %v, want 6", result[len(result)-1])
				}
			},
		},
		{
			name:    "single value",
			src:     vec.Vector{42},
			dstSize: 5,
			validate: func(t *testing.T, result vec.Vector) {
				for i := range result {
					if result[i] != 42 {
						t.Errorf("result[%d] = %v, want 42", i, result[i])
					}
				}
			},
		},
		{
			name:    "two points",
			src:     vec.Vector{0, 4},
			dstSize: 5,
			validate: func(t *testing.T, result vec.Vector) {
				// Should fallback to linear
				if result[0] != 0 {
					t.Errorf("Start point: got %v, want 0", result[0])
				}
				if result[len(result)-1] != 4 {
					t.Errorf("End point: got %v, want 4", result[len(result)-1])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dst := vec.New(tt.dstSize)
			result := CubicUpsample(tt.src, dst)

			if result == nil {
				t.Fatal("CubicUpsample returned nil")
			}

			if tt.validate != nil {
				tt.validate(t, result)
			}
		})
	}
}

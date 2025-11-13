package interpolation

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
)

func TestLinearMatrixUpsample(t *testing.T) {
	tests := []struct {
		name   string
		src    mat.Matrix
		dst    mat.Matrix
		verify func(t *testing.T, result mat.Matrix)
	}{
		{
			name: "2x2 to 4x4",
			src:  mat.New(2, 2, 0, 10, 10, 0),
			dst:  mat.New(4, 4),
			verify: func(t *testing.T, result mat.Matrix) {
				// Check corners match
				if result[0][0] != 0 {
					t.Errorf("Top-left: got %v, want 0", result[0][0])
				}
				if result[0][3] != 10 {
					t.Errorf("Top-right: got %v, want 10", result[0][3])
				}
				if result[3][0] != 10 {
					t.Errorf("Bottom-left: got %v, want 10", result[3][0])
				}
				if result[3][3] != 0 {
					t.Errorf("Bottom-right: got %v, want 0", result[3][3])
				}
			},
		},
		{
			name: "1x1 to 3x3",
			src:  mat.New(1, 1, 42),
			dst:  mat.New(3, 3),
			verify: func(t *testing.T, result mat.Matrix) {
				for i := range result {
					for j := range result[i] {
						if result[i][j] != 42 {
							t.Errorf("result[%d][%d] = %v, want 42", i, j, result[i][j])
						}
					}
				}
			},
		},
		{
			name: "3x3 to 6x6",
			src: mat.New(3, 3,
				0, 0, 0,
				0, 1, 0,
				0, 0, 0),
			dst: mat.New(6, 6),
			verify: func(t *testing.T, result mat.Matrix) {
				// Center should be the maximum value (but may be smoothed)
				center := result[3][3]
				maxVal := center
				for i := range result {
					for j := range result[i] {
						if result[i][j] > maxVal {
							maxVal = result[i][j]
						}
					}
				}
				if center < maxVal*0.8 {
					t.Errorf("Center value should be near max: got %v (max: %v)", center, maxVal)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := LinearMatrixUpsample(tt.src, tt.dst)

			if result == nil {
				t.Fatal("LinearMatrixUpsample returned nil")
			}

			if tt.verify != nil {
				tt.verify(t, result)
			}
		})
	}
}

func TestBicubicMatrixUpsample(t *testing.T) {
	tests := []struct {
		name   string
		src    mat.Matrix
		dst    mat.Matrix
		verify func(t *testing.T, result mat.Matrix)
	}{
		{
			name: "2x2 to 4x4",
			src:  mat.New(2, 2, 0, 10, 10, 0),
			dst:  mat.New(4, 4),
			verify: func(t *testing.T, result mat.Matrix) {
				// Check corners match
				if abs(result[0][0]) > 0.1 {
					t.Errorf("Top-left: got %v, want ~0", result[0][0])
				}
				if abs(result[0][3]-10) > 0.1 {
					t.Errorf("Top-right: got %v, want ~10", result[0][3])
				}
				if abs(result[3][0]-10) > 0.1 {
					t.Errorf("Bottom-left: got %v, want ~10", result[3][0])
				}
				if abs(result[3][3]) > 0.1 {
					t.Errorf("Bottom-right: got %v, want ~0", result[3][3])
				}
			},
		},
		{
			name: "smooth transition",
			src: mat.New(3, 3,
				0, 5, 10,
				5, 7, 10,
				10, 10, 10),
			dst: mat.New(6, 6),
			verify: func(t *testing.T, result mat.Matrix) {
				// Check monotonicity in diagonal
				for i := 1; i < 6; i++ {
					prev := result[i-1][i-1]
					curr := result[i][i]
					if curr < prev {
						t.Errorf("Non-monotonic at [%d][%d]: %v -> %v", i-1, i-1, prev, curr)
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := BicubicMatrixUpsample(tt.src, tt.dst)

			if result == nil {
				t.Fatal("BicubicMatrixUpsample returned nil")
			}

			if tt.verify != nil {
				tt.verify(t, result)
			}
		})
	}
}

func TestMatrixUpsample_Invalid(t *testing.T) {
	// Test with empty matrix
	result := LinearMatrixUpsample(nil, mat.New(3, 3))
	if result != nil {
		t.Error("Expected nil for empty source")
	}

	result = LinearMatrixUpsample(mat.New(2, 2), nil)
	if result != nil {
		t.Error("Expected nil for empty destination")
	}
}

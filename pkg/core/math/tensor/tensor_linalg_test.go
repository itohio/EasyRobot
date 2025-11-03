package tensor

import (
	"math"
	"testing"
)

func TestMatMul2D(t *testing.T) {
	tests := []struct {
		name     string
		t1       *Tensor
		t2       *Tensor
		expected []float32
		expShape []int
	}{
		{
			name:     "2x3 × 3x2 = 2x2",
			t1:       &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}},
			t2:       &Tensor{Dim: []int{3, 2}, Data: []float32{1, 2, 3, 4, 5, 6}},
			expected: []float32{22, 28, 49, 64}, // [1*1+2*3+3*5, 1*2+2*4+3*6, 4*1+5*3+6*5, 4*2+5*4+6*6]
			expShape: []int{2, 2},
		},
		{
			name:     "1x4 × 4x1 = 1x1",
			t1:       &Tensor{Dim: []int{1, 4}, Data: []float32{1, 2, 3, 4}},
			t2:       &Tensor{Dim: []int{4, 1}, Data: []float32{1, 2, 3, 4}},
			expected: []float32{30}, // 1*1+2*2+3*3+4*4
			expShape: []int{1, 1},
		},
		{
			name:     "3x2 × 2x4 = 3x4",
			t1:       &Tensor{Dim: []int{3, 2}, Data: []float32{1, 2, 3, 4, 5, 6}},
			t2:       &Tensor{Dim: []int{2, 4}, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}},
			expected: []float32{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68},
			expShape: []int{3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.MatMul(tt.t2)

			if len(result.Dim) != len(tt.expShape) {
				t.Fatalf("Shape length mismatch: got %v, expected %v", result.Dim, tt.expShape)
			}

			for i := range tt.expShape {
				if result.Dim[i] != tt.expShape[i] {
					t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], tt.expShape[i])
				}
			}

			for i := range tt.expected {
				if !floatEqual(result.Data[i], tt.expected[i]) {
					t.Errorf("Data[%d] = %f, expected %f", i, result.Data[i], tt.expected[i])
				}
			}
		})
	}
}

func TestMatMulBatched(t *testing.T) {
	t.Run("2x3x2 × 2x2x4 = 2x3x4", func(t *testing.T) {
		// Batch of 2, each is 3×2 and 2×4
		t1 := &Tensor{Dim: []int{2, 3, 2}, Data: []float32{
			1, 2, 3, 4, 5, 6, // First batch: 3×2
			1, 2, 3, 4, 5, 6, // Second batch: 3×2
		}}
		t2 := &Tensor{Dim: []int{2, 2, 4}, Data: []float32{
			1, 2, 3, 4, 5, 6, 7, 8, // First batch: 2×4
			1, 2, 3, 4, 5, 6, 7, 8, // Second batch: 2×4
		}}

		result := t1.MatMul(t2)

		expectedShape := []int{2, 3, 4}
		if len(result.Dim) != len(expectedShape) {
			t.Fatalf("Shape mismatch: got %v, expected %v", result.Dim, expectedShape)
		}

		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}

		// Check first element of result: [0,0,0] = [1,2] × [1;5] = 1*1+2*5 = 11
		if !floatEqual(result.Data[0], 11.0) {
			t.Errorf("result[0] = %f, expected 11.0", result.Data[0])
		}
	})

	t.Run("3x2 × 2x2x4 = 2x3x4 (broadcast first)", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{3, 2}, Data: []float32{1, 2, 3, 4, 5, 6}}
		t2 := &Tensor{Dim: []int{2, 2, 4}, Data: []float32{
			1, 2, 3, 4, 5, 6, 7, 8, // First batch
			1, 2, 3, 4, 5, 6, 7, 8, // Second batch
		}}

		result := t1.MatMul(t2)

		expectedShape := []int{2, 3, 4}
		if len(result.Dim) != len(expectedShape) {
			t.Fatalf("Shape mismatch: got %v, expected %v", result.Dim, expectedShape)
		}

		// Result should have shape [2, 3, 4]
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

func TestMatMulTo(t *testing.T) {
	t.Run("create new tensor", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
		t2 := &Tensor{Dim: []int{3, 2}, Data: []float32{1, 2, 3, 4, 5, 6}}

		result := t1.MatMulTo(t2, nil)

		if result == nil {
			t.Fatal("MatMulTo returned nil")
		}

		expectedShape := []int{2, 2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})

	t.Run("use destination tensor", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
		t2 := &Tensor{Dim: []int{3, 2}, Data: []float32{1, 2, 3, 4, 5, 6}}
		dst := &Tensor{Dim: []int{2, 2}, Data: make([]float32, 4)}

		result := t1.MatMulTo(t2, dst)

		if result != dst {
			t.Errorf("MatMulTo should return dst")
		}

		// Check that dst was filled
		if dst.Data[0] == 0 {
			t.Error("dst should be filled with result")
		}
	})
}

func TestTranspose2D(t *testing.T) {
	tests := []struct {
		name     string
		t        *Tensor
		expected []float32
		expShape []int
	}{
		{
			name:     "2x3 transpose",
			t:        &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}},
			expected: []float32{1, 4, 2, 5, 3, 6}, // [1,2,3; 4,5,6]^T = [1,4; 2,5; 3,6]
			expShape: []int{3, 2},
		},
		{
			name:     "3x2 transpose",
			t:        &Tensor{Dim: []int{3, 2}, Data: []float32{1, 2, 3, 4, 5, 6}},
			expected: []float32{1, 3, 5, 2, 4, 6}, // [1,2; 3,4; 5,6]^T = [1,3,5; 2,4,6]
			expShape: []int{2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Transpose()

			if len(result.Dim) != len(tt.expShape) {
				t.Fatalf("Shape length mismatch: got %v, expected %v", result.Dim, tt.expShape)
			}

			for i := range tt.expShape {
				if result.Dim[i] != tt.expShape[i] {
					t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], tt.expShape[i])
				}
			}

			for i := range tt.expected {
				if !floatEqual(result.Data[i], tt.expected[i]) {
					t.Errorf("Data[%d] = %f, expected %f", i, result.Data[i], tt.expected[i])
				}
			}
		})
	}
}

func TestTransposeTo(t *testing.T) {
	t.Run("create new tensor", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}

		result := t1.TransposeTo(nil)

		if result == nil {
			t.Fatal("TransposeTo returned nil")
		}

		expectedShape := []int{3, 2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})

	t.Run("use destination tensor", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
		dst := &Tensor{Dim: []int{3, 2}, Data: make([]float32, 6)}

		result := t1.TransposeTo(dst)

		if result != dst {
			t.Errorf("TransposeTo should return dst")
		}

		// Check that dst was filled
		if dst.Data[0] == 0 {
			t.Error("dst should be filled with result")
		}
	})
}

func TestDot(t *testing.T) {
	t.Run("vector dot product", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}}
		t2 := &Tensor{Dim: []int{3}, Data: []float32{4, 5, 6}}

		result := t1.Dot(t2)
		expected := float32(1*4 + 2*5 + 3*6) // 4 + 10 + 18 = 32

		if !floatEqual(result, expected) {
			t.Errorf("Dot = %f, expected %f", result, expected)
		}
	})

	t.Run("matrix Frobenius inner product", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
		t2 := &Tensor{Dim: []int{2, 3}, Data: []float32{1, 1, 1, 1, 1, 1}}

		result := t1.Dot(t2)
		expected := float32(1 + 2 + 3 + 4 + 5 + 6) // 21

		if !floatEqual(result, expected) {
			t.Errorf("Frobenius inner product = %f, expected %f", result, expected)
		}
	})
}

func TestNorm(t *testing.T) {
	tests := []struct {
		name     string
		t        *Tensor
		ord      int
		expected float32
	}{
		{
			name:     "L1 norm of vector",
			t:        &Tensor{Dim: []int{3}, Data: []float32{3, -4, 5}},
			ord:      0,
			expected: 12.0, // |3| + |-4| + |5| = 12
		},
		{
			name:     "L2 norm of vector",
			t:        &Tensor{Dim: []int{3}, Data: []float32{3, 4, 0}},
			ord:      1,
			expected: 5.0, // sqrt(3^2 + 4^2 + 0^2) = 5
		},
		{
			name:     "Frobenius norm of matrix",
			t:        &Tensor{Dim: []int{2, 2}, Data: []float32{3, 4, 0, 0}},
			ord:      2,
			expected: 5.0, // sqrt(3^2 + 4^2 + 0^2 + 0^2) = 5
		},
		{
			name:     "L2 norm of matrix (same as Frobenius)",
			t:        &Tensor{Dim: []int{2, 2}, Data: []float32{1, 0, 0, 1}},
			ord:      1,
			expected: float32(math.Sqrt(2)), // sqrt(1 + 0 + 0 + 1) = sqrt(2)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Norm(tt.ord)

			if !floatEqual(result, tt.expected) {
				t.Errorf("Norm(%d) = %f, expected %f", tt.ord, result, tt.expected)
			}
		})
	}
}

func TestNormalize(t *testing.T) {
	t.Run("normalize vector", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{3}, Data: []float32{3, 4, 0}}
		result := t1.Normalize(0)

		// After normalization, L2 norm should be 1
		norm := result.Norm(1)
		if !floatEqual(norm, 1.0) {
			t.Errorf("Normalized vector norm = %f, expected 1.0", norm)
		}

		// Check first element: 3/5 = 0.6
		expectedFirst := float32(3.0 / 5.0)
		if !floatEqual(result.Data[0], expectedFirst) {
			t.Errorf("result[0] = %f, expected %f", result.Data[0], expectedFirst)
		}
	})

	t.Run("normalize matrix along rows", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{2, 3}, Data: []float32{3, 4, 0, 0, 0, 0}}
		result := t1.Normalize(0)

		// First row should have L2 norm = 1
		firstRowNorm := float32(math.Sqrt(float64(result.Data[0]*result.Data[0] + result.Data[1]*result.Data[1] + result.Data[2]*result.Data[2])))
		if !floatEqual(firstRowNorm, 1.0) {
			t.Errorf("First row norm = %f, expected 1.0", firstRowNorm)
		}

		// First row: [3,4,0] normalized = [3/5, 4/5, 0]
		expectedFirst := float32(3.0 / 5.0)
		if !floatEqual(result.Data[0], expectedFirst) {
			t.Errorf("result[0] = %f, expected %f", result.Data[0], expectedFirst)
		}
	})

	t.Run("normalize matrix along columns", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{2, 3}, Data: []float32{3, 0, 0, 4, 0, 0}}
		result := t1.Normalize(1)

		// First column should have L2 norm = 1
		firstColNorm := float32(math.Sqrt(float64(result.Data[0]*result.Data[0] + result.Data[3]*result.Data[3])))
		if !floatEqual(firstColNorm, 1.0) {
			t.Errorf("First column norm = %f, expected 1.0", firstColNorm)
		}
	})
}

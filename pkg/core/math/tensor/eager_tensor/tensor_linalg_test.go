package eager_tensor

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestMatMul2D(t *testing.T) {
	tests := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		expected []float32
		expShape []int
	}{
		{
			name:     "2x3 × 3x2 = 2x2",
			t1:       FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6}),
			t2:       FromFloat32(types.NewShape(3, 2), []float32{1, 2, 3, 4, 5, 6}),
			expected: []float32{22, 28, 49, 64}, // [1*1+2*3+3*5, 1*2+2*4+3*6, 4*1+5*3+6*5, 4*2+5*4+6*6]
			expShape: []int{2, 2},
		},
		{
			name:     "1x4 × 4x1 = 1x1",
			t1:       FromFloat32(types.NewShape(1, 4), []float32{1, 2, 3, 4}),
			t2:       FromFloat32(types.NewShape(4, 1), []float32{1, 2, 3, 4}),
			expected: []float32{30}, // 1*1+2*2+3*3+4*4
			expShape: []int{1, 1},
		},
		{
			name:     "3x2 × 2x4 = 3x4",
			t1:       FromFloat32(types.NewShape(3, 2), []float32{1, 2, 3, 4, 5, 6}),
			t2:       FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			expected: []float32{11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68},
			expShape: []int{3, 4},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.MatMul(nil, tt.t2)

			resultShape := result.Shape()
			assert.Equal(t, len(tt.expShape), len(resultShape), "Shape length mismatch")

			for i := range tt.expShape {
				assert.Equal(t, tt.expShape[i], resultShape[i], "Shape dimension %d mismatch", i)
			}

			resultData := result.Data().([]float32)
			for i := range tt.expected {
				assert.InDelta(t, tt.expected[i], resultData[i], 1e-6, "Data[%d] mismatch", i)
			}
		})
	}
}

func TestMatMulBatched(t *testing.T) {
	t.Run("2x3x2 × 2x2x4 = 2x3x4", func(t *testing.T) {
		// Batch of 2, each is 3×2 and 2×4
		t1 := FromFloat32(types.NewShape(2, 3, 2), []float32{
			1, 2, 3, 4, 5, 6, // First batch: 3×2
			1, 2, 3, 4, 5, 6, // Second batch: 3×2
		})
		t2 := FromFloat32(types.NewShape(2, 2, 4), []float32{
			1, 2, 3, 4, 5, 6, 7, 8, // First batch: 2×4
			1, 2, 3, 4, 5, 6, 7, 8, // Second batch: 2×4
		})

		result := t1.MatMul(nil, t2)

		expectedShape := []int{2, 3, 4}
		resultShape := result.Shape()
		if len(resultShape) != len(expectedShape) {
			t.Fatalf("Shape mismatch: got %v, expected %v", resultShape, expectedShape)
		}

		for i := range expectedShape {
			if resultShape[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, resultShape[i], expectedShape[i])
			}
		}

		// Check first element of result: [0,0,0] = [1,2] × [1;5] = 1*1+2*5 = 11
		resultData := result.Data().([]float32)
		assert.InDelta(t, 11.0, resultData[0], 1e-6, "result[0] = %f, expected 11.0", resultData[0])
	})

	t.Run("3x2 × 2x2x4 = 2x3x4 (broadcast first)", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(3, 2), []float32{1, 2, 3, 4, 5, 6})
		t2 := FromFloat32(types.NewShape(2, 2, 4), []float32{
			1, 2, 3, 4, 5, 6, 7, 8, // First batch
			1, 2, 3, 4, 5, 6, 7, 8, // Second batch
		})

		result := t1.MatMul(nil, t2)

		expectedShape := []int{2, 3, 4}
		resultShape := result.Shape()
		if len(resultShape) != len(expectedShape) {
			t.Fatalf("Shape mismatch: got %v, expected %v", resultShape, expectedShape)
		}

		// Result should have shape [2, 3, 4]
		for i := range expectedShape {
			if resultShape[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, resultShape[i], expectedShape[i])
			}
		}
	})
}

func TestMatMulTo(t *testing.T) {
	t.Run("create new tensor", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
		t2 := FromFloat32(types.NewShape(3, 2), []float32{1, 2, 3, 4, 5, 6})

		result := t1.MatMul(nil, t2)

		expectedShape := []int{2, 2}
		resultShape := result.Shape()
		for i := range expectedShape {
			if resultShape[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, resultShape[i], expectedShape[i])
			}
		}
	})

	t.Run("use destination tensor", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
		t2 := FromFloat32(types.NewShape(3, 2), []float32{1, 2, 3, 4, 5, 6})
		dst := New(types.FP32, types.NewShape(2, 2))

		result := t1.MatMul(&dst, t2)

		assert.Equal(t, &dst, result, "MatMulTo should return dst")

		// Check that dst was filled
		dstData := dst.Data().([]float32)
		assert.NotEqual(t, float32(0), dstData[0], "dst should be filled with result")
	})
}

func TestTranspose2D(t *testing.T) {
	tests := []struct {
		name     string
		t        Tensor
		expected []float32
		expShape []int
	}{
		{
			name:     "2x3 transpose",
			t:        FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6}),
			expected: []float32{1, 4, 2, 5, 3, 6}, // [1,2,3; 4,5,6]^T = [1,4; 2,5; 3,6]
			expShape: []int{3, 2},
		},
		{
			name:     "3x2 transpose",
			t:        FromFloat32(types.NewShape(3, 2), []float32{1, 2, 3, 4, 5, 6}),
			expected: []float32{1, 3, 5, 2, 4, 6}, // [1,2; 3,4; 5,6]^T = [1,3,5; 2,4,6]
			expShape: []int{2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Transpose(nil, nil)

			resultShape := result.Shape()
			assert.Equal(t, len(tt.expShape), len(resultShape), "Shape length mismatch")

			for i := range tt.expShape {
				assert.Equal(t, tt.expShape[i], resultShape[i], "Shape dimension %d mismatch", i)
			}

			resultData := result.Data().([]float32)
			for i := range tt.expected {
				assert.InDelta(t, tt.expected[i], resultData[i], 1e-5, "Data[%d] = %f, expected %f", i, resultData[i], tt.expected[i])
			}
		})
	}
}

func TestTransposeTo(t *testing.T) {
	t.Run("create new tensor", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})

		result := t1.Transpose(nil, nil)

		if result == nil {
			t.Fatal("TransposeTo returned nil")
		}

		expectedShape := []int{3, 2}
		resultShape := result.Shape()
		for i := range expectedShape {
			if resultShape[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, resultShape[i], expectedShape[i])
			}
		}
	})

	t.Run("use destination tensor", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
		dst := New(types.FP32, types.NewShape(3, 2))

		result := t1.Transpose(&dst, nil)

		assert.Equal(t, &dst, result, "TransposeTo should return dst")

		// Check that dst was filled
		dstData := dst.Data().([]float32)
		assert.NotEqual(t, float32(0), dstData[0], "dst should be filled with result")
	})
}

func TestDot(t *testing.T) {
	t.Run("vector dot product", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(3), []float32{1, 2, 3})
		t2 := FromFloat32(types.NewShape(3), []float32{4, 5, 6})

		result := t1.Dot(t2)
		expected := float32(1*4 + 2*5 + 3*6) // 4 + 10 + 18 = 32

		assert.InDelta(t, expected, result, 1e-5, "Dot = %f, expected %f", result, expected)
	})

	t.Run("matrix Frobenius inner product", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
		t2 := FromFloat32(types.NewShape(2, 3), []float32{1, 1, 1, 1, 1, 1})

		result := t1.Dot(t2)
		expected := float32(1 + 2 + 3 + 4 + 5 + 6) // 21

		assert.InDelta(t, expected, result, 1e-5, "Frobenius inner product = %f, expected %f", result, expected)
	})
}

func TestNorm(t *testing.T) {
	tests := []struct {
		name     string
		t        Tensor
		ord      int
		expected float32
	}{
		{
			name:     "L1 norm of vector",
			t:        FromFloat32(types.NewShape(3), []float32{3, -4, 5}),
			ord:      0,
			expected: 12.0, // |3| + |-4| + |5| = 12
		},
		{
			name:     "L2 norm of vector",
			t:        FromFloat32(types.NewShape(3), []float32{3, 4, 0}),
			ord:      1,
			expected: 5.0, // sqrt(3^2 + 4^2 + 0^2) = 5
		},
		{
			name:     "Frobenius norm of matrix",
			t:        FromFloat32(types.NewShape(2, 2), []float32{3, 4, 0, 0}),
			ord:      2,
			expected: 5.0, // sqrt(3^2 + 4^2 + 0^2 + 0^2) = 5
		},
		{
			name:     "L2 norm of matrix (same as Frobenius)",
			t:        FromFloat32(types.NewShape(2, 2), []float32{1, 0, 0, 1}),
			ord:      1,
			expected: float32(math.Sqrt(2)), // sqrt(1 + 0 + 0 + 1) = sqrt(2)
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Norm(tt.ord)

			assert.InDelta(t, tt.expected, result, 1e-5, "Norm(%d) = %f, expected %f", tt.ord, result, tt.expected)
		})
	}
}

func TestNormalize(t *testing.T) {
	t.Run("normalize vector", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(3), []float32{3, 4, 0})
		result := t1.Normalize(nil, 0)

		// After normalization, L2 norm should be 1
		norm := result.Norm(1)
		assert.InDelta(t, 1.0, norm, 1e-5, "Normalized vector norm = %f, expected 1.0", norm)

		// Check first element: 3/5 = 0.6
		expectedFirst := float32(3.0 / 5.0)
		resultData := result.Data().([]float32)
		assert.InDelta(t, expectedFirst, resultData[0], 1e-5, "result[0] = %f, expected %f", resultData[0], expectedFirst)
	})

	t.Run("normalize matrix along rows", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 3), []float32{3, 4, 0, 0, 0, 0})
		result := t1.Normalize(nil, 0)

		// First row should have L2 norm = 1
		resultData := result.Data().([]float32)
		firstRowNorm := float32(math.Sqrt(float64(resultData[0]*resultData[0] + resultData[1]*resultData[1] + resultData[2]*resultData[2])))
		assert.InDelta(t, 1.0, firstRowNorm, 1e-5, "First row norm = %f, expected 1.0", firstRowNorm)

		// First row: [3,4,0] normalized = [3/5, 4/5, 0]
		expectedFirst := float32(3.0 / 5.0)
		resultData = result.Data().([]float32)
		assert.InDelta(t, expectedFirst, resultData[0], 1e-5, "result[0] = %f, expected %f", resultData[0], expectedFirst)
	})

	t.Run("normalize matrix along columns", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 3), []float32{3, 0, 0, 4, 0, 0})
		result := t1.Normalize(nil, 1)

		// First column should have L2 norm = 1
		resultData := result.Data().([]float32)
		firstColNorm := float32(math.Sqrt(float64(resultData[0]*resultData[0] + resultData[3]*resultData[3])))
		assert.InDelta(t, 1.0, firstColNorm, 1e-5, "First column norm = %f, expected 1.0", firstColNorm)
	})
}

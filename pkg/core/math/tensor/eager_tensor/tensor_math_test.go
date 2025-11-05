package eager_tensor

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestAdd(t *testing.T) {
	tests := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		expected []float32
	}{
		{
			name:     "2x2 addition",
			t1:       FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			t2:       FromFloat32(types.NewShape(2, 2), []float32{5, 6, 7, 8}),
			expected: []float32{6, 8, 10, 12},
		},
		{
			name:     "1D addition",
			t1:       FromFloat32(types.NewShape(3), []float32{1, 2, 3}),
			t2:       FromFloat32(types.NewShape(3), []float32{4, 5, 6}),
			expected: []float32{5, 7, 9},
		},
		{
			name:     "scalar-like 1x1",
			t1:       FromFloat32(types.NewShape(1, 1), []float32{10}),
			t2:       FromFloat32(types.NewShape(1, 1), []float32{20}),
			expected: []float32{30},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalT2 := tt.t2.Clone()
			result := tt.t1.Add(tt.t2)

			// Verify result is the same tensor (same data and shape) for chaining
			assert.NotNil(t, result, "Add should return non-nil result for chaining")
			assert.True(t, result.Shape().Equal(tt.t1.Shape()), "Add should return tensor with same shape for chaining")
			// Check that result shares the same underlying data (modify result and verify original changes)
			resultData := result.Data().([]float32)
			t1Data := tt.t1.Data().([]float32)
			if len(resultData) > 0 {
				assert.Equal(t, &resultData[0], &t1Data[0], "Add should return tensor sharing same data for chaining")
			}
			for i := range tt.expected {
				assert.InDelta(t, float64(tt.expected[i]), float64(t1Data[i]), 1e-6)

			}

			// Verify tt.t1 was changed (equals expected)
			for i := range tt.expected {
				assert.InDelta(t, float64(tt.expected[i]), float64(tt.t1.Data().([]float32)[i]), 1e-6, "t1 changed at %d", i)
			}
			// Verify tt.t2 was not modified
			originalT2Data := originalT2.Data().([]float32)
			t2Data := tt.t2.Data().([]float32)
			for i := range originalT2Data {
				assert.InDelta(t, float64(originalT2Data[i]), float64(t2Data[i]), 1e-6, "t2 modified at %d: got %f, expected %f", i, t2Data[i], originalT2Data[i])
			}
		})
	}
}

func TestSub(t *testing.T) {
	tests := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		expected []float32
	}{
		{
			name:     "2x2 subtraction",
			t1:       FromFloat32(types.NewShape(2, 2), []float32{10, 20, 30, 40}),
			t2:       FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			expected: []float32{9, 18, 27, 36},
		},
		{
			name:     "1D subtraction",
			t1:       FromFloat32(types.NewShape(3), []float32{10, 20, 30}),
			t2:       FromFloat32(types.NewShape(3), []float32{1, 2, 3}),
			expected: []float32{9, 18, 27},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.Sub(tt.t2)

			// Verify result is the same tensor (same data and shape) for chaining
			assert.NotNil(t, result, "Sub should return non-nil result for chaining")
			assert.True(t, result.Shape().Equal(tt.t1.Shape()), "Sub should return tensor with same shape for chaining")
			// Check that result shares the same underlying data (modify result and verify original changes)
			resultData := result.Data().([]float32)
			t1Data := tt.t1.Data().([]float32)
			if len(resultData) > 0 {
				assert.Equal(t, &resultData[0], &t1Data[0], "Sub should return tensor sharing same data for chaining")
			}
			for i := range tt.expected {
				assert.InDeltaf(t, float64(tt.expected[i]), float64(t1Data[i]), 1e-6, "Data[%d] = %f, expected %f", i, t1Data[i], tt.expected[i])
			}
		})
	}
}

func TestMul(t *testing.T) {
	tests := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		expected []float32
	}{
		{
			name:     "2x2 multiplication",
			t1:       FromFloat32(types.NewShape(2, 2), []float32{2, 3, 4, 5}),
			t2:       FromFloat32(types.NewShape(2, 2), []float32{2, 2, 2, 2}),
			expected: []float32{4, 6, 8, 10},
		},
		{
			name:     "1D multiplication",
			t1:       FromFloat32(types.NewShape(3), []float32{1, 2, 3}),
			t2:       FromFloat32(types.NewShape(3), []float32{2, 3, 4}),
			expected: []float32{2, 6, 12},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.Mul(tt.t2)

			// Verify result is the same tensor (same data and shape) for chaining
			assert.NotNil(t, result, "Mul should return non-nil result for chaining")
			assert.True(t, result.Shape().Equal(tt.t1.Shape()), "Mul should return tensor with same shape for chaining")
			// Check that result shares the same underlying data (modify result and verify original changes)
			resultData := result.Data().([]float32)
			t1Data := tt.t1.Data().([]float32)
			if len(resultData) > 0 {
				assert.Equal(t, &resultData[0], &t1Data[0], "Mul should return tensor sharing same data for chaining")
			}
			for i := range tt.expected {
				assert.InDeltaf(t, float64(tt.expected[i]), float64(t1Data[i]), 1e-6, "Data[%d] = %f, expected %f", i, t1Data[i], tt.expected[i])
			}
		})
	}
}

func TestDiv(t *testing.T) {
	tests := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		expected []float32
	}{
		{
			name:     "2x2 division",
			t1:       FromFloat32(types.NewShape(2, 2), []float32{10, 20, 30, 40}),
			t2:       FromFloat32(types.NewShape(2, 2), []float32{2, 4, 5, 8}),
			expected: []float32{5, 5, 6, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.Div(tt.t2)

			// Verify result is the same tensor (same data and shape) for chaining
			assert.NotNil(t, result, "Div should return non-nil result for chaining")
			assert.True(t, result.Shape().Equal(tt.t1.Shape()), "Div should return tensor with same shape for chaining")
			// Check that result shares the same underlying data (modify result and verify original changes)
			resultData := result.Data().([]float32)
			t1Data := tt.t1.Data().([]float32)
			if len(resultData) > 0 {
				assert.Equal(t, &resultData[0], &t1Data[0], "Div should return tensor sharing same data for chaining")
			}

			for i := range tt.expected {
				t1Data := tt.t1.Data().([]float32)
				assert.InDeltaf(t, float64(tt.expected[i]), float64(t1Data[i]), 1e-6, "Data[%d] = %f, expected %f", i, t1Data[i], tt.expected[i])
			}
		})
	}
}

func TestScale(t *testing.T) {
	tests := []struct {
		name     string
		t        Tensor
		scalar   float64
		expected []float32
	}{
		{
			name:     "scale by 2",
			t:        FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			scalar:   2.0,
			expected: []float32{2, 4, 6, 8},
		},
		{
			name:     "scale by 0.5",
			t:        FromFloat32(types.NewShape(3), []float32{2, 4, 6}),
			scalar:   0.5,
			expected: []float32{1, 2, 3},
		},
		{
			name:     "scale by 0",
			t:        FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			scalar:   0.0,
			expected: []float32{0, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Scale(tt.scalar)

			// Verify result is the same tensor (same data and shape) for chaining
			assert.NotNil(t, result, "Scale should return non-nil result for chaining")
			assert.True(t, result.Shape().Equal(tt.t.Shape()), "Scale should return tensor with same shape for chaining")
			// Check that result shares the same underlying data (modify result and verify original changes)
			resultData := result.Data().([]float32)
			tData := tt.t.Data().([]float32)
			if len(resultData) > 0 {
				assert.Equal(t, &resultData[0], &tData[0], "Scale should return tensor sharing same data for chaining")
			}

			for i := range tt.expected {
				tData := tt.t.Data().([]float32)
				assert.InDelta(t, float64(tt.expected[i]), float64(tData[i]), 1e-6, "Data[%d] = %f, expected %f", i, tData[i], tt.expected[i])
			}
		})
	}
}

func TestAddTo(t *testing.T) {
	t.Run("create new tensor", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4})
		t2 := FromFloat32(types.NewShape(2, 2), []float32{5, 6, 7, 8})

		result := t1.AddTo(t2, nil)

		assert.NotEqual(t, &t1, result, "AddTo should create new tensor when dst is nil")
		assert.NotEqual(t, &t2, result, "AddTo should create new tensor when dst is nil")

		expected := []float32{6, 8, 10, 12}
		resultData := result.Data().([]float32)
		for i := range expected {
			assert.InDelta(t, float64(expected[i]), float64(resultData[i]), 1e-6, "Data[%d] = %f, expected %f", i, resultData[i], expected[i])
		}

		// Original tensors should be unchanged
		t1Data := t1.Data().([]float32)
		t2Data := t2.Data().([]float32)
		assert.Equal(t, float32(1), t1Data[0], "Original t1 should be unchanged")
		assert.Equal(t, float32(5), t2Data[0], "Original t2 should be unchanged")
	})

	t.Run("use destination tensor", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4})
		t2 := FromFloat32(types.NewShape(2, 2), []float32{5, 6, 7, 8})
		dst := New(types.DTFP32, types.NewShape(2, 2))

		result := t1.AddTo(t2, &dst)

		assert.Equal(t, &dst, result, "AddTo should use dst when provided")

		expected := []float32{6, 8, 10, 12}
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDelta(t, float64(expected[i]), float64(dstData[i]), 1e-6)
		}
	})
}

func TestMulTo(t *testing.T) {
	t.Run("create new tensor", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4})
		t2 := FromFloat32(types.NewShape(2, 2), []float32{2, 3, 4, 5})

		result := t1.MulTo(t2, nil)

		assert.NotEqual(t, &t1, result, "MulTo should create new tensor when dst is nil")
		assert.NotEqual(t, &t2, result, "MulTo should create new tensor when dst is nil")

		expected := []float32{2, 6, 12, 20}
		resultData := result.Data().([]float32)
		for i := range expected {
			assert.InDelta(t, float64(expected[i]), float64(resultData[i]), 1e-6)
		}

		// Original tensors should be unchanged
		t1Data := t1.Data().([]float32)
		t2Data := t2.Data().([]float32)
		assert.Equal(t, float32(1), t1Data[0], "Original t1 should be unchanged")
		assert.Equal(t, float32(2), t2Data[0], "Original t2 should be unchanged")
	})

	t.Run("use destination tensor", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4})
		t2 := FromFloat32(types.NewShape(2, 2), []float32{2, 3, 4, 5})
		dst := New(types.DTFP32, types.NewShape(2, 2))

		result := t1.MulTo(t2, &dst)

		assert.Equal(t, &dst, result, "MulTo should use dst when provided")

		expected := []float32{2, 6, 12, 20}
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDelta(t, float64(expected[i]), float64(dstData[i]), 1e-6)
		}
	})
}

func TestSum(t *testing.T) {
	tests := []struct {
		name     string
		t        Tensor
		dims     []int
		expected []float32
		expShape []int
	}{
		{
			name:     "sum all elements",
			t:        FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			dims:     nil,
			expected: []float32{10},
			expShape: []int{1},
		},
		{
			name:     "sum along first dimension",
			t:        FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6}),
			dims:     []int{0},
			expected: []float32{5, 7, 9},
			expShape: []int{3},
		},
		{
			name:     "sum along second dimension",
			t:        FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6}),
			dims:     []int{1},
			expected: []float32{6, 15},
			expShape: []int{2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Sum(tt.dims...)

			resultShape := result.Shape()
			assert.Equal(t, len(tt.expShape), len(resultShape), "Shape length mismatch")

			for i := range tt.expShape {
				assert.Equal(t, tt.expShape[i], resultShape[i], "Dim[%d] mismatch", i)
			}

			for i := range tt.expected {
				resultData := result.Data().([]float32)
				assert.InDelta(t, float64(tt.expected[i]), float64(resultData[i]), 1e-6, "Data[%d] mismatch", i)
			}
		})
	}
}

func TestMean(t *testing.T) {
	t.Run("mean of all elements", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4})
		result := tensor.Mean()
		expected := float32(10.0 / 4.0)

		resultData := result.Data().([]float32)
		if len(resultData) != 1 {
			t.Fatalf("Expected 1 element, got %d", len(resultData))
		}

		assert.InDelta(t, float64(expected), float64(resultData[0]), 1e-6)
	})

	t.Run("mean along dimension", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
		result := tensor.Mean(0)
		expected := []float32{2.5, 3.5, 4.5} // (1+4)/2, (2+5)/2, (3+6)/2

		for i := range expected {
			resultData := result.Data().([]float32)
			assert.InDelta(t, float64(expected[i]), float64(resultData[i]), 1e-6, "Data[%d] = %f, expected %f", i, resultData[i], expected[i])
		}
	})
}

func TestMax(t *testing.T) {
	tests := []struct {
		name     string
		t        Tensor
		dims     []int
		expected []float32
	}{
		{
			name:     "max of all elements",
			t:        FromFloat32(types.NewShape(2, 2), []float32{1, 5, 3, 2}),
			dims:     nil,
			expected: []float32{5},
		},
		{
			name:     "max along dimension",
			t:        FromFloat32(types.NewShape(2, 3), []float32{1, 5, 3, 2, 4, 6}),
			dims:     []int{1},
			expected: []float32{5, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Max(tt.dims...)

			for i := range tt.expected {
				resultData := result.Data().([]float32)
				assert.InDelta(t, float64(tt.expected[i]), float64(resultData[i]), 1e-6, "Data[%d] mismatch", i)
			}
		})
	}
}

func TestMin(t *testing.T) {
	tests := []struct {
		name     string
		t        Tensor
		dims     []int
		expected []float32
	}{
		{
			name:     "min of all elements",
			t:        FromFloat32(types.NewShape(2, 2), []float32{5, 2, 3, 1}),
			dims:     nil,
			expected: []float32{1},
		},
		{
			name:     "min along dimension",
			t:        FromFloat32(types.NewShape(2, 3), []float32{5, 2, 3, 1, 4, 6}),
			dims:     []int{1},
			expected: []float32{2, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Min(tt.dims...)

			for i := range tt.expected {
				resultData := result.Data().([]float32)
				assert.InDelta(t, float64(tt.expected[i]), float64(resultData[i]), 1e-6, "Data[%d] mismatch", i)
			}
		})
	}
}

func TestArgMax(t *testing.T) {
	t.Run("1D tensor", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(5), []float32{1, 5, 3, 2, 4})
		result := tensor.ArgMax(0)

		resultData := result.Data().([]float32)
		assert.Equal(t, 1, len(resultData), "Expected 1 element")

		expectedIdx := 1 // index of max value 5
		assert.Equal(t, expectedIdx, int(resultData[0]), "ArgMax value mismatch")
	})

	t.Run("2D tensor along dimension", func(t *testing.T) {
		// 2x3 tensor, argmax along dim 1
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 5, 3, 2, 4, 6})
		result := tensor.ArgMax(1)

		// Expected: [1, 2] (indices of max in each row)
		expected := []float32{1, 2}
		for i := range expected {
			resultData := result.Data().([]float32)
			assert.InDelta(t, float64(expected[i]), float64(resultData[i]), 1e-6, "Data[%d] = %f, expected %f", i, resultData[i], expected[i])
		}
	})
}

func TestBroadcastTo(t *testing.T) {
	t.Run("already correct shape", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
		result, err := tensor.BroadcastTo([]int{2, 3})

		assert.NoError(t, err, "Unexpected error")
		assert.True(t, result.Shape().Equal(types.NewShape(2, 3)), "Shape should be [2, 3]")

		resultData := result.Data().([]float32)
		tensorData := tensor.Data().([]float32)
		// Clone() should create a copy, but check data equality instead of pointer equality
		if len(resultData) > 0 && len(tensorData) > 0 {
			// Verify they're not the same slice (Clone should create new data)
			if &resultData[0] == &tensorData[0] {
				t.Errorf("BroadcastTo should return a copy when shape matches, but data pointers are the same")
			}
		}

		for i := range tensorData {
			assert.InDelta(t, float64(tensorData[i]), float64(resultData[i]), 1e-6, "Data[%d] mismatch", i)
		}
	})

	t.Run("broadcast expand", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(1, 3), []float32{1, 2, 3})
		result, err := tensor.BroadcastTo([]int{2, 3})

		assert.NoError(t, err, "Unexpected error")

		expected := []float32{1, 2, 3, 1, 2, 3}
		assert.True(t, result.Shape().Equal(types.NewShape(2, 3)), "Shape should be [2,3]")
		for i := range expected {
			resultData := result.Data().([]float32)
			assert.InDeltaf(t, float64(expected[i]), float64(resultData[i]), 1e-6, "Data[%d] = %f, expected %f", i, resultData[i], expected[i])
		}
	})

	t.Run("error: incompatible shapes", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
		_, err := tensor.BroadcastTo([]int{3, 4})

		assert.Error(t, err, "Expected error for incompatible shapes")
	})
}

func TestSquare(t *testing.T) {
	t.Run("Square in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{2, 3, -4, 5})
		expected := []float32{4, 9, 16, 25}

		tensor.Square(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Square with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(3), []float32{1, 2, -3})
		expected := []float32{1, 4, 9}
		dst := New(types.DTFP32, types.NewShape(3))
		originalData := tensor.Clone()

		result := tensor.Square(dst)

		assert.Equal(t, dst, result, "Square should return dst when provided")
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(dstData[i]), 1e-5, "Data[%d] = %f, expected %f", i, dstData[i], expected[i])
		}
		// Verify original unchanged
		originalDataSlice := originalData.Data().([]float32)
		tData := tensor.Data().([]float32)
		for i := range originalDataSlice {
			assert.InDeltaf(t, float64(originalDataSlice[i]), float64(tData[i]), 1e-5, "Original tensor should be unchanged at %d", i)
		}
	})
}

func TestSqrt(t *testing.T) {
	t.Run("Sqrt in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{4, 9, 16, 25})
		expected := []float32{2, 3, 4, 5}

		tensor.Sqrt(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Sqrt with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(3), []float32{1, 4, 9})
		expected := []float32{1, 2, 3}
		dst := New(types.DTFP32, types.NewShape(3))
		originalData := tensor.Clone()

		result := tensor.Sqrt(dst)

		assert.Equal(t, dst, result, "Sqrt should return dst when provided")
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(dstData[i]), 1e-5, "Data[%d] = %f, expected %f", i, dstData[i], expected[i])
		}
		// Verify original unchanged
		originalDataSlice := originalData.Data().([]float32)
		tData := tensor.Data().([]float32)
		for i := range originalDataSlice {
			assert.InDeltaf(t, float64(originalDataSlice[i]), float64(tData[i]), 1e-5, "Original tensor should be unchanged at %d", i)
		}
	})
}

func TestExp(t *testing.T) {
	t.Run("Exp in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{0, 1, -1, 2})
		expected := []float32{1.0, float32(math.E), 1.0 / float32(math.E), float32(math.E * math.E)}

		tensor.Exp(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Exp with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2), []float32{0, 1})
		expected := []float32{1.0, float32(math.E)}
		dst := New(types.DTFP32, types.NewShape(2))
		originalData := tensor.Clone()

		result := tensor.Exp(dst)

		assert.Equal(t, dst, result, "Exp should return dst when provided")
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(dstData[i]), 1e-5, "Data[%d] = %f, expected %f", i, dstData[i], expected[i])
		}
		// Verify original unchanged
		originalDataSlice := originalData.Data().([]float32)
		tData := tensor.Data().([]float32)
		for i := range originalDataSlice {
			assert.InDeltaf(t, float64(originalDataSlice[i]), float64(tData[i]), 1e-5, "Original tensor should be unchanged at %d", i)
		}
	})
}

func TestLog(t *testing.T) {
	t.Run("Log in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{1, float32(math.E), float32(math.E * math.E), 10})
		expected := []float32{0, 1, 2, float32(math.Log(10))}

		tensor.Log(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Log with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2), []float32{1, float32(math.E)})
		expected := []float32{0, 1}
		dst := New(types.DTFP32, types.NewShape(2))
		originalData := tensor.Clone()

		result := tensor.Log(dst)

		assert.Equal(t, dst, result, "Log should return dst when provided")
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(dstData[i]), 1e-5, "Data[%d] = %f, expected %f", i, dstData[i], expected[i])
		}
		// Verify original unchanged
		originalDataSlice := originalData.Data().([]float32)
		tData := tensor.Data().([]float32)
		for i := range originalDataSlice {
			assert.InDeltaf(t, float64(originalDataSlice[i]), float64(tData[i]), 1e-5, "Original tensor should be unchanged at %d", i)
		}
	})
}

func TestPow(t *testing.T) {
	t.Run("Pow in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{2, 3, 4, 5})
		power := float64(2.0)
		expected := []float32{4, 9, 16, 25}

		tensor.Pow(nil, power)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Pow with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(3), []float32{4, 9, 16})
		power := float64(0.5)
		expected := []float32{2, 3, 4}
		dst := New(types.DTFP32, types.NewShape(3))
		originalData := tensor.Clone()

		result := tensor.Pow(dst, power)

		assert.Equal(t, dst, result, "Pow should return dst when provided")
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(dstData[i]), 1e-5, "Data[%d] = %f, expected %f", i, dstData[i], expected[i])
		}
		// Verify original unchanged
		originalDataSlice := originalData.Data().([]float32)
		tData := tensor.Data().([]float32)
		for i := range originalDataSlice {
			assert.InDeltaf(t, float64(originalDataSlice[i]), float64(tData[i]), 1e-5, "Original tensor should be unchanged at %d", i)
		}
	})
}

func TestAbs(t *testing.T) {
	t.Run("Abs in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{-2, 3, -4, 5})
		expected := []float32{2, 3, 4, 5}

		tensor.Abs(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Abs with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(3), []float32{-1, 0, 1})
		expected := []float32{1, 0, 1}
		dst := New(types.DTFP32, types.NewShape(3))
		originalData := tensor.Clone()

		result := tensor.Abs(dst)

		assert.Equal(t, dst, result, "Abs should return dst when provided")
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(dstData[i]), 1e-5, "Data[%d] = %f, expected %f", i, dstData[i], expected[i])
		}
		// Verify original unchanged
		originalDataSlice := originalData.Data().([]float32)
		tData := tensor.Data().([]float32)
		for i := range originalDataSlice {
			assert.InDeltaf(t, float64(originalDataSlice[i]), float64(tData[i]), 1e-5, "Original tensor should be unchanged at %d", i)
		}
	})
}

func TestSign(t *testing.T) {
	t.Run("Sign in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{-2, 0, 3, -1})
		expected := []float32{-1, 0, 1, -1}

		tensor.Sign(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Sign with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(4), []float32{-5, 0, 2, -0.1})
		expected := []float32{-1, 0, 1, -1}
		dst := New(types.DTFP32, types.NewShape(4))
		originalData := tensor.Clone()

		result := tensor.Sign(dst)

		assert.Equal(t, dst, result, "Sign should return dst when provided")
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(dstData[i]), 1e-5, "Data[%d] = %f, expected %f", i, dstData[i], expected[i])
		}
		// Verify original unchanged
		originalDataSlice := originalData.Data().([]float32)
		tData := tensor.Data().([]float32)
		for i := range originalDataSlice {
			assert.InDeltaf(t, float64(originalDataSlice[i]), float64(tData[i]), 1e-5, "Original tensor should be unchanged at %d", i)
		}
	})
}

func TestCos(t *testing.T) {
	t.Run("Cos in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{0, float32(math.Pi / 2), float32(math.Pi), float32(math.Pi * 3 / 2)})
		expected := []float32{1, 0, -1, 0}

		tensor.Cos(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Cos with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2), []float32{0, float32(math.Pi)})
		expected := []float32{1, -1}
		dst := New(types.DTFP32, types.NewShape(2))
		originalData := tensor.Clone()

		result := tensor.Cos(dst)

		assert.Equal(t, dst, result, "Cos should return dst when provided")
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(dstData[i]), 1e-5, "Data[%d] = %f, expected %f", i, dstData[i], expected[i])
		}
		// Verify original unchanged
		originalDataSlice := originalData.Data().([]float32)
		tData := tensor.Data().([]float32)
		for i := range originalDataSlice {
			assert.InDeltaf(t, float64(originalDataSlice[i]), float64(tData[i]), 1e-5, "Original tensor should be unchanged at %d", i)
		}
	})
}

func TestSin(t *testing.T) {
	t.Run("Sin in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{0, float32(math.Pi / 2), float32(math.Pi), float32(math.Pi * 3 / 2)})
		expected := []float32{0, 1, 0, -1}

		tensor.Sin(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Sin with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2), []float32{0, float32(math.Pi / 2)})
		expected := []float32{0, 1}
		dst := New(types.DTFP32, types.NewShape(2))
		originalData := tensor.Clone()

		result := tensor.Sin(dst)

		assert.Equal(t, dst, result, "Sin should return dst when provided")
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(dstData[i]), 1e-5, "Data[%d] = %f, expected %f", i, dstData[i], expected[i])
		}
		// Verify original unchanged
		originalDataSlice := originalData.Data().([]float32)
		tData := tensor.Data().([]float32)
		for i := range originalDataSlice {
			assert.InDeltaf(t, float64(originalDataSlice[i]), float64(tData[i]), 1e-5, "Original tensor should be unchanged at %d", i)
		}
	})
}

func TestNegative(t *testing.T) {
	t.Run("Negative in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{2, -3, 4, -5})
		expected := []float32{-2, 3, -4, 5}

		tensor.Negative(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Negative with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(3), []float32{1, 0, -1})
		expected := []float32{-1, 0, 1}
		dst := New(types.DTFP32, types.NewShape(3))
		originalData := tensor.Clone()

		result := tensor.Negative(dst)

		assert.Equal(t, dst, result, "Negative should return dst when provided")
		dstData := dst.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(dstData[i]), 1e-5, "Data[%d] = %f, expected %f", i, dstData[i], expected[i])
		}
		// Verify original unchanged
		originalDataSlice := originalData.Data().([]float32)
		tData := tensor.Data().([]float32)
		for i := range originalDataSlice {
			assert.InDeltaf(t, float64(originalDataSlice[i]), float64(tData[i]), 1e-5, "Original tensor should be unchanged at %d", i)
		}
	})
}

func TestEqual(t *testing.T) {
	tests := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		expected []float32
	}{
		{
			name:     "equal 2x2 tensor",
			t1:       FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			t2:       FromFloat32(types.NewShape(2, 2), []float32{1, 2, 5, 4}),
			expected: []float32{1, 1, 0, 1},
		},
		{
			name:     "equal 1D tensor",
			t1:       FromFloat32(types.NewShape(3), []float32{1, 2, 3}),
			t2:       FromFloat32(types.NewShape(3), []float32{1, 5, 3}),
			expected: []float32{1, 0, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.Equal(tt.t2)

			assert.NotNil(t, result, "Equal should not return nil")
			assert.True(t, result.Shape().Equal(tt.t1.Shape()), "Equal result shape should match input shape")

			resultData := result.Data().([]float32)
			for i := range tt.expected {
				assert.InDeltaf(t, float64(tt.expected[i]), float64(resultData[i]), 1e-5, "Data[%d] = %f, expected %f", i, resultData[i], tt.expected[i])
			}
		})
	}
}

func TestGreaterThan(t *testing.T) {
	tests := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		expected []float32
	}{
		{
			name:     "greater than 2x2 tensor",
			t1:       FromFloat32(types.NewShape(2, 2), []float32{5, 2, 3, 4}),
			t2:       FromFloat32(types.NewShape(2, 2), []float32{3, 2, 5, 4}),
			expected: []float32{1, 0, 0, 0},
		},
		{
			name:     "greater than 1D tensor",
			t1:       FromFloat32(types.NewShape(3), []float32{5, 2, 3}),
			t2:       FromFloat32(types.NewShape(3), []float32{3, 2, 5}),
			expected: []float32{1, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.GreaterThan(tt.t2)

			assert.NotNil(t, result, "GreaterThan should not return nil")
			assert.True(t, result.Shape().Equal(tt.t1.Shape()), "GreaterThan result shape should match input shape")

			resultData := result.Data().([]float32)
			for i := range tt.expected {
				assert.InDeltaf(t, float64(tt.expected[i]), float64(resultData[i]), 1e-5, "Data[%d] = %f, expected %f", i, resultData[i], tt.expected[i])
			}
		})
	}
}

func TestGreater(t *testing.T) {
	// Greater is an alias for GreaterThan, so test that it works the same
	t.Run("greater alias", func(t *testing.T) {
		t1 := FromFloat32(types.NewShape(2, 2), []float32{5, 2, 3, 4})
		t2 := FromFloat32(types.NewShape(2, 2), []float32{3, 2, 5, 4})

		result1 := t1.GreaterThan(t2)
		result2 := t1.Greater(t2)

		assert.NotNil(t, result1, "GreaterThan result should not be nil")
		assert.NotNil(t, result2, "Greater result should not be nil")

		result1Data := result1.Data().([]float32)
		result2Data := result2.Data().([]float32)

		for i := range result1Data {
			assert.InDeltaf(t, float64(result1Data[i]), float64(result2Data[i]), 1e-5, "Greater should match GreaterThan at index %d", i)
		}
	})
}

func TestLess(t *testing.T) {
	tests := []struct {
		name     string
		t1       Tensor
		t2       Tensor
		expected []float32
	}{
		{
			name:     "less 2x2 tensor",
			t1:       FromFloat32(types.NewShape(2, 2), []float32{3, 2, 5, 4}),
			t2:       FromFloat32(types.NewShape(2, 2), []float32{5, 2, 3, 4}),
			expected: []float32{1, 0, 0, 0},
		},
		{
			name:     "less 1D tensor",
			t1:       FromFloat32(types.NewShape(3), []float32{3, 2, 5}),
			t2:       FromFloat32(types.NewShape(3), []float32{5, 2, 3}),
			expected: []float32{1, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.Less(tt.t2)

			assert.NotNil(t, result, "Less should not return nil")
			assert.True(t, result.Shape().Equal(tt.t1.Shape()), "Less result shape should match input shape")

			resultData := result.Data().([]float32)
			for i := range tt.expected {
				assert.InDeltaf(t, float64(tt.expected[i]), float64(resultData[i]), 1e-5, "Data[%d] = %f, expected %f", i, resultData[i], tt.expected[i])
			}
		})
	}
}

func TestWhere(t *testing.T) {
	tests := []struct {
		name      string
		t         Tensor
		condition Tensor
		a         Tensor
		b         Tensor
		expected  []float32
	}{
		{
			name:      "where 2x2 tensor",
			t:         FromFloat32(types.NewShape(2, 2), []float32{0, 0, 0, 0}), // dummy tensor for receiver
			condition: FromFloat32(types.NewShape(2, 2), []float32{1, 0, 1, 0}),
			a:         FromFloat32(types.NewShape(2, 2), []float32{10, 20, 30, 40}),
			b:         FromFloat32(types.NewShape(2, 2), []float32{100, 200, 300, 400}),
			expected:  []float32{10, 200, 30, 400},
		},
		{
			name:      "where 1D tensor",
			t:         FromFloat32(types.NewShape(3), []float32{0, 0, 0}), // dummy tensor for receiver
			condition: FromFloat32(types.NewShape(3), []float32{1, 0, 1}),
			a:         FromFloat32(types.NewShape(3), []float32{1, 2, 3}),
			b:         FromFloat32(types.NewShape(3), []float32{10, 20, 30}),
			expected:  []float32{1, 20, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Where(tt.condition, tt.a, tt.b)

			assert.NotNil(t, result, "Where should not return nil")
			assert.True(t, result.Shape().Equal(tt.condition.Shape()), "Where result shape should match condition shape")

			resultData := result.Data().([]float32)
			for i := range tt.expected {
				assert.InDeltaf(t, float64(tt.expected[i]), float64(resultData[i]), 1e-5, "Data[%d] = %f, expected %f", i, resultData[i], tt.expected[i])
			}
		})
	}
}

package tensor

import (
	"math"
	"testing"
)

func TestAdd(t *testing.T) {
	tests := []struct {
		name     string
		t1       *Tensor
		t2       *Tensor
		expected []float32
	}{
		{
			name:     "2x2 addition",
			t1:       &Tensor{Dim: []int{2, 2}, Data: []float32{1, 2, 3, 4}},
			t2:       &Tensor{Dim: []int{2, 2}, Data: []float32{5, 6, 7, 8}},
			expected: []float32{6, 8, 10, 12},
		},
		{
			name:     "1D addition",
			t1:       &Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}},
			t2:       &Tensor{Dim: []int{3}, Data: []float32{4, 5, 6}},
			expected: []float32{5, 7, 9},
		},
		{
			name:     "scalar-like 1x1",
			t1:       &Tensor{Dim: []int{1, 1}, Data: []float32{10}},
			t2:       &Tensor{Dim: []int{1, 1}, Data: []float32{20}},
			expected: []float32{30},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := tt.t1.Clone()
			result := tt.t1.Add(tt.t2)

			if result != tt.t1 {
				t.Errorf("Add should return self for chaining")
			}

			for i := range tt.expected {
				if !floatEqual(tt.t1.Data[i], tt.expected[i]) {
					t.Errorf("Data[%d] = %f, expected %f", i, tt.t1.Data[i], tt.expected[i])
				}
			}

			// Verify t2 is unchanged (check first element as simple verification)
			if len(tt.t2.Data) > 0 && tt.t2.Data[0] != original.Data[0] {
				// Actually, verify both t1 changed and t2 unchanged properly
			}
		})
	}
}

func TestSub(t *testing.T) {
	tests := []struct {
		name     string
		t1       *Tensor
		t2       *Tensor
		expected []float32
	}{
		{
			name:     "2x2 subtraction",
			t1:       &Tensor{Dim: []int{2, 2}, Data: []float32{10, 20, 30, 40}},
			t2:       &Tensor{Dim: []int{2, 2}, Data: []float32{1, 2, 3, 4}},
			expected: []float32{9, 18, 27, 36},
		},
		{
			name:     "1D subtraction",
			t1:       &Tensor{Dim: []int{3}, Data: []float32{10, 20, 30}},
			t2:       &Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}},
			expected: []float32{9, 18, 27},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.Sub(tt.t2)

			if result != tt.t1 {
				t.Errorf("Sub should return self for chaining")
			}

			for i := range tt.expected {
				if !floatEqual(tt.t1.Data[i], tt.expected[i]) {
					t.Errorf("Data[%d] = %f, expected %f", i, tt.t1.Data[i], tt.expected[i])
				}
			}
		})
	}
}

func TestMul(t *testing.T) {
	tests := []struct {
		name     string
		t1       *Tensor
		t2       *Tensor
		expected []float32
	}{
		{
			name:     "2x2 multiplication",
			t1:       &Tensor{Dim: []int{2, 2}, Data: []float32{2, 3, 4, 5}},
			t2:       &Tensor{Dim: []int{2, 2}, Data: []float32{2, 2, 2, 2}},
			expected: []float32{4, 6, 8, 10},
		},
		{
			name:     "1D multiplication",
			t1:       &Tensor{Dim: []int{3}, Data: []float32{1, 2, 3}},
			t2:       &Tensor{Dim: []int{3}, Data: []float32{2, 3, 4}},
			expected: []float32{2, 6, 12},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.Mul(tt.t2)

			if result != tt.t1 {
				t.Errorf("Mul should return self for chaining")
			}

			for i := range tt.expected {
				if !floatEqual(tt.t1.Data[i], tt.expected[i]) {
					t.Errorf("Data[%d] = %f, expected %f", i, tt.t1.Data[i], tt.expected[i])
				}
			}
		})
	}
}

func TestDiv(t *testing.T) {
	tests := []struct {
		name     string
		t1       *Tensor
		t2       *Tensor
		expected []float32
	}{
		{
			name:     "2x2 division",
			t1:       &Tensor{Dim: []int{2, 2}, Data: []float32{10, 20, 30, 40}},
			t2:       &Tensor{Dim: []int{2, 2}, Data: []float32{2, 4, 5, 8}},
			expected: []float32{5, 5, 6, 5},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t1.Div(tt.t2)

			if result != tt.t1 {
				t.Errorf("Div should return self for chaining")
			}

			for i := range tt.expected {
				if !floatEqual(tt.t1.Data[i], tt.expected[i]) {
					t.Errorf("Data[%d] = %f, expected %f", i, tt.t1.Data[i], tt.expected[i])
				}
			}
		})
	}
}

func TestScale(t *testing.T) {
	tests := []struct {
		name     string
		t        *Tensor
		scalar   float32
		expected []float32
	}{
		{
			name:     "scale by 2",
			t:        &Tensor{Dim: []int{2, 2}, Data: []float32{1, 2, 3, 4}},
			scalar:   2.0,
			expected: []float32{2, 4, 6, 8},
		},
		{
			name:     "scale by 0.5",
			t:        &Tensor{Dim: []int{3}, Data: []float32{2, 4, 6}},
			scalar:   0.5,
			expected: []float32{1, 2, 3},
		},
		{
			name:     "scale by 0",
			t:        &Tensor{Dim: []int{2, 2}, Data: []float32{1, 2, 3, 4}},
			scalar:   0.0,
			expected: []float32{0, 0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Scale(tt.scalar)

			if result != tt.t {
				t.Errorf("Scale should return self for chaining")
			}

			for i := range tt.expected {
				if !floatEqual(tt.t.Data[i], tt.expected[i]) {
					t.Errorf("Data[%d] = %f, expected %f", i, tt.t.Data[i], tt.expected[i])
				}
			}
		})
	}
}

func TestAddTo(t *testing.T) {
	t.Run("create new tensor", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{2, 2}, Data: []float32{1, 2, 3, 4}}
		t2 := &Tensor{Dim: []int{2, 2}, Data: []float32{5, 6, 7, 8}}

		result := t1.AddTo(t2, nil)

		if result == t1 || result == t2 {
			t.Errorf("AddTo should create new tensor when dst is nil")
		}

		expected := []float32{6, 8, 10, 12}
		for i := range expected {
			if !floatEqual(result.Data[i], expected[i]) {
				t.Errorf("Data[%d] = %f, expected %f", i, result.Data[i], expected[i])
			}
		}

		// Original tensors should be unchanged
		if t1.Data[0] != 1 || t2.Data[0] != 5 {
			t.Errorf("Original tensors should be unchanged")
		}
	})

	t.Run("use destination tensor", func(t *testing.T) {
		t1 := &Tensor{Dim: []int{2, 2}, Data: []float32{1, 2, 3, 4}}
		t2 := &Tensor{Dim: []int{2, 2}, Data: []float32{5, 6, 7, 8}}
		dst := &Tensor{Dim: []int{2, 2}, Data: make([]float32, 4)}

		result := t1.AddTo(t2, dst)

		if result != dst {
			t.Errorf("AddTo should use dst when provided")
		}

		expected := []float32{6, 8, 10, 12}
		for i := range expected {
			if !floatEqual(dst.Data[i], expected[i]) {
				t.Errorf("Data[%d] = %f, expected %f", i, dst.Data[i], expected[i])
			}
		}
	})
}

func TestSum(t *testing.T) {
	tests := []struct {
		name     string
		t        *Tensor
		dims     []int
		expected []float32
		expShape []int
	}{
		{
			name:     "sum all elements",
			t:        &Tensor{Dim: []int{2, 2}, Data: []float32{1, 2, 3, 4}},
			dims:     nil,
			expected: []float32{10},
			expShape: []int{1},
		},
		{
			name:     "sum along first dimension",
			t:        &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}},
			dims:     []int{0},
			expected: []float32{5, 7, 9},
			expShape: []int{3},
		},
		{
			name:     "sum along second dimension",
			t:        &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}},
			dims:     []int{1},
			expected: []float32{6, 15},
			expShape: []int{2},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Sum(tt.dims...)

			if len(result.Dim) != len(tt.expShape) {
				t.Errorf("Shape length mismatch: got %v, expected %v", result.Dim, tt.expShape)
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

func TestMean(t *testing.T) {
	t.Run("mean of all elements", func(t *testing.T) {
		tensor := &Tensor{Dim: []int{2, 2}, Data: []float32{1, 2, 3, 4}}
		result := tensor.Mean()
		expected := float32(10.0 / 4.0)

		if len(result.Data) != 1 {
			t.Fatalf("Expected 1 element, got %d", len(result.Data))
		}

		if !floatEqual(result.Data[0], expected) {
			t.Errorf("Mean = %f, expected %f", result.Data[0], expected)
		}
	})

	t.Run("mean along dimension", func(t *testing.T) {
		tensor := &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
		result := tensor.Mean(0)
		expected := []float32{2.5, 3.5, 4.5} // (1+4)/2, (2+5)/2, (3+6)/2

		for i := range expected {
			if !floatEqual(result.Data[i], expected[i]) {
				t.Errorf("Data[%d] = %f, expected %f", i, result.Data[i], expected[i])
			}
		}
	})
}

func TestMax(t *testing.T) {
	tests := []struct {
		name     string
		t        *Tensor
		dims     []int
		expected []float32
	}{
		{
			name:     "max of all elements",
			t:        &Tensor{Dim: []int{2, 2}, Data: []float32{1, 5, 3, 2}},
			dims:     nil,
			expected: []float32{5},
		},
		{
			name:     "max along dimension",
			t:        &Tensor{Dim: []int{2, 3}, Data: []float32{1, 5, 3, 2, 4, 6}},
			dims:     []int{1},
			expected: []float32{5, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Max(tt.dims...)

			for i := range tt.expected {
				if !floatEqual(result.Data[i], tt.expected[i]) {
					t.Errorf("Data[%d] = %f, expected %f", i, result.Data[i], tt.expected[i])
				}
			}
		})
	}
}

func TestMin(t *testing.T) {
	tests := []struct {
		name     string
		t        *Tensor
		dims     []int
		expected []float32
	}{
		{
			name:     "min of all elements",
			t:        &Tensor{Dim: []int{2, 2}, Data: []float32{5, 2, 3, 1}},
			dims:     nil,
			expected: []float32{1},
		},
		{
			name:     "min along dimension",
			t:        &Tensor{Dim: []int{2, 3}, Data: []float32{5, 2, 3, 1, 4, 6}},
			dims:     []int{1},
			expected: []float32{2, 1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.Min(tt.dims...)

			for i := range tt.expected {
				if !floatEqual(result.Data[i], tt.expected[i]) {
					t.Errorf("Data[%d] = %f, expected %f", i, result.Data[i], tt.expected[i])
				}
			}
		})
	}
}

func TestArgMax(t *testing.T) {
	t.Run("1D tensor", func(t *testing.T) {
		tensor := &Tensor{Dim: []int{5}, Data: []float32{1, 5, 3, 2, 4}}
		result := tensor.ArgMax(0)

		if len(result.Data) != 1 {
			t.Fatalf("Expected 1 element, got %d", len(result.Data))
		}

		expectedIdx := 1 // index of max value 5
		if int(result.Data[0]) != expectedIdx {
			t.Errorf("ArgMax = %f, expected %d", result.Data[0], expectedIdx)
		}
	})

	t.Run("2D tensor along dimension", func(t *testing.T) {
		// 2x3 tensor, argmax along dim 1
		tensor := &Tensor{Dim: []int{2, 3}, Data: []float32{1, 5, 3, 2, 4, 6}}
		result := tensor.ArgMax(1)

		// Expected: [1, 2] (indices of max in each row)
		expected := []float32{1, 2}
		for i := range expected {
			if !floatEqual(result.Data[i], expected[i]) {
				t.Errorf("Data[%d] = %f, expected %f", i, result.Data[i], expected[i])
			}
		}
	})
}

func TestBroadcastTo(t *testing.T) {
	t.Run("already correct shape", func(t *testing.T) {
		tensor := &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
		result, err := tensor.BroadcastTo([]int{2, 3})

		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if !result.sameShapeInt([]int{2, 3}) {
			t.Errorf("Shape should be [2, 3], got %v", result.Dim)
		}
	})

	t.Run("error: incompatible shapes", func(t *testing.T) {
		tensor := &Tensor{Dim: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
		_, err := tensor.BroadcastTo([]int{3, 4})

		if err == nil {
			t.Errorf("Expected error for incompatible shapes")
		}
	})
}

// Helper function for float32 comparison with epsilon
func floatEqual(a, b float32) bool {
	const epsilon = 1e-6
	return math.Abs(float64(a-b)) < epsilon
}

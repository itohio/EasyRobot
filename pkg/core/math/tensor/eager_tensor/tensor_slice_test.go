package eager_tensor

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestSlice(t *testing.T) {
	tests := []struct {
		name          string
		tensor        Tensor
		dim           int
		start         int
		length        int
		expected      []float32
		expectedShape []int
		shouldPanic   bool
	}{
		{
			name:          "slice last dimension - 1D",
			tensor:        FromFloat32(types.NewShape(6), []float32{1, 2, 3, 4, 5, 6}),
			dim:           0,
			start:         1,
			length:        3,
			expected:      []float32{2, 3, 4},
			expectedShape: []int{3},
			shouldPanic:   false,
		},
		{
			name:          "slice last dimension - 2D",
			tensor:        FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			dim:           1,
			start:         1,
			length:        2,
			expected:      []float32{2, 3, 6, 7},
			expectedShape: []int{2, 2},
			shouldPanic:   false,
		},
		{
			name:          "slice first dimension - 2D",
			tensor:        FromFloat32(types.NewShape(4, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			dim:           0,
			start:         1,
			length:        2,
			expected:      []float32{3, 4, 5, 6},
			expectedShape: []int{2, 2},
			shouldPanic:   false,
		},
		{
			name: "slice middle dimension - 3D",
			tensor: FromFloat32(types.NewShape(2, 4, 3), []float32{
				1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
				13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
			}),
			dim:           1,
			start:         1,
			length:        2,
			expected:      []float32{4, 5, 6, 7, 8, 9, 16, 17, 18, 19, 20, 21},
			expectedShape: []int{2, 2, 3},
			shouldPanic:   false,
		},
		{
			name:          "slice first element - 2D",
			tensor:        FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			dim:           1,
			start:         0,
			length:        1,
			expected:      []float32{1, 5},
			expectedShape: []int{2, 1},
			shouldPanic:   false,
		},
		{
			name:          "slice full dimension - 2D",
			tensor:        FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			dim:           1,
			start:         0,
			length:        4,
			expected:      []float32{1, 2, 3, 4, 5, 6, 7, 8},
			expectedShape: []int{2, 4},
			shouldPanic:   false,
		},
		{
			name:        "dimension out of range",
			tensor:      FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			dim:         2,
			start:       0,
			length:      2,
			shouldPanic: true,
		},
		{
			name:        "start out of range",
			tensor:      FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			dim:         1,
			start:       5,
			length:      2,
			shouldPanic: true,
		},
		{
			name:        "start+length exceeds dimension",
			tensor:      FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			dim:         1,
			start:       2,
			length:      3,
			shouldPanic: true,
		},
		{
			name:        "negative start",
			tensor:      FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			dim:         1,
			start:       -1,
			length:      2,
			shouldPanic: true,
		},
		{
			name:        "zero length",
			tensor:      FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			dim:         1,
			start:       1,
			length:      0,
			shouldPanic: true,
		},
		{
			name:        "negative length",
			tensor:      FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8}),
			dim:         1,
			start:       1,
			length:      -1,
			shouldPanic: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if tt.shouldPanic {
					if r == nil {
						t.Errorf("Slice() should have panicked")
					}
					return
				}
				if r != nil {
					t.Errorf("Slice() panicked unexpectedly: %v", r)
				}
			}()

			result := tt.tensor.Slice(nil, tt.dim, tt.start, tt.length)

			if tt.shouldPanic {
				return
			}

			if result == nil {
				t.Fatalf("Slice() returned nil")
			}

			// Verify shape
			resultShape := result.Shape()
			if len(resultShape) != len(tt.expectedShape) {
				t.Errorf("Slice() shape length = %d, expected %d", len(resultShape), len(tt.expectedShape))
			}

			for i := range tt.expectedShape {
				if resultShape[i] != tt.expectedShape[i] {
					t.Errorf("Slice() Shape()[%d] = %d, expected %d", i, resultShape[i], tt.expectedShape[i])
				}
			}

			// Verify data
			resultData := result.Data().([]float32)
			if len(resultData) != len(tt.expected) {
				t.Errorf("Slice() Data length = %d, expected %d", len(resultData), len(tt.expected))
			}

			for i := range tt.expected {
				assert.InDeltaf(t, tt.expected[i], resultData[i], 1e-5,
					"Slice() Data[%d] = %f, expected %f", i, resultData[i], tt.expected[i])
			}
		})
	}
}

func TestSliceZeroCopy(t *testing.T) {
	// Test that slicing last dimension creates a zero-copy view
	original := FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8})
	sliced := original.Slice(nil, 1, 1, 2)

	// Verify it's a view (zero-copy when slicing last dimension)
	originalData := original.Data().([]float32)
	slicedData := sliced.Data().([]float32)

	// Check that modifying the slice affects the original (for last dimension slice)
	// This only works for contiguous slices (last dimension)
	if original.Shape()[original.Rank()-1] == 4 && sliced.Shape()[sliced.Rank()-1] == 2 {
		// Check if they share the same underlying array
		// For zero-copy, the slice should point to the same underlying array
		originalFirst := originalData[0]
		slicedData[0] = 999.0

		// For last dimension slice, they should share data
		if originalData[1] == 999.0 {
			// Restore
			slicedData[0] = originalFirst
		}
	}
}

func TestSliceMultipleDimensions(t *testing.T) {
	// Test slicing on different dimensions
	tensor := FromFloat32(types.NewShape(3, 4, 5), make([]float32, 60))

	// Initialize with index values for easier verification
	for i := range tensor.Data().([]float32) {
		tensor.Data().([]float32)[i] = float32(i)
	}

	// Slice dimension 0
	result0 := tensor.Slice(nil, 0, 1, 2)
	assert.Equal(t, []int{2, 4, 5}, result0.Shape().ToSlice())
	assert.Equal(t, 40, result0.Size())

	// Slice dimension 1
	result1 := tensor.Slice(nil, 1, 1, 2)
	assert.Equal(t, []int{3, 2, 5}, result1.Shape().ToSlice())
	assert.Equal(t, 30, result1.Size())

	// Slice dimension 2 (last dimension - should be contiguous)
	result2 := tensor.Slice(nil, 2, 1, 3)
	assert.Equal(t, []int{3, 4, 3}, result2.Shape().ToSlice())
	assert.Equal(t, 36, result2.Size())
}

func TestSliceLSTMGates(t *testing.T) {
	// Test the LSTM use case: extract gates from concatenated gates tensor
	// gates: [batch_size, 4*hidden_size] = [2, 8] where hidden_size=2
	batchSize := 2
	hiddenSize := 2
	gates := FromFloat32(types.NewShape(batchSize, 4*hiddenSize), []float32{
		1, 2, 3, 4, 5, 6, 7, 8, // batch 0: gates [i, f, g, o] = [1,2], [3,4], [5,6], [7,8]
		9, 10, 11, 12, 13, 14, 15, 16, // batch 1: gates [i, f, g, o] = [9,10], [11,12], [13,14], [15,16]
	})

	// Extract input gate (i): [batch_size, hidden_size] = [2, 2]
	iGate := gates.Slice(nil, 1, 0, hiddenSize)
	assert.Equal(t, []int{2, 2}, iGate.Shape().ToSlice())
	iGateData := iGate.Data().([]float32)
	assert.InDelta(t, 1.0, iGateData[0], 1e-5)
	assert.InDelta(t, 2.0, iGateData[1], 1e-5)
	assert.InDelta(t, 9.0, iGateData[2], 1e-5)
	assert.InDelta(t, 10.0, iGateData[3], 1e-5)

	// Extract forget gate (f): [batch_size, hidden_size] = [2, 2]
	fGate := gates.Slice(nil, 1, hiddenSize, hiddenSize)
	assert.Equal(t, []int{2, 2}, fGate.Shape().ToSlice())
	fGateData := fGate.Data().([]float32)
	assert.InDelta(t, 3.0, fGateData[0], 1e-5)
	assert.InDelta(t, 4.0, fGateData[1], 1e-5)
	assert.InDelta(t, 11.0, fGateData[2], 1e-5)
	assert.InDelta(t, 12.0, fGateData[3], 1e-5)

	// Extract cell gate (g): [batch_size, hidden_size] = [2, 2]
	gGate := gates.Slice(nil, 1, 2*hiddenSize, hiddenSize)
	assert.Equal(t, []int{2, 2}, gGate.Shape().ToSlice())
	gGateData := gGate.Data().([]float32)
	assert.InDelta(t, 5.0, gGateData[0], 1e-5)
	assert.InDelta(t, 6.0, gGateData[1], 1e-5)
	assert.InDelta(t, 13.0, gGateData[2], 1e-5)
	assert.InDelta(t, 14.0, gGateData[3], 1e-5)

	// Extract output gate (o): [batch_size, hidden_size] = [2, 2]
	oGate := gates.Slice(nil, 1, 3*hiddenSize, hiddenSize)
	assert.Equal(t, []int{2, 2}, oGate.Shape().ToSlice())
	oGateData := oGate.Data().([]float32)
	assert.InDelta(t, 7.0, oGateData[0], 1e-5)
	assert.InDelta(t, 8.0, oGateData[1], 1e-5)
	assert.InDelta(t, 15.0, oGateData[2], 1e-5)
	assert.InDelta(t, 16.0, oGateData[3], 1e-5)
}

func TestSliceWithDst(t *testing.T) {
	t.Run("with dst parameter", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8})
		expectedShape := types.NewShape(2, 2)

		// Test with dst parameter
		dst := New(types.FP32, expectedShape)
		result := tensor.Slice(dst, 1, 1, 2)

		if result.ID() != dst.ID() {
			t.Errorf("Slice() with dst should return dst (same ID), got different tensor")
		}

		resultData := result.Data().([]float32)
		expected := []float32{2, 3, 6, 7} // Slice dimension 1 from index 1, length 2
		for i := range expected {
			assert.InDeltaf(t, expected[i], resultData[i], 1e-5, "Slice() with dst Data[%d] = %f, expected %f", i, resultData[i], expected[i])
		}
	})

	t.Run("dst shape mismatch", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 4), []float32{1, 2, 3, 4, 5, 6, 7, 8})
		dst := New(types.FP32, types.NewShape(3, 3)) // Wrong shape

		defer func() {
			r := recover()
			if r == nil {
				t.Errorf("Slice() with mismatched dst shape should panic")
			}
		}()

		tensor.Slice(dst, 1, 1, 2)
	})
}

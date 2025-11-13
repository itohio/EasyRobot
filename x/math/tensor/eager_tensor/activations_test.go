package eager_tensor

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestReLU(t *testing.T) {
	t.Run("ReLU in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{-1, 2, -3, 4})
		expected := []float32{0, 2, 0, 4}

		tensor.ReLU(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("ReLU with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(3), []float32{-1, 0, 1})
		expected := []float32{0, 0, 1}
		dst := New(types.FP32, types.NewShape(3))
		originalData := tensor.Clone()

		result := tensor.ReLU(dst)

		assert.Equal(t, dst, result, "ReLU should return dst when provided")
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

func TestSigmoid(t *testing.T) {
	t.Run("Sigmoid in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{0, 1, -1, 2})
		expected := []float32{0.5, 1.0 / (1.0 + float32(math.Exp(-1))), 1.0 / (1.0 + float32(math.Exp(1))), 1.0 / (1.0 + float32(math.Exp(-2)))}

		tensor.Sigmoid(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Sigmoid with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2), []float32{0, -1})
		expected := []float32{0.5, 1.0 / (1.0 + float32(math.Exp(1)))}
		dst := New(types.FP32, types.NewShape(2))
		originalData := tensor.Clone()

		result := tensor.Sigmoid(dst)

		assert.Equal(t, dst, result, "Sigmoid should return dst when provided")
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

func TestTanh(t *testing.T) {
	t.Run("Tanh in-place", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 2), []float32{0, 1, -1, 2})
		expected := []float32{0, float32(math.Tanh(1)), float32(math.Tanh(-1)), float32(math.Tanh(2))}

		tensor.Tanh(nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Tanh with destination", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2), []float32{0, -1})
		expected := []float32{0, float32(math.Tanh(-1))}
		dst := New(types.FP32, types.NewShape(2))
		originalData := tensor.Clone()

		result := tensor.Tanh(dst)

		assert.Equal(t, dst, result, "Tanh should return dst when provided")
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

func TestSoftmax(t *testing.T) {
	t.Run("Softmax 1D", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(3), []float32{1, 2, 3})
		exp1 := math.Exp(1)
		exp2 := math.Exp(2)
		exp3 := math.Exp(3)
		sum := exp1 + exp2 + exp3
		expected := []float32{float32(exp1 / sum), float32(exp2 / sum), float32(exp3 / sum)}

		tensor.Softmax(0, nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
		// Verify sum equals 1
		sumResult := float32(0)
		for i := range tData {
			sumResult += tData[i]
		}
		assert.InDelta(t, 1.0, float64(sumResult), 1e-5, "Softmax should sum to 1")
	})

	t.Run("Softmax 2D along dim 0 (rows)", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
		exp1, exp4 := math.Exp(1), math.Exp(4)
		exp2, exp5 := math.Exp(2), math.Exp(5)
		exp3, exp6 := math.Exp(3), math.Exp(6)
		expected := []float32{
			float32(exp1 / (exp1 + exp4)),
			float32(exp2 / (exp2 + exp5)),
			float32(exp3 / (exp3 + exp6)),
			float32(exp4 / (exp1 + exp4)),
			float32(exp5 / (exp2 + exp5)),
			float32(exp6 / (exp3 + exp6)),
		}

		tensor.Softmax(0, nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})

	t.Run("Softmax 2D along dim 1 (columns)", func(t *testing.T) {
		tensor := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
		exp1, exp2, exp3 := math.Exp(1), math.Exp(2), math.Exp(3)
		exp4, exp5, exp6 := math.Exp(4), math.Exp(5), math.Exp(6)
		sumRow1 := exp1 + exp2 + exp3
		sumRow2 := exp4 + exp5 + exp6
		expected := []float32{
			float32(exp1 / sumRow1),
			float32(exp2 / sumRow1),
			float32(exp3 / sumRow1),
			float32(exp4 / sumRow2),
			float32(exp5 / sumRow2),
			float32(exp6 / sumRow2),
		}

		tensor.Softmax(1, nil)

		tData := tensor.Data().([]float32)
		for i := range expected {
			assert.InDeltaf(t, float64(expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], expected[i])
		}
	})
}

func TestDropoutForward(t *testing.T) {
	tests := []struct {
		name     string
		t        Tensor
		mask     Tensor
		expected []float32
	}{
		{
			name:     "DropoutForward with mask",
			t:        FromFloat32(types.NewShape(2, 2), []float32{1, 2, 3, 4}),
			mask:     FromFloat32(types.NewShape(2, 2), []float32{2.0, 0.0, 2.0, 1.0}),
			expected: []float32{2, 0, 6, 4},
		},
		{
			name:     "DropoutForward 1D",
			t:        FromFloat32(types.NewShape(3), []float32{1, 2, 3}),
			mask:     FromFloat32(types.NewShape(3), []float32{1.0, 0.0, 2.0}),
			expected: []float32{1, 0, 6},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.t.DropoutForward(nil, tt.mask)

			// Verify result is the same tensor for chaining
			assert.NotNil(t, result, "DropoutForward should return non-nil result")
			assert.True(t, result.Shape().Equal(tt.t.Shape()), "DropoutForward should return tensor with same shape")

			tData := tt.t.Data().([]float32)
			for i := range tt.expected {
				assert.InDeltaf(t, float64(tt.expected[i]), float64(tData[i]), 1e-5, "Data[%d] = %f, expected %f", i, tData[i], tt.expected[i])
			}
		})
	}
}

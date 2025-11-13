package eager_tensor

import (
	"testing"

	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestBatchNormForward(t *testing.T) {
	t.Run("BatchNormForward with gamma and beta", func(t *testing.T) {
		// Create input tensor [batch=2, features=3]
		input := FromFloat32(types.NewShape(2, 3), []float32{
			1.0, 2.0, 3.0, // batch 0
			4.0, 5.0, 6.0, // batch 1
		})
		gamma := FromFloat32(types.NewShape(3), []float32{1.0, 2.0, 0.5})
		beta := FromFloat32(types.NewShape(3), []float32{0.1, 0.2, 0.3})
		dst := New(types.FP32, types.NewShape(2, 3))
		eps := 1e-5

		result := input.BatchNormForward(dst, gamma, beta, eps)

		assert.Equal(t, dst, result, "BatchNormForward should return dst when provided")
		assert.NotNil(t, result, "Result should not be nil")

		// Verify result shape
		assert.Equal(t, types.NewShape(2, 3), result.Shape())

		// The result should be normalized
		resultData := result.Data().([]float32)
		assert.Equal(t, 6, len(resultData))
		// We can't easily test exact values without computing expected results,
		// but we can test that the operation completed without error
		for _, v := range resultData {
			assert.True(t, v != 0, "Normalized values should not be zero")
		}
	})

	t.Run("BatchNormForward without gamma/beta", func(t *testing.T) {
		input := FromFloat32(types.NewShape(2, 2), []float32{
			1.0, 2.0,
			3.0, 4.0,
		})
		dst := New(types.FP32, types.NewShape(2, 2))

		result := input.BatchNormForward(dst, nil, nil, 1e-5)

		assert.Equal(t, dst, result)
		assert.Equal(t, types.NewShape(2, 2), result.Shape())
	})

	t.Run("BatchNormForward invalid shape", func(t *testing.T) {
		input := FromFloat32(types.NewShape(5), []float32{1, 2, 3, 4, 5})
		dst := New(types.FP32, types.NewShape(5))

		result := input.BatchNormForward(dst, nil, nil, 1e-5)

		// Should copy input when shape is invalid
		inputData := input.Data().([]float32)
		resultData := result.Data().([]float32)
		for i := range inputData {
			assert.Equal(t, inputData[i], resultData[i])
		}
	})
}

func TestLayerNormForward(t *testing.T) {
	t.Run("LayerNormForward with gamma and beta", func(t *testing.T) {
		// Create input tensor [batch=2, seq=3, features=4]
		input := FromFloat32(types.NewShape(2, 3, 4), []float32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, // batch 0
			13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, // batch 1
		})
		gamma := FromFloat32(types.NewShape(4), []float32{1.0, 1.5, 2.0, 0.5})
		beta := FromFloat32(types.NewShape(4), []float32{0.1, 0.2, 0.3, 0.4})
		dst := New(types.FP32, types.NewShape(2, 3, 4))

		result := input.LayerNormForward(dst, gamma, beta, 1e-5)

		assert.Equal(t, dst, result)
		assert.Equal(t, types.NewShape(2, 3, 4), result.Shape())

		resultData := result.Data().([]float32)
		assert.Equal(t, 24, len(resultData))
	})

	t.Run("LayerNormForward without gamma/beta", func(t *testing.T) {
		input := FromFloat32(types.NewShape(2, 3), []float32{
			1, 2, 3,
			4, 5, 6,
		})
		dst := New(types.FP32, types.NewShape(2, 3))

		result := input.LayerNormForward(dst, nil, nil, 1e-5)

		assert.Equal(t, dst, result)
		assert.Equal(t, types.NewShape(2, 3), result.Shape())
	})
}

func TestRMSNormForward(t *testing.T) {
	t.Run("RMSNormForward with gamma", func(t *testing.T) {
		input := FromFloat32(types.NewShape(2, 3), []float32{
			1, 2, 3,
			4, 5, 6,
		})
		gamma := FromFloat32(types.NewShape(3), []float32{1.0, 2.0, 0.5})
		dst := New(types.FP32, types.NewShape(2, 3))

		result := input.RMSNormForward(dst, gamma, 1e-5)

		assert.Equal(t, dst, result)
		assert.Equal(t, types.NewShape(2, 3), result.Shape())
	})

	t.Run("RMSNormForward without gamma", func(t *testing.T) {
		input := FromFloat32(types.NewShape(2, 2), []float32{
			1, 2,
			3, 4,
		})
		dst := New(types.FP32, types.NewShape(2, 2))

		result := input.RMSNormForward(dst, nil, 1e-5)

		assert.Equal(t, dst, result)
		assert.Equal(t, types.NewShape(2, 2), result.Shape())
	})
}

func TestInstanceNorm2D(t *testing.T) {
	t.Run("InstanceNorm2D basic", func(t *testing.T) {
		// [batch=1, channels=2, height=2, width=2]
		input := FromFloat32(types.NewShape(1, 2, 2, 2), []float32{
			1, 2, 3, 4, // channel 0
			5, 6, 7, 8, // channel 1
		})
		gamma := FromFloat32(types.NewShape(2), []float32{1.0, 2.0})
		beta := FromFloat32(types.NewShape(2), []float32{0.1, 0.2})
		dst := New(types.FP32, types.NewShape(1, 2, 2, 2))

		result := input.InstanceNorm2D(dst, gamma, beta, 1e-5)

		assert.Equal(t, dst, result)
		assert.Equal(t, types.NewShape(1, 2, 2, 2), result.Shape())
	})

	t.Run("InstanceNorm2D invalid shape", func(t *testing.T) {
		input := FromFloat32(types.NewShape(2, 3), []float32{1, 2, 3, 4, 5, 6})
		dst := New(types.FP32, types.NewShape(2, 3))

		assert.Panics(t, func() {
			input.InstanceNorm2D(dst, nil, nil, 1e-5)
		})
	})
}

func TestGroupNormForward(t *testing.T) {
	t.Run("GroupNormForward basic", func(t *testing.T) {
		// [batch=1, channels=4, height=2, width=2]
		input := FromFloat32(types.NewShape(1, 4, 2, 2), []float32{
			1, 2, 3, 4, 5, 6, 7, 8, // channel 0,1
			9, 10, 11, 12, 13, 14, 15, 16, // channel 2,3
		})
		gamma := FromFloat32(types.NewShape(4), []float32{1.0, 2.0, 0.5, 1.5})
		beta := FromFloat32(types.NewShape(4), []float32{0.1, 0.2, 0.3, 0.4})
		dst := New(types.FP32, types.NewShape(1, 4, 2, 2))

		result := input.GroupNormForward(dst, gamma, beta, 2, 1e-5) // 2 groups

		assert.Equal(t, dst, result)
		assert.Equal(t, types.NewShape(1, 4, 2, 2), result.Shape())
	})

	t.Run("GroupNormForward invalid groups", func(t *testing.T) {
		input := FromFloat32(types.NewShape(1, 3, 2, 2), []float32{
			1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
		})
		dst := New(types.FP32, types.NewShape(1, 3, 2, 2))

		result := input.GroupNormForward(dst, nil, nil, 2, 1e-5) // 3 channels, 2 groups - invalid

		// Should copy input when groups don't divide channels
		inputData := input.Data().([]float32)
		resultData := result.Data().([]float32)
		for i := range inputData {
			assert.Equal(t, inputData[i], resultData[i])
		}
	})
}

// Benchmarks for normalization operations
func BenchmarkBatchNormForward(b *testing.B) {
	input := FromFloat32(types.NewShape(32, 128), make([]float32, 32*128))
	gamma := FromFloat32(types.NewShape(128), make([]float32, 128))
	beta := FromFloat32(types.NewShape(128), make([]float32, 128))
	dst := New(types.FP32, types.NewShape(32, 128))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		input.BatchNormForward(dst, gamma, beta, 1e-5)
	}
}

func BenchmarkLayerNormForward(b *testing.B) {
	input := FromFloat32(types.NewShape(32, 64, 128), make([]float32, 32*64*128))
	gamma := FromFloat32(types.NewShape(128), make([]float32, 128))
	beta := FromFloat32(types.NewShape(128), make([]float32, 128))
	dst := New(types.FP32, types.NewShape(32, 64, 128))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		input.LayerNormForward(dst, gamma, beta, 1e-5)
	}
}

func BenchmarkRMSNormForward(b *testing.B) {
	input := FromFloat32(types.NewShape(32, 64, 128), make([]float32, 32*64*128))
	gamma := FromFloat32(types.NewShape(128), make([]float32, 128))
	dst := New(types.FP32, types.NewShape(32, 64, 128))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		input.RMSNormForward(dst, gamma, 1e-5)
	}
}

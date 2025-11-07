package eager_tensor

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestConv2D(t *testing.T) {
	t.Run("basic 2D convolution", func(t *testing.T) {
		// Input: [1, 1, 4, 4] (batch, channels, height, width)
		input := FromFloat32(types.NewShape(1, 1, 4, 4), []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		})

		// Kernel: [1, 1, 3, 3] (outChannels, inChannels, kernelH, kernelW)
		kernel := FromFloat32(types.NewShape(1, 1, 3, 3), []float32{
			1, 0, -1,
			1, 0, -1,
			1, 0, -1,
		})

		// Bias: [1]
		bias := FromFloat32(types.NewShape(1), []float32{0})

		result := input.Conv2D(nil, kernel, bias, []int{1, 1}, []int{0, 0})

		expectedShape := []int{1, 1, 2, 2} // outHeight = (4+0-3)/1+1 = 2, outWidth = 2
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")

		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})

	t.Run("with padding and stride", func(t *testing.T) {
		// Input: [1, 1, 3, 3]
		input := FromFloat32(types.NewShape(1, 1, 3, 3), []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		})

		// Kernel: [1, 1, 2, 2]
		kernel := FromFloat32(types.NewShape(1, 1, 2, 2), []float32{
			1, 1,
			1, 1,
		})

		result := input.Conv2D(nil, kernel, Empty(kernel.DataType()), []int{1, 1}, []int{1, 1})

		// With padding=1, outHeight = (3+2*1-2)/1+1 = 4
		expectedShape := []int{1, 1, 4, 4}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})

	t.Run("with multiple channels", func(t *testing.T) {
		// Input: [1, 2, 3, 3] (batch, inChannels, height, width)
		input := New(types.FP32, types.NewShape(1, 2, 3, 3))
		inputData := input.Data().([]float32)
		for i := range inputData {
			inputData[i] = float32(i + 1)
		}

		// Kernel: [1, 2, 2, 2] (outChannels, inChannels, kernelH, kernelW)
		kernel := FromFloat32(types.NewShape(1, 2, 2, 2), []float32{
			1, 1, 1, 1, // channel 0, filter 0
			1, 1, 1, 1, // channel 1, filter 0
		})

		result := input.Conv2D(nil, kernel, Empty(kernel.DataType()), []int{1, 1}, []int{0, 0})

		// outHeight = (3+0-2)/1+1 = 2, outWidth = 2
		expectedShape := []int{1, 1, 2, 2}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})
}

func TestConv2DTransposed(t *testing.T) {
	t.Run("basic transposed convolution", func(t *testing.T) {
		// Input: [1, 1, 2, 2]
		input := FromFloat32(types.NewShape(1, 1, 2, 2), []float32{
			1, 2,
			3, 4,
		})

		// Kernel: [1, 1, 2, 2] (inChannels, outChannels, kernelH, kernelW)
		kernel := FromFloat32(types.NewShape(1, 1, 2, 2), []float32{
			1, 1,
			1, 1,
		})

		result := input.Conv2DTransposed(nil, kernel, Empty(kernel.DataType()), []int{1, 1}, []int{0, 0})

		// outHeight = (2-1)*1 - 2*0 + 2 = 3
		expectedShape := []int{1, 1, 3, 3}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})
}

func TestConv1D(t *testing.T) {
	t.Run("basic 1D convolution", func(t *testing.T) {
		// Input: [2, 5] (inChannels, length)
		input := FromFloat32(types.NewShape(2, 5), []float32{
			1, 2, 3, 4, 5, // channel 0
			6, 7, 8, 9, 10, // channel 1
		})

		// Kernel: [1, 2, 3] (outChannels, inChannels, kernelLen)
		kernel := FromFloat32(types.NewShape(1, 2, 3), []float32{
			1, 1, 1, // channel 0, filter 0
			1, 1, 1, // channel 1, filter 0
		})

		result := input.Conv1D(nil, kernel, Empty(kernel.DataType()), 1, 0)

		// outLen = (5+0-3)/1+1 = 3
		expectedShape := []int{1, 3} // [outChannels, outLen]
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})

	t.Run("with batch", func(t *testing.T) {
		// Input: [2, 2, 5] (batch, inChannels, length)
		input := New(types.FP32, types.NewShape(2, 2, 5))
		inputData := input.Data().([]float32)
		for i := range inputData {
			inputData[i] = float32(i + 1)
		}

		kernel := FromFloat32(types.NewShape(1, 2, 3), []float32{1, 1, 1, 1, 1, 1})

		result := input.Conv1D(nil, kernel, Empty(kernel.DataType()), 1, 0)

		expectedShape := []int{2, 1, 3} // [batch, outChannels, outLen]
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})
}

func TestMaxPool2D(t *testing.T) {
	t.Run("basic max pooling", func(t *testing.T) {
		input := FromFloat32(types.NewShape(1, 1, 4, 4), []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		})

		result := input.MaxPool2D(nil, []int{2, 2}, []int{2, 2}, []int{0, 0})

		// outHeight = (4+0-2)/2+1 = 2, outWidth = 2
		expectedShape := []int{1, 1, 2, 2}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}

		// Check that max pooling works (first window should be max of [1,2,5,6] = 6)
		resultData := result.Data().([]float32)
		if resultData[0] != 6.0 {
			t.Errorf("MaxPool[0] = %f, expected 6.0", resultData[0])
		}
	})

	t.Run("with stride and padding", func(t *testing.T) {
		input := FromFloat32(types.NewShape(1, 1, 3, 3), []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		})

		result := input.MaxPool2D(nil, []int{2, 2}, []int{1, 1}, []int{1, 1})

		// outHeight = (3+2*1-2)/1+1 = 4, outWidth = 4
		expectedShape := []int{1, 1, 4, 4}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})
}

func TestAvgPool2D(t *testing.T) {
	t.Run("basic average pooling", func(t *testing.T) {
		input := FromFloat32(types.NewShape(1, 1, 4, 4), []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		})

		result := input.AvgPool2D(nil, []int{2, 2}, []int{2, 2}, []int{0, 0})

		// outHeight = (4+0-2)/2+1 = 2, outWidth = 2
		expectedShape := []int{1, 1, 2, 2}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}

		// Check that average pooling works (first window: avg of [1,2,5,6] = 3.5)
		expected := float32(3.5)
		resultData := result.Data().([]float32)
		assert.InDelta(t, expected, resultData[0], 1e-5, "AvgPool[0] = %f, expected %f", resultData[0], expected)
	})
}

func TestGlobalAvgPool2D(t *testing.T) {
	t.Run("global average pooling", func(t *testing.T) {
		input := FromFloat32(types.NewShape(1, 2, 2, 2), []float32{ // [batch, channels, height, width]
			1, 2, // channel 0, row 0
			3, 4, // channel 0, row 1
			5, 6, // channel 1, row 0
			7, 8, // channel 1, row 1
		})

		result := input.GlobalAvgPool2D(nil)

		// Output: [batch, channels] = [1, 2]
		expectedShape := []int{1, 2}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}

		resultData := result.Data().([]float32)
		// Channel 0: avg of [1,2,3,4] = 2.5
		expected0 := float32(2.5)
		assert.InDelta(t, expected0, resultData[0], 1e-5, "GlobalAvgPool[0] = %f, expected %f", resultData[0], expected0)

		// Channel 1: avg of [5,6,7,8] = 6.5
		expected1 := float32(6.5)
		assert.InDelta(t, expected1, resultData[1], 1e-5, "GlobalAvgPool[1] = %f, expected %f", resultData[1], expected1)
	})
}

func TestIm2Col(t *testing.T) {
	t.Run("basic Im2Col", func(t *testing.T) {
		// Input: [1, 1, 3, 3]
		input := FromFloat32(types.NewShape(1, 1, 3, 3), []float32{
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
		})

		result := input.Im2Col(nil, []int{2, 2}, []int{1, 1}, []int{0, 0})

		// Output: [batch*outHeight*outWidth, channels*kernelH*kernelW]
		// outHeight = (3+0-2)/1+1 = 2, outWidth = 2
		// colHeight = 1*2*2 = 4, colWidth = 1*2*2 = 4
		expectedShape := []int{4, 4}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})
}

func TestCol2Im(t *testing.T) {
	t.Run("basic Col2Im", func(t *testing.T) {
		// Input columns: [4, 4] (from 1 batch, 2x2 output, 1 channel, 2x2 kernel)
		col := New(types.FP32, types.NewShape(4, 4))
		colData := col.Data().([]float32)
		for i := range colData {
			colData[i] = float32(i + 1)
		}

		// Output shape: [1, 1, 3, 3]
		result := col.Col2Im(nil, []int{1, 1, 3, 3}, []int{2, 2}, []int{1, 1}, []int{0, 0})

		expectedShape := []int{1, 1, 3, 3}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})
}

func TestDepthwiseConv2D(t *testing.T) {
	t.Run("basic depthwise convolution", func(t *testing.T) {
		// Input: [1, 2, 3, 3] - batch=1, channels=2, height=3, width=3
		input := FromFloat32(types.NewShape(1, 2, 3, 3), []float32{
			// Channel 0
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			// Channel 1
			10, 11, 12,
			13, 14, 15,
			16, 17, 18,
		})

		// Kernel: [2, 1, 2, 2] - channels=2, 1, kernelH=2, kernelW=2
		kernel := FromFloat32(types.NewShape(2, 1, 2, 2), []float32{
			// Channel 0 kernel
			1, 1,
			1, 1,
			// Channel 1 kernel
			2, 2,
			2, 2,
		})

		bias := FromFloat32(types.NewShape(2), []float32{0.5, 1.0})

		result := input.DepthwiseConv2D(kernel, bias, []int{1, 1}, []int{0, 0})

		// Output: [1, 2, 2, 2] - same channels, reduced spatial size
		expectedShape := []int{1, 2, 2, 2}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}

		// Verify output size
		expectedSize := 1 * 2 * 2 * 2
		resultData := result.Data().([]float32)
		if len(resultData) != expectedSize {
			t.Errorf("Data length = %d, expected %d", len(resultData), expectedSize)
		}
	})

	t.Run("depthwise with 3D kernel", func(t *testing.T) {
		input := FromFloat32(types.NewShape(1, 2, 2, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8})

		// Kernel: [2, 2, 2] - channels=2, kernelH=2, kernelW=2
		kernel := FromFloat32(types.NewShape(2, 2, 2), []float32{
			1, 1, // Channel 0 kernel
			1, 1,
			2, 2, // Channel 1 kernel
			2, 2,
		})

		result := input.DepthwiseConv2D(kernel, Empty(kernel.DataType()), []int{1, 1}, []int{0, 0})
		expectedShape := []int{1, 2, 1, 1}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})
}

func TestGroupConv2D(t *testing.T) {
	t.Run("grouped convolution with 2 groups", func(t *testing.T) {
		// Input: [1, 4, 2, 2] - batch=1, channels=4, height=2, width=2
		input := New(types.FP32, types.NewShape(1, 4, 2, 2))
		inputData := input.Data().([]float32)
		for i := range inputData {
			inputData[i] = float32(i + 1)
		}

		// Kernel: [4, 2, 1, 1] - outChannels=4, inChannels/groups=2, kernelH=1, kernelW=1
		// With groups=2, we have 2 groups, each processing 2 input channels to produce 2 output channels
		kernel := New(types.FP32, types.NewShape(4, 2, 1, 1))
		kernelData := kernel.Data().([]float32)
		for i := range kernelData {
			kernelData[i] = float32(i + 1)
		}

		result := input.GroupConv2D(kernel, Empty(kernel.DataType()), []int{1, 1}, []int{0, 0}, 2)

		// Output: [1, 4, 2, 2] - same spatial size
		expectedShape := []int{1, 4, 2, 2}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})
}

func TestDilatedConv2D(t *testing.T) {
	t.Run("dilated convolution", func(t *testing.T) {
		// Input: [1, 1, 5, 5]
		input := New(types.FP32, types.NewShape(1, 1, 5, 5))
		inputData := input.Data().([]float32)
		for i := range inputData {
			inputData[i] = float32(i + 1)
		}

		// Kernel: [1, 1, 3, 3]
		kernel := FromFloat32(types.NewShape(1, 1, 3, 3), []float32{1, 0, 0, 0, 1, 0, 0, 0, 1})

		// Dilation: [2, 2] - effective kernel size becomes (3-1)*2+1 = 5
		result := input.DilatedConv2D(kernel, Empty(kernel.DataType()), []int{1, 1}, []int{0, 0}, []int{2, 2})

		// With dilation=2 and kernel=3x3, effective kernel is 5x5
		// Input 5x5 with effective kernel 5x5 and stride 1, padding 0 gives 1x1 output
		expectedShape := []int{1, 1, 1, 1}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})
}

func TestConv3D(t *testing.T) {
	t.Run("basic 3D convolution", func(t *testing.T) {
		// Input: [1, 1, 3, 3, 3] - batch=1, channels=1, depth=3, height=3, width=3
		input := New(types.FP32, types.NewShape(1, 1, 3, 3, 3))
		inputData := input.Data().([]float32)
		for i := range inputData {
			inputData[i] = float32(i + 1)
		}

		// Kernel: [1, 1, 2, 2, 2] - outChannels=1, inChannels=1, kernelD=2, kernelH=2, kernelW=2
		kernel := FromFloat32(types.NewShape(1, 1, 2, 2, 2), []float32{1, 1, 1, 1, 1, 1, 1, 1})

		bias := FromFloat32(types.NewShape(1), []float32{0.0})

		result := input.Conv3D(kernel, bias, []int{1, 1, 1}, []int{0, 0, 0})

		// Output: [1, 1, 2, 2, 2] - depth reduced by 1, spatial dimensions reduced by 1
		expectedShape := []int{1, 1, 2, 2, 2}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}

		// Verify output size
		expectedSize := 1 * 1 * 2 * 2 * 2
		resultData := result.Data().([]float32)
		if len(resultData) != expectedSize {
			t.Errorf("Data length = %d, expected %d", len(resultData), expectedSize)
		}
	})
}

func TestAdaptiveAvgPool2D(t *testing.T) {
	t.Run("adaptive average pooling to 2x2", func(t *testing.T) {
		// Input: [1, 1, 4, 4]
		input := FromFloat32(types.NewShape(1, 1, 4, 4), []float32{
			1, 2, 3, 4,
			5, 6, 7, 8,
			9, 10, 11, 12,
			13, 14, 15, 16,
		})

		// Output size: [2, 2]
		result := input.AdaptiveAvgPool2D(nil, []int{2, 2})

		// Output: [1, 1, 2, 2]
		expectedShape := []int{1, 1, 2, 2}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}

		// Verify output values
		// Top-left region: avg of [1,2,5,6] = 3.5
		// Top-right region: avg of [3,4,7,8] = 5.5
		// Bottom-left region: avg of [9,10,13,14] = 11.5
		// Bottom-right region: avg of [11,12,15,16] = 13.5
		expected := []float32{3.5, 5.5, 11.5, 13.5}
		resultData := result.Data().([]float32)
		for i := range expected {
			assert.InDelta(t, expected[i], resultData[i], 1e-5, "Data[%d] = %f, expected %f", i, resultData[i], expected[i])
		}
	})

	t.Run("adaptive average pooling to 1x1", func(t *testing.T) {
		input := FromFloat32(types.NewShape(1, 2, 3, 3), []float32{
			// Channel 0
			1, 2, 3,
			4, 5, 6,
			7, 8, 9,
			// Channel 1
			10, 20, 30,
			40, 50, 60,
			70, 80, 90,
		})

		result := input.AdaptiveAvgPool2D(nil, []int{1, 1})

		// Output: [1, 2, 1, 1]
		expectedShape := []int{1, 2, 1, 1}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}

		// Channel 0: avg of all 9 values = 5.0
		// Channel 1: avg of all 9 values = 50.0
		expected := []float32{5.0, 50.0}
		resultData := result.Data().([]float32)
		for i := range expected {
			assert.InDelta(t, expected[i], resultData[i], 1e-5, "Data[%d] = %f, expected %f", i, resultData[i], expected[i])
		}
	})

	t.Run("adaptive average pooling to larger size", func(t *testing.T) {
		input := FromFloat32(types.NewShape(1, 1, 2, 2), []float32{1, 2, 3, 4})

		result := input.AdaptiveAvgPool2D(nil, []int{4, 4})

		// Output: [1, 1, 4, 4]
		expectedShape := []int{1, 1, 4, 4}
		resultShape := result.Shape()
		assert.Equal(t, len(expectedShape), len(resultShape), "Shape length mismatch")
		for i := range expectedShape {
			assert.Equal(t, expectedShape[i], resultShape[i], "Shape dimension %d mismatch", i)
		}
	})
}

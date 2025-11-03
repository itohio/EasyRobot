package tensor

import (
	"testing"
)

func TestConv2D(t *testing.T) {
	t.Run("basic 2D convolution", func(t *testing.T) {
		// Input: [1, 1, 4, 4] (batch, channels, height, width)
		input := &Tensor{
			Dim: []int{1, 1, 4, 4},
			Data: []float32{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
				13, 14, 15, 16,
			},
		}

		// Kernel: [1, 1, 3, 3] (outChannels, inChannels, kernelH, kernelW)
		kernel := &Tensor{
			Dim: []int{1, 1, 3, 3},
			Data: []float32{
				1, 0, -1,
				1, 0, -1,
				1, 0, -1,
			},
		}

		// Bias: [1]
		bias := &Tensor{
			Dim:  []int{1},
			Data: []float32{0},
		}

		result := input.Conv2D(kernel, bias, []int{1, 1}, []int{0, 0})

		expectedShape := []int{1, 1, 2, 2} // outHeight = (4+0-3)/1+1 = 2, outWidth = 2
		if len(result.Dim) != len(expectedShape) {
			t.Fatalf("Shape mismatch: got %v, expected %v", result.Dim, expectedShape)
		}

		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})

	t.Run("with padding and stride", func(t *testing.T) {
		// Input: [1, 1, 3, 3]
		input := &Tensor{
			Dim: []int{1, 1, 3, 3},
			Data: []float32{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			},
		}

		// Kernel: [1, 1, 2, 2]
		kernel := &Tensor{
			Dim: []int{1, 1, 2, 2},
			Data: []float32{
				1, 1,
				1, 1,
			},
		}

		result := input.Conv2D(kernel, nil, []int{1, 1}, []int{1, 1})

		// With padding=1, outHeight = (3+2*1-2)/1+1 = 4
		expectedShape := []int{1, 1, 4, 4}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})

	t.Run("with multiple channels", func(t *testing.T) {
		// Input: [1, 2, 3, 3] (batch, inChannels, height, width)
		input := &Tensor{
			Dim:  []int{1, 2, 3, 3},
			Data: make([]float32, 1*2*3*3),
		}
		for i := range input.Data {
			input.Data[i] = float32(i + 1)
		}

		// Kernel: [1, 2, 2, 2] (outChannels, inChannels, kernelH, kernelW)
		kernel := &Tensor{
			Dim: []int{1, 2, 2, 2},
			Data: []float32{
				1, 1, 1, 1, // channel 0, filter 0
				1, 1, 1, 1, // channel 1, filter 0
			},
		}

		result := input.Conv2D(kernel, nil, []int{1, 1}, []int{0, 0})

		// outHeight = (3+0-2)/1+1 = 2, outWidth = 2
		expectedShape := []int{1, 1, 2, 2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

func TestConv2DTransposed(t *testing.T) {
	t.Run("basic transposed convolution", func(t *testing.T) {
		// Input: [1, 1, 2, 2]
		input := &Tensor{
			Dim: []int{1, 1, 2, 2},
			Data: []float32{
				1, 2,
				3, 4,
			},
		}

		// Kernel: [1, 1, 2, 2] (inChannels, outChannels, kernelH, kernelW)
		kernel := &Tensor{
			Dim: []int{1, 1, 2, 2},
			Data: []float32{
				1, 1,
				1, 1,
			},
		}

		result := input.Conv2DTransposed(kernel, nil, []int{1, 1}, []int{0, 0})

		// outHeight = (2-1)*1 - 2*0 + 2 = 3
		expectedShape := []int{1, 1, 3, 3}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

func TestConv1D(t *testing.T) {
	t.Run("basic 1D convolution", func(t *testing.T) {
		// Input: [2, 5] (inChannels, length)
		input := &Tensor{
			Dim: []int{2, 5},
			Data: []float32{
				1, 2, 3, 4, 5, // channel 0
				6, 7, 8, 9, 10, // channel 1
			},
		}

		// Kernel: [1, 2, 3] (outChannels, inChannels, kernelLen)
		kernel := &Tensor{
			Dim: []int{1, 2, 3},
			Data: []float32{
				1, 1, 1, // channel 0, filter 0
				1, 1, 1, // channel 1, filter 0
			},
		}

		result := input.Conv1D(kernel, nil, 1, 0)

		// outLen = (5+0-3)/1+1 = 3
		expectedShape := []int{1, 3} // [outChannels, outLen]
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})

	t.Run("with batch", func(t *testing.T) {
		// Input: [2, 2, 5] (batch, inChannels, length)
		input := &Tensor{
			Dim:  []int{2, 2, 5},
			Data: make([]float32, 2*2*5),
		}
		for i := range input.Data {
			input.Data[i] = float32(i + 1)
		}

		kernel := &Tensor{
			Dim:  []int{1, 2, 3},
			Data: []float32{1, 1, 1, 1, 1, 1},
		}

		result := input.Conv1D(kernel, nil, 1, 0)

		expectedShape := []int{2, 1, 3} // [batch, outChannels, outLen]
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

func TestMaxPool2D(t *testing.T) {
	t.Run("basic max pooling", func(t *testing.T) {
		input := &Tensor{
			Dim: []int{1, 1, 4, 4},
			Data: []float32{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
				13, 14, 15, 16,
			},
		}

		result := input.MaxPool2D([]int{2, 2}, []int{2, 2}, []int{0, 0})

		// outHeight = (4+0-2)/2+1 = 2, outWidth = 2
		expectedShape := []int{1, 1, 2, 2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}

		// Check that max pooling works (first window should be max of [1,2,5,6] = 6)
		if result.Data[0] != 6.0 {
			t.Errorf("MaxPool[0] = %f, expected 6.0", result.Data[0])
		}
	})

	t.Run("with stride and padding", func(t *testing.T) {
		input := &Tensor{
			Dim: []int{1, 1, 3, 3},
			Data: []float32{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			},
		}

		result := input.MaxPool2D([]int{2, 2}, []int{1, 1}, []int{1, 1})

		// outHeight = (3+2*1-2)/1+1 = 4, outWidth = 4
		expectedShape := []int{1, 1, 4, 4}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

func TestAvgPool2D(t *testing.T) {
	t.Run("basic average pooling", func(t *testing.T) {
		input := &Tensor{
			Dim: []int{1, 1, 4, 4},
			Data: []float32{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
				13, 14, 15, 16,
			},
		}

		result := input.AvgPool2D([]int{2, 2}, []int{2, 2}, []int{0, 0})

		// outHeight = (4+0-2)/2+1 = 2, outWidth = 2
		expectedShape := []int{1, 1, 2, 2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}

		// Check that average pooling works (first window: avg of [1,2,5,6] = 3.5)
		expected := float32(3.5)
		if !floatEqual(result.Data[0], expected) {
			t.Errorf("AvgPool[0] = %f, expected %f", result.Data[0], expected)
		}
	})
}

func TestGlobalAvgPool2D(t *testing.T) {
	t.Run("global average pooling", func(t *testing.T) {
		input := &Tensor{
			Dim: []int{1, 2, 2, 2}, // [batch, channels, height, width]
			Data: []float32{
				1, 2, // channel 0, row 0
				3, 4, // channel 0, row 1
				5, 6, // channel 1, row 0
				7, 8, // channel 1, row 1
			},
		}

		result := input.GlobalAvgPool2D()

		// Output: [batch, channels] = [1, 2]
		expectedShape := []int{1, 2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}

		// Channel 0: avg of [1,2,3,4] = 2.5
		expected0 := float32(2.5)
		if !floatEqual(result.Data[0], expected0) {
			t.Errorf("GlobalAvgPool[0] = %f, expected %f", result.Data[0], expected0)
		}

		// Channel 1: avg of [5,6,7,8] = 6.5
		expected1 := float32(6.5)
		if !floatEqual(result.Data[1], expected1) {
			t.Errorf("GlobalAvgPool[1] = %f, expected %f", result.Data[1], expected1)
		}
	})
}

func TestIm2Col(t *testing.T) {
	t.Run("basic Im2Col", func(t *testing.T) {
		// Input: [1, 1, 3, 3]
		input := &Tensor{
			Dim: []int{1, 1, 3, 3},
			Data: []float32{
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
			},
		}

		result := input.Im2Col([]int{2, 2}, []int{1, 1}, []int{0, 0})

		// Output: [batch*outHeight*outWidth, channels*kernelH*kernelW]
		// outHeight = (3+0-2)/1+1 = 2, outWidth = 2
		// colHeight = 1*2*2 = 4, colWidth = 1*2*2 = 4
		expectedShape := []int{4, 4}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

func TestCol2Im(t *testing.T) {
	t.Run("basic Col2Im", func(t *testing.T) {
		// Input columns: [4, 4] (from 1 batch, 2x2 output, 1 channel, 2x2 kernel)
		col := &Tensor{
			Dim:  []int{4, 4},
			Data: make([]float32, 16),
		}
		for i := range col.Data {
			col.Data[i] = float32(i + 1)
		}

		// Output shape: [1, 1, 3, 3]
		result := col.Col2Im([]int{1, 1, 3, 3}, []int{2, 2}, []int{1, 1}, []int{0, 0})

		expectedShape := []int{1, 1, 3, 3}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

func TestDepthwiseConv2D(t *testing.T) {
	t.Run("basic depthwise convolution", func(t *testing.T) {
		// Input: [1, 2, 3, 3] - batch=1, channels=2, height=3, width=3
		input := &Tensor{
			Dim: []int{1, 2, 3, 3},
			Data: []float32{
				// Channel 0
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				// Channel 1
				10, 11, 12,
				13, 14, 15,
				16, 17, 18,
			},
		}

		// Kernel: [2, 1, 2, 2] - channels=2, 1, kernelH=2, kernelW=2
		kernel := &Tensor{
			Dim: []int{2, 1, 2, 2},
			Data: []float32{
				// Channel 0 kernel
				1, 1,
				1, 1,
				// Channel 1 kernel
				2, 2,
				2, 2,
			},
		}

		bias := &Tensor{
			Dim:  []int{2},
			Data: []float32{0.5, 1.0},
		}

		result := input.DepthwiseConv2D(kernel, bias, []int{1, 1}, []int{0, 0})

		// Output: [1, 2, 2, 2] - same channels, reduced spatial size
		expectedShape := []int{1, 2, 2, 2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}

		// Verify output size
		expectedSize := 1 * 2 * 2 * 2
		if len(result.Data) != expectedSize {
			t.Errorf("Data length = %d, expected %d", len(result.Data), expectedSize)
		}
	})

	t.Run("depthwise with 3D kernel", func(t *testing.T) {
		input := &Tensor{
			Dim:  []int{1, 2, 2, 2},
			Data: []float32{1, 2, 3, 4, 5, 6, 7, 8},
		}

		// Kernel: [2, 2, 2] - channels=2, kernelH=2, kernelW=2
		kernel := &Tensor{
			Dim: []int{2, 2, 2},
			Data: []float32{
				1, 1, // Channel 0 kernel
				1, 1,
				2, 2, // Channel 1 kernel
				2, 2,
			},
		}

		result := input.DepthwiseConv2D(kernel, nil, []int{1, 1}, []int{0, 0})
		expectedShape := []int{1, 2, 1, 1}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

func TestGroupConv2D(t *testing.T) {
	t.Run("grouped convolution with 2 groups", func(t *testing.T) {
		// Input: [1, 4, 2, 2] - batch=1, channels=4, height=2, width=2
		input := &Tensor{
			Dim:  []int{1, 4, 2, 2},
			Data: make([]float32, 1*4*2*2),
		}
		for i := range input.Data {
			input.Data[i] = float32(i + 1)
		}

		// Kernel: [4, 2, 1, 1] - outChannels=4, inChannels/groups=2, kernelH=1, kernelW=1
		// With groups=2, we have 2 groups, each processing 2 input channels to produce 2 output channels
		kernel := &Tensor{
			Dim:  []int{4, 2, 1, 1},
			Data: make([]float32, 4*2*1*1),
		}
		for i := range kernel.Data {
			kernel.Data[i] = float32(i + 1)
		}

		result := input.GroupConv2D(kernel, nil, []int{1, 1}, []int{0, 0}, 2)

		// Output: [1, 4, 2, 2] - same spatial size
		expectedShape := []int{1, 4, 2, 2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

func TestDilatedConv2D(t *testing.T) {
	t.Run("dilated convolution", func(t *testing.T) {
		// Input: [1, 1, 5, 5]
		input := &Tensor{
			Dim:  []int{1, 1, 5, 5},
			Data: make([]float32, 1*1*5*5),
		}
		for i := range input.Data {
			input.Data[i] = float32(i + 1)
		}

		// Kernel: [1, 1, 3, 3]
		kernel := &Tensor{
			Dim:  []int{1, 1, 3, 3},
			Data: []float32{1, 0, 0, 0, 1, 0, 0, 0, 1},
		}

		// Dilation: [2, 2] - effective kernel size becomes (3-1)*2+1 = 5
		result := input.DilatedConv2D(kernel, nil, []int{1, 1}, []int{0, 0}, []int{2, 2})

		// With dilation=2 and kernel=3x3, effective kernel is 5x5
		// Input 5x5 with effective kernel 5x5 and stride 1, padding 0 gives 1x1 output
		expectedShape := []int{1, 1, 1, 1}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

func TestConv3D(t *testing.T) {
	t.Run("basic 3D convolution", func(t *testing.T) {
		// Input: [1, 1, 3, 3, 3] - batch=1, channels=1, depth=3, height=3, width=3
		input := &Tensor{
			Dim:  []int{1, 1, 3, 3, 3},
			Data: make([]float32, 1*1*3*3*3),
		}
		for i := range input.Data {
			input.Data[i] = float32(i + 1)
		}

		// Kernel: [1, 1, 2, 2, 2] - outChannels=1, inChannels=1, kernelD=2, kernelH=2, kernelW=2
		kernel := &Tensor{
			Dim:  []int{1, 1, 2, 2, 2},
			Data: []float32{1, 1, 1, 1, 1, 1, 1, 1},
		}

		bias := &Tensor{
			Dim:  []int{1},
			Data: []float32{0.0},
		}

		result := input.Conv3D(kernel, bias, []int{1, 1, 1}, []int{0, 0, 0})

		// Output: [1, 1, 2, 2, 2] - depth reduced by 1, spatial dimensions reduced by 1
		expectedShape := []int{1, 1, 2, 2, 2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}

		// Verify output size
		expectedSize := 1 * 1 * 2 * 2 * 2
		if len(result.Data) != expectedSize {
			t.Errorf("Data length = %d, expected %d", len(result.Data), expectedSize)
		}
	})
}

func TestAdaptiveAvgPool2D(t *testing.T) {
	t.Run("adaptive average pooling to 2x2", func(t *testing.T) {
		// Input: [1, 1, 4, 4]
		input := &Tensor{
			Dim: []int{1, 1, 4, 4},
			Data: []float32{
				1, 2, 3, 4,
				5, 6, 7, 8,
				9, 10, 11, 12,
				13, 14, 15, 16,
			},
		}

		// Output size: [2, 2]
		result := input.AdaptiveAvgPool2D([]int{2, 2})

		// Output: [1, 1, 2, 2]
		expectedShape := []int{1, 1, 2, 2}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}

		// Verify output values
		// Top-left region: avg of [1,2,5,6] = 3.5
		// Top-right region: avg of [3,4,7,8] = 5.5
		// Bottom-left region: avg of [9,10,13,14] = 11.5
		// Bottom-right region: avg of [11,12,15,16] = 13.5
		expected := []float32{3.5, 5.5, 11.5, 13.5}
		for i := range expected {
			if !floatEqual(result.Data[i], expected[i]) {
				t.Errorf("Data[%d] = %f, expected %f", i, result.Data[i], expected[i])
			}
		}
	})

	t.Run("adaptive average pooling to 1x1", func(t *testing.T) {
		input := &Tensor{
			Dim: []int{1, 2, 3, 3},
			Data: []float32{
				// Channel 0
				1, 2, 3,
				4, 5, 6,
				7, 8, 9,
				// Channel 1
				10, 20, 30,
				40, 50, 60,
				70, 80, 90,
			},
		}

		result := input.AdaptiveAvgPool2D([]int{1, 1})

		// Output: [1, 2, 1, 1]
		expectedShape := []int{1, 2, 1, 1}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}

		// Channel 0: avg of all 9 values = 5.0
		// Channel 1: avg of all 9 values = 50.0
		expected := []float32{5.0, 50.0}
		for i := range expected {
			if !floatEqual(result.Data[i], expected[i]) {
				t.Errorf("Data[%d] = %f, expected %f", i, result.Data[i], expected[i])
			}
		}
	})

	t.Run("adaptive average pooling to larger size", func(t *testing.T) {
		input := &Tensor{
			Dim:  []int{1, 1, 2, 2},
			Data: []float32{1, 2, 3, 4},
		}

		result := input.AdaptiveAvgPool2D([]int{4, 4})

		// Output: [1, 1, 4, 4]
		expectedShape := []int{1, 1, 4, 4}
		for i := range expectedShape {
			if result.Dim[i] != expectedShape[i] {
				t.Errorf("Dim[%d] = %d, expected %d", i, result.Dim[i], expectedShape[i])
			}
		}
	})
}

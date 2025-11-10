package fp32

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIm2Col(t *testing.T) {
	tests := []struct {
		name                string
		im                  []float32
		batchSize, channels int
		height, width       int
		kernelH, kernelW    int
		padH, padW          int
		strideH, strideW    int
		wantCol             []float32
	}{
		{
			name: "simple 1x1 image, 1x1 kernel, no padding",
			// Single pixel image: [[1]]
			im:        []float32{1.0},
			batchSize: 1,
			channels:  1,
			height:    1,
			width:     1,
			kernelH:   1,
			kernelW:   1,
			padH:      0,
			padW:      0,
			strideH:   1,
			strideW:   1,
			// Output: 1 patch of 1 element
			wantCol: []float32{1.0},
		},
		{
			name: "2x2 image, 1x1 kernel, no padding",
			// Image: [[1, 2],
			//         [3, 4]]
			im:        []float32{1.0, 2.0, 3.0, 4.0},
			batchSize: 1,
			channels:  1,
			height:    2,
			width:     2,
			kernelH:   1,
			kernelW:   1,
			padH:      0,
			padW:      0,
			strideH:   1,
			strideW:   1,
			// Output: 4 patches of 1 element each
			wantCol: []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name: "2x2 image, 2x2 kernel, no padding",
			// Image: [[1, 2],
			//         [3, 4]]
			im:        []float32{1.0, 2.0, 3.0, 4.0},
			batchSize: 1,
			channels:  1,
			height:    2,
			width:     2,
			kernelH:   2,
			kernelW:   2,
			padH:      0,
			padW:      0,
			strideH:   1,
			strideW:   1,
			// Output: 1 patch of 4 elements: [1, 2, 3, 4]
			wantCol: []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name: "2x2 image, 2x2 kernel, with padding",
			// Image: [[1, 2],
			//         [3, 4]]
			// With padding=1, padded image:
			// [[0, 0, 0, 0],
			//  [0, 1, 2, 0],
			//  [0, 3, 4, 0],
			//  [0, 0, 0, 0]]
			// Output size: 3x3 = 9 patches
			// Each patch is 2x2 kernel, stored row-major: [top-left, top-right, bottom-left, bottom-right]
			im:        []float32{1.0, 2.0, 3.0, 4.0},
			batchSize: 1,
			channels:  1,
			height:    2,
			width:     2,
			kernelH:   2,
			kernelW:   2,
			padH:      1,
			padW:      1,
			strideH:   1,
			strideW:   1,
			// Patches (row, col) stored as [kh=0,kw=0, kh=0,kw=1, kh=1,kw=0, kh=1,kw=1]:
			// (0,0): [0, 0, 0, 1]
			// (0,1): [0, 0, 1, 2]
			// (0,2): [0, 0, 2, 0]
			// (1,0): [0, 1, 0, 3]
			// (1,1): [1, 2, 3, 4]
			// (1,2): [2, 0, 4, 0]
			// (2,0): [0, 3, 0, 0]
			// (2,1): [3, 4, 0, 0]
			// (2,2): [4, 0, 0, 0]
			wantCol: []float32{
				0, 0, 0, 1, // patch (0,0)
				0, 0, 1, 2, // patch (0,1)
				0, 0, 2, 0, // patch (0,2)
				0, 1, 0, 3, // patch (1,0)
				1, 2, 3, 4, // patch (1,1)
				2, 0, 4, 0, // patch (1,2)
				0, 3, 0, 0, // patch (2,0)
				3, 4, 0, 0, // patch (2,1)
				4, 0, 0, 0, // patch (2,2)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Calculate output dimensions
			outHeight := (tt.height+2*tt.padH-tt.kernelH)/tt.strideH + 1
			outWidth := (tt.width+2*tt.padW-tt.kernelW)/tt.strideW + 1
			kernelSize := tt.channels * tt.kernelH * tt.kernelW
			im2colSize := tt.batchSize * outHeight * outWidth

			col := make([]float32, im2colSize*kernelSize)
			Im2Col(col, tt.im, tt.batchSize, tt.channels, tt.height, tt.width,
				tt.kernelH, tt.kernelW, tt.padH, tt.padW, tt.strideH, tt.strideW)

			assert.InDeltaSlice(t, tt.wantCol, col, 1e-5)
		})
	}
}

func TestCol2Im(t *testing.T) {
	tests := []struct {
		name                string
		col                 []float32
		batchSize, channels int
		height, width       int
		kernelH, kernelW    int
		padH, padW          int
		strideH, strideW    int
		wantIm              []float32
	}{
		{
			name:      "simple 1x1, 1x1 kernel",
			col:       []float32{1.0},
			batchSize: 1,
			channels:  1,
			height:    1,
			width:     1,
			kernelH:   1,
			kernelW:   1,
			padH:      0,
			padW:      0,
			strideH:   1,
			strideW:   1,
			wantIm:    []float32{1.0},
		},
		{
			name:      "2x2 image, 2x2 kernel",
			col:       []float32{1.0, 2.0, 3.0, 4.0},
			batchSize: 1,
			channels:  1,
			height:    2,
			width:     2,
			kernelH:   2,
			kernelW:   2,
			padH:      0,
			padW:      0,
			strideH:   1,
			strideW:   1,
			// Single patch accumulates to all positions
			wantIm: []float32{1.0, 2.0, 3.0, 4.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			im := make([]float32, tt.batchSize*tt.channels*tt.height*tt.width)
			Col2Im(im, tt.col, tt.batchSize, tt.channels, tt.height, tt.width,
				tt.kernelH, tt.kernelW, tt.padH, tt.padW, tt.strideH, tt.strideW)

			assert.InDeltaSlice(t, tt.wantIm, im, 1e-5)
		})
	}
}

func TestCol2ImClearsDestination(t *testing.T) {
	col := []float32{1, 2, 3, 4}
	im := make([]float32, 4)
	for i := range im {
		im[i] = 99
	}

	Col2Im(im, col, 1, 1, 2, 2, 1, 1, 0, 0, 1, 1)

	assert.InDeltaSlice(t, []float32{1, 2, 3, 4}, im, 1e-6)
}

func TestAvgPool2DBackwardClearsDestination(t *testing.T) {
	gradInput := make([]float32, 9)
	for i := range gradInput {
		gradInput[i] = 100 + float32(i)
	}
	gradOutput := []float32{1, 1, 1, 1}

	AvgPool2DBackward(gradInput, gradOutput, 1, 1, 3, 3, 2, 2, 2, 2, 2, 2, 1, 1)

	expected := []float32{
		1, 0.5, 0.5,
		0.5, 0.25, 0.25,
		0.5, 0.25, 0.25,
	}
	assert.InDeltaSlice(t, expected, gradInput, 1e-6)
}

func TestMaxPool2DBackwardClearsDestination(t *testing.T) {
	input := []float32{
		1, 2, 3, 4,
		5, 9, 8, 7,
		0, 6, 5, 4,
		3, 2, 1, 0,
	}
	output := make([]float32, 4)
	indices := make([]int32, 4)
	MaxPool2DWithIndices(output, input, indices, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0)

	gradInput := make([]float32, len(input))
	for i := range gradInput {
		gradInput[i] = 50 + float32(i)
	}
	gradOutput := []float32{1, 1, 1, 1}

	MaxPool2DBackward(gradInput, gradOutput, indices, input, 1, 1, 4, 4, 2, 2, 2, 2, 2, 2, 0, 0)

	expected := []float32{
		0, 0, 0, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 0, 0, 0,
	}
	assert.InDeltaSlice(t, expected, gradInput, 1e-6)
}

func TestConv2D(t *testing.T) {
	tests := []struct {
		name                    string
		input, weights          []float32
		batchSize               int
		inChannels, outChannels int
		inHeight, inWidth       int
		outHeight, outWidth     int
		kernelH, kernelW        int
		strideH, strideW        int
		padH, padW              int
		bias                    []float32
		wantOutput              []float32
	}{
		{
			name: "simple 1x1 convolution, 1x1 kernel",
			// Input: [1, 1, 1, 1] - single pixel, single channel
			input: []float32{1.0},
			// Weights: [1, 1, 1, 1] - 1 output channel, 1 input channel, 1x1 kernel
			weights:     []float32{2.0},
			batchSize:   1,
			inChannels:  1,
			outChannels: 1,
			inHeight:    1,
			inWidth:     1,
			outHeight:   1,
			outWidth:    1,
			kernelH:     1,
			kernelW:     1,
			strideH:     1,
			strideW:     1,
			padH:        0,
			padW:        0,
			bias:        nil,
			// Output: 1 * 2 = 2
			wantOutput: []float32{2.0},
		},
		{
			name: "2x2 image, 1 output channel, 1x1 kernel",
			// Input: [1, 1, 2, 2] - 2x2 image, 1 channel
			// Layout: [batch, channel, height, width]
			input: []float32{1.0, 2.0, 3.0, 4.0},
			// Weights: [1, 1, 1, 1] - 1 output, 1 input, 1x1 kernel
			weights:     []float32{0.5},
			batchSize:   1,
			inChannels:  1,
			outChannels: 1,
			inHeight:    2,
			inWidth:     2,
			outHeight:   2,
			outWidth:    2,
			kernelH:     1,
			kernelW:     1,
			strideH:     1,
			strideW:     1,
			padH:        0,
			padW:        0,
			bias:        nil,
			// Output: input * 0.5 = [0.5, 1.0, 1.5, 2.0]
			wantOutput: []float32{0.5, 1.0, 1.5, 2.0},
		},
		{
			name:        "with bias",
			input:       []float32{1.0},
			weights:     []float32{2.0},
			batchSize:   1,
			inChannels:  1,
			outChannels: 1,
			inHeight:    1,
			inWidth:     1,
			outHeight:   1,
			outWidth:    1,
			kernelH:     1,
			kernelW:     1,
			strideH:     1,
			strideW:     1,
			padH:        0,
			padW:        0,
			bias:        []float32{10.0},
			// Output: 1 * 2 + 10 = 12
			wantOutput: []float32{12.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputSize := tt.batchSize * tt.outChannels * tt.outHeight * tt.outWidth
			output := make([]float32, outputSize)

			Conv2D(output, tt.input, tt.weights, tt.batchSize, tt.inChannels, tt.outChannels,
				tt.inHeight, tt.inWidth, tt.outHeight, tt.outWidth,
				tt.kernelH, tt.kernelW, tt.strideH, tt.strideW, tt.padH, tt.padW, tt.bias)

			assert.InDeltaSlice(t, tt.wantOutput, output, 1e-5)
		})
	}
}

func TestConv2DTransposed(t *testing.T) {
	tests := []struct {
		name                    string
		input, weights          []float32
		batchSize               int
		inChannels, outChannels int
		inHeight, inWidth       int
		outHeight, outWidth     int
		kernelH, kernelW        int
		strideH, strideW        int
		padH, padW              int
		bias                    []float32
		wantOutput              []float32
	}{
		{
			name:        "simple 1x1 transposed convolution",
			input:       []float32{2.0},
			weights:     []float32{3.0}, // [inChannels=1, outChannels=1, 1, 1]
			batchSize:   1,
			inChannels:  1,
			outChannels: 1,
			inHeight:    1,
			inWidth:     1,
			outHeight:   1,
			outWidth:    1,
			kernelH:     1,
			kernelW:     1,
			strideH:     1,
			strideW:     1,
			padH:        0,
			padW:        0,
			bias:        nil,
			// Output: 2 * 3 = 6
			wantOutput: []float32{6.0},
		},
		{
			name:        "with bias",
			input:       []float32{2.0},
			weights:     []float32{3.0},
			batchSize:   1,
			inChannels:  1,
			outChannels: 1,
			inHeight:    1,
			inWidth:     1,
			outHeight:   1,
			outWidth:    1,
			kernelH:     1,
			kernelW:     1,
			strideH:     1,
			strideW:     1,
			padH:        0,
			padW:        0,
			bias:        []float32{5.0},
			// Output: 2 * 3 + 5 = 11
			wantOutput: []float32{11.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			outputSize := tt.batchSize * tt.outChannels * tt.outHeight * tt.outWidth
			output := make([]float32, outputSize)

			Conv2DTransposed(output, tt.input, tt.weights, tt.batchSize, tt.inChannels, tt.outChannels,
				tt.inHeight, tt.inWidth, tt.outHeight, tt.outWidth,
				tt.kernelH, tt.kernelW, tt.strideH, tt.strideW, tt.padH, tt.padW, tt.bias)

			assert.InDeltaSlice(t, tt.wantOutput, output, 1e-5)
		})
	}
}

func TestTensorEmpty(t *testing.T) {
	im := []float32{1, 2, 3, 4}
	col := make([]float32, 4)
	input := []float32{1, 2, 3, 4}
	weights := []float32{1, 2, 3, 4}
	output := make([]float32, 4)

	require.NotPanics(t, func() {
		// Empty dimensions should not panic
		Im2Col(col, im, 0, 1, 2, 2, 1, 1, 0, 0, 1, 1)
		Im2Col(col, im, 1, 0, 2, 2, 1, 1, 0, 0, 1, 1)
		Col2Im(im, col, 0, 1, 2, 2, 1, 1, 0, 0, 1, 1)
		Conv2D(output, input, weights, 0, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, nil)
		Conv2DTransposed(output, input, weights, 0, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, nil)
	})
}

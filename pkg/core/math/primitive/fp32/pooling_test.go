package fp32

import "testing"

func TestAvgPool2DForwardWithPadding(t *testing.T) {
	input := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	batchSize, channels, height, width := 1, 1, 3, 3
	kernelH, kernelW := 2, 2
	strideH, strideW := 2, 2
	padH, padW := 1, 1
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	dst := make([]float32, batchSize*channels*outHeight*outWidth)
	AvgPool2D(dst, input, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)

	expected := []float32{
		1, 2.5,
		5.5, 7,
	}

	for i, got := range dst {
		want := expected[i]
		if absDiff(float64(got), float64(want)) > 1e-6 {
			t.Fatalf("dst[%d]=%f want %f", i, got, want)
		}
	}
}

func TestAvgPool2DBackwardWithPadding(t *testing.T) {
	batchSize, channels, inHeight, inWidth := 1, 1, 3, 3
	kernelH, kernelW := 2, 2
	strideH, strideW := 2, 2
	padH, padW := 1, 1
	outHeight := (inHeight+2*padH-kernelH)/strideH + 1
	outWidth := (inWidth+2*padW-kernelW)/strideW + 1

	gradOutput := make([]float32, batchSize*channels*outHeight*outWidth)
	for i := range gradOutput {
		gradOutput[i] = 1
	}

	gradInput := make([]float32, batchSize*channels*inHeight*inWidth)
	AvgPool2DBackward(gradInput, gradOutput, batchSize, channels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW)

	expected := []float32{
		1, 0.5, 0.5,
		0.5, 0.25, 0.25,
		0.5, 0.25, 0.25,
	}

	for i, got := range gradInput {
		want := expected[i]
		if absDiff(float64(got), float64(want)) > 1e-6 {
			t.Fatalf("gradInput[%d]=%f want %f", i, got, want)
		}
	}
}

func TestMaxPool2DForward(t *testing.T) {
	input := []float32{
		1, 2, 3, 4,
		5, 9, 8, 7,
		0, 6, 5, 4,
		3, 2, 1, 0,
	}
	batchSize, channels, height, width := 1, 1, 4, 4
	kernelH, kernelW := 2, 2
	strideH, strideW := 2, 2
	padH, padW := 0, 0
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	dst := make([]float32, batchSize*channels*outHeight*outWidth)
	MaxPool2D(dst, input, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)

	expected := []float32{
		9, 8,
		6, 5,
	}

	for i, got := range dst {
		want := expected[i]
		if absDiff(float64(got), float64(want)) > 1e-6 {
			t.Fatalf("dst[%d]=%f want %f", i, got, want)
		}
	}
}

func TestMaxPool2DBackward(t *testing.T) {
	input := []float32{
		1, 2, 3, 4,
		5, 9, 8, 7,
		0, 6, 5, 4,
		3, 2, 1, 0,
	}
	batchSize, channels, height, width := 1, 1, 4, 4
	kernelH, kernelW := 2, 2
	strideH, strideW := 2, 2
	padH, padW := 0, 0
	outHeight := (height+2*padH-kernelH)/strideH + 1
	outWidth := (width+2*padW-kernelW)/strideW + 1

	dst := make([]float32, batchSize*channels*outHeight*outWidth)
	indices := make([]int32, batchSize*channels*outHeight*outWidth)
	MaxPool2DWithIndices(dst, input, indices, batchSize, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW)

	gradOutput := make([]float32, len(dst))
	for i := range gradOutput {
		gradOutput[i] = 1
	}

	gradInput := make([]float32, batchSize*channels*height*width)
	MaxPool2DBackward(gradInput, gradOutput, indices, input, batchSize, channels, height, width, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW)

	expected := []float32{
		0, 0, 0, 0,
		0, 1, 1, 0,
		0, 1, 1, 0,
		0, 0, 0, 0,
	}

	for i, got := range gradInput {
		want := expected[i]
		if absDiff(float64(got), float64(want)) > 1e-6 {
			t.Fatalf("gradInput[%d]=%f want %f", i, got, want)
		}
	}
}

func TestGlobalAvgPool2D(t *testing.T) {
	input := []float32{
		1, 2, 3,
		4, 5, 6,
		7, 8, 9,
	}
	batchSize, channels, height, width := 1, 1, 3, 3

	dst := make([]float32, batchSize*channels)
	GlobalAvgPool2D(dst, input, batchSize, channels, height, width)

	expected := []float32{5}
	if absDiff(float64(dst[0]), float64(expected[0])) > 1e-6 {
		t.Fatalf("global average = %f want %f", dst[0], expected[0])
	}
}

func absDiff(a, b float64) float64 {
	if a > b {
		return a - b
	}
	return b - a
}

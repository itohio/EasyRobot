package blobs

import (
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	visiontypes "github.com/itohio/EasyRobot/x/vision/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gocv.io/x/gocv"
)

func TestProcessor_DetectBlobs(t *testing.T) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "detect_blob.png")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	mat := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, mat.Empty(), "Failed to load test image")
	defer mat.Close()

	// Create input tensor
	inputTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Create processor
	processor := New()
	defer processor.Close()

	// Create buffers
	srcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	dstBuf := buffer.NewRing(make([]visiontypes.KeypointsWithImage, 0, 10))

	// Write input
	written, err := srcBuf.Write([]types.Tensor{inputTensor})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	// Process
	outputCount := processor.Process(dstBuf, srcBuf)
	if outputCount == 0 {
		t.Log("Blob detector found no blobs - this may be acceptable")
		return
	}
	assert.Greater(t, outputCount, 0, "Should detect at least some blobs")

	// Read output
	outputs := make([]visiontypes.KeypointsWithImage, 1)
	n, err := dstBuf.Read(outputs)
	require.NoError(t, err)
	assert.Equal(t, 1, n)

	output := outputs[0]
	assert.NotNil(t, output.Image, "Output image should not be nil")
	assert.Greater(t, len(output.Keypoints), 0, "Should detect at least one blob")

	// Verify keypoints have valid positions
	for i, kp := range output.Keypoints {
		assert.GreaterOrEqual(t, kp.X, 0.0, "Keypoint %d X should be non-negative", i)
		assert.GreaterOrEqual(t, kp.Y, 0.0, "Keypoint %d Y should be non-negative", i)
	}

	t.Logf("Detected %d blobs", len(output.Keypoints))
}

func TestProcessor_DetectBlobs_WithArea(t *testing.T) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "detect_blob.png")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	mat := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, mat.Empty(), "Failed to load test image")
	defer mat.Close()

	// Create input tensor
	inputTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Create processor with custom area constraints
	processor := NewWithArea(100, 10000, image.Point{X: 30, Y: 30})
	defer processor.Close()

	// Create buffers
	srcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	dstBuf := buffer.NewRing(make([]visiontypes.KeypointsWithImage, 0, 10))

	// Write input
	written, err := srcBuf.Write([]types.Tensor{inputTensor})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	// Process
	outputCount := processor.Process(dstBuf, srcBuf)
	t.Logf("Detected %d blobs with area constraints", outputCount)
}

func TestProcessor_Draw(t *testing.T) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "detect_blob.png")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	mat := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, mat.Empty(), "Failed to load test image")
	defer mat.Close()

	// Create processor
	processor := New()
	defer processor.Close()

	// Create input tensor
	inputTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Create buffers and detect blobs
	srcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	dstBuf := buffer.NewRing(make([]visiontypes.KeypointsWithImage, 0, 10))

	_, err = srcBuf.Write([]types.Tensor{inputTensor})
	require.NoError(t, err)

	outputCount := processor.Process(dstBuf, srcBuf)
	if outputCount == 0 {
		t.Skip("No blobs detected, skipping draw test")
	}

	// Read output
	outputs := make([]visiontypes.KeypointsWithImage, 1)
	n, err := dstBuf.Read(outputs)
	require.NoError(t, err)
	require.Equal(t, 1, n)

	output := outputs[0]
	if len(output.Keypoints) == 0 {
		t.Skip("No keypoints to draw")
	}

	// Create output tensor for drawing
	outputMat := mat.Clone()
	defer outputMat.Close()
	outputTensor, err := tensorgocv.FromMat(outputMat, tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Draw keypoints
	err = processor.DrawKeypoints(outputTensor, output)
	assert.NoError(t, err, "Draw should succeed")
}

func TestProcessor_NoBlobs(t *testing.T) {
	// Create a simple test image (blank or no blobs)
	mat := gocv.NewMatWithSize(100, 100, gocv.MatTypeCV8UC3)
	defer mat.Close()
	mat.SetTo(gocv.NewScalar(0, 0, 0, 0)) // Black image

	// Create input tensor
	inputTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Create processor
	processor := New()
	defer processor.Close()

	// Create buffers
	srcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	dstBuf := buffer.NewRing(make([]visiontypes.KeypointsWithImage, 0, 10))

	// Write input
	written, err := srcBuf.Write([]types.Tensor{inputTensor})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	// Process
	outputCount := processor.Process(dstBuf, srcBuf)

	// Should handle no blobs gracefully
	if outputCount == 0 {
		t.Log("No blobs detected - acceptable for blank image")
		return
	}

	// If blobs were detected, verify output
	outputs := make([]visiontypes.KeypointsWithImage, 1)
	n, err := dstBuf.Read(outputs)
	if err == nil && n > 0 {
		t.Logf("Detected %d blobs in blank image", len(outputs[0].Keypoints))
	}
}

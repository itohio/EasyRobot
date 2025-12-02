package corners

import (
	"image"
	"os"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gocv.io/x/gocv"
)

func TestProcessor_DetectCorners(t *testing.T) {
	// Test image path - chessboard pattern for calibration
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "stereo_calib", "left04.jpg")
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

	// Test grid size (typical chessboard pattern)
	gridSize := image.Point{X: 9, Y: 6} // 9x6 internal corners

	// Create processor
	processor := New(gridSize)
	defer processor.Close()

	// Create buffers
	srcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	dstBuf := buffer.NewRing(make([]types.Tensor, 0, 10))

	// Write input
	written, err := srcBuf.Write([]types.Tensor{inputTensor})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	// Process
	outputCount := processor.Process(dstBuf, srcBuf)
	assert.Equal(t, 1, outputCount, "Should output 1 tensor")

	// Read output
	outputTensors := make([]types.Tensor, 1)
	read, err := dstBuf.Read(outputTensors)
	require.NoError(t, err)
	assert.Equal(t, 1, read)

	// Verify output
	outputTensor := outputTensors[0]
	require.NotNil(t, outputTensor, "Output tensor should not be nil")

	// Verify output is gocv tensor
	gocvOutput, ok := outputTensor.(tensorgocv.Tensor)
	require.True(t, ok, "Output should be gocv tensor")

	// Get underlying Mat
	outputMat, err := gocvOutput.Mat()
	require.NoError(t, err)
	assert.False(t, outputMat.Empty(), "Output Mat should not be empty")

	// Verify it's a points Mat (Nx1, 2-channel CV_32FC2)
	assert.Equal(t, 1, outputMat.Cols(), "Corners Mat should have 1 column")
	assert.Equal(t, gocv.MatTypeCV32FC2, outputMat.Type(), "Corners Mat should be CV_32FC2")

	// Expected number of corners: gridSize.X * gridSize.Y
	expectedCorners := gridSize.X * gridSize.Y
	assert.Equal(t, expectedCorners, outputMat.Rows(), "Should detect %d corners", expectedCorners)
}

func TestProcessor_DetectCorners_NoPattern(t *testing.T) {
	// Create a blank image (no chessboard pattern)
	mat := gocv.NewMatWithSize(640, 480, gocv.MatTypeCV8UC3)
	defer mat.Close()
	mat.SetTo(gocv.NewScalar(128, 128, 128, 0)) // Gray image

	// Create input tensor
	inputTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	gridSize := image.Point{X: 9, Y: 6}

	// Create processor
	processor := New(gridSize)
	defer processor.Close()

	// Create buffers
	srcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	dstBuf := buffer.NewRing(make([]types.Tensor, 0, 10))

	// Write input
	written, err := srcBuf.Write([]types.Tensor{inputTensor})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	// Process - should not find corners
	outputCount := processor.Process(dstBuf, srcBuf)
	assert.Equal(t, 0, outputCount, "Should not detect corners in blank image")
}

func TestProcessor_Detect_Direct(t *testing.T) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "stereo_calib", "left04.jpg")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	mat := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, mat.Empty(), "Failed to load test image")
	defer mat.Close()

	gridSize := image.Point{X: 9, Y: 6}

	// Test direct Detect function
	processor := New(gridSize)
	defer processor.Close()

	found, corners, err := processor.Detect(mat)
	require.NoError(t, err)
	assert.True(t, found, "Should find chessboard corners")
	assert.NotEmpty(t, corners, "Corners should not be empty")
	assert.Equal(t, gridSize.X*gridSize.Y, len(corners), "Should detect correct number of corners")
}

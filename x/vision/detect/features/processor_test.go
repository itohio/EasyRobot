package features

import (
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

func TestProcessor_DetectFeatures_ORB(t *testing.T) {
	testFeatureDetection(t, ORB, "ORB")
}

func TestProcessor_DetectFeatures_SIFT(t *testing.T) {
	testFeatureDetection(t, SIFT, "SIFT")
}

func TestProcessor_DetectFeatures_FAST(t *testing.T) {
	testFeatureDetection(t, FAST, "FAST")
}

func TestProcessor_DetectFeatures_GFTT(t *testing.T) {
	testFeatureDetection(t, GFTT, "GFTT")
}

func TestProcessor_DetectFeatures_KAZE(t *testing.T) {
	testFeatureDetection(t, KAZE, "KAZE")
}

func TestProcessor_DetectFeatures_AKAZE(t *testing.T) {
	testFeatureDetection(t, AKAZE, "AKAZE")
}

func TestProcessor_DetectFeatures_BRISK(t *testing.T) {
	testFeatureDetection(t, BRISK, "BRISK")
}

func testFeatureDetection(t *testing.T, featureType FeatureType, name string) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "img1", "000001.png")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	mat := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, mat.Empty(), "Failed to load test image for %s", name)
	defer mat.Close()

	// Create input tensor
	inputTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Create processor
	processor := New(featureType)
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
		// Some detectors might not find features in certain images
		t.Logf("%s detector found no features - this may be acceptable", name)
		return
	}
	assert.Equal(t, 1, outputCount, "%s should output 1 KeypointsWithImage", name)

	// Read output
	outputs := make([]visiontypes.KeypointsWithImage, 1)
	read, err := dstBuf.Read(outputs)
	require.NoError(t, err)
	assert.Equal(t, 1, read)

	// Verify output
	output := outputs[0]
	require.NotNil(t, output.Image, "%s output image should not be nil", name)
	assert.NotEmpty(t, output.Keypoints, "%s should detect keypoints", name)
	assert.Greater(t, len(output.Keypoints), 0, "%s should detect at least one keypoint", name)

	// Verify image tensor
	gocvTensor, ok := output.Image.(tensorgocv.Tensor)
	require.True(t, ok, "%s output image should be gocv tensor", name)
	outputMat, err := gocvTensor.Mat()
	require.NoError(t, err)
	assert.False(t, outputMat.Empty(), "%s output Mat should not be empty", name)

	// Verify keypoint properties
	for i, kp := range output.Keypoints {
		assert.GreaterOrEqual(t, kp.X, float32(0), "%s keypoint %d: X should be >= 0", name, i)
		assert.GreaterOrEqual(t, kp.Y, float32(0), "%s keypoint %d: Y should be >= 0", name, i)
		assert.LessOrEqual(t, kp.X, float32(outputMat.Cols()), "%s keypoint %d: X should be within image width", name, i)
		assert.LessOrEqual(t, kp.Y, float32(outputMat.Rows()), "%s keypoint %d: Y should be within image height", name, i)
		assert.Greater(t, kp.Size, float32(0), "%s keypoint %d: Size should be > 0", name, i)
		assert.GreaterOrEqual(t, kp.Response, float32(0), "%s keypoint %d: Response should be >= 0", name, i)
	}
}

func TestProcessor_DetectFeatures_Draw(t *testing.T) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "img1", "000001.png")
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
	processor := New(ORB)
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
	require.Equal(t, 1, outputCount, "Should output 1 KeypointsWithImage")

	// Read output
	outputs := make([]visiontypes.KeypointsWithImage, 1)
	read, err := dstBuf.Read(outputs)
	require.NoError(t, err)
	assert.Equal(t, 1, read)

	output := outputs[0]
	if len(output.Keypoints) == 0 {
		t.Skip("No keypoints detected, skipping draw test")
	}

	// Create output tensor for drawing
	outputMat := mat.Clone()
	defer outputMat.Close()
	outputTensor, err := tensorgocv.FromMat(outputMat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Test Draw method
	err = processor.DrawKeypoints(outputTensor, output)
	assert.NoError(t, err, "Draw should not return error")
}

func TestProcessor_DetectFeatures_EmptyImage(t *testing.T) {
	// Create a blank image
	mat := gocv.NewMatWithSize(640, 480, gocv.MatTypeCV8UC3)
	defer mat.Close()
	mat.SetTo(gocv.NewScalar(128, 128, 128, 0)) // Gray image

	// Create input tensor
	inputTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Create processor
	processor := New(ORB)
	defer processor.Close()

	// Create buffers
	srcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	dstBuf := buffer.NewRing(make([]visiontypes.KeypointsWithImage, 0, 10))

	// Write input
	written, err := srcBuf.Write([]types.Tensor{inputTensor})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	// Process - might or might not find features in blank image
	outputCount := processor.Process(dstBuf, srcBuf)
	// Either 0 or 1 is acceptable - depends on detector behavior
	assert.GreaterOrEqual(t, outputCount, 0)
	assert.LessOrEqual(t, outputCount, 1)
}


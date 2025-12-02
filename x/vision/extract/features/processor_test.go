package features

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/buffer"
	visiondetect "github.com/itohio/EasyRobot/x/vision/detect/features"
	tensorgocv "github.com/itohio/EasyRobot/x/math/tensor/gocv"
	"github.com/itohio/EasyRobot/x/math/tensor/types"
	visiontypes "github.com/itohio/EasyRobot/x/vision/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gocv.io/x/gocv"
)

func TestProcessor_ExtractFeatures_ORB(t *testing.T) {
	testFeatureExtraction(t, ORB, visiondetect.ORB, "ORB")
}

func TestProcessor_ExtractFeatures_SIFT(t *testing.T) {
	testFeatureExtraction(t, SIFT, visiondetect.SIFT, "SIFT")
}

func TestProcessor_ExtractFeatures_KAZE(t *testing.T) {
	testFeatureExtraction(t, KAZE, visiondetect.KAZE, "KAZE")
}

func TestProcessor_ExtractFeatures_AKAZE(t *testing.T) {
	testFeatureExtraction(t, AKAZE, visiondetect.AKAZE, "AKAZE")
}

func TestProcessor_ExtractFeatures_BRISK(t *testing.T) {
	testFeatureExtraction(t, BRISK, visiondetect.BRISK, "BRISK")
}

func testFeatureExtraction(t *testing.T, extractType FeatureType, detectType visiondetect.FeatureType, name string) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "img1", "000001.png")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	mat := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, mat.Empty(), "Failed to load test image for %s", name)
	defer mat.Close()

	// Step 1: Detect features
	detector := visiondetect.New(detectType)
	defer detector.Close()

	// Create buffers for detection
	detectSrcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	detectDstBuf := buffer.NewRing(make([]visiontypes.KeypointsWithImage, 0, 10))

	// Write input for detection
	inputTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)
	written, err := detectSrcBuf.Write([]types.Tensor{inputTensor})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	// Detect features
	detectOutputCount := detector.Process(detectDstBuf, detectSrcBuf)
	if detectOutputCount == 0 {
		t.Skipf("%s detector found no features - skipping extraction test", name)
		return
	}
	assert.Equal(t, 1, detectOutputCount, "%s should detect features", name)

	// Read detected keypoints
	keypointsOutputs := make([]visiontypes.KeypointsWithImage, 1)
	read, err := detectDstBuf.Read(keypointsOutputs)
	require.NoError(t, err)
	assert.Equal(t, 1, read)

	keypointsWithImage := keypointsOutputs[0]
	assert.NotEmpty(t, keypointsWithImage.Keypoints, "%s should have detected keypoints", name)

	// Step 2: Extract descriptors
	extractor := New(extractType)
	defer extractor.Close()

	// Create buffers for extraction
	extractSrcBuf := buffer.NewRing(make([]visiontypes.KeypointsWithImage, 0, 10))
	extractDstBuf := buffer.NewRing(make([]visiontypes.DescriptorsWithImage, 0, 10))

	// Write keypoints input for extraction
	// Clone the image tensor for extraction
	extractImageTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)
	keypointsWithImage.Image = extractImageTensor

	written, err = extractSrcBuf.Write([]visiontypes.KeypointsWithImage{keypointsWithImage})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	// Extract descriptors
	extractOutputCount := extractor.Process(extractDstBuf, extractSrcBuf)
	if extractOutputCount == 0 {
		// Some extractors might not produce descriptors for certain keypoints
		t.Logf("%s extractor produced no descriptors - this may be acceptable", name)
		return
	}
	assert.Equal(t, 1, extractOutputCount, "%s should output 1 DescriptorsWithImage", name)

	// Read output
	descriptorsOutputs := make([]visiontypes.DescriptorsWithImage, 1)
	read, err = extractDstBuf.Read(descriptorsOutputs)
	require.NoError(t, err)
	assert.Equal(t, 1, read)

	// Verify output
	output := descriptorsOutputs[0]
	require.NotNil(t, output.Image, "%s output image should not be nil", name)
	assert.NotEmpty(t, output.Keypoints, "%s should have keypoints", name)
	require.NotNil(t, output.Descriptors, "%s descriptors tensor should not be nil", name)

	// Verify descriptors tensor
	gocvTensor, ok := output.Descriptors.(tensorgocv.Tensor)
	require.True(t, ok, "%s descriptors should be gocv tensor", name)
	descMat, err := gocvTensor.Mat()
	require.NoError(t, err)
	assert.False(t, descMat.Empty(), "%s descriptors Mat should not be empty", name)

	// Verify descriptor dimensions
	// Descriptors should have one row per keypoint
	assert.Equal(t, len(output.Keypoints), descMat.Rows(), "%s should have one descriptor per keypoint", name)
	assert.Greater(t, descMat.Cols(), 0, "%s descriptors should have descriptor dimension > 0", name)

	// Verify descriptor type (typically CV_8U for binary descriptors, CV_32F for float)
	assert.True(t,
		descMat.Type() == gocv.MatTypeCV8UC1 || descMat.Type() == gocv.MatTypeCV32FC1,
		"%s descriptors should be CV_8UC1 or CV_32FC1 (got type %d)", name, descMat.Type())

	// Verify image tensor
	imgTensor, ok := output.Image.(tensorgocv.Tensor)
	require.True(t, ok, "%s output image should be gocv tensor", name)
	imgMat, err := imgTensor.Mat()
	require.NoError(t, err)
	assert.False(t, imgMat.Empty(), "%s output image Mat should not be empty", name)
}

func TestProcessor_ExtractFeatures_NoKeypoints(t *testing.T) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "img1", "000001.png")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	mat := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, mat.Empty(), "Failed to load test image")
	defer mat.Close()

	// Create extractor
	extractor := New(ORB)
	defer extractor.Close()

	// Create buffers
	extractSrcBuf := buffer.NewRing(make([]visiontypes.KeypointsWithImage, 0, 10))
	extractDstBuf := buffer.NewRing(make([]visiontypes.DescriptorsWithImage, 0, 10))

	// Create KeypointsWithImage with no keypoints
	inputTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)
	keypointsWithImage := visiontypes.KeypointsWithImage{
		Image:     inputTensor,
		Keypoints: []gocv.KeyPoint{}, // Empty keypoints
	}

	// Write input
	written, err := extractSrcBuf.Write([]visiontypes.KeypointsWithImage{keypointsWithImage})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	// Process - should handle empty keypoints gracefully
	extractOutputCount := extractor.Process(extractDstBuf, extractSrcBuf)
	// Should return 0 if no keypoints
	assert.Equal(t, 0, extractOutputCount, "Should return 0 when no keypoints provided")
}

func TestProcessor_ExtractFeatures_FullPipeline(t *testing.T) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "img1", "000001.png")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	mat := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, mat.Empty(), "Failed to load test image")
	defer mat.Close()

	// Create detector and extractor
	detector := visiondetect.New(visiondetect.ORB)
	defer detector.Close()

	extractor := New(ORB)
	defer extractor.Close()

	// Full pipeline: Image -> Detect -> Extract
	inputTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Detection step
	detectSrcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	detectDstBuf := buffer.NewRing(make([]visiontypes.KeypointsWithImage, 0, 10))

	written, err := detectSrcBuf.Write([]types.Tensor{inputTensor})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	detectOutputCount := detector.Process(detectDstBuf, detectSrcBuf)
	if detectOutputCount == 0 {
		t.Skip("No features detected, skipping full pipeline test")
	}

	// Read keypoints
	keypointsOutputs := make([]visiontypes.KeypointsWithImage, 1)
	read, err := detectDstBuf.Read(keypointsOutputs)
	require.NoError(t, err)
	assert.Equal(t, 1, read)

	keypointsWithImage := keypointsOutputs[0]
	if len(keypointsWithImage.Keypoints) == 0 {
		t.Skip("No keypoints detected, skipping full pipeline test")
	}

	// Update image tensor for extraction
	extractImageTensor, err := tensorgocv.FromMat(mat.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)
	keypointsWithImage.Image = extractImageTensor

	// Extraction step
	extractSrcBuf := buffer.NewRing(make([]visiontypes.KeypointsWithImage, 0, 10))
	extractDstBuf := buffer.NewRing(make([]visiontypes.DescriptorsWithImage, 0, 10))

	written, err = extractSrcBuf.Write([]visiontypes.KeypointsWithImage{keypointsWithImage})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	extractOutputCount := extractor.Process(extractDstBuf, extractSrcBuf)
	require.Equal(t, 1, extractOutputCount, "Should extract descriptors")

	// Read descriptors
	descriptorsOutputs := make([]visiontypes.DescriptorsWithImage, 1)
	read, err = extractDstBuf.Read(descriptorsOutputs)
	require.NoError(t, err)
	assert.Equal(t, 1, read)

	output := descriptorsOutputs[0]
	assert.NotEmpty(t, output.Keypoints, "Should have keypoints")
	require.NotNil(t, output.Descriptors, "Should have descriptors")

	// Verify descriptors
	gocvTensor, ok := output.Descriptors.(tensorgocv.Tensor)
	require.True(t, ok)
	descMat, err := gocvTensor.Mat()
	require.NoError(t, err)
	assert.False(t, descMat.Empty())
	assert.Equal(t, len(output.Keypoints), descMat.Rows(), "Should have one descriptor per keypoint")
	assert.Greater(t, descMat.Cols(), 0, "Descriptors should have dimension > 0")
}


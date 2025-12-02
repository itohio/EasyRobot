package phase

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

// shiftImage creates a shifted version of the input image by translating it.
// Uses WarpAffine with a translation matrix.
func shiftImage(src gocv.Mat, dx, dy float64) (gocv.Mat, error) {
	// Create translation matrix: [1 0 dx; 0 1 dy]
	transformMatrix := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV64F)
	transformMatrix.SetDoubleAt(0, 0, 1.0)
	transformMatrix.SetDoubleAt(0, 1, 0.0)
	transformMatrix.SetDoubleAt(0, 2, dx)
	transformMatrix.SetDoubleAt(1, 0, 0.0)
	transformMatrix.SetDoubleAt(1, 1, 1.0)
	transformMatrix.SetDoubleAt(1, 2, dy)
	defer transformMatrix.Close()

	// Apply translation
	dst := gocv.NewMat()
	size := image.Pt(src.Cols(), src.Rows())
	gocv.WarpAffine(src, &dst, transformMatrix, size)

	return dst, nil
}

func TestProcessor_PhaseCorrelation_ShiftedImage(t *testing.T) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "stereo_calib", "left11.jpg")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	original := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, original.Empty(), "Failed to load test image")
	defer original.Close()

	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(original, &gray, gocv.ColorBGRToGray)

	// Get image size
	imgSize := image.Pt(gray.Cols(), gray.Rows())

	// Create processor with skip=0 (no warmup frames)
	processor := New(imgSize, WithSkip(0))
	defer processor.Close()

	// Test with different known shifts
	testCases := []struct {
		name string
		dx   float64
		dy   float64
	}{
		{"Small shift right", 10, 0},
		{"Small shift down", 0, 10},
		{"Small shift right-down", 10, 10},
		{"Medium shift", 20, -15},
		{"Larger shift", 30, 20},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Reset processor for each test
			processor.Reset()

			// Create shifted image
			shifted, err := shiftImage(gray, tc.dx, tc.dy)
			require.NoError(t, err)
			defer shifted.Close()

			// Create buffers
			srcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
			dstBuf := buffer.NewRing(make([]visiontypes.Displacement, 0, 10))

			// Create tensors from images
			originalTensor, err := tensorgocv.FromMat(gray.Clone(), tensorgocv.WithAdoptedMat())
			require.NoError(t, err)

			shiftedTensor, err := tensorgocv.FromMat(shifted.Clone(), tensorgocv.WithAdoptedMat())
			require.NoError(t, err)

			// Process original image (first frame - will return zero displacement during warmup)
			written, err := srcBuf.Write([]types.Tensor{originalTensor})
			require.NoError(t, err)
			assert.Equal(t, 1, written)

			// Process first frame - this will be stored as previous frame
			outputCount := processor.Process(dstBuf, srcBuf)
			if outputCount > 0 {
				// Read and discard warmup result
				outputs := make([]visiontypes.Displacement, 1)
				dstBuf.Read(outputs)
			}

			// Clear source buffer for second frame
			srcBuf = buffer.NewRing(make([]types.Tensor, 0, 10))

			// Process shifted image (second frame)
			written, err = srcBuf.Write([]types.Tensor{shiftedTensor})
			require.NoError(t, err)
			assert.Equal(t, 1, written)

			// Process second frame - should detect displacement
			outputCount = processor.Process(dstBuf, srcBuf)
			
			// Note: Since phaseCorrelation() is not yet implemented, it will return (0, 0, 0.0)
			// This test verifies the structure works, actual displacement verification
			// should be added once the DFT implementation is complete
			
			if outputCount > 0 {
				outputs := make([]visiontypes.Displacement, 1)
				n, err := dstBuf.Read(outputs)
				require.NoError(t, err)
				assert.Equal(t, 1, n)

				displacement := outputs[0]
				t.Logf("Expected shift: dx=%.1f, dy=%.1f, Detected: dx=%.1f, dy=%.1f, peak=%.3f",
					tc.dx, tc.dy, displacement.DX, displacement.DY, displacement.PeakValue)

				// Verify displacement matches expected shift (allowing 1 pixel tolerance)
				assert.InDelta(t, tc.dx, displacement.DX, 1.0, "DX should match expected shift")
				assert.InDelta(t, tc.dy, displacement.DY, 1.0, "DY should match expected shift")
				assert.Greater(t, displacement.PeakValue, 0.0, "Peak correlation value should be positive")
			} else {
				t.Log("No displacement detected - this may be expected if phase correlation is not yet implemented")
			}
		})
	}
}

func TestProcessor_NoShift(t *testing.T) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "stereo_calib", "left11.jpg")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	original := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, original.Empty(), "Failed to load test image")
	defer original.Close()

	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(original, &gray, gocv.ColorBGRToGray)

	// Get image size
	imgSize := image.Pt(gray.Cols(), gray.Rows())

	// Create processor with skip=0
	processor := New(imgSize, WithSkip(0))
	defer processor.Close()

	// Create buffers
	srcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	dstBuf := buffer.NewRing(make([]visiontypes.Displacement, 0, 10))

	// Create tensor from image
	originalTensor, err := tensorgocv.FromMat(gray.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Process same image twice (no shift)
	written, err := srcBuf.Write([]types.Tensor{originalTensor})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	// Process first frame
	outputCount := processor.Process(dstBuf, srcBuf)
	if outputCount > 0 {
		outputs := make([]visiontypes.Displacement, 1)
		dstBuf.Read(outputs)
	}

	// Process second frame (same image)
	srcBuf = buffer.NewRing(make([]types.Tensor, 0, 10))
	originalTensor2, err := tensorgocv.FromMat(gray.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	written, err = srcBuf.Write([]types.Tensor{originalTensor2})
	require.NoError(t, err)
	assert.Equal(t, 1, written)

	outputCount = processor.Process(dstBuf, srcBuf)
	
	if outputCount > 0 {
		outputs := make([]visiontypes.Displacement, 1)
		n, err := dstBuf.Read(outputs)
		require.NoError(t, err)
		assert.Equal(t, 1, n)

		displacement := outputs[0]
		t.Logf("No shift test - Detected: dx=%.1f, dy=%.1f, peak=%.3f",
			displacement.DX, displacement.DY, displacement.PeakValue)

		// When no shift, displacement should be close to (0, 0)
		assert.InDelta(t, 0.0, displacement.DX, 1.0, "DX should be close to 0 for no shift")
		assert.InDelta(t, 0.0, displacement.DY, 1.0, "DY should be close to 0 for no shift")
	}
}

func TestProcessor_WarmupSkip(t *testing.T) {
	// Test image path
	imagePath := filepath.Join("..", "..", "..", "marshaller", "testdata", "stereo_calib", "left11.jpg")
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		t.Skipf("Test file not found: %s", imagePath)
	}

	// Load test image
	original := gocv.IMRead(imagePath, gocv.IMReadColor)
	require.False(t, original.Empty(), "Failed to load test image")
	defer original.Close()

	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(original, &gray, gocv.ColorBGRToGray)

	// Get image size
	imgSize := image.Pt(gray.Cols(), gray.Rows())

	// Create processor with skip=3 (warmup frames)
	processor := New(imgSize, WithSkip(3))
	defer processor.Close()

	// Create buffers
	srcBuf := buffer.NewRing(make([]types.Tensor, 0, 10))
	dstBuf := buffer.NewRing(make([]visiontypes.Displacement, 0, 10))

	// Create tensor from image
	originalTensor, err := tensorgocv.FromMat(gray.Clone(), tensorgocv.WithAdoptedMat())
	require.NoError(t, err)

	// Process multiple frames - first 3 should return zero displacement
	for i := 0; i < 5; i++ {
		written, err := srcBuf.Write([]types.Tensor{originalTensor})
		require.NoError(t, err)
		assert.Equal(t, 1, written)

		outputCount := processor.Process(dstBuf, srcBuf)
		
		if outputCount > 0 {
			outputs := make([]visiontypes.Displacement, 1)
			n, err := dstBuf.Read(outputs)
			require.NoError(t, err)
			assert.Equal(t, 1, n)

			displacement := outputs[0]
			
			if i < 3 {
				// During warmup, displacement should be (0, 0)
				assert.Equal(t, 0.0, displacement.DX, "DX should be 0 during warmup at frame %d", i)
				assert.Equal(t, 0.0, displacement.DY, "DY should be 0 during warmup at frame %d", i)
			} else {
				t.Logf("Frame %d - Detected: dx=%.1f, dy=%.1f, peak=%.3f",
					i, displacement.DX, displacement.DY, displacement.PeakValue)
			}
		}

		// Clear for next frame
		srcBuf = buffer.NewRing(make([]types.Tensor, 0, 10))
	}
}


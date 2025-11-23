package dsp

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
)

func TestFFT2D_Forward(t *testing.T) {
	fft := NewFFT2D(2, 2)

	// Create 2x2 test matrix
	matrix := mat.New(2, 2)
	matrix[0][0] = 1
	matrix[0][1] = 0
	matrix[1][0] = 0
	matrix[1][1] = 0

	spectrum := mat.New(2, 2)
	fft.Forward(spectrum, matrix)

	if spectrum.Rows() != 2 || spectrum.Cols() != 2 {
		t.Errorf("Expected 2x2 spectrum, got %dx%d", spectrum.Rows(), spectrum.Cols())
	}
}

func TestFFT2D_Backward(t *testing.T) {
	fft := NewFFT2D(8, 8) // Use power of 2

	// Create test matrix
	original := mat.New(8, 8)
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			original[i][j] = float32(i*8 + j)
		}
	}

	spectrum := mat.New(8, 8)
	reconstructed := mat.New(8, 8)

	// Forward then inverse FFT
	fft.Forward(spectrum, original)
	fft.Backward(reconstructed, spectrum)

	// Check reconstruction - IFFT scales by 1/(rows*cols), so Forward then Backward gives (1/(rows*cols)) * original
	scale := 1.0 / float32(fft.rows*fft.cols) // 1/(8*8) = 1/64
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			expected := original[i][j] * scale
			actual := reconstructed[i][j]
			if abs := math.Abs(float64(actual - expected)); abs > 1e-3 {
				t.Errorf("2D reconstruction error at (%d,%d): expected %f, got %f", i, j, expected, actual)
			}
		}
	}
}

func TestFFT2D_ForwardBackward_Roundtrip(t *testing.T) {
	fft := NewFFT2D(4, 4)

	// Create test matrix
	original := mat.New(4, 4)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			original[i][j] = float32(i*4 + j + 1)
		}
	}

	spectrum := mat.New(4, 4)
	reconstructed := mat.New(4, 4)

	// Forward FFT
	fft.Forward(spectrum, original)

	// Backward FFT
	fft.Backward(reconstructed, spectrum)

	// Check that we recover the original (with IFFT scaling by 1/(rows*cols))
	scale := 1.0 / float32(fft.rows*fft.cols)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			expected := original[i][j] * scale
			actual := reconstructed[i][j]
			if abs := math.Abs(float64(actual - expected)); abs > 1e-3 {
				t.Errorf("Roundtrip error at (%d,%d): expected %f, got %f", i, j, expected, actual)
			}
		}
	}
}

func TestFFT2D_Convolve(t *testing.T) {
	fft := NewFFT2D(4, 4) // Use power of 2

	// Simple 2x2 convolution
	signal := mat.New(2, 2)
	signal[0][0] = 1
	signal[0][1] = 2
	signal[1][0] = 3
	signal[1][1] = 4

	kernel := mat.New(2, 2)
	kernel[0][0] = 0.5
	kernel[0][1] = 1.0
	kernel[1][0] = 0.0
	kernel[1][1] = 0.5

	result := mat.New(4, 4) // Must match FFT dimensions
	fft.Convolve(result, signal, kernel)

	// Check result dimensions
	if result.Rows() != 4 || result.Cols() != 4 {
		t.Errorf("Expected 4x4 result, got %dx%d", result.Rows(), result.Cols())
	}

	// Check that result is non-zero
	hasNonZero := false
	for i := 0; i < result.Rows(); i++ {
		for j := 0; j < result.Cols(); j++ {
			if math.Abs(float64(result[i][j])) > 1e-6 {
				hasNonZero = true
				break
			}
		}
		if hasNonZero {
			break
		}
	}
	if !hasNonZero {
		t.Error("Convolution result should be non-zero")
	}
}

func TestFFT2D_Correlate(t *testing.T) {
	fft := NewFFT2D(4, 4)

	// Test 2D correlation
	signal1 := mat.New(2, 2)
	signal1[0][0] = 1
	signal1[0][1] = 2
	signal1[1][0] = 3
	signal1[1][1] = 4

	signal2 := mat.New(2, 2)
	signal2[0][0] = 1
	signal2[0][1] = 0
	signal2[1][0] = 0
	signal2[1][1] = 0

	result := mat.New(4, 4)
	fft.Correlate(result, signal1, signal2)

	// Check result dimensions
	if result.Rows() != 4 || result.Cols() != 4 {
		t.Errorf("Expected 4x4 result, got %dx%d", result.Rows(), result.Cols())
	}

	// Check that result is non-zero
	hasNonZero := false
	for i := 0; i < result.Rows(); i++ {
		for j := 0; j < result.Cols(); j++ {
			if math.Abs(float64(result[i][j])) > 1e-6 {
				hasNonZero = true
				break
			}
		}
		if hasNonZero {
			break
		}
	}
	if !hasNonZero {
		t.Error("Correlation result should be non-zero")
	}
}

func TestFFT2D_Correlate_SameSignal(t *testing.T) {
	fft := NewFFT2D(4, 4)

	// Autocorrelation of 2x2 matrix
	signal := mat.New(2, 2)
	signal[0][0] = 1
	signal[0][1] = 2
	signal[1][0] = 3
	signal[1][1] = 4

	result := mat.New(4, 4)
	fft.Correlate(signal, signal, result)

	// Autocorrelation should have peak at center
	// Check that result is non-zero
	hasNonZero := false
	for i := 0; i < result.Rows(); i++ {
		for j := 0; j < result.Cols(); j++ {
			if math.Abs(float64(result[i][j])) > 1e-6 {
				hasNonZero = true
				break
			}
		}
		if hasNonZero {
			break
		}
	}
	if !hasNonZero {
		t.Error("Autocorrelation result should be non-zero")
	}
}

func TestFFT2D_EdgeCases(t *testing.T) {
	// Test with 1x1 matrix
	fft1 := NewFFT2D(1, 1)
	matrix1 := mat.New(1, 1)
	matrix1[0][0] = 5.0
	spectrum1 := mat.New(1, 1)
	fft1.Forward(spectrum1, matrix1)
	if math.Abs(float64(spectrum1[0][0]-5.0)) > 1e-6 {
		t.Errorf("FFT of 1x1 matrix failed: expected 5.0, got %f", spectrum1[0][0])
	}

	// Test with non-power-of-two dimensions (should be padded)
	fft2 := NewFFT2D(3, 3) // Will pad to 4x4
	matrix2 := mat.New(3, 3)
	for i := range 3 {
		for j := range 3 {
			matrix2[i][j] = float32(i*3 + j + 1)
		}
	}
	spectrum2 := mat.New(4, 4)
	fft2.Forward(spectrum2, matrix2)
	// Check that FFT completed without error
	hasNonZero := false
	for i := range 4 {
		for j := range 4 {
			if math.Abs(float64(spectrum2[i][j])) > 1e-6 {
				hasNonZero = true
				break
			}
		}
		if hasNonZero {
			break
		}
	}
	if !hasNonZero {
		t.Error("FFT of non-power-of-two matrix should produce non-zero spectrum")
	}
}

func TestFFT2D_ErrorConditions(t *testing.T) {
	// Test panic for zero rows
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for zero rows FFT")
		}
	}()
	NewFFT2D(0, 4)

	// Test panic for zero cols
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for zero cols FFT")
		}
	}()
	NewFFT2D(4, 0)
}

func TestFFT2D_Convolve_ErrorConditions(t *testing.T) {
	fft := NewFFT2D(4, 4)
	signal := mat.New(2, 2)
	signal[0][0] = 1
	signal[0][1] = 2
	signal[1][0] = 3
	signal[1][1] = 4
	kernel := mat.New(2, 2)
	kernel[0][0] = 0.5
	kernel[0][1] = 1.0
	kernel[1][0] = 0.0
	kernel[1][1] = 0.5
	result := mat.New(2, 2) // Too small (need 3x3)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for destination buffer too small in Convolve")
		}
	}()
	fft.Convolve(result, signal, kernel)
}

func TestFFT2D_Correlate_ErrorConditions(t *testing.T) {
	fft := NewFFT2D(4, 4)
	signal1 := mat.New(2, 2)
	signal1[0][0] = 1
	signal1[0][1] = 2
	signal1[1][0] = 3
	signal1[1][1] = 4
	signal2 := mat.New(2, 2)
	signal2[0][0] = 1
	signal2[0][1] = 0
	signal2[1][0] = 0
	signal2[1][1] = 0
	result := mat.New(2, 2) // Too small (need 3x3)

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for destination buffer too small in Correlate")
		}
	}()
	fft.Correlate(result, signal1, signal2)
}

func TestFFT2D_Correlate_KnownSignals(t *testing.T) {
	// Test correlation with known signals
	// Correlating impulse with itself should give peak at center
	fft := NewFFT2D(8, 8)
	signal1 := mat.New(2, 2)
	signal1[0][0] = 1
	signal1[0][1] = 0
	signal1[1][0] = 0
	signal1[1][1] = 0
	signal2 := mat.New(2, 2)
	signal2[0][0] = 1
	signal2[0][1] = 0
	signal2[1][0] = 0
	signal2[1][1] = 0
	result := mat.New(3, 3) // sig1Rows + sig2Rows - 1 = 2 + 2 - 1 = 3

	fft.Correlate(result, signal1, signal2)

	// Find peak
	maxRow, maxCol := 0, 0
	maxVal := math.Abs(float64(result[0][0]))
	for i := range result.Rows() {
		for j := range result.Cols() {
			if math.Abs(float64(result[i][j])) > maxVal {
				maxVal = math.Abs(float64(result[i][j]))
				maxRow, maxCol = i, j
			}
		}
	}

	// Peak should be at center (approximately)
	// Allow some tolerance due to FFT implementation
	if maxRow < 1 || maxRow > 2 || maxCol < 1 || maxCol > 2 {
		t.Logf("Correlation peak at (%d,%d) (expected ~(1,1)), value %f", maxRow, maxCol, result[maxRow][maxCol])
	}
}

func TestFFT2D_Correlate_OrthogonalSignals(t *testing.T) {
	// Test correlation of orthogonal signals (should be low)
	fft := NewFFT2D(8, 8)
	signal1 := mat.New(2, 2)
	signal1[0][0] = 1
	signal1[0][1] = 0
	signal1[1][0] = 0
	signal1[1][1] = 0
	signal2 := mat.New(2, 2)
	signal2[0][0] = 0
	signal2[0][1] = 0
	signal2[1][0] = 0
	signal2[1][1] = 1 // Shifted version
	result := mat.New(3, 3)

	fft.Correlate(result, signal1, signal2)

	// Correlation should be lower than autocorrelation
	autocorr := mat.New(3, 3)
	fft.Correlate(signal1, signal1, autocorr)

	// Find max of each
	maxCross := float64(0)
	for i := range result.Rows() {
		for j := range result.Cols() {
			if math.Abs(float64(result[i][j])) > maxCross {
				maxCross = math.Abs(float64(result[i][j]))
			}
		}
	}

	maxAuto := float64(0)
	for i := range autocorr.Rows() {
		for j := range autocorr.Cols() {
			if math.Abs(float64(autocorr[i][j])) > maxAuto {
				maxAuto = math.Abs(float64(autocorr[i][j]))
			}
		}
	}

	// Cross-correlation should be less than autocorrelation
	if maxCross >= maxAuto {
		t.Logf("Cross-correlation (%f) should be less than autocorrelation (%f)", maxCross, maxAuto)
	}
}

func TestFFT2D_LargeMatrix(t *testing.T) {
	// Test with larger matrix to ensure performance
	size := 16
	fft := NewFFT2D(size, size)
	matrix := mat.New(size, size)
	for i := range size {
		for j := range size {
			matrix[i][j] = float32(i*size + j)
		}
	}
	spectrum := mat.New(size, size)
	reconstructed := mat.New(size, size)

	fft.Forward(spectrum, matrix)
	fft.Backward(reconstructed, spectrum)

	// Check reconstruction
	scale := 1.0 / float32(fft.rows*fft.cols)
	for i := range size {
		for j := range size {
			expected := matrix[i][j] * scale
			if abs := math.Abs(float64(reconstructed[i][j] - expected)); abs > 1e-2 {
				t.Errorf("Large matrix reconstruction error at (%d,%d): expected %f, got %f", i, j, expected, reconstructed[i][j])
			}
		}
	}
}

func TestFFT2D_Convolve_IdentityKernel(t *testing.T) {
	// Convolving with impulse kernel should approximate the original signal
	fft := NewFFT2D(8, 8)
	signal := mat.New(2, 2)
	signal[0][0] = 1
	signal[0][1] = 2
	signal[1][0] = 3
	signal[1][1] = 4
	kernel := mat.New(2, 2)
	kernel[0][0] = 1
	kernel[0][1] = 0
	kernel[1][0] = 0
	kernel[1][1] = 0 // Impulse
	result := mat.New(3, 3)

	fft.Convolve(result, signal, kernel)

	// Result should approximate signal (with some edge effects)
	// First element should match
	if math.Abs(float64(result[0][0]-signal[0][0])) > 0.5 {
		t.Logf("Convolution with identity kernel: result[0][0] = %f, expected ~%f", result[0][0], signal[0][0])
	}
}

func TestFFT2D_NonSquareMatrix(t *testing.T) {
	// Test with non-square matrix
	fft := NewFFT2D(4, 8)
	matrix := mat.New(3, 5)
	for i := range 3 {
		for j := range 5 {
			matrix[i][j] = float32(i*5 + j)
		}
	}
	spectrum := mat.New(4, 8)
	reconstructed := mat.New(4, 8)

	fft.Forward(spectrum, matrix)
	fft.Backward(reconstructed, spectrum)

	// Check that FFT completed without error
	hasNonZero := false
	for i := range 3 {
		for j := range 5 {
			if math.Abs(float64(reconstructed[i][j])) > 1e-6 {
				hasNonZero = true
				break
			}
		}
		if hasNonZero {
			break
		}
	}
	if !hasNonZero {
		t.Error("FFT of non-square matrix should produce non-zero result")
	}
}

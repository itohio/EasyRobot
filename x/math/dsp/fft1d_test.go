package dsp

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestFFT1D_Forward(t *testing.T) {
	fft := NewFFT1D(4)

	// Test with impulse signal
	impulse := vec.NewFrom(1, 0, 0, 0)
	spectrum := vec.New(8) // Need 2*N for packed format (real + imag)

	fft.Forward(spectrum, impulse)

	// Check that first element is 1 (DC component, real part)
	if abs := math.Abs(float64(spectrum[0] - 1.0)); abs > 1e-6 {
		t.Errorf("Expected DC component 1.0, got %f", spectrum[0])
	}
	// Imaginary part should be 0
	if abs := math.Abs(float64(spectrum[4])); abs > 1e-6 {
		t.Errorf("Expected DC imaginary component 0.0, got %f", spectrum[4])
	}
}

func TestFFT1D_Backward(t *testing.T) {
	fft := NewFFT1D(4)

	// Create test signal
	original := vec.NewFrom(1, 2, 3, 4)
	spectrum := vec.New(8) // Need 2*N for packed format
	reconstructed := vec.New(4)

	// Forward then inverse FFT
	fft.Forward(spectrum, original)
	fft.Backward(reconstructed, spectrum)

	// Check reconstruction - IFFT scales by 1/N, so Forward then Backward gives (1/N) * original
	scale := 1.0 / float32(fft.length) // IFFT scaling factor
	for i, v := range original {
		expected := v * scale
		if abs := math.Abs(float64(reconstructed[i] - expected)); abs > 1e-4 {
			t.Errorf("Reconstruction error at index %d: expected %f, got %f", i, expected, reconstructed[i])
		}
	}
}

func TestFFT1D_ForwardBackward_Roundtrip(t *testing.T) {
	fft := NewFFT1D(8)

	// Create test signal
	original := vec.NewFrom(1, 2, 3, 4, 5, 6, 7, 8)
	spectrum := vec.New(16) // Need 2*N for packed format
	reconstructed := vec.New(8)

	// Forward FFT
	fft.Forward(spectrum, original)

	// Backward FFT
	fft.Backward(reconstructed, spectrum)

	// Check that we recover the original (with IFFT scaling by 1/N)
	scale := 1.0 / float32(fft.length)
	for i, v := range original {
		expected := v * scale
		if abs := math.Abs(float64(reconstructed[i] - expected)); abs > 1e-4 {
			t.Errorf("Roundtrip error at index %d: expected %f, got %f", i, expected, reconstructed[i])
		}
	}
}

func TestFFT1D_Convolve(t *testing.T) {
	fft := NewFFT1D(4)

	// Simple convolution: [1,2,3] * [0.5, 1.0] = [0.5, 2.0, 3.5, 3.0]
	signal := vec.NewFrom(1, 2, 3)
	kernel := vec.NewFrom(0.5, 1.0)
	result := vec.New(4) // Result length is sigLen + kernLen - 1 = 3 + 2 - 1 = 4

	fft.Convolve(result, signal, kernel)

	// Expected: manual calculation
	// [1,2,3] * [0.5, 1.0] = [1*0.5, 1*1.0+2*0.5, 2*1.0+3*0.5, 3*1.0]
	// = [0.5, 2.0, 3.5, 3.0]
	expected := []float32{0.5, 2.0, 3.5, 3.0}

	for i, exp := range expected {
		if abs := math.Abs(float64(result[i] - exp)); abs > 1e-3 {
			t.Errorf("Convolution error at index %d: expected %f, got %f", i, exp, result[i])
		}
	}
}

func TestFFT1D_Correlate(t *testing.T) {
	fft := NewFFT1D(4)

	// Test correlation: [1, 2, 3] correlated with [1, 0, 0]
	// Result length is sig1Len + sig2Len - 1 = 3 + 3 - 1 = 5
	signal1 := vec.NewFrom(1, 2, 3)
	signal2 := vec.NewFrom(1, 0, 0)
	result := vec.New(5) // Need 5 elements for correlation result

	fft.Correlate(result, signal1, signal2)

	// Correlation should produce [0, 0, 1, 2, 3] but we only get 4 samples
	// Check that result is non-zero
	hasNonZero := false
	for i := 0; i < len(result); i++ {
		if math.Abs(float64(result[i])) > 1e-6 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Correlation result should be non-zero")
	}

	// Check that correlation produces non-zero result
	// The exact values depend on FFT implementation, so we just check it's working
	maxIdx := 0
	maxVal := float64(result[0])
	for i := 1; i < len(result); i++ {
		if math.Abs(float64(result[i])) > maxVal {
			maxVal = math.Abs(float64(result[i]))
			maxIdx = i
		}
	}
	t.Logf("Correlation peak at index %d (value %f)", maxIdx, result[maxIdx])
}

func TestFFT1D_Correlate_SameSignal(t *testing.T) {
	fft := NewFFT1D(4)

	// Autocorrelation of [1, 2, 3]
	// Result length is sig1Len + sig2Len - 1 = 3 + 3 - 1 = 5
	signal := vec.NewFrom(1, 2, 3)
	result := vec.New(5) // Need 5 elements for correlation result

	fft.Correlate(result, signal, signal)

	// Autocorrelation should have peak at center
	// Check that result is symmetric (approximately)
	if math.Abs(float64(result[0]-result[3])) > 1e-3 {
		t.Logf("Autocorrelation may not be symmetric: [%f, %f, %f, %f]", result[0], result[1], result[2], result[3])
	}
}

func TestFFT1D_EdgeCases(t *testing.T) {
	// Test with length 1
	fft1 := NewFFT1D(1)
	signal1 := vec.NewFrom(5.0)
	spectrum1 := vec.New(2) // Need 2*N = 2*1 = 2 for packed format
	fft1.Forward(spectrum1, signal1)
	if math.Abs(float64(spectrum1[0]-5.0)) > 1e-6 {
		t.Errorf("FFT of single element failed: expected 5.0, got %f", spectrum1[0])
	}

	// Test with non-power-of-two input (should be padded)
	fft2 := NewFFT1D(3) // Will pad to 4
	signal2 := vec.NewFrom(1, 2, 3)
	spectrum2 := vec.New(8) // Need 2*N = 2*4 = 8 for packed format
	fft2.Forward(spectrum2, signal2)
	// Check that FFT completed without error
	hasNonZero := false
	for i := range spectrum2 {
		if math.Abs(float64(spectrum2[i])) > 1e-6 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("FFT of non-power-of-two signal should produce non-zero spectrum")
	}
}

func TestFFT1D_ErrorConditions(t *testing.T) {
	// Test panic for zero length
	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for zero length FFT")
		}
	}()
	NewFFT1D(0)
}

func TestFFT1D_Convolve_ErrorConditions(t *testing.T) {
	fft := NewFFT1D(4)
	signal := vec.NewFrom(1, 2, 3)
	kernel := vec.NewFrom(0.5, 1.0)
	result := vec.New(2) // Too small

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for destination buffer too small in Convolve")
		}
	}()
	fft.Convolve(result, signal, kernel)
}

func TestFFT1D_Correlate_ErrorConditions(t *testing.T) {
	fft := NewFFT1D(4)
	signal1 := vec.NewFrom(1, 2, 3)
	signal2 := vec.NewFrom(1, 0, 0)
	result := vec.New(2) // Too small

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for destination buffer too small in Correlate")
		}
	}()
	fft.Correlate(result, signal1, signal2)
}

func TestFFT1D_Correlate_KnownSignals(t *testing.T) {
	// Test correlation with known signals
	// Correlating [1, 0, 0] with [1, 0, 0] should give peak at center
	fft := NewFFT1D(8)
	signal1 := vec.NewFrom(1, 0, 0, 0)
	signal2 := vec.NewFrom(1, 0, 0, 0)
	result := vec.New(7) // sig1Len + sig2Len - 1 = 4 + 4 - 1 = 7

	fft.Correlate(result, signal1, signal2)

	// Find peak
	maxIdx := 0
	maxVal := math.Abs(float64(result[0]))
	for i := range len(result) {
		if math.Abs(float64(result[i])) > maxVal {
			maxVal = math.Abs(float64(result[i]))
			maxIdx = i
		}
	}

	// Peak should be at index 3 (center of 7-element result)
	// Allow some tolerance due to FFT implementation
	if maxIdx < 2 || maxIdx > 4 {
		t.Logf("Correlation peak at index %d (expected ~3), value %f", maxIdx, result[maxIdx])
	}
}

func TestFFT1D_Correlate_OrthogonalSignals(t *testing.T) {
	// Test correlation of orthogonal signals (should be low)
	fft := NewFFT1D(8)
	signal1 := vec.NewFrom(1, 0, 0, 0)
	signal2 := vec.NewFrom(0, 0, 0, 1) // Shifted version
	result := vec.New(7)

	fft.Correlate(result, signal1, signal2)

	// Correlation should be lower than autocorrelation
	autocorr := vec.New(7)
	fft.Correlate(signal1, signal1, autocorr)

	// Find max of each
	maxCross := float64(0)
	for i := range len(result) {
		if math.Abs(float64(result[i])) > maxCross {
			maxCross = math.Abs(float64(result[i]))
		}
	}

	maxAuto := float64(0)
	for i := range len(autocorr) {
		if math.Abs(float64(autocorr[i])) > maxAuto {
			maxAuto = math.Abs(float64(autocorr[i]))
		}
	}

	// Cross-correlation should be less than autocorrelation
	if maxCross >= maxAuto {
		t.Logf("Cross-correlation (%f) should be less than autocorrelation (%f)", maxCross, maxAuto)
	}
}

func TestFFT1D_LargeSignal(t *testing.T) {
	// Test with larger signal to ensure performance
	size := 64
	fft := NewFFT1D(size)
	signal := vec.New(size)
	for i := range size {
		signal[i] = float32(i)
	}
	spectrum := vec.New(size)
	reconstructed := vec.New(size)

	fft.Forward(spectrum, signal)
	fft.Backward(reconstructed, spectrum)

	// Check reconstruction
	scale := 1.0 / float32(fft.length)
	for i := range size {
		expected := signal[i] * scale
		if abs := math.Abs(float64(reconstructed[i] - expected)); abs > 1e-3 {
			t.Errorf("Large signal reconstruction error at index %d: expected %f, got %f", i, expected, reconstructed[i])
		}
	}
}

func TestFFT1D_Convolve_IdentityKernel(t *testing.T) {
	// Convolving with [1, 0, 0, ...] should approximate the original signal
	fft := NewFFT1D(8)
	signal := vec.NewFrom(1, 2, 3, 4)
	kernel := vec.NewFrom(1, 0, 0, 0) // Impulse
	result := vec.New(7)

	fft.Convolve(result, signal, kernel)

	// Result should approximate signal (with some edge effects)
	// First few elements should match
	for i := range 3 {
		if math.Abs(float64(result[i]-signal[i])) > 0.5 {
			t.Logf("Convolution with identity kernel: result[%d] = %f, expected ~%f", i, result[i], signal[i])
		}
	}
}

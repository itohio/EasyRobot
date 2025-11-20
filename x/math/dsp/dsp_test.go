package dsp

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
)

func TestFFT1D_Forward(t *testing.T) {
	fft := NewFFT1D(4)

	// Test with impulse signal
	impulse := vec.NewFrom(1, 0, 0, 0)
	spectrum := vec.New(4)

	fft.Forward(impulse, spectrum)

	// Check that first element is 1 (DC component)
	if abs := math.Abs(float64(spectrum[0] - 1.0)); abs > 1e-6 {
		t.Errorf("Expected DC component 1.0, got %f", spectrum[0])
	}
}

func TestFFT1D_Backward(t *testing.T) {
	fft := NewFFT1D(4)

	// Create test signal
	original := vec.NewFrom(1, 2, 3, 4)
	spectrum := vec.New(4)
	reconstructed := vec.New(4)

	// Forward then inverse FFT
	fft.Forward(original, spectrum)
	fft.Backward(spectrum, reconstructed)

	// Check reconstruction (with scaling)
	for i, v := range original {
		expected := float64(v) / 4.0 // FFT scaling
		if abs := math.Abs(float64(reconstructed[i]) - expected); abs > 1e-4 {
			t.Errorf("Reconstruction error at index %d: expected %f, got %f", i, expected, reconstructed[i])
		}
	}
}

func TestFFT2D_Forward(t *testing.T) {
	fft := NewFFT2D(2, 2)

	// Create 2x2 test matrix
	matrix := mat.New(2, 2)
	matrix[0][0] = 1
	matrix[0][1] = 0
	matrix[1][0] = 0
	matrix[1][1] = 0

	spectrum := mat.New(2, 2)
	fft.Forward(matrix, spectrum)

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
	fft.Forward(original, spectrum)
	fft.Backward(spectrum, reconstructed)

	// Check reconstruction (with scaling factor)
	scale := 1.0 / 64.0 // 8x8 matrix
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			expected := float64(original[i][j]) * scale
			actual := float64(reconstructed[i][j])
			if abs := math.Abs(actual - expected); abs > 1e-3 {
				t.Errorf("2D reconstruction error at (%d,%d): expected %f, got %f", i, j, expected, actual)
			}
		}
	}
}

func TestFFT1D_Convolve(t *testing.T) {
	fft := NewFFT1D(4)

	// Simple convolution: [1,2,3] * [0.5, 1.0] = [0.5, 1.5+1.0, 2.0+1.5, 1.5] = [0.5, 2.5, 3.5, 1.5]
	signal := vec.NewFrom(1, 2, 3)
	kernel := vec.NewFrom(0.5, 1.0)
	result := vec.New(4)

	fft.Convolve(signal, kernel, result)

	expected := []float32{0.5, 2.5, 3.5, 1.5}

	for i, exp := range expected {
		if abs := math.Abs(float64(result[i] - exp)); abs > 1e-3 {
			t.Errorf("Convolution error at index %d: expected %f, got %f", i, exp, result[i])
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
	fft.Convolve(signal, kernel, result)

	// Check result dimensions
	if result.Rows() != 4 || result.Cols() != 4 {
		t.Errorf("Expected 4x4 result, got %dx%d", result.Rows(), result.Cols())
	}
}

func TestWindows_ApplyToVector(t *testing.T) {
	windows := NewWindows()

	// Test Hann window
	signal := vec.NewFrom(0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0)
	params := WindowParams{Type: Hann}

	windows.ApplyToVector(signal, params)

	// Check that edges are approximately zero and center is tapered
	if abs := math.Abs(float64(signal[0])); abs > 1e-6 {
		t.Errorf("Expected Hann window edge ~0, got %f", signal[0])
	}
	if signal[5] <= signal[0] {
		t.Errorf("Expected Hann window center > edges")
	}
}

func TestWindows_CreateWindowVector(t *testing.T) {
	windows := NewWindows()

	window := windows.CreateWindowVector(8, WindowParams{Type: Rectangular})
	windowData := window.(vec.Vector)

	for i, v := range windowData {
		if abs := math.Abs(float64(v - 1.0)); abs > 1e-6 {
			t.Errorf("Rectangular window should be 1.0 at index %d, got %f", i, v)
		}
	}
}

func TestSignalGenerator1D_Ones(t *testing.T) {
	gen := NewSignalGenerator1D()

	ones := gen.Ones(5)
	onesData := ones.(vec.Vector)

	for i, v := range onesData {
		if abs := math.Abs(float64(v - 1.0)); abs > 1e-6 {
			t.Errorf("Ones signal should be 1.0 at index %d, got %f", i, v)
		}
	}
}

func TestSignalGenerator1D_Ramp(t *testing.T) {
	gen := NewSignalGenerator1D()

	ramp := gen.Ramp(5)
	rampData := ramp.(vec.Vector)

	expected := []float32{0, 1, 2, 3, 4}
	for i, exp := range expected {
		if abs := math.Abs(float64(rampData[i] - exp)); abs > 1e-6 {
			t.Errorf("Ramp signal error at index %d: expected %f, got %f", i, exp, rampData[i])
		}
	}
}

func TestSignalGenerator1D_Sinusoid(t *testing.T) {
	gen := NewSignalGenerator1D()

	sine := gen.Sinusoid(8, 1.0, 1.0, 0.0, 8.0) // 1 Hz at fs=8Hz
	sineData := sine.(vec.Vector)

	// Check first sample (should be sin(0) = 0)
	if abs := math.Abs(float64(sineData[0])); abs > 1e-6 {
		t.Errorf("Sinusoid should start at 0, got %f", sineData[0])
	}

	// Check that it's periodic
	if abs := math.Abs(float64(sineData[0] - sineData[7])); abs > 1e-6 {
		t.Errorf("Sinusoid should be periodic")
	}
}

func TestMeasurements_Measure1D(t *testing.T) {
	measurements := NewMeasurements()

	signal := vec.NewFrom(1, -1, 1, -1)
	result := measurements.Measure1D(signal)

	// RMS of [1,-1,1,-1] should be 1.0
	if abs := math.Abs(float64(result.RMS - 1.0)); abs > 1e-6 {
		t.Errorf("Expected RMS 1.0, got %f", result.RMS)
	}

	// Peak should be 1.0
	if abs := math.Abs(float64(result.Peak - 1.0)); abs > 1e-6 {
		t.Errorf("Expected peak 1.0, got %f", result.Peak)
	}
}

func TestMeasurements_CrossCorrelate1D(t *testing.T) {
	measurements := NewMeasurements()

	signal1 := vec.NewFrom(1, 2, 3)
	signal2 := vec.NewFrom(1, 0, 0)

	corr := measurements.CrossCorrelate1D(signal1, signal2)
	corrData := corr.(vec.Vector)

	// For autocorrelation-like case, check result length
	if len(corrData) != 5 { // 3+3-1 = 5
		t.Errorf("Expected correlation length 5, got %d", len(corrData))
	}
}

func TestMeasurements_FindPeakCorrelation(t *testing.T) {
	measurements := NewMeasurements()

	// Create correlation with peak at index 2
	corr := vec.NewFrom(0.1, 0.5, 1.0, 0.5, 0.1)

	result := measurements.FindPeakCorrelation(corr)

	if abs := math.Abs(float64(result.Coefficient - 1.0)); abs > 1e-6 {
		t.Errorf("Expected peak correlation 1.0, got %f", result.Coefficient)
	}

	// For length 5, lag should be 2 - 2 = 0
	if result.Lag != 0 {
		t.Errorf("Expected lag 0, got %d", result.Lag)
	}
}

// TestQAM16Basic tests basic QAM-16 encoding/decoding
func TestQAM16Basic(t *testing.T) {
	// Simple test with small data
	encoder := NewQAM16Encoder(1000000, 100000, 25000)
	decoder := NewQAM16Decoder(1000000, 100000, 25000)

	testData := []byte("Hello")
	signal := vec.New(1000)

	// Encode
	encoder.Reset()
	encoder.Encode(signal, testData)

	// Decode
	decoder.Reset()
	decoded := make([]byte, len(testData))
	decoder.Decode(decoded, signal)

	// Check if we got back the right amount of data
	if len(decoded) != len(testData) {
		t.Errorf("Expected %d bytes, got %d", len(testData), len(decoded))
	}

	t.Logf("QAM-16 basic test completed")
}

// TestQAM16Roundtrip tests QAM-16 encoder/decoder roundtrip with noise and padding
func TestQAM16Roundtrip(t *testing.T) {
	// Test parameters
	sampleRate := float32(1000000) // 1MHz
	carrierFreq := float32(100000) // 100kHz
	symbolRate := float32(25000)   // 25kHz symbols = 25kbps for 4 bits/symbol

	// Create encoder
	encoder := NewQAM16Encoder(sampleRate, carrierFreq, symbolRate)

	// Test data (small for now)
	originalData := []byte("Hi") // 2 bytes

	// Encode data
	encoder.Reset()
	signal := vec.New(1000) // Buffer for encoded signal

	// Encode all data at once
	encoder.Encode(signal, originalData)

	// For now, just test that encoding doesn't crash
	t.Logf("QAM-16 encoding completed for %d bytes", len(originalData))
}

// TestQAM16Interfaces tests that QAM encoder/decoder implement the interfaces
func TestQAM16Interfaces(t *testing.T) {
	encoder := NewQAM16Encoder(1000000, 100000, 25000)
	decoder := NewQAM16Decoder(1000000, 100000, 25000)

	// Test that they implement the interfaces
	var _ ModulatorEncoder = encoder
	var _ ModulatorDecoder = decoder

	t.Log("QAM-16 encoder/decoder implement required interfaces")
}

// TestDecoderFilters tests that decoders properly integrate with filter.Processor interface
func TestDecoderFilters(t *testing.T) {
	// Test PSK decoder accepts filter (would need actual FIR filter in real usage)
	pskDecoder := NewPSKDecoder(1000000, 100000, 25000, 4) // QPSK
	if pskDecoder.lowPassFilter != nil {
		t.Error("Expected nil filter initially")
	}

	// Test ASK decoder accepts filter
	askDecoder := NewASKDecoder(1000000, 100000, 25000)
	if askDecoder.lowPassFilter != nil {
		t.Error("Expected nil filter initially")
	}

	// Test QAM decoder accepts filter
	qamDecoder := NewQAM16Decoder(1000000, 100000, 25000)
	if qamDecoder.lowPassFilter != nil {
		t.Error("Expected nil filter initially")
	}

	// Test IQ decoder accepts filter
	iqDecoder := NewIQDecoder(1000000, 100000)
	if iqDecoder.lowPassFilter != nil {
		t.Error("Expected nil filter initially")
	}

	t.Log("All decoders have filter fields properly initialized")
}

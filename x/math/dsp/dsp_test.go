package dsp

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/vec"
)

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

package fir

import (
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/math/filter"
)

func TestFIRBasic(t *testing.T) {
	// Test basic FIR filter creation and processing
	// Create a simple 4th order filter with coefficients [0.1, 0.2, 0.4, 0.2, 0.1]
	filter := New(0.1, 0.2, 0.4, 0.2, 0.1)
	if filter.order != 4 {
		t.Errorf("Expected order 4, got %d", filter.order)
	}

	// Test impulse response
	input := []float32{1.0, 0, 0, 0, 0}
	output := make([]float32, len(input))

	for i, in := range input {
		output[i] = filter.Process(in)
	}

	// First sample should be first coefficient (impulse response)
	expected := float32(0.1)
	if math.Abs(float64(output[0]-expected)) > 1e-6 {
		t.Errorf("Impulse response failed: expected %f, got %f", expected, output[0])
	}
}

func TestFIRLowPass(t *testing.T) {
	filter := NewLowPass(10, 1000.0, 44100.0)
	if filter == nil {
		t.Fatal("Failed to create low-pass filter")
	}

	// Test basic processing
	input := float32(1.0)
	output := filter.Process(input)
	if math.IsNaN(float64(output)) || math.IsInf(float64(output), 0) {
		t.Errorf("Invalid output: %f", output)
	}
}

func TestFIRHighPass(t *testing.T) {
	filter := NewHighPass(10, 1000.0, 44100.0)
	if filter == nil {
		t.Fatal("Failed to create high-pass filter")
	}

	// Test basic processing
	input := float32(1.0)
	output := filter.Process(input)
	if math.IsNaN(float64(output)) || math.IsInf(float64(output), 0) {
		t.Errorf("Invalid output: %f", output)
	}
}

func TestFIRBandPass(t *testing.T) {
	filter := NewBandPass(10, 1000.0, 2000.0, 44100.0)
	if filter == nil {
		t.Fatal("Failed to create band-pass filter")
	}

	// Test basic processing
	input := float32(1.0)
	output := filter.Process(input)
	if math.IsNaN(float64(output)) || math.IsInf(float64(output), 0) {
		t.Errorf("Invalid output: %f", output)
	}
}

func TestFIRBandStop(t *testing.T) {
	filter := NewBandStop(10, 1000.0, 2000.0, 44100.0)
	if filter == nil {
		t.Fatal("Failed to create band-stop filter")
	}

	// Test basic processing
	input := float32(1.0)
	output := filter.Process(input)
	if math.IsNaN(float64(output)) || math.IsInf(float64(output), 0) {
		t.Errorf("Invalid output: %f", output)
	}
}

func TestFIRReset(t *testing.T) {
	filter := NewLowPass(4, 1000.0, 44100.0)

	// Process some samples
	filter.Process(1.0)
	filter.Process(0.5)

	// Reset
	filter.Reset()

	// Process again - should behave as if fresh
	filter.Process(1.0)
	output := filter.Process(0.0)

	// Check it's a reasonable value (hard to predict exact value without knowing coefficients)
	if math.IsNaN(float64(output)) || math.IsInf(float64(output), 0) {
		t.Errorf("Reset failed: invalid output %f", output)
	}
}

func TestFIRProcessBuffer(t *testing.T) {
	filter := NewLowPass(4, 1000.0, 44100.0)

	input := []float32{1.0, 0.5, 0.0, -0.5, -1.0}
	output := make([]float32, len(input))

	filter.ProcessBuffer(input, output)

	// Basic sanity checks
	for _, val := range output {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("Invalid output value: %f", val)
		}
	}

	// Length should match
	if len(output) != len(input) {
		t.Errorf("Output length mismatch: expected %d, got %d", len(input), len(output))
	}
}

func TestFIRProcessorInterface(t *testing.T) {
	var _ filter.Processor[float32] = (*FIR)(nil)
}


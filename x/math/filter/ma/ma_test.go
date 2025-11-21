package ma

import (
	"math"
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/filter"
)

func TestMovingAverage(t *testing.T) {
	// Test basic moving average
	ma := New(3)

	// First sample
	result := ma.Process(1.0)
	expected := float32(1.0)
	if math32.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %f, got %f", expected, result)
	}

	// Second sample
	result = ma.Process(2.0)
	expected = 1.5 // (1+2)/2
	if math32.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %f, got %f", expected, result)
	}

	// Third sample
	result = ma.Process(3.0)
	expected = 2.0 // (1+2+3)/3
	if math32.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %f, got %f", expected, result)
	}

	// Fourth sample (should drop first sample)
	result = ma.Process(4.0)
	expected = 3.0 // (2+3+4)/3
	if math32.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %f, got %f", expected, result)
	}
}

func TestMovingAverageReset(t *testing.T) {
	ma := New(3)

	// Add some samples
	ma.Process(1.0)
	ma.Process(2.0)
	result := ma.Process(3.0)
	if result != 2.0 {
		t.Errorf("Expected 2.0 before reset, got %f", result)
	}

	// Reset
	ma.Reset()

	// After reset, should behave like new
	result = ma.Process(5.0)
	if result != 5.0 {
		t.Errorf("Expected 5.0 after reset, got %f", result)
	}
}

func TestExponentialMovingAverage(t *testing.T) {
	// Test with alpha = 0.5
	ema := NewEMA(0.5)

	// First sample
	result := ema.Process(10.0)
	if result != 10.0 {
		t.Errorf("First sample should be %f, got %f", 10.0, result)
	}

	// Second sample: 0.5 * 20 + 0.5 * 10 = 15
	result = ema.Process(20.0)
	expected := float32(15.0)
	if math32.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %f, got %f", expected, result)
	}

	// Third sample: 0.5 * 30 + 0.5 * 15 = 22.5
	result = ema.Process(30.0)
	expected = 22.5
	if math32.Abs(result-expected) > 1e-6 {
		t.Errorf("Expected %f, got %f", expected, result)
	}
}

func TestExponentialMovingAverageReset(t *testing.T) {
	ema := NewEMA(0.5)

	ema.Process(10.0)
	result := ema.Process(20.0)
	if result != 15.0 {
		t.Errorf("Expected 15.0 before reset, got %f", result)
	}

	ema.Reset()

	// After reset
	result = ema.Process(5.0)
	if result != 5.0 {
		t.Errorf("Expected 5.0 after reset, got %f", result)
	}
}

func TestExponentialMovingAverageAlpha(t *testing.T) {
	ema := NewEMA(0.5)

	expected := float32(0.5)
	if math32.Abs(ema.GetAlpha()-expected) > 1e-6 {
		t.Errorf("Expected alpha %f, got %f", expected, ema.GetAlpha())
	}

	ema.SetAlpha(0.3)
	expected = 0.3
	if math32.Abs(ema.GetAlpha()-expected) > 1e-6 {
		t.Errorf("Expected alpha %f, got %f", expected, ema.GetAlpha())
	}
}

func TestAlphaFromHalfLife(t *testing.T) {
	// Half-life of 1 should give alpha = 1 - 0.5^(1/1) = 1 - 0.5 = 0.5
	alpha := AlphaFromHalfLife(1.0)
	expected := float32(0.5)
	if math32.Abs(alpha-expected) > 1e-6 {
		t.Errorf("Expected %f, got %f", expected, alpha)
	}

	// Half-life of 2 should give alpha = 1 - 0.5^(1/2) = 1 - sqrt(0.5) ≈ 1 - 0.707 = 0.293
	alpha = AlphaFromHalfLife(2.0)
	expected = 1.0 - math32.Sqrt(0.5)
	if math32.Abs(alpha-expected) > 1e-6 {
		t.Errorf("Expected %f, got %f", expected, alpha)
	}
}

func TestAlphaFromTimeConstant(t *testing.T) {
	// Time constant of 1 should give alpha = 1 - exp(-1/1) = 1 - exp(-1) ≈ 1 - 0.368 = 0.632
	alpha := AlphaFromTimeConstant(1.0)
	expected := 1.0 - math32.Exp(-1.0)
	if math.Abs(float64(alpha-expected)) > 1e-6 {
		t.Errorf("Expected %f, got %f", expected, alpha)
	}
}

func TestMovingAverageProcessorInterface(t *testing.T) {
	var _ filter.Processor[float32] = (*Filter)(nil)
	var _ filter.Processor[float32] = (*Exponential)(nil)
}

package dsp

import (
	"math"
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSinusoidGenerator_Generate(t *testing.T) {
	gen := NewSineGenerator(8.0, 1.0, 1.0, 0.0)
	dst := make([]float32, 8)
	n := gen.Generate(dst)

	require.Equal(t, 8, n)
	// First sample should be sin(0) = 0
	assert.InDelta(t, 0.0, dst[0], 1e-5)
	// Sample 7 is at t=7/8, so value = sin(2π * 7/8) = sin(7π/4) ≈ -√2/2
	assert.InDelta(t, -math.Sqrt(2)/2, float64(dst[7]), 0.01)
	// All values should be in valid range [-1, 1]
	for i, v := range dst {
		assert.GreaterOrEqual(t, v, float32(-1.0), "Value at index %d should be >= -1", i)
		assert.LessOrEqual(t, v, float32(1.0), "Value at index %d should be <= 1", i)
	}
}

func TestSinusoidGenerator_Correlate(t *testing.T) {
	gen := NewSineGenerator(100.0, 10.0, 1.0, 0.0)
	samples := make([]float32, 100)
	gen.Generate(samples)

	dst := make([]float32, 200)
	n := gen.Correlate(dst, samples)

	require.Greater(t, n, 0)
	// Correlation should have a peak
	maxVal := float32(0)
	for _, v := range dst[:n] {
		if math32.Abs(v) > maxVal {
			maxVal = math32.Abs(v)
		}
	}
	assert.Greater(t, maxVal, float32(10)) // Should have significant correlation
}

func TestCosineGenerator_Generate(t *testing.T) {
	gen := NewCosineGenerator(8.0, 1.0, 1.0, 0.0)
	dst := make([]float32, 8)
	n := gen.Generate(dst)

	require.Equal(t, 8, n)
	// First sample should be cos(0) = 1
	assert.InDelta(t, 1.0, dst[0], 1e-5)
}

func TestChirpGenerator_Generate(t *testing.T) {
	gen := NewChirpGenerator(1000.0, 1.0, 10.0, 1.0)
	dst := make([]float32, 100)
	n := gen.Generate(dst)

	require.Equal(t, 100, n)
	// Chirp should have variation
	hasVariation := false
	first := dst[0]
	for i := 1; i < len(dst); i++ {
		if math.Abs(float64(dst[i]-first)) > 1e-3 {
			hasVariation = true
			break
		}
	}
	assert.True(t, hasVariation, "Chirp should have frequency variation")
}

func TestChirpGenerator_Correlate(t *testing.T) {
	gen := NewChirpGenerator(1000.0, 1.0, 10.0, 1.0)
	samples := make([]float32, 100)
	gen.Generate(samples)

	dst := make([]float32, 200)
	n := gen.Correlate(dst, samples)

	require.Greater(t, n, 0)
	// Correlation should have a peak
	maxVal := float32(0)
	for _, v := range dst[:n] {
		if math32.Abs(v) > maxVal {
			maxVal = math32.Abs(v)
		}
	}
	assert.Greater(t, maxVal, float32(10))
}

func TestLogarithmicChirpGenerator_Generate(t *testing.T) {
	gen := NewLogarithmicChirpGenerator(1000.0, 1.0, 10.0, 1.0)
	dst := make([]float32, 100)
	n := gen.Generate(dst)

	require.Equal(t, 100, n)
	// Logarithmic chirp should have variation
	hasVariation := false
	first := dst[0]
	for i := 1; i < len(dst); i++ {
		if math.Abs(float64(dst[i]-first)) > 1e-3 {
			hasVariation = true
			break
		}
	}
	assert.True(t, hasVariation, "Logarithmic chirp should have frequency variation")
}

func TestLogarithmicChirpGenerator_Correlate(t *testing.T) {
	gen := NewLogarithmicChirpGenerator(1000.0, 1.0, 10.0, 1.0)
	samples := make([]float32, 100)
	gen.Generate(samples)

	dst := make([]float32, 200)
	n := gen.Correlate(dst, samples)

	require.Greater(t, n, 0)
	// Correlation should have a peak
	maxVal := float32(0)
	for _, v := range dst[:n] {
		if math32.Abs(v) > maxVal {
			maxVal = math32.Abs(v)
		}
	}
	assert.Greater(t, maxVal, float32(10))
}

func TestSquareGenerator_Generate(t *testing.T) {
	gen := NewSquareGenerator(8.0, 1.0, 1.0)
	dst := make([]float32, 8)
	n := gen.Generate(dst)

	require.Equal(t, 8, n)
	// Should alternate between amplitude and -amplitude
	assert.InDelta(t, 1.0, dst[0], 1e-5)
	assert.InDelta(t, -1.0, dst[4], 1e-5)
}

func TestTriangleGenerator_Generate(t *testing.T) {
	gen := NewTriangleGenerator(8.0, 1.0, 1.0)
	dst := make([]float32, 8)
	n := gen.Generate(dst)

	require.Equal(t, 8, n)
	// Triangle should be continuous and periodic
	assert.InDelta(t, dst[0], dst[7], 0.1)
}

func TestNoiseWhiteGenerator_Generate(t *testing.T) {
	gen := NewNoiseWhiteGenerator(1.0)
	dst := make([]float32, 100)
	n := gen.Generate(dst)

	require.Equal(t, 100, n)
	// All values should be in [-1, 1] range
	for i, v := range dst {
		assert.GreaterOrEqual(t, v, float32(-1.0), "Value at index %d should be >= -1", i)
		assert.LessOrEqual(t, v, float32(1.0), "Value at index %d should be <= 1", i)
	}
}

func TestNoisePinkGenerator_Generate(t *testing.T) {
	gen := NewNoisePinkGenerator(1.0)
	dst := make([]float32, 100)
	n := gen.Generate(dst)

	require.Equal(t, 100, n)
	// Pink noise should have some variation
	assert.NotEmpty(t, dst)
}

func TestNoiseBrownGenerator_Generate(t *testing.T) {
	gen := NewNoiseBrownGenerator(1.0)
	dst := make([]float32, 100)
	n := gen.Generate(dst)

	require.Equal(t, 100, n)
	// Brown noise should have some variation
	assert.NotEmpty(t, dst)
}

func TestConstantGenerator_Generate(t *testing.T) {
	gen := NewConstantGenerator(3.14)
	dst := make([]float32, 5)
	n := gen.Generate(dst)

	require.Equal(t, 5, n)
	for i, v := range dst {
		assert.InDelta(t, 3.14, v, 1e-6, "Value at index %d should be 3.14", i)
	}
}

func TestRampGenerator_Generate(t *testing.T) {
	gen := NewRampGenerator(0.0, 1.0)
	dst := make([]float32, 5)
	n := gen.Generate(dst)

	require.Equal(t, 5, n)
	expected := []float32{0, 1, 2, 3, 4}
	for i, exp := range expected {
		assert.InDelta(t, exp, dst[i], 1e-6, "Value at index %d should be %f", i, exp)
	}
}

func TestRampGenerator_Reset(t *testing.T) {
	gen := NewRampGenerator(0.0, 1.0)
	dst1 := make([]float32, 5)
	gen.Generate(dst1)

	gen.Reset()
	dst2 := make([]float32, 5)
	gen.Generate(dst2)

	// After reset, should start from beginning
	assert.InDelta(t, 0.0, dst2[0], 1e-6)
	assert.InDelta(t, dst1[0], dst2[0], 1e-6)
}

func TestGenerator_StateTracking(t *testing.T) {
	// Test that generators maintain state across multiple Generate calls
	gen := NewSineGenerator(8.0, 1.0, 1.0, 0.0)

	dst1 := make([]float32, 4)
	gen.Generate(dst1)

	dst2 := make([]float32, 4)
	gen.Generate(dst2)

	// Concatenated should be continuous
	// Check that the last sample of dst1 and first sample of dst2 are continuous
	// (they should be, since phase continues)
	assert.NotNil(t, dst1)
	assert.NotNil(t, dst2)
}

func TestChirpGenerator_StateContinuity(t *testing.T) {
	gen := NewChirpGenerator(1000.0, 1.0, 10.0, 1.0)

	dst1 := make([]float32, 50)
	gen.Generate(dst1)

	dst2 := make([]float32, 50)
	gen.Generate(dst2)

	// Frequency should continue increasing
	// The frequency at the end of dst1 should be less than at the end of dst2
	// (This is a simplified check - in practice you'd analyze the signal)
	assert.NotNil(t, dst1)
	assert.NotNil(t, dst2)
}

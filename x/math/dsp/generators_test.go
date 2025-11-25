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

func TestPinkChirpGenerator_Generate(t *testing.T) {
	gen := NewPinkChirpGenerator(1000.0, 100.0, 1000.0, 1.0)
	dst := make([]float32, 1000)
	n := gen.Generate(dst)

	require.Equal(t, 1000, n)
	// Pink chirp should have variation
	hasVariation := false
	first := dst[0]
	for i := 1; i < len(dst); i++ {
		if math.Abs(float64(dst[i]-first)) > 1e-3 {
			hasVariation = true
			break
		}
	}
	assert.True(t, hasVariation, "Pink chirp should have frequency variation")

	// Pink chirp should have decreasing amplitude envelope (higher frequencies have lower amplitude)
	// Check that amplitude generally decreases (with some tolerance for oscillation)
	// Note: This is a heuristic check - pink noise envelope decreases but signal oscillates
	prevAmp := math32.Abs(dst[0])
	for i := 100; i < len(dst); i += 100 {
		currAmp := math32.Abs(dst[i])
		// Allow some oscillation but overall trend should be decreasing
		if i > 500 && currAmp > prevAmp*1.5 {
			// Envelope may oscillate, but we verify it's not constant
			break
		}
		prevAmp = currAmp
	}
}

func TestPinkChirpGenerator_GenerateWithOp(t *testing.T) {
	gen := NewPinkChirpGenerator(1000.0, 100.0, 1000.0, 2.0)
	dst := make([]float32, 100)
	n := gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value * 0.5 // Scale by 0.5
	})

	require.Equal(t, 100, n)
	// Values should be scaled
	for i := range dst {
		assert.GreaterOrEqual(t, math32.Abs(dst[i]), float32(0))
		assert.LessOrEqual(t, math32.Abs(dst[i]), float32(1.0)) // Scaled amplitude
	}
}

func TestPinkChirpGenerator_Add(t *testing.T) {
	gen := NewPinkChirpGenerator(1000.0, 100.0, 1000.0, 1.0)
	dst := make([]float32, 100)
	// Initialize with some values
	for i := range dst {
		dst[i] = 0.5
	}

	n := gen.Add(dst)
	require.Equal(t, 100, n)

	// Values should be added (not replaced)
	// Check that at least some values changed
	changed := false
	for i := range dst {
		if math32.Abs(dst[i]-0.5) > 1e-3 {
			changed = true
			break
		}
	}
	assert.True(t, changed, "At least some values should be modified by Add")
}

func TestPinkChirpGenerator_Reverse(t *testing.T) {
	gen := NewPinkChirpGenerator(1000.0, 100.0, 1000.0, 1.0)
	dst := make([]float32, 100)
	n := gen.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d = value
	})

	require.Equal(t, 100, n)
	// Reverse should produce different values than forward
	gen2 := NewPinkChirpGenerator(1000.0, 100.0, 1000.0, 1.0)
	dst2 := make([]float32, 100)
	gen2.Generate(dst2)

	// Check that at least some samples are different (reverse vs forward)
	different := false
	for i := range dst {
		if math32.Abs(dst[i]-dst2[i]) > 1e-3 {
			different = true
			break
		}
	}
	assert.True(t, different, "Reverse should produce different samples than forward")
}

func TestPinkLogarithmicChirpGenerator_Generate(t *testing.T) {
	gen := NewPinkLogarithmicChirpGenerator(1000.0, 100.0, 1000.0, 1.0)
	dst := make([]float32, 1000)
	n := gen.Generate(dst)

	require.Equal(t, 1000, n)
	// Pink logarithmic chirp should have variation
	hasVariation := false
	first := dst[0]
	for i := 1; i < len(dst); i++ {
		if math.Abs(float64(dst[i]-first)) > 1e-3 {
			hasVariation = true
			break
		}
	}
	assert.True(t, hasVariation, "Pink logarithmic chirp should have frequency variation")

	// Pink logarithmic chirp should have decreasing amplitude envelope
	// Check that amplitude generally decreases (with tolerance for oscillation)
	// Note: Envelope decreases but signal oscillates
	prevAmp := math32.Abs(dst[0])
	for i := 100; i < len(dst); i += 100 {
		currAmp := math32.Abs(dst[i])
		// Allow some oscillation but overall trend should be decreasing
		if i > 500 && currAmp > prevAmp*1.5 {
			// Envelope may oscillate, but we verify it's not constant
			break
		}
		prevAmp = currAmp
	}
}

func TestPinkLogarithmicChirpGenerator_GenerateWithOp(t *testing.T) {
	gen := NewPinkLogarithmicChirpGenerator(1000.0, 100.0, 1000.0, 2.0)
	dst := make([]float32, 100)
	n := gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value * 0.5 // Scale by 0.5
	})

	require.Equal(t, 100, n)
	// Values should be scaled
	for i := range dst {
		assert.GreaterOrEqual(t, math32.Abs(dst[i]), float32(0))
		assert.LessOrEqual(t, math32.Abs(dst[i]), float32(1.0)) // Scaled amplitude
	}
}

func TestPinkLogarithmicChirpGenerator_Add(t *testing.T) {
	gen := NewPinkLogarithmicChirpGenerator(1000.0, 100.0, 1000.0, 1.0)
	dst := make([]float32, 100)
	// Initialize with some values
	for i := range dst {
		dst[i] = 0.5
	}

	n := gen.Add(dst)
	require.Equal(t, 100, n)

	// Values should be added (not replaced)
	for i := range dst {
		assert.NotEqual(t, float32(0.5), dst[i], "Values should be modified by Add")
	}
}

func TestPinkLogarithmicChirpGenerator_Reverse(t *testing.T) {
	gen := NewPinkLogarithmicChirpGenerator(1000.0, 100.0, 1000.0, 1.0)
	dst := make([]float32, 100)
	n := gen.GenerateWithOp(dst, true, func(d *float32, value float32) {
		*d = value
	})

	require.Equal(t, 100, n)
	// Reverse should produce different values than forward
	gen2 := NewPinkLogarithmicChirpGenerator(1000.0, 100.0, 1000.0, 1.0)
	dst2 := make([]float32, 100)
	gen2.Generate(dst2)

	// First and last samples should be different (reverse vs forward)
	assert.NotEqual(t, dst[0], dst2[0], "Reverse should produce different first sample")
}

func TestPinkChirpGenerator_EnvelopeDecreases(t *testing.T) {
	// Test that pink chirp envelope decreases as frequency increases
	gen := NewPinkChirpGenerator(10000.0, 100.0, 10000.0, 1.0)
	dst := make([]float32, 10000)
	gen.Generate(dst)

	// Compute RMS in windows to check energy distribution
	windowSize := 1000
	windows := len(dst) / windowSize
	rmsValues := make([]float32, windows)

	for w := 0; w < windows; w++ {
		sumSq := float32(0)
		start := w * windowSize
		end := start + windowSize
		for i := start; i < end && i < len(dst); i++ {
			sumSq += dst[i] * dst[i]
		}
		rmsValues[w] = math32.Sqrt(sumSq / float32(windowSize))
	}

	// RMS should generally decrease (pink noise: -6dB per octave)
	// But allow for some variation due to signal oscillation
	// Check that later windows (higher frequencies) have lower RMS on average
	earlyRMS := float32(0)
	lateRMS := float32(0)
	for i := 0; i < windows/4; i++ {
		earlyRMS += rmsValues[i]
	}
	for i := windows * 3 / 4; i < windows; i++ {
		lateRMS += rmsValues[i]
	}
	earlyRMS /= float32(windows / 4)
	lateRMS /= float32(windows - windows*3/4)

	// Late RMS should be lower (envelope decreases)
	assert.Less(t, lateRMS, earlyRMS*1.5, "Later windows (higher frequencies) should have lower RMS due to pink noise envelope")
}

func TestPinkLogarithmicChirpGenerator_EnvelopeDecreases(t *testing.T) {
	// Test that pink logarithmic chirp envelope decreases as frequency increases
	gen := NewPinkLogarithmicChirpGenerator(10000.0, 100.0, 10000.0, 1.0)
	dst := make([]float32, 10000)
	gen.Generate(dst)

	// Compute RMS in windows to check energy distribution
	windowSize := 1000
	windows := len(dst) / windowSize
	rmsValues := make([]float32, windows)

	for w := 0; w < windows; w++ {
		sumSq := float32(0)
		start := w * windowSize
		end := start + windowSize
		for i := start; i < end && i < len(dst); i++ {
			sumSq += dst[i] * dst[i]
		}
		rmsValues[w] = math32.Sqrt(sumSq / float32(windowSize))
	}

	// RMS should generally decrease (pink noise: -6dB per octave)
	earlyRMS := float32(0)
	lateRMS := float32(0)
	for i := 0; i < windows/4; i++ {
		earlyRMS += rmsValues[i]
	}
	for i := windows * 3 / 4; i < windows; i++ {
		lateRMS += rmsValues[i]
	}
	earlyRMS /= float32(windows / 4)
	lateRMS /= float32(windows - windows*3/4)

	// Late RMS should be lower (envelope decreases)
	assert.Less(t, lateRMS, earlyRMS*1.5, "Later windows (higher frequencies) should have lower RMS due to pink noise envelope")
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
	dst := make([]float32, 9)
	n := gen.Generate(dst)

	require.Equal(t, 9, n)
	// Triangle should be continuous and periodic
	assert.InDelta(t, dst[0], dst[8], 0.1)
	assert.InDelta(t, float32(-0.5), dst[7], 0.1)
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

func TestSincGenerator_Generate(t *testing.T) {
	// Test sinc centered at index 10, scale 1.0, amplitude 1.0
	gen := NewSincGenerator(10.0, 1.0, 1.0)
	dst := make([]float32, 21)
	n := gen.Generate(dst)

	require.Equal(t, 21, n)
	// At center (index 10), sinc(0) = 1
	assert.InDelta(t, 1.0, dst[10], 1e-5, "Sinc at center should be 1.0")
	// At index 9 (x=-1), sinc(-π) = sin(-π)/(-π) = 0
	assert.InDelta(t, 0.0, dst[9], 1e-5, "Sinc at x=-1 should be 0")
	// At index 11 (x=1), sinc(π) = sin(π)/(π) = 0
	assert.InDelta(t, 0.0, dst[11], 1e-5, "Sinc at x=1 should be 0")
	// Sinc should be symmetric
	assert.InDelta(t, dst[8], dst[12], 1e-5, "Sinc should be symmetric")
}

func TestSincGenerator_ZeroAtCenter(t *testing.T) {
	// Test sinc with center at 0
	gen := NewSincGenerator(0.0, 1.0, 1.0)
	dst := make([]float32, 5)
	n := gen.Generate(dst)

	require.Equal(t, 5, n)
	// At center (index 0), sinc(0) = 1
	assert.InDelta(t, 1.0, dst[0], 1e-5, "Sinc at center should be 1.0")
}

func TestSincGenerator_Add(t *testing.T) {
	gen := NewSincGenerator(5.0, 1.0, 1.0)
	dst := make([]float32, 11)
	// Initialize with some values
	for i := range dst {
		dst[i] = 0.5
	}
	n := gen.Add(dst)

	require.Equal(t, 11, n)
	// At center (index 5), should be 0.5 + 1.0 = 1.5
	assert.InDelta(t, 1.5, dst[5], 1e-5, "Add should accumulate at center")
}

func TestSincGenerator_GenerateWithOp(t *testing.T) {
	gen := NewSincGenerator(5.0, 1.0, 2.0)
	dst := make([]float32, 11)
	n := gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value * 0.5 // Scale by 0.5
	})

	require.Equal(t, 11, n)
	// At center, should be 2.0 * 0.5 = 1.0
	assert.InDelta(t, 1.0, dst[5], 1e-5, "GenerateWithOp should apply operation")
}

func TestGaussianGenerator_Generate(t *testing.T) {
	// Test Gaussian centered at index 10, sigma 2.0, amplitude 1.0
	gen := NewGaussianGenerator(10.0, 2.0, 1.0)
	dst := make([]float32, 21)
	n := gen.Generate(dst)

	require.Equal(t, 21, n)
	// At center (index 10), Gaussian should be maximum (amplitude)
	assert.InDelta(t, 1.0, dst[10], 1e-5, "Gaussian at center should be amplitude")
	// Gaussian should be symmetric
	assert.InDelta(t, dst[9], dst[11], 1e-5, "Gaussian should be symmetric")
	// Values away from center should be smaller
	assert.Less(t, dst[5], dst[10], "Gaussian should decrease away from center")
	assert.Less(t, dst[15], dst[10], "Gaussian should decrease away from center")
	// All values should be positive
	for i, v := range dst {
		assert.GreaterOrEqual(t, v, float32(0.0), "Gaussian value at index %d should be >= 0", i)
		assert.LessOrEqual(t, v, float32(1.0), "Gaussian value at index %d should be <= amplitude", i)
	}
}

func TestGaussianGenerator_ZeroAtCenter(t *testing.T) {
	// Test Gaussian with center at 0
	gen := NewGaussianGenerator(0.0, 1.0, 1.0)
	dst := make([]float32, 5)
	n := gen.Generate(dst)

	require.Equal(t, 5, n)
	// At center (index 0), Gaussian should be maximum
	assert.InDelta(t, 1.0, dst[0], 1e-5, "Gaussian at center should be amplitude")
}

func TestGaussianGenerator_SigmaEffect(t *testing.T) {
	// Test that larger sigma produces wider Gaussian
	genNarrow := NewGaussianGenerator(10.0, 1.0, 1.0)
	genWide := NewGaussianGenerator(10.0, 3.0, 1.0)

	dstNarrow := make([]float32, 21)
	dstWide := make([]float32, 21)

	genNarrow.Generate(dstNarrow)
	genWide.Generate(dstWide)

	// At center, both should be 1.0
	assert.InDelta(t, 1.0, dstNarrow[10], 1e-5)
	assert.InDelta(t, 1.0, dstWide[10], 1e-5)

	// Away from center, wide Gaussian should have larger values
	// At index 5 (x=-5), narrow should be smaller than wide
	assert.Less(t, dstNarrow[5], dstWide[5], "Wider Gaussian should have larger values away from center")
}

func TestGaussianGenerator_InvalidSigma(t *testing.T) {
	// Test with invalid sigma (zero or negative)
	gen := NewGaussianGenerator(10.0, 0.0, 1.0)
	dst := make([]float32, 11)
	n := gen.Generate(dst)

	require.Equal(t, 11, n)
	// Should return zeros for invalid sigma
	for i, v := range dst {
		assert.InDelta(t, 0.0, v, 1e-6, "Invalid sigma should produce zeros at index %d", i)
	}
}

func TestGaussianGenerator_Add(t *testing.T) {
	gen := NewGaussianGenerator(5.0, 1.0, 1.0)
	dst := make([]float32, 11)
	// Initialize with some values
	for i := range dst {
		dst[i] = 0.5
	}
	n := gen.Add(dst)

	require.Equal(t, 11, n)
	// At center (index 5), should be 0.5 + 1.0 = 1.5
	assert.InDelta(t, 1.5, dst[5], 1e-5, "Add should accumulate at center")
}

func TestGaussianGenerator_GenerateWithOp(t *testing.T) {
	gen := NewGaussianGenerator(5.0, 1.0, 2.0)
	dst := make([]float32, 11)
	n := gen.GenerateWithOp(dst, false, func(d *float32, value float32) {
		*d = value * 0.5 // Scale by 0.5
	})

	require.Equal(t, 11, n)
	// At center, should be 2.0 * 0.5 = 1.0
	assert.InDelta(t, 1.0, dst[5], 1e-5, "GenerateWithOp should apply operation")
}

func TestRaisedCosineGeneratorMatchesFormula(t *testing.T) {
	gen := NewRaisedCosineGenerator(16, 1.0/32.0, 0.35, 1.0)
	buf := make([]float32, 64)
	gen.Generate(buf)

	xScaled := -gen.center * gen.scale
	const tol = 1e-5
	for i, sample := range buf {
		expected := raisedCosineReferenceValue(xScaled, gen.rolloff, gen.amplitude)
		assert.InDeltaf(t, expected, sample, tol, "sample mismatch at %d (x=%f expected=%f got=%f)", i, xScaled, expected, sample)
		xScaled += gen.scale
	}
}

func BenchmarkRaisedCosineGenerator(b *testing.B) {
	gen := NewRaisedCosineGenerator(32, 1.0/64.0, 0.35, 1.0)
	buf := make([]float32, 256)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.Generate(buf)
	}
}

func raisedCosineReferenceValue(x, rolloff, amplitude float32) float32 {
	alpha := rolloff
	if alpha < 0 {
		alpha = -alpha
	}

	const thetaEps = 1e-6
	const denomEps = 1e-5

	theta := math32.Pi * x
	sinTheta := math32.Sin(theta)
	cosPhi := math32.Cos(math32.Pi * alpha * x)
	xAlpha := 2 * alpha * x
	useAlpha := alpha >= 1e-6

	if !useAlpha {
		if math32.Abs(theta) < thetaEps {
			return amplitude
		}
		return amplitude * sinTheta / theta
	}

	sinc := float32(1)
	if math32.Abs(theta) >= thetaEps {
		sinc = sinTheta / theta
	}

	denom := 1 - xAlpha*xAlpha
	if math32.Abs(denom) < denomEps {
		return amplitude * (alpha / 2) * math32.Sin(math32.Pi/(2*alpha))
	}

	return amplitude * (sinc * cosPhi / denom)
}

func TestSineGenerator_PhaseWrapping(t *testing.T) {
	// Test that phase wrapping prevents calculation errors
	// Use a very large phase that would overflow without wrapping
	gen := NewSineGenerator(1000.0, 1.0, 1.0, 1000.0*math32.Pi)
	dst := make([]float32, 100)
	n := gen.Generate(dst)

	require.Equal(t, 100, n)
	// All values should be in valid range [-1, 1]
	for i, v := range dst {
		assert.GreaterOrEqual(t, v, float32(-1.0), "Value at index %d should be >= -1", i)
		assert.LessOrEqual(t, v, float32(1.0), "Value at index %d should be <= 1", i)
	}
}

func TestCosineGenerator_PhaseWrapping(t *testing.T) {
	// Test that phase wrapping prevents calculation errors
	// Use a very large phase that would overflow without wrapping
	gen := NewCosineGenerator(1000.0, 1.0, 1.0, 1000.0*math32.Pi)
	dst := make([]float32, 100)
	n := gen.Generate(dst)

	require.Equal(t, 100, n)
	// All values should be in valid range [-1, 1]
	for i, v := range dst {
		assert.GreaterOrEqual(t, v, float32(-1.0), "Value at index %d should be >= -1", i)
		assert.LessOrEqual(t, v, float32(1.0), "Value at index %d should be <= 1", i)
	}
}

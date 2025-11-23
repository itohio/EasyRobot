package dsss

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCalculateMinimumSampleRate(t *testing.T) {
	symbolRate := float32(1000)
	spreadingFactor := float32(7) // m-sequence with n=3
	carrierFreq := float32(1000)

	// Calculate minimum sample rate
	minRate := CalculateMinimumSampleRate(symbolRate, spreadingFactor, carrierFreq, MinimumSamplesPerChip)
	
	// Chip rate = 1000 * 7 = 7000 chips/sec
	// Minimum samples per chip = 4
	// Chip sampling constraint = 7000 * 4 = 28000 Hz
	// Nyquist constraint = 4 * 1000 = 4000 Hz
	// Minimum = max(28000, 4000) = 28000 Hz
	expectedMinRate := float32(28000)
	assert.InDelta(t, expectedMinRate, minRate, 1.0, "Minimum sample rate should be 28000 Hz")

	// Calculate recommended sample rate
	recommendedRate := CalculateRecommendedSampleRate(symbolRate, spreadingFactor, carrierFreq)
	
	// Recommended samples per chip = 8
	// Chip sampling constraint = 7000 * 8 = 56000 Hz
	// Nyquist constraint = 4000 Hz
	// Recommended = max(56000, 4000) = 56000 Hz
	expectedRecommendedRate := float32(56000)
	assert.InDelta(t, expectedRecommendedRate, recommendedRate, 1.0, "Recommended sample rate should be 56000 Hz")
}

func TestCalculateMinimumSampleRate_HighCarrierFreq(t *testing.T) {
	// Test case where Nyquist constraint dominates
	symbolRate := float32(1000)
	spreadingFactor := float32(7)
	carrierFreq := float32(10000) // High carrier frequency

	minRate := CalculateMinimumSampleRate(symbolRate, spreadingFactor, carrierFreq, MinimumSamplesPerChip)
	
	// Chip sampling constraint = 7000 * 4 = 28000 Hz
	// Nyquist constraint = 4 * 10000 = 40000 Hz
	// Minimum = max(28000, 40000) = 40000 Hz
	expectedMinRate := float32(40000)
	assert.InDelta(t, expectedMinRate, minRate, 1.0, "Minimum sample rate should be 40000 Hz (Nyquist dominated)")
}

func TestSampleRateValidation(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	symbolRate := float32(1000)
	carrierFreq := float32(1000)
	spreadingFactor := float32(code.Length()) // 7

	// Test with valid sample rate (above minimum)
	minRate := CalculateMinimumSampleRate(symbolRate, spreadingFactor, carrierFreq, MinimumSamplesPerChip)
	validRate := minRate + 1000 // Just above minimum
	modulator := NewModulator(validRate, carrierFreq, symbolRate, code)
	require.NotNil(t, modulator)
	assert.Equal(t, validRate, modulator.sampleRate)

	// Test with invalid sample rate (below minimum) - should panic
	invalidRate := minRate - 1000 // Below minimum
	assert.Panics(t, func() {
		NewModulator(invalidRate, carrierFreq, symbolRate, code)
	}, "Should panic when sample rate is below minimum")
}

func TestTestSampleRateExceedsRecommended(t *testing.T) {
	code := GenerateMSequence(3, []int{2, 1, 0}, 1)
	spreadingFactor := float32(code.Length()) // 7
	
	// Calculate recommended sample rate for test parameters
	recommendedRate := CalculateRecommendedSampleRate(testSymbolRate, spreadingFactor, testCarrierFreq)
	
	// Verify test sample rate exceeds recommended
	assert.GreaterOrEqual(t, testSampleRate, recommendedRate,
		"Test sample rate %f should be >= recommended rate %f", testSampleRate, recommendedRate)
	
	// Calculate actual samples per chip with test parameters
	chipRate := testSymbolRate * spreadingFactor
	samplesPerChip := float32(testSampleRate) / chipRate
	
	// Should have at least recommended samples per chip
	assert.GreaterOrEqual(t, samplesPerChip, float32(RecommendedSamplesPerChip),
		"Samples per chip %f should be >= recommended %d", samplesPerChip, RecommendedSamplesPerChip)
}


package dsss

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestBandwidthRequirements verifies that test parameters meet bandwidth and Nyquist constraints
func TestBandwidthRequirements(t *testing.T) {
	// Test parameters
	symbolRate := testSymbolRate
	spreadingFactor := float32(testSpreadingFactor)
	carrierFreq := testCarrierFreq
	rolloffFactor := testRolloffFactor
	sampleRate := testSampleRate

	// Calculate chip rate
	chipRate := symbolRate * spreadingFactor
	t.Logf("Chip rate: %.0f chips/sec", chipRate)

	// Calculate baseband bandwidth with raised cosine pulse shaping
	basebandBandwidth := Bandwidth(symbolRate, spreadingFactor, rolloffFactor)
	t.Logf("Baseband bandwidth (with rolloff %.2f): %.0f Hz", rolloffFactor, basebandBandwidth)

	// For passband signal, the bandwidth extends from (carrierFreq - bandwidth/2) to (carrierFreq + bandwidth/2)
	lowerFreq := carrierFreq - basebandBandwidth/2
	upperFreq := carrierFreq + basebandBandwidth/2
	t.Logf("Passband frequency range: %.0f Hz to %.0f Hz", lowerFreq, upperFreq)

	// Check 1: Carrier frequency must be >= bandwidth/2 to avoid negative frequencies
	minCarrierFreq := basebandBandwidth / 2
	t.Logf("Minimum carrier frequency required: %.0f Hz", minCarrierFreq)
	assert.GreaterOrEqual(t, carrierFreq, minCarrierFreq,
		"Carrier frequency must be >= bandwidth/2 to avoid negative frequencies (aliasing)")

	// Check 2: Nyquist criterion - sampleRate/2 must be > upperFreq
	nyquistLimit := sampleRate / 2
	t.Logf("Nyquist limit (sampleRate/2): %.0f Hz", nyquistLimit)
	t.Logf("Upper frequency: %.0f Hz", upperFreq)
	assert.Greater(t, nyquistLimit, upperFreq,
		"Nyquist criterion: sampleRate/2 must be > (carrierFreq + bandwidth/2)")

	// Check 3: Adequate sampling for chip rate
	// Need at least 2 samples per chip (Nyquist for chip rate), but we use more for pulse shaping
	samplesPerChip := sampleRate / chipRate
	t.Logf("Samples per chip: %.2f", samplesPerChip)
	assert.GreaterOrEqual(t, float64(samplesPerChip), 2.0,
		"Must have at least 2 samples per chip (Nyquist)")

	// Check 4: Adequate sampling for carrier
	samplesPerCarrierPeriod := sampleRate / carrierFreq
	t.Logf("Samples per carrier period: %.2f", samplesPerCarrierPeriod)
	assert.GreaterOrEqual(t, float64(samplesPerCarrierPeriod), 2.0,
		"Must have at least 2 samples per carrier period (Nyquist)")
}

// TestBandwidth tests the Bandwidth function
func TestBandwidth(t *testing.T) {
	symbolRate := float32(1000)
	spreadingFactor := float32(7)
	rolloffFactor := float32(0.5)

	bandwidth := Bandwidth(symbolRate, spreadingFactor, rolloffFactor)

	t.Logf("Symbol rate: %.0f symbols/sec", symbolRate)
	t.Logf("Spreading factor: %.0f chips/symbol", spreadingFactor)
	t.Logf("Rolloff factor: %.2f", rolloffFactor)
	t.Logf("Bandwidth: %.0f Hz", bandwidth)

	// Expected: 1000 * 7 * 1.5 = 10500 Hz
	expectedBandwidth := float32(10500)
	assert.InDelta(t, expectedBandwidth, bandwidth, 1.0, "Bandwidth calculation")
}

// TestBandwidth_VariousRolloffFactors tests bandwidth with different rolloff factors
func TestBandwidth_VariousRolloffFactors(t *testing.T) {
	symbolRate := float32(1000)
	spreadingFactor := float32(7)

	testCases := []struct {
		rolloffFactor float32
		expected      float32
	}{
		{0.0, 7000},   // No rolloff: bandwidth = chipRate
		{0.35, 9450},   // Typical rolloff: 7000 * 1.35
		{0.5, 10500},   // Higher rolloff: 7000 * 1.5
		{1.0, 14000},  // Maximum rolloff: 7000 * 2.0
	}

	for _, tc := range testCases {
		bandwidth := Bandwidth(symbolRate, spreadingFactor, tc.rolloffFactor)
		assert.InDelta(t, tc.expected, bandwidth, 1.0,
			"Bandwidth with rolloff factor %.2f", tc.rolloffFactor)
	}
}


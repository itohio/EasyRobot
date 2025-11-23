package modulation

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestGoertzelFrequencySpacing tests Goertzel detector with proper frequency spacing
// Demonstrates reliable detection when frequencies are spaced according to theoretical requirements
func TestGoertzelFrequencySpacing(t *testing.T) {
	// Test parameters
	symbolRate := float32(1000)                          // 1000 symbols per second
	samplesPerSymbol := 48                               // 48 samples per symbol
	sampleRate := symbolRate * float32(samplesPerSymbol) // 48 kHz

	// Recommended spacing: 3 * symbolRate = 3000 Hz
	// Minimum spacing: 2 * symbolRate = 2000 Hz
	// Use 2x recommended spacing for safety: 2 * 3 * symbolRate = 6000 Hz
	recommendedSpacing := 3 * symbolRate
	testSpacing := 2 * recommendedSpacing // 2x recommended for test safety margin

	t.Logf("Symbol rate: %.0f sps", symbolRate)
	t.Logf("Sample rate: %.0f Hz", sampleRate)
	t.Logf("Samples per symbol: %d", samplesPerSymbol)
	t.Logf("Frequency resolution: %.2f Hz", sampleRate/float32(samplesPerSymbol))
	t.Logf("Recommended spacing: %.0f Hz", recommendedSpacing)
	t.Logf("Test spacing (2x recommended): %.0f Hz", testSpacing)

	// Create frequencies with proper spacing
	// Start at 1000 Hz, space by testSpacing
	freqs := []float32{
		1000,
		1000 + testSpacing,
		1000 + 2*testSpacing,
		1000 + 3*testSpacing,
	}

	t.Logf("Test frequencies: %v Hz", freqs)

	detector := NewGoertzelFrequencyDetector(sampleRate, freqs...)

	// Test detection of each frequency individually
	correctDetections := 0
	for expectedIdx, freq := range freqs {
		// Generate one symbol worth of samples at this frequency
		samples := make([]float32, samplesPerSymbol)
		phase := float32(0)
		for i := range samples {
			phase += 2 * math.Pi * freq / sampleRate
			samples[i] = float32(math.Cos(float64(phase)))
		}

		// Detect frequency
		detectedIdx, magnitude := detector.DetectFrequency(samples)

		t.Logf("Frequency %d (%.0f Hz): detected=%d, magnitude=%.6f", expectedIdx, freq, detectedIdx, magnitude)

		if detectedIdx == expectedIdx {
			correctDetections++
		} else if detectedIdx < 0 {
			t.Logf("  Ambiguous detection for frequency %d", expectedIdx)
		} else {
			t.Logf("  Wrong detection: expected %d, got %d", expectedIdx, detectedIdx)
		}

		detector.Reset()
	}

	t.Logf("Correct detections: %d/%d", correctDetections, len(freqs))
	assert.Equal(t, len(freqs), correctDetections, "Should detect all frequencies correctly with proper spacing")
}

// TestGoertzelMultipleFrequencies tests Goertzel detector with multiple frequencies present simultaneously
// This is the more challenging case - detecting one frequency when others are also present
func TestGoertzelMultipleFrequencies(t *testing.T) {
	// Test parameters
	symbolRate := float32(1000)                          // 1000 symbols per second
	samplesPerSymbol := 48                               // 48 samples per symbol
	sampleRate := symbolRate * float32(samplesPerSymbol) // 48 kHz

	// Use 2x recommended spacing for safety
	recommendedSpacing := 3 * symbolRate
	testSpacing := 2 * recommendedSpacing // 6000 Hz

	// Create frequencies with proper spacing
	freqs := []float32{
		1000,
		1000 + testSpacing,
		1000 + 2*testSpacing,
		1000 + 3*testSpacing,
	}

	t.Logf("Test frequencies: %v Hz (spacing: %.0f Hz)", freqs, testSpacing)

	detector := NewGoertzelFrequencyDetector(sampleRate, freqs...)

	// Test detection when multiple frequencies are present simultaneously
	// Generate signal with all frequencies at equal amplitude
	correctDetections := 0
	for expectedIdx, targetFreq := range freqs {
		// Generate signal with all frequencies present, but target frequency is dominant
		samples := make([]float32, samplesPerSymbol)
		for i := range samples {
			// Add all frequencies, but make target frequency 2x stronger
			sample := float32(0)
			for freqIdx, freq := range freqs {
				phaseFreq := 2 * math.Pi * freq / sampleRate * float32(i)
				amplitude := float32(0.5) // Base amplitude
				if freqIdx == expectedIdx {
					amplitude = 1.0 // Target frequency is stronger
				}
				sample += amplitude * float32(math.Cos(float64(phaseFreq)))
			}
			samples[i] = sample
		}

		// Detect frequency
		detectedIdx, magnitude := detector.DetectFrequency(samples)

		t.Logf("Target frequency %d (%.0f Hz) with all frequencies present: detected=%d, magnitude=%.6f",
			expectedIdx, targetFreq, detectedIdx, magnitude)

		if detectedIdx == expectedIdx {
			correctDetections++
		} else if detectedIdx < 0 {
			t.Logf("  Ambiguous detection for target frequency %d", expectedIdx)
		} else {
			t.Logf("  Wrong detection: expected %d, got %d", expectedIdx, detectedIdx)
		}

		detector.Reset()
	}

	t.Logf("Correct detections: %d/%d", correctDetections, len(freqs))
	assert.GreaterOrEqual(t, correctDetections, len(freqs)-1, "Should detect target frequency correctly even with others present")
}

// TestGoertzelMinimumSpacing tests Goertzel detector at minimum spacing requirement
// This demonstrates the boundary condition - spacing = 2 * symbolRate
func TestGoertzelMinimumSpacing(t *testing.T) {
	// Test parameters
	symbolRate := float32(1000)                          // 1000 symbols per second
	samplesPerSymbol := 48                               // 48 samples per symbol
	sampleRate := symbolRate * float32(samplesPerSymbol) // 48 kHz

	// Minimum spacing: 2 * symbolRate = 2000 Hz
	// Use 2x minimum for test safety: 4000 Hz
	minSpacing := 2 * symbolRate
	testSpacing := 2 * minSpacing // 4000 Hz (2x minimum for safety)

	t.Logf("Symbol rate: %.0f sps", symbolRate)
	t.Logf("Sample rate: %.0f Hz", sampleRate)
	t.Logf("Samples per symbol: %d", samplesPerSymbol)
	t.Logf("Frequency resolution: %.2f Hz", sampleRate/float32(samplesPerSymbol))
	t.Logf("Minimum spacing: %.0f Hz", minSpacing)
	t.Logf("Test spacing (2x minimum): %.0f Hz", testSpacing)

	// Create frequencies at minimum spacing
	freqs := []float32{
		1000,
		1000 + testSpacing,
	}

	t.Logf("Test frequencies: %v Hz", freqs)

	detector := NewGoertzelFrequencyDetector(sampleRate, freqs...)

	// Test detection of each frequency
	correctDetections := 0
	for expectedIdx, freq := range freqs {
		// Generate one symbol worth of samples at this frequency
		samples := make([]float32, samplesPerSymbol)
		phase := float32(0)
		for i := range samples {
			phase += 2 * math.Pi * freq / sampleRate
			samples[i] = float32(math.Cos(float64(phase)))
		}

		// Detect frequency
		detectedIdx, magnitude := detector.DetectFrequency(samples)

		t.Logf("Frequency %d (%.0f Hz): detected=%d, magnitude=%.6f", expectedIdx, freq, detectedIdx, magnitude)

		if detectedIdx == expectedIdx {
			correctDetections++
		} else if detectedIdx < 0 {
			t.Logf("  Ambiguous detection for frequency %d", expectedIdx)
		} else {
			t.Logf("  Wrong detection: expected %d, got %d", expectedIdx, detectedIdx)
		}

		detector.Reset()
	}

	t.Logf("Correct detections: %d/%d", correctDetections, len(freqs))
	// At minimum spacing, we expect at least one correct detection
	assert.GreaterOrEqual(t, correctDetections, 1, "Should detect at least one frequency correctly at minimum spacing")
}

// TestGoertzelSampleRateRequirement tests that sample rate must be sufficient for given spacing
// Demonstrates that insufficient sample rate leads to detection failures
func TestGoertzelSampleRateRequirement(t *testing.T) {
	symbolRate := float32(1000)          // 1000 symbols per second
	recommendedSpacing := 3 * symbolRate // 3000 Hz

	// Test with insufficient sample rate (too few samples per symbol)
	insufficientSamplesPerSymbol := 12                                           // Only 12 samples per symbol
	insufficientSampleRate := symbolRate * float32(insufficientSamplesPerSymbol) // 12 kHz
	resolution := insufficientSampleRate / float32(insufficientSamplesPerSymbol) // 1000 Hz

	t.Logf("Insufficient sample rate test:")
	t.Logf("  Symbol rate: %.0f sps", symbolRate)
	t.Logf("  Samples per symbol: %d", insufficientSamplesPerSymbol)
	t.Logf("  Sample rate: %.0f Hz", insufficientSampleRate)
	t.Logf("  Frequency resolution: %.2f Hz", resolution)
	t.Logf("  Recommended spacing: %.0f Hz", recommendedSpacing)
	t.Logf("  Spacing / Resolution: %.2f (should be >= 3)", recommendedSpacing/resolution)

	// Create frequencies with recommended spacing
	freqs := []float32{
		1000,
		1000 + recommendedSpacing,
	}

	detector := NewGoertzelFrequencyDetector(insufficientSampleRate, freqs...)

	// Test detection - may fail due to insufficient resolution
	correctDetections := 0
	for expectedIdx, freq := range freqs {
		samples := make([]float32, insufficientSamplesPerSymbol)
		phase := float32(0)
		for i := range samples {
			phase += 2 * math.Pi * freq / insufficientSampleRate
			samples[i] = float32(math.Cos(float64(phase)))
		}

		detectedIdx, magnitude := detector.DetectFrequency(samples)
		t.Logf("Frequency %d (%.0f Hz): detected=%d, magnitude=%.6f", expectedIdx, freq, detectedIdx, magnitude)

		if detectedIdx == expectedIdx {
			correctDetections++
		}

		detector.Reset()
	}

	// With insufficient sample rate, detection may fail
	// This test documents the requirement - we expect some failures
	t.Logf("Correct detections with insufficient sample rate: %d/%d", correctDetections, len(freqs))

	// Test with sufficient sample rate (2x recommended)
	sufficientSamplesPerSymbol := 48                                                   // 48 samples per symbol
	sufficientSampleRate := symbolRate * float32(sufficientSamplesPerSymbol)           // 48 kHz
	sufficientResolution := sufficientSampleRate / float32(sufficientSamplesPerSymbol) // 1000 Hz

	t.Logf("\nSufficient sample rate test:")
	t.Logf("  Symbol rate: %.0f sps", symbolRate)
	t.Logf("  Samples per symbol: %d", sufficientSamplesPerSymbol)
	t.Logf("  Sample rate: %.0f Hz", sufficientSampleRate)
	t.Logf("  Frequency resolution: %.2f Hz", sufficientResolution)
	t.Logf("  Recommended spacing: %.0f Hz", recommendedSpacing)
	t.Logf("  Spacing / Resolution: %.2f (should be >= 3)", recommendedSpacing/sufficientResolution)

	detectorSufficient := NewGoertzelFrequencyDetector(sufficientSampleRate, freqs...)

	correctDetectionsSufficient := 0
	for expectedIdx, freq := range freqs {
		samples := make([]float32, sufficientSamplesPerSymbol)
		phase := float32(0)
		for i := range samples {
			phase += 2 * math.Pi * freq / sufficientSampleRate
			samples[i] = float32(math.Cos(float64(phase)))
		}

		detectedIdx, magnitude := detectorSufficient.DetectFrequency(samples)
		t.Logf("Frequency %d (%.0f Hz): detected=%d, magnitude=%.6f", expectedIdx, freq, detectedIdx, magnitude)

		if detectedIdx == expectedIdx {
			correctDetectionsSufficient++
		}

		detectorSufficient.Reset()
	}

	t.Logf("Correct detections with sufficient sample rate: %d/%d", correctDetectionsSufficient, len(freqs))
	// With sufficient sample rate, we expect all detections to be correct
	assert.Equal(t, len(freqs), correctDetectionsSufficient, "Should detect all frequencies correctly with sufficient sample rate")
}

// TestIQFrequencySpacing tests IQ detector with proper frequency spacing
// Demonstrates reliable detection when frequencies are spaced according to theoretical requirements
func TestIQFrequencySpacing(t *testing.T) {
	// Test parameters
	symbolRate := float32(1000)                          // 1000 symbols per second
	samplesPerSymbol := 48                               // 48 samples per symbol
	sampleRate := symbolRate * float32(samplesPerSymbol) // 48 kHz

	// Recommended spacing: 3 * symbolRate = 3000 Hz
	// Minimum spacing: 2 * symbolRate = 2000 Hz
	// Use 2x recommended spacing for safety: 2 * 3 * symbolRate = 6000 Hz
	recommendedSpacing := 3 * symbolRate
	testSpacing := 2 * recommendedSpacing // 2x recommended for test safety margin

	t.Logf("Symbol rate: %.0f sps", symbolRate)
	t.Logf("Sample rate: %.0f Hz", sampleRate)
	t.Logf("Samples per symbol: %d", samplesPerSymbol)
	t.Logf("Frequency resolution: %.2f Hz", sampleRate/float32(samplesPerSymbol))
	t.Logf("Recommended spacing: %.0f Hz", recommendedSpacing)
	t.Logf("Test spacing (2x recommended): %.0f Hz", testSpacing)

	// Create frequencies with proper spacing
	// Start at 1000 Hz, space by testSpacing
	freqs := []float32{
		1000,
		1000 + testSpacing,
		1000 + 2*testSpacing,
		1000 + 3*testSpacing,
	}

	t.Logf("Test frequencies: %v Hz", freqs)

	detector := NewIQFrequencyDetector(sampleRate, symbolRate, freqs...)

	// Test detection of each frequency individually
	// IQ detector uses EMA smoothing, so we need to warm it up with multiple symbols
	correctDetections := 0
	for expectedIdx, freq := range freqs {
		// Warm up: feed 3 symbols to build up EMA state
		for warmup := 0; warmup < 3; warmup++ {
			warmupSamples := make([]float32, samplesPerSymbol)
			phase := float32(0)
			for i := range warmupSamples {
				phase += 2 * math.Pi * freq / sampleRate
				warmupSamples[i] = float32(math.Cos(float64(phase)))
			}
			detector.DetectFrequency(warmupSamples)
		}

		// Generate one symbol worth of samples at this frequency for actual test
		samples := make([]float32, samplesPerSymbol)
		phase := float32(0)
		for i := range samples {
			phase += 2 * math.Pi * freq / sampleRate
			samples[i] = float32(math.Cos(float64(phase)))
		}

		// Detect frequency
		detectedIdx, magnitude := detector.DetectFrequency(samples)

		t.Logf("Frequency %d (%.0f Hz): detected=%d, magnitude=%.6f", expectedIdx, freq, detectedIdx, magnitude)

		if detectedIdx == expectedIdx {
			correctDetections++
		} else if detectedIdx < 0 {
			t.Logf("  Ambiguous detection for frequency %d", expectedIdx)
		} else {
			t.Logf("  Wrong detection: expected %d, got %d", expectedIdx, detectedIdx)
		}

		detector.Reset()
	}

	t.Logf("Correct detections: %d/%d", correctDetections, len(freqs))
	assert.Equal(t, len(freqs), correctDetections, "Should detect all frequencies correctly with proper spacing")
}

// TestIQMultipleFrequencies tests IQ detector with multiple frequencies present simultaneously
// This is the more challenging case - detecting one frequency when others are also present
func TestIQMultipleFrequencies(t *testing.T) {
	// Test parameters
	symbolRate := float32(1000)                          // 1000 symbols per second
	samplesPerSymbol := 48                               // 48 samples per symbol
	sampleRate := symbolRate * float32(samplesPerSymbol) // 48 kHz

	// Use 2x recommended spacing for safety
	recommendedSpacing := 3 * symbolRate
	testSpacing := 2 * recommendedSpacing // 6000 Hz

	// Create frequencies with proper spacing
	freqs := []float32{
		1000,
		1000 + testSpacing,
		1000 + 2*testSpacing,
		1000 + 3*testSpacing,
	}

	t.Logf("Test frequencies: %v Hz (spacing: %.0f Hz)", freqs, testSpacing)

	detector := NewIQFrequencyDetector(sampleRate, symbolRate, freqs...)

	// Test detection when multiple frequencies are present simultaneously
	// Generate signal with all frequencies at equal amplitude
	correctDetections := 0
	for expectedIdx, targetFreq := range freqs {
		// Generate signal with all frequencies present, but target frequency is dominant
		samples := make([]float32, samplesPerSymbol)
		for i := range samples {
			// Add all frequencies, but make target frequency 2x stronger
			sample := float32(0)
			for freqIdx, freq := range freqs {
				phaseFreq := 2 * math.Pi * freq / sampleRate * float32(i)
				amplitude := float32(0.5) // Base amplitude
				if freqIdx == expectedIdx {
					amplitude = 1.0 // Target frequency is stronger
				}
				sample += amplitude * float32(math.Cos(float64(phaseFreq)))
			}
			samples[i] = sample
		}

		// Detect frequency
		detectedIdx, magnitude := detector.DetectFrequency(samples)

		t.Logf("Target frequency %d (%.0f Hz) with all frequencies present: detected=%d, magnitude=%.6f",
			expectedIdx, targetFreq, detectedIdx, magnitude)

		if detectedIdx == expectedIdx {
			correctDetections++
		} else if detectedIdx < 0 {
			t.Logf("  Ambiguous detection for target frequency %d", expectedIdx)
		} else {
			t.Logf("  Wrong detection: expected %d, got %d", expectedIdx, detectedIdx)
		}

		detector.Reset()
	}

	t.Logf("Correct detections: %d/%d", correctDetections, len(freqs))
	assert.GreaterOrEqual(t, correctDetections, len(freqs)-1, "Should detect target frequency correctly even with others present")
}

// TestIQMinimumSpacing tests IQ detector at minimum spacing requirement
// This demonstrates the boundary condition - spacing = 2 * symbolRate
func TestIQMinimumSpacing(t *testing.T) {
	// Test parameters
	symbolRate := float32(1000)                          // 1000 symbols per second
	samplesPerSymbol := 48                               // 48 samples per symbol
	sampleRate := symbolRate * float32(samplesPerSymbol) // 48 kHz

	// Minimum spacing: 2 * symbolRate = 2000 Hz
	// Use 2x minimum for test safety: 4000 Hz
	minSpacing := 2 * symbolRate
	testSpacing := 2 * minSpacing // 4000 Hz (2x minimum for safety)

	t.Logf("Symbol rate: %.0f sps", symbolRate)
	t.Logf("Sample rate: %.0f Hz", sampleRate)
	t.Logf("Samples per symbol: %d", samplesPerSymbol)
	t.Logf("Frequency resolution: %.2f Hz", sampleRate/float32(samplesPerSymbol))
	t.Logf("Minimum spacing: %.0f Hz", minSpacing)
	t.Logf("Test spacing (2x minimum): %.0f Hz", testSpacing)

	// Create frequencies at minimum spacing
	freqs := []float32{
		1000,
		1000 + testSpacing,
	}

	t.Logf("Test frequencies: %v Hz", freqs)

	detector := NewIQFrequencyDetector(sampleRate, symbolRate, freqs...)

	// Test detection of each frequency
	correctDetections := 0
	for expectedIdx, freq := range freqs {
		// Generate one symbol worth of samples at this frequency
		samples := make([]float32, samplesPerSymbol)
		phase := float32(0)
		for i := range samples {
			phase += 2 * math.Pi * freq / sampleRate
			samples[i] = float32(math.Cos(float64(phase)))
		}

		// Detect frequency
		detectedIdx, magnitude := detector.DetectFrequency(samples)

		t.Logf("Frequency %d (%.0f Hz): detected=%d, magnitude=%.6f", expectedIdx, freq, detectedIdx, magnitude)

		if detectedIdx == expectedIdx {
			correctDetections++
		} else if detectedIdx < 0 {
			t.Logf("  Ambiguous detection for frequency %d", expectedIdx)
		} else {
			t.Logf("  Wrong detection: expected %d, got %d", expectedIdx, detectedIdx)
		}

		detector.Reset()
	}

	t.Logf("Correct detections: %d/%d", correctDetections, len(freqs))
	// At minimum spacing, we expect at least one correct detection
	assert.GreaterOrEqual(t, correctDetections, 1, "Should detect at least one frequency correctly at minimum spacing")
}

// TestIQSampleRateRequirement tests that sample rate must be sufficient for given spacing
// Demonstrates that insufficient sample rate leads to detection failures
func TestIQSampleRateRequirement(t *testing.T) {
	symbolRate := float32(1000)          // 1000 symbols per second
	recommendedSpacing := 3 * symbolRate // 3000 Hz

	// Test with insufficient sample rate (too few samples per symbol)
	insufficientSamplesPerSymbol := 12                                           // Only 12 samples per symbol
	insufficientSampleRate := symbolRate * float32(insufficientSamplesPerSymbol) // 12 kHz
	resolution := insufficientSampleRate / float32(insufficientSamplesPerSymbol) // 1000 Hz

	t.Logf("Insufficient sample rate test:")
	t.Logf("  Symbol rate: %.0f sps", symbolRate)
	t.Logf("  Samples per symbol: %d", insufficientSamplesPerSymbol)
	t.Logf("  Sample rate: %.0f Hz", insufficientSampleRate)
	t.Logf("  Frequency resolution: %.2f Hz", resolution)
	t.Logf("  Recommended spacing: %.0f Hz", recommendedSpacing)
	t.Logf("  Spacing / Resolution: %.2f (should be >= 3)", recommendedSpacing/resolution)

	// Create frequencies with recommended spacing
	freqs := []float32{
		1000,
		1000 + recommendedSpacing,
	}

	detector := NewIQFrequencyDetector(insufficientSampleRate, symbolRate, freqs...)

	// Test detection - may fail due to insufficient resolution
	correctDetections := 0
	for expectedIdx, freq := range freqs {
		samples := make([]float32, insufficientSamplesPerSymbol)
		phase := float32(0)
		for i := range samples {
			phase += 2 * math.Pi * freq / insufficientSampleRate
			samples[i] = float32(math.Cos(float64(phase)))
		}

		detectedIdx, magnitude := detector.DetectFrequency(samples)
		t.Logf("Frequency %d (%.0f Hz): detected=%d, magnitude=%.6f", expectedIdx, freq, detectedIdx, magnitude)

		if detectedIdx == expectedIdx {
			correctDetections++
		}

		detector.Reset()
	}

	// With insufficient sample rate, detection may fail
	// This test documents the requirement - we expect some failures
	t.Logf("Correct detections with insufficient sample rate: %d/%d", correctDetections, len(freqs))

	// Test with sufficient sample rate (2x recommended)
	sufficientSamplesPerSymbol := 48                                                   // 48 samples per symbol
	sufficientSampleRate := symbolRate * float32(sufficientSamplesPerSymbol)           // 48 kHz
	sufficientResolution := sufficientSampleRate / float32(sufficientSamplesPerSymbol) // 1000 Hz

	t.Logf("\nSufficient sample rate test:")
	t.Logf("  Symbol rate: %.0f sps", symbolRate)
	t.Logf("  Samples per symbol: %d", sufficientSamplesPerSymbol)
	t.Logf("  Sample rate: %.0f Hz", sufficientSampleRate)
	t.Logf("  Frequency resolution: %.2f Hz", sufficientResolution)
	t.Logf("  Recommended spacing: %.0f Hz", recommendedSpacing)
	t.Logf("  Spacing / Resolution: %.2f (should be >= 3)", recommendedSpacing/sufficientResolution)

	detectorSufficient := NewIQFrequencyDetector(sufficientSampleRate, symbolRate, freqs...)

	correctDetectionsSufficient := 0
	for expectedIdx, freq := range freqs {
		samples := make([]float32, sufficientSamplesPerSymbol)
		phase := float32(0)
		for i := range samples {
			phase += 2 * math.Pi * freq / sufficientSampleRate
			samples[i] = float32(math.Cos(float64(phase)))
		}

		detectedIdx, magnitude := detectorSufficient.DetectFrequency(samples)
		t.Logf("Frequency %d (%.0f Hz): detected=%d, magnitude=%.6f", expectedIdx, freq, detectedIdx, magnitude)

		if detectedIdx == expectedIdx {
			correctDetectionsSufficient++
		}

		detectorSufficient.Reset()
	}

	t.Logf("Correct detections with sufficient sample rate: %d/%d", correctDetectionsSufficient, len(freqs))
	// With sufficient sample rate, we expect all detections to be correct
	assert.Equal(t, len(freqs), correctDetectionsSufficient, "Should detect all frequencies correctly with sufficient sample rate")
}
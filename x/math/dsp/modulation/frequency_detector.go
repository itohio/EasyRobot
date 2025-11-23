package modulation

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/dsp"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// GoertzelFrequencyDetector uses Goertzel algorithm for frequency detection.
// Phase-independent and efficient for detecting specific frequencies.
//
// Frequency Spacing Requirements:
//
// The Goertzel algorithm requires sufficient frequency resolution to reliably distinguish
// between different frequencies. The frequency resolution is determined by:
//
//	resolution = sampleRate / N
//
// where N is the number of samples used for detection (typically samples per symbol).
//
// For reliable detection when multiple frequencies are present:
//   - Minimum spacing: 2 * resolution = 2 * sampleRate / N
//   - Recommended spacing: 3 * resolution = 3 * sampleRate / N (for safety margin)
//
// Since N = sampleRate / symbolRate (samples per symbol), this simplifies to:
//   - Minimum spacing: 2 * symbolRate
//   - Recommended spacing: 3 * symbolRate
//
// Sample Rate Requirements:
//
// For a given frequency spacing (Δf) and symbol rate (R_s), the required sample rate is:
//
//	sampleRate >= Δf * N / 2
//	sampleRate >= Δf * (sampleRate / symbolRate) / 2
//	sampleRate >= Δf / (2 * symbolRate) * sampleRate
//
// Rearranging: sampleRate >= Δf * (sampleRate / symbolRate) / 2
// This means: sampleRate >= Δf / (2 * symbolRate) * sampleRate
// Which simplifies to: 1 >= Δf / (2 * symbolRate)
// Therefore: Δf <= 2 * symbolRate (minimum spacing requirement)
//
// For recommended spacing with safety margin:
//
//	Δf >= 3 * symbolRate (recommended spacing)
//
// Example:
//
//	For symbol rate R_s = 1000 sps:
//	- Minimum frequency spacing: 2000 Hz
//	- Recommended frequency spacing: 3000 Hz
//	- For 8-FSK with 250 Hz spacing: Need 4x sample rate (192 kHz) to achieve
//	  resolution = 192000 / 192 = 1000 Hz, allowing detection of 250 Hz spacing
//	  (though still tight - coherent detection recommended for closely spaced frequencies)
//
// Note: For closely spaced frequencies (< 2 * symbolRate), coherent detection
// (using correlators with phase tracking) is recommended over Goertzel.
type GoertzelFrequencyDetector struct {
	frequencies []float32
	sampleRate  float32
}

// NewGoertzelFrequencyDetector creates a new Goertzel-based frequency detector
// Uses the dsp.Goertzel function for correct algorithm implementation
func NewGoertzelFrequencyDetector(sampleRate float32, frequencies ...float32) GoertzelFrequencyDetector {
	return GoertzelFrequencyDetector{
		frequencies: frequencies,
		sampleRate:  sampleRate,
	}
}

// DetectFrequency detects which frequency has the highest magnitude
// Returns the index of the detected frequency and its magnitude
// If the detection is ambiguous (similar magnitudes), returns -1 to indicate noise
// Uses dsp.Goertzel for correct algorithm implementation
func (gfd *GoertzelFrequencyDetector) DetectFrequency(samples []float32) (freqIdx int, magnitude float32) {
	n := len(gfd.frequencies)
	if n == 0 {
		return -1, 0
	}

	magnitudes := make([]float32, n)
	maxMagnitude := float32(0)
	bestFreqIdx := 0

	// Convert samples to vec.Vector (zero-copy cast since Vector is []float32)
	signalVec := vec.Vector(samples)

	// Calculate magnitude for each frequency using dsp.Goertzel
	for i := range n {
		result := dsp.Goertzel(signalVec, gfd.frequencies[i], gfd.sampleRate)
		mag := result.Amplitude
		magnitudes[i] = mag
		if mag > maxMagnitude {
			maxMagnitude = mag
			bestFreqIdx = i
		}
	}

	// Check if detection is ambiguous (likely noise)
	// Find second-best magnitude
	secondBest := float32(0)
	for i := range magnitudes {
		if i != bestFreqIdx {
			if mag := magnitudes[i]; mag > secondBest {
				secondBest = mag
			}
		}
	}

	// If the best and second-best are too close, it's likely noise
	// For a good FSK signal, the correct frequency should be significantly stronger
	// Use adaptive threshold based on magnitude:
	// - For high magnitudes (clean signal), use very lenient threshold (98%) - trust the detection
	// - For low magnitudes (noisy signal), use more strict threshold (90%) to reject noise
	var ratioThreshold float32 = 0.98 // Very lenient for clean signals
	if maxMagnitude < 0.1 {
		// Lower magnitude suggests noise - use more strict threshold for noisy signals
		ratioThreshold = 0.90
	}
	if maxMagnitude < 0.05 {
		// Very low magnitude - even more strict (85%)
		ratioThreshold = 0.85
	}

	// For clean signals, always trust the detection if magnitude is reasonable
	// With normalization (2.0/N), clean signals typically have magnitude around 0.04-0.1
	// For 48 samples: 2.0/48 = 0.0417, so clean signals should be around 0.04
	if maxMagnitude > 0.02 {
		// Clean signal - trust the detection, return best frequency
		// Don't do any ambiguity checks for clean signals
		return bestFreqIdx, maxMagnitude
	}

	// For lower magnitudes, do ambiguity checks but be lenient
	// Only check ratio if we have a meaningful second-best
	if maxMagnitude > 0.005 && secondBest > 0 {
		ratio := secondBest / maxMagnitude
		// If second-best is too close to best, it's ambiguous (noise)
		// Use lenient threshold
		if ratio > ratioThreshold {
			return -1, maxMagnitude // Return -1 to indicate noise/ambiguous
		}
	}

	// Check absolute magnitude threshold - very low for any signal
	magnitudeThreshold := float32(0.001) // Very low threshold
	if maxMagnitude < magnitudeThreshold {
		return -1, maxMagnitude // Too low, likely noise
	}

	// Additional check: require minimum difference between best and second-best
	// But be very lenient
	if maxMagnitude > 0.005 && secondBest > 0 {
		diff := maxMagnitude - secondBest
		diffThreshold := float32(0.0005) // Very lenient
		if diff < diffThreshold {
			return -1, maxMagnitude // Too close, likely noise
		}
	}

	return bestFreqIdx, maxMagnitude
}

// Reset resets the detector state (no state to reset for Goertzel)
func (gfd *GoertzelFrequencyDetector) Reset() {
	// Goertzel is stateless, nothing to reset
}

// CorrelationFrequencyDetector uses correlation for frequency detection
// Alternative implementation that can be more robust in some cases
type CorrelationFrequencyDetector struct {
	frequencies []float32
	sampleRate  float32
}

// NewCorrelationFrequencyDetector creates a new correlation-based frequency detector
func NewCorrelationFrequencyDetector(sampleRate float32, frequencies ...float32) CorrelationFrequencyDetector {
	return CorrelationFrequencyDetector{
		frequencies: frequencies,
		sampleRate:  sampleRate,
	}
}

// DetectFrequency detects which frequency has the highest correlation
func (cfd *CorrelationFrequencyDetector) DetectFrequency(samples []float32) (freqIdx int, magnitude float32) {
	maxMagnitude := float32(0)
	bestFreqIdx := 0

	// Reslice samples for BCE hint
	n := len(samples)
	samples = samples[:n]
	invN := 1.0 / float32(n)

	for i, freq := range cfd.frequencies {
		var sumI, sumQ float32
		omega := 2 * math32.Pi * freq / cfd.sampleRate

		// Correlate with cosine and sine (phase-independent)
		// Hot path: correlation loop
		for j := range n {
			phase := omega * float32(j)
			cosPhase := math32.Cos(phase)
			sinPhase := math32.Sin(phase)
			sumI += samples[j] * cosPhase
			sumQ += samples[j] * sinPhase
		}

		magnitude := math32.Sqrt(sumI*sumI + sumQ*sumQ)
		magnitude *= invN // Normalize

		if magnitude > maxMagnitude {
			maxMagnitude = magnitude
			bestFreqIdx = i
		}
	}

	return bestFreqIdx, maxMagnitude
}

// Reset resets the detector state (no state to reset for correlation)
func (cfd *CorrelationFrequencyDetector) Reset() {
	// Correlation is stateless, nothing to reset
}

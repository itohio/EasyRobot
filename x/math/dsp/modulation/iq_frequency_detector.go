package modulation

import (
	"github.com/chewxy/math32"
)

// IQFrequencyDetector uses I/Q correlation with pre-computed reference signals
// for frequency detection. Phase-independent and efficient for detecting specific frequencies.
// Similar to gocoustics BFSK decoder - more robust than Goertzel for FSK.
//
// Frequency Spacing Requirements:
//
// The I/Q correlation detector requires sufficient frequency resolution to reliably distinguish
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
// (using correlators with phase tracking) is recommended.
type IQFrequencyDetector struct {
	frequencies []float32
	sampleRate  float32
	symbolRate  float32

	// Pre-computed I/Q reference signals for each frequency
	// Each frequency has [I, Q] reference signals
	referenceSignals [][2][]float32 // [freqIdx][I/Q][samples]

	// EMA smoothing parameters
	W float32 // EMA window size (0.33 * samplesPerSymbol)

	// State for EMA and hysteresis
	smoothedMagnitudes []float32 // EMA-smoothed magnitudes for each frequency
	lastFreqIdx        int       // Last detected frequency (for hysteresis)
	hysteresisThresh   float32   // Hysteresis threshold (0.1)

	// Energy threshold
	energyThreshold float32
}

// NewIQFrequencyDetector creates a new I/Q correlation-based frequency detector.
// Uses techniques from gocoustics BFSK decoder: I/Q correlation, EMA smoothing, hysteresis.
func NewIQFrequencyDetector(sampleRate, symbolRate float32, frequencies ...float32) IQFrequencyDetector {
	if len(frequencies) < 2 {
		panic("IQFrequencyDetector requires at least 2 frequencies")
	}

	samplesPerSymbol := int(sampleRate / symbolRate)
	correlationWindow := samplesPerSymbol / 2 // Half symbol period, like gocoustics

	// Pre-compute I/Q reference signals for each frequency
	referenceSignals := make([][2][]float32, len(frequencies))
	for freqIdx, freq := range frequencies {
		I := make([]float32, correlationWindow)
		Q := make([]float32, correlationWindow)
		omega := 2 * math32.Pi * freq / sampleRate

		// Reslice for BCE hint
		I = I[:correlationWindow]
		Q = Q[:correlationWindow]

		for i := range correlationWindow {
			phase := omega * float32(i)
			I[i] = math32.Cos(phase)
			Q[i] = math32.Sin(phase)
		}

		referenceSignals[freqIdx] = [2][]float32{I, Q}
	}

	// EMA window size: 0.33 * samplesPerSymbol (like gocoustics)
	W := 0.33 * float32(samplesPerSymbol)
	if W < 1 {
		W = 1
	}

	return IQFrequencyDetector{
		frequencies:        frequencies,
		sampleRate:         sampleRate,
		symbolRate:         symbolRate,
		referenceSignals:   referenceSignals,
		W:                  W,
		smoothedMagnitudes: make([]float32, len(frequencies)),
		lastFreqIdx:        -1,
		hysteresisThresh:   0.1, // Like gocoustics
		energyThreshold:    0.01,
	}
}

// DetectFrequency detects which frequency has the highest correlation
// Uses I/Q correlation with EMA smoothing and hysteresis (like gocoustics)
func (iqfd *IQFrequencyDetector) DetectFrequency(samples []float32) (freqIdx int, magnitude float32) {
	correlationWindow := len(iqfd.referenceSignals[0][0])
	if len(samples) < correlationWindow {
		// Not enough samples
		return -1, 0
	}

	// Use the last correlationWindow samples
	sampleWindow := samples[len(samples)-correlationWindow:]
	// Reslice for BCE hint
	sampleWindow = sampleWindow[:correlationWindow]
	invWindow := 1.0 / float32(correlationWindow)

	// Calculate I/Q correlation for each frequency
	nFreqs := len(iqfd.frequencies)
	magnitudes := make([]float32, nFreqs)
	for freqIdx := range nFreqs {
		var sumI, sumQ float32
		Iref := iqfd.referenceSignals[freqIdx][0]
		Qref := iqfd.referenceSignals[freqIdx][1]

		// Reslice references for BCE hint
		Iref = Iref[:correlationWindow]
		Qref = Qref[:correlationWindow]

		// Hot path: Correlate input with reference I/Q signals
		for i := range correlationWindow {
			sumI += sampleWindow[i] * Iref[i]
			sumQ += sampleWindow[i] * Qref[i]
		}

		// Calculate magnitude: I^2 + Q^2
		mag := math32.Sqrt(sumI*sumI + sumQ*sumQ)
		mag *= invWindow // Normalize

		// Apply EMA smoothing (like gocoustics)
		iqfd.smoothedMagnitudes[freqIdx] = (iqfd.smoothedMagnitudes[freqIdx]*(iqfd.W-1) + mag) / iqfd.W
		magnitudes[freqIdx] = iqfd.smoothedMagnitudes[freqIdx]
	}

	// Find best frequency
	maxMagnitude := float32(0)
	bestFreqIdx := 0
	for i := range magnitudes {
		if mag := magnitudes[i]; mag > maxMagnitude {
			maxMagnitude = mag
			bestFreqIdx = i
		}
	}

	// Apply hysteresis (like gocoustics)
	// If we have a previous detection, use hysteresis to avoid rapid switching
	if iqfd.lastFreqIdx >= 0 && iqfd.lastFreqIdx < len(magnitudes) {
		lastMag := magnitudes[iqfd.lastFreqIdx]
		diff := maxMagnitude - lastMag

		if iqfd.lastFreqIdx == bestFreqIdx {
			// Same frequency - check if difference is significant enough to switch
			// (This shouldn't happen, but handle it)
		} else {
			// Different frequency - apply hysteresis
			if iqfd.lastFreqIdx < len(magnitudes) {
				// If last was bestFreqIdx, require diff > hysteresisThresh to switch
				// If last was not bestFreqIdx, require diff > -hysteresisThresh to stay
				if diff < iqfd.hysteresisThresh {
					// Not enough difference - keep last frequency
					bestFreqIdx = iqfd.lastFreqIdx
					maxMagnitude = lastMag
				}
			}
		}
	}

	// Check energy threshold (like gocoustics: (I+Q)/2 > threshold)
	// Calculate average energy across all frequencies
	avgEnergy := float32(0)
	for i := range magnitudes {
		avgEnergy += magnitudes[i]
	}
	avgEnergy /= float32(len(magnitudes))

	if avgEnergy < iqfd.energyThreshold {
		// Too low energy - likely noise
		return -1, maxMagnitude
	}

	// Update last frequency
	iqfd.lastFreqIdx = bestFreqIdx

	return bestFreqIdx, maxMagnitude
}

// Reset resets the detector state
func (iqfd *IQFrequencyDetector) Reset() {
	for i := range iqfd.smoothedMagnitudes {
		iqfd.smoothedMagnitudes[i] = 0
	}
	iqfd.lastFreqIdx = -1
}

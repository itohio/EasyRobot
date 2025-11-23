package modulation

import (
	"math"
)

// IQFrequencyDetector uses I/Q correlation with pre-computed reference signals
// Similar to gocoustics BFSK decoder - more robust than Goertzel for FSK
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
	lastFreqIdx        int        // Last detected frequency (for hysteresis)
	hysteresisThresh   float32    // Hysteresis threshold (0.1)

	// Energy threshold
	energyThreshold float32
}

// NewIQFrequencyDetector creates a new I/Q correlation-based frequency detector
// Uses techniques from gocoustics BFSK decoder: I/Q correlation, EMA smoothing, hysteresis
func NewIQFrequencyDetector(sampleRate, symbolRate float32, frequencies ...float32) *IQFrequencyDetector {
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
		omega := 2 * math.Pi * freq / sampleRate

		for i := 0; i < correlationWindow; i++ {
			phase := omega * float32(i)
			I[i] = float32(math.Cos(float64(phase)))
			Q[i] = float32(math.Sin(float64(phase)))
		}

		referenceSignals[freqIdx] = [2][]float32{I, Q}
	}

	// EMA window size: 0.33 * samplesPerSymbol (like gocoustics)
	W := float32(0.33 * float64(samplesPerSymbol))
	if W < 1 {
		W = 1
	}

	return &IQFrequencyDetector{
		frequencies:        frequencies,
		sampleRate:         sampleRate,
		symbolRate:          symbolRate,
		referenceSignals:  referenceSignals,
		W:                   W,
		smoothedMagnitudes:  make([]float32, len(frequencies)),
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

	// Calculate I/Q correlation for each frequency
	magnitudes := make([]float32, len(iqfd.frequencies))
	for freqIdx := range iqfd.frequencies {
		var sumI, sumQ float32
		Iref := iqfd.referenceSignals[freqIdx][0]
		Qref := iqfd.referenceSignals[freqIdx][1]

		// Correlate input with reference I/Q signals
		for i := 0; i < correlationWindow; i++ {
			sumI += sampleWindow[i] * Iref[i]
			sumQ += sampleWindow[i] * Qref[i]
		}

		// Calculate magnitude: I^2 + Q^2
		mag := float32(math.Sqrt(float64(sumI*sumI + sumQ*sumQ)))
		mag /= float32(correlationWindow) // Normalize

		// Apply EMA smoothing (like gocoustics)
		iqfd.smoothedMagnitudes[freqIdx] = (iqfd.smoothedMagnitudes[freqIdx]*(iqfd.W-1) + mag) / iqfd.W
		magnitudes[freqIdx] = iqfd.smoothedMagnitudes[freqIdx]
	}

	// Find best frequency
	maxMagnitude := float32(0)
	bestFreqIdx := 0
	for i, mag := range magnitudes {
		if mag > maxMagnitude {
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
	for _, mag := range magnitudes {
		avgEnergy += mag
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


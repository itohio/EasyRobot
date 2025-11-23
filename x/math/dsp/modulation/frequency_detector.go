package modulation

import (
	"math"
)

// goertzelCoeff holds Goertzel algorithm coefficients
type goertzelCoeff struct {
	omega    float32
	coeff    float32
	cosOmega float32
	sinOmega float32
}

// GoertzelFrequencyDetector uses Goertzel algorithm for frequency detection
// Phase-independent and efficient for detecting specific frequencies
type GoertzelFrequencyDetector struct {
	frequencies    []float32
	sampleRate     float32
	goertzelCoeffs []goertzelCoeff
}

// NewGoertzelFrequencyDetector creates a new Goertzel-based frequency detector
func NewGoertzelFrequencyDetector(sampleRate float32, frequencies ...float32) *GoertzelFrequencyDetector {
	coeffs := make([]goertzelCoeff, len(frequencies))
	for i, freq := range frequencies {
		omega := 2 * math.Pi * freq / sampleRate
		coeff := 2 * float32(math.Cos(float64(omega)))
		coeffs[i] = goertzelCoeff{
			omega:    omega,
			coeff:    coeff,
			cosOmega: float32(math.Cos(float64(omega))),
			sinOmega: float32(math.Sin(float64(omega))),
		}
	}
	
	return &GoertzelFrequencyDetector{
		frequencies:    frequencies,
		sampleRate:     sampleRate,
		goertzelCoeffs: coeffs,
	}
}

// DetectFrequency detects which frequency has the highest magnitude
// Returns the index of the detected frequency and its magnitude
// If the detection is ambiguous (similar magnitudes), returns -1 to indicate noise
func (gfd *GoertzelFrequencyDetector) DetectFrequency(samples []float32) (freqIdx int, magnitude float32) {
	magnitudes := make([]float32, len(gfd.goertzelCoeffs))
	maxMagnitude := float32(0)
	bestFreqIdx := 0
	
	// Calculate magnitude for each frequency
	for freqIdx, coeff := range gfd.goertzelCoeffs {
		mag := gfd.goertzelProcess(samples, coeff)
		magnitudes[freqIdx] = mag
		if mag > maxMagnitude {
			maxMagnitude = mag
			bestFreqIdx = freqIdx
		}
	}
	
	// Check if detection is ambiguous (likely noise)
	// Find second-best magnitude
	secondBest := float32(0)
	for i, mag := range magnitudes {
		if i != bestFreqIdx && mag > secondBest {
			secondBest = mag
		}
	}
	
	// If the best and second-best are too close, it's likely noise
	// For a good FSK signal, the correct frequency should be significantly stronger
	// Use adaptive threshold based on magnitude:
	// - For high magnitudes (clean signal), use lenient threshold (95%)
	// - For low magnitudes (noisy signal), use more lenient threshold (90%) to allow detection
	var ratioThreshold float32 = 0.95
	if maxMagnitude < 0.1 {
		// Lower magnitude suggests noise - use more lenient threshold for noisy signals
		ratioThreshold = 0.90 // More lenient for noisy signals
	}
	if maxMagnitude < 0.05 {
		// Very low magnitude - even more lenient (85%)
		ratioThreshold = 0.85
	}
	
	if maxMagnitude > 0.02 && secondBest > 0 { // Lower threshold from 0.05 to 0.02
		ratio := secondBest / maxMagnitude
		// If second-best is too close to best, it's ambiguous (noise)
		if ratio > ratioThreshold {
			return -1, maxMagnitude // Return -1 to indicate noise/ambiguous
		}
	}
	
	// Check absolute magnitude threshold - lower for noisy signals
	// Valid FSK signals should have reasonable magnitude
	magnitudeThreshold := float32(0.01)
	if maxMagnitude > 0.05 {
		// Higher magnitude - use standard threshold
		magnitudeThreshold = 0.01
	} else {
		// Lower magnitude - use even lower threshold for noisy signals
		magnitudeThreshold = 0.005
	}
	if maxMagnitude < magnitudeThreshold {
		return -1, maxMagnitude // Too low, likely noise
	}
	
	// Additional check: require minimum difference between best and second-best
	// This helps reject noise where magnitudes are similar
	// Use adaptive threshold based on magnitude
	if maxMagnitude > 0.02 && secondBest > 0 { // Lower threshold from 0.05 to 0.02
		diff := maxMagnitude - secondBest
		// Require at least 2% absolute difference for low magnitudes (more lenient)
		diffThreshold := float32(0.002) // Lower from 0.005 to 0.002
		if maxMagnitude < 0.1 && diff < diffThreshold {
			return -1, maxMagnitude // Too close, likely noise
		}
	}
	
	return bestFreqIdx, maxMagnitude
}

// Reset resets the detector state (no state to reset for Goertzel)
func (gfd *GoertzelFrequencyDetector) Reset() {
	// Goertzel is stateless, nothing to reset
}

// goertzelProcess processes samples using Goertzel algorithm
func (gfd *GoertzelFrequencyDetector) goertzelProcess(samples []float32, coeff goertzelCoeff) float32 {
	var q1, q2 float32
	
	for _, sample := range samples {
		q0 := sample + coeff.coeff*q1 - q2
		q2 = q1
		q1 = q0
	}
	
	real := q1 - q2*coeff.cosOmega
	imag := q2 * coeff.sinOmega
	
	magnitude := float32(math.Sqrt(float64(real*real + imag*imag)))
	magnitude *= 2.0 / float32(len(samples)) // Normalize
	
	return magnitude
}

// CorrelationFrequencyDetector uses correlation for frequency detection
// Alternative implementation that can be more robust in some cases
type CorrelationFrequencyDetector struct {
	frequencies []float32
	sampleRate  float32
}

// NewCorrelationFrequencyDetector creates a new correlation-based frequency detector
func NewCorrelationFrequencyDetector(sampleRate float32, frequencies ...float32) *CorrelationFrequencyDetector {
	return &CorrelationFrequencyDetector{
		frequencies: frequencies,
		sampleRate:  sampleRate,
	}
}

// DetectFrequency detects which frequency has the highest correlation
func (cfd *CorrelationFrequencyDetector) DetectFrequency(samples []float32) (freqIdx int, magnitude float32) {
	maxMagnitude := float32(0)
	bestFreqIdx := 0
	
	for freqIdx, freq := range cfd.frequencies {
		var sumI, sumQ float32
		omega := 2 * math.Pi * freq / cfd.sampleRate
		
		// Correlate with cosine and sine (phase-independent)
		for i, sample := range samples {
			phase := omega * float32(i)
			cosPhase := float32(math.Cos(float64(phase)))
			sinPhase := float32(math.Sin(float64(phase)))
			sumI += sample * cosPhase
			sumQ += sample * sinPhase
		}
		
		magnitude := float32(math.Sqrt(float64(sumI*sumI + sumQ*sumQ)))
		magnitude /= float32(len(samples)) // Normalize
		
		if magnitude > maxMagnitude {
			maxMagnitude = magnitude
			bestFreqIdx = freqIdx
		}
	}
	
	return bestFreqIdx, maxMagnitude
}

// Reset resets the detector state (no state to reset for correlation)
func (cfd *CorrelationFrequencyDetector) Reset() {
	// Correlation is stateless, nothing to reset
}


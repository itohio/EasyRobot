package mask

import (
	"github.com/chewxy/math32"
)

// quantizeAmplitude quantizes the amplitude value to the nearest level.
func quantizeAmplitude(value float32, levels []float32) int {
	if len(levels) == 0 {
		return 0
	}
	if len(levels) == 1 {
		return 0
	}

	// Find nearest level
	bestIdx := 0
	bestDist := math32.Abs(value - levels[0])
	for i := 1; i < len(levels); i++ {
		dist := math32.Abs(value - levels[i])
		if dist < bestDist {
			bestDist = dist
			bestIdx = i
		}
	}

	return bestIdx
}

// symbolEnergy calculates the average energy of a symbol period.
func symbolEnergy(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}
	var energy float32
	for _, s := range samples {
		energy += s * s
	}
	return energy / float32(len(samples))
}

// envelopeDetection computes the envelope of a passband signal.
// For real signals, this is simply the absolute value or RMS.
// For proper envelope detection with carrier, we'd need I/Q components.
func envelopeDetection(samples []float32) float32 {
	if len(samples) == 0 {
		return 0
	}
	// Simple RMS envelope
	var sum float32
	for _, s := range samples {
		sum += s * s
	}
	return math32.Sqrt(sum / float32(len(samples)))
}

// normalizeAmplitude normalizes an amplitude value using AGC reference.
func normalizeAmplitude(value, agcReference float32) float32 {
	if agcReference <= 0 {
		return value
	}
	return value / agcReference
}


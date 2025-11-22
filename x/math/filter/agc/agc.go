package agc

import (
	"math"
)

// Filter implements automatic gain control
type Filter struct {
	gain        float32
	targetLevel float32
	timeConst   float32
	alpha       float32 // Smoothing factor
}

// New creates a new AGC filter
func New(targetLevel, timeConst float32) *Filter {
	return &Filter{
		gain:        1.0,
		targetLevel: targetLevel,
		timeConst:   timeConst,
		alpha:       timeConst, // Will be updated based on sample rate
	}
}

// SetSampleRate sets the sample rate for AGC time constant
func (f *Filter) SetSampleRate(sampleRate float32) {
	// Update alpha based on sample rate
	f.alpha = f.timeConst * sampleRate
	if f.alpha > 1.0 {
		f.alpha = 1.0
	}
}

// Process processes a sample with AGC
func (f *Filter) Process(sample float32) float32 {
	// Measure signal level
	level := float32(math.Abs(float64(sample)))

	// Update gain to reach target level
	if level > 0.0001 {
		desiredGain := f.targetLevel / level
		f.gain = f.gain*(1.0-f.alpha) + desiredGain*f.alpha

		// Limit gain range
		if f.gain > 10.0 {
			f.gain = 10.0
		} else if f.gain < 0.1 {
			f.gain = 0.1
		}
	}

	return sample * f.gain
}

// Reset resets AGC state
func (f *Filter) Reset() {
	f.gain = 1.0
}

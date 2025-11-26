package internal

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
)

var _ Processor = (*AGC)(nil)

// AGC implements Automatic Gain Control for complex64 signals.
// It normalizes the magnitude of complex samples to a target level.
type AGC struct {
	gain        float32
	targetLevel float32
	alpha       float32 // Smoothing factor for gain updates
	batchSize   int     // Batch size for processing
}

// NewAGC creates a new AGC processor for complex64 signals.
// targetLevel: desired magnitude level (typical: 1.0)
// timeConst: time constant for gain adjustment (typical: 0.001)
func NewAGC(targetLevel, timeConst float32) AGC {
	return NewAGCWithBatchSize(targetLevel, timeConst, 16)
}

// NewAGCWithBatchSize creates a new AGC processor with custom batch size.
func NewAGCWithBatchSize(targetLevel, timeConst float32, batchSize int) AGC {
	if batchSize <= 0 {
		batchSize = 16
	}
	return AGC{
		gain:        1.0,
		targetLevel: targetLevel,
		alpha:       timeConst,
		batchSize:   batchSize,
	}
}

// SetSampleRate sets the sample rate for AGC time constant calculation.
func (a *AGC) SetSampleRate(sampleRate float32) {
	// Update alpha based on sample rate
	alpha := a.alpha * sampleRate
	if alpha > 1.0 {
		alpha = 1.0
	}
	a.alpha = alpha
}

// Process implements Processor.
// Performs AGC processing on complex64 samples.
func (a *AGC) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	return processInPlace(dst, src, a.batchSize, func(items []complex64) {
		for i := range items {
			sample := items[i]
			iVal := float32(real(sample))
			qVal := float32(imag(sample))

			// Measure magnitude
			mag := math32.Hypot(iVal, qVal)

			// Update gain based on magnitude
			if mag > 1e-6 {
				desiredGain := a.targetLevel / mag
				a.gain = a.gain*(1.0-a.alpha) + desiredGain*a.alpha

				// Limit gain range to prevent instability
				if a.gain > 10.0 {
					a.gain = 10.0
				} else if a.gain < 0.1 {
					a.gain = 0.1
				}
			}

			// Apply gain
			items[i] = complex64(complex(iVal*a.gain, qVal*a.gain))
		}
	})
}

// Reset clears the internal state.
func (a *AGC) Reset() {
	a.gain = 1.0
}


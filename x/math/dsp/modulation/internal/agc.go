package internal

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
)

var _ Processor = (*AGC)(nil)

// AGC implements Automatic Gain Control for complex64 signals.
// It normalizes the magnitude of complex samples to a target level.
// Uses PowerTracker internally to track signal power.
type AGC struct {
	powerTracker PowerTracker[complex64] // Embedded power tracker
	gain         float32
	targetLevel  float32
	targetPower  float32 // Target level squared (for power-based gain calculation)
	alpha        float32 // Smoothing factor for gain updates
	batchSize    int     // Batch size for processing
}

// NewAGC creates a new AGC processor for complex64 signals.
// targetLevel: desired magnitude level (typical: 1.0)
// timeConst: time constant for gain adjustment (typical: 0.001)
// sampleRate: sample rate in Hz
func NewAGC(targetLevel, timeConst, sampleRate float32) AGC {
	return NewAGCWithBatchSize(targetLevel, timeConst, sampleRate, 16)
}

// NewAGCWithBatchSize creates a new AGC processor with custom batch size.
// targetLevel: desired magnitude level (typical: 1.0)
// timeConst: time constant for gain adjustment (typical: 0.001)
// sampleRate: sample rate in Hz
// batchSize: batch size for processing
func NewAGCWithBatchSize(targetLevel, timeConst, sampleRate float32, batchSize int) AGC {
	if batchSize <= 0 {
		batchSize = 16
	}
	// Calculate alpha based on time constant and sample rate
	alpha := timeConst * sampleRate
	if alpha > 1.0 {
		alpha = 1.0
	}
	return AGC{
		powerTracker: NewPowerTracker[complex64](alpha),
		gain:         1.0,
		targetLevel:  targetLevel,
		targetPower:  targetLevel * targetLevel, // Power is magnitude squared
		alpha:        alpha,
		batchSize:    batchSize,
	}
}

// Process implements Processor.
// Performs AGC processing on complex64 samples.
func (a *AGC) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	return processInPlace(dst, src, a.batchSize, func(items []complex64) {
		for i := range items {
			sample := items[i]
			iVal := float32(real(sample))
			qVal := float32(imag(sample))

			// Calculate power (magnitude squared) and update tracker
			power := iVal*iVal + qVal*qVal
			a.powerTracker.UpdatePower(power)

			// Calculate gain from tracked power (EMA)
			// Gain = sqrt(targetPower / trackedPower) = targetLevel / sqrt(trackedPower)
			trackedPower := a.powerTracker.EMA()
			if trackedPower > 1e-12 {
				desiredGain := math32.Sqrt(a.targetPower / trackedPower)
				// Apply EMA smoothing to gain updates
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
	a.powerTracker.Reset()
	a.gain = 1.0
}

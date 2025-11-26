package internal

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
)

var _ Processor = (*CFOCorrector)(nil)

// CFOCorrector implements Carrier Frequency Offset (CFO) correction.
// Applies frequency shift to compensate for carrier frequency offset.
type CFOCorrector struct {
	// Frequency offset state
	frequencyOffset float32 // Frequency offset in Hz
	sampleRate      float32 // Sample rate in Hz

	// Phase accumulator for frequency mixing
	phaseAccumulator float32 // Current phase
	phaseIncrement   float32 // Phase increment per sample

	// Cached sin/cos for rotation
	sin       float32
	cos       float32
	batchSize int // Batch size for processing

}

// NewCFOCorrector creates a new CFO corrector.
// frequencyOffset: carrier frequency offset in Hz
// sampleRate: sample rate in Hz
func NewCFOCorrector(frequencyOffset, sampleRate float32) CFOCorrector {
	return NewCFOCorrectorWithBatchSize(frequencyOffset, sampleRate, 16)
}

// NewCFOCorrectorWithBatchSize creates a new CFO corrector with custom batch size.
func NewCFOCorrectorWithBatchSize(frequencyOffset, sampleRate float32, batchSize int) CFOCorrector {
	if batchSize <= 0 {
		batchSize = 16
	}
	phaseIncrement := 2 * math32.Pi * frequencyOffset / sampleRate

	cfo := CFOCorrector{
		frequencyOffset: frequencyOffset,
		sampleRate:      sampleRate,
		phaseIncrement:  phaseIncrement,
		cos:             1.0,
		batchSize:       batchSize,
	}
	cfo.updateTrig()
	return cfo
}

// updateTrig updates cached sin/cos values.
func (c *CFOCorrector) updateTrig() {
	c.sin, c.cos = math32.Sincos(c.phaseAccumulator)
}

// Process implements Processor.
// Corrects frequency offset by mixing with complex exponential.
func (c *CFOCorrector) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	return processInPlace(dst, src, c.batchSize, func(items []complex64) {
		for idx := range items {
			sample := items[idx]
			iVal := float32(real(sample))
			qVal := float32(imag(sample))

			// Mix with complex exponential: multiply by exp(-j*2π*f_offset*t)
			// This derotates the signal by the frequency offset
			iCorrected := iVal*c.cos + qVal*c.sin
			qCorrected := -iVal*c.sin + qVal*c.cos

			items[idx] = complex64(complex(iCorrected, qCorrected))

			// Update phase accumulator
			c.phaseAccumulator += c.phaseIncrement
			// Wrap to [-π, π) for accuracy
			for c.phaseAccumulator > math32.Pi {
				c.phaseAccumulator -= 2 * math32.Pi
				c.updateTrig()
			}
			for c.phaseAccumulator < -math32.Pi {
				c.phaseAccumulator += 2 * math32.Pi
				c.updateTrig()
			}
		}
	})
}

// Reset clears the internal state (frequency offset is preserved).
func (c *CFOCorrector) Reset() {
	c.phaseAccumulator = 0
	c.updateTrig()
}

// SetFrequencyOffset sets the frequency offset in Hz.
func (c *CFOCorrector) SetFrequencyOffset(offsetHz float32) {
	c.frequencyOffset = offsetHz
	c.phaseIncrement = 2 * math32.Pi * offsetHz / c.sampleRate
}

// FrequencyOffset returns the current frequency offset in Hz.
func (c *CFOCorrector) FrequencyOffset() float32 {
	return c.frequencyOffset
}

// Phase returns the current phase accumulator in radians.
func (c *CFOCorrector) Phase() float32 {
	return c.phaseAccumulator
}


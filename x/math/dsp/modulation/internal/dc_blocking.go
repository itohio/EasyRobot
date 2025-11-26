package internal

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/buffer"
)

var _ Processor = (*DCBlocking)(nil)

// DCBlocking implements a DC blocking filter (high-pass) for complex64 signals.
// Removes DC offset from I and Q components independently.
type DCBlocking struct {
	// High-pass filter state (IIR)
	alpha        float32 // Filter coefficient (typically 0.995 to 0.999)
	iDcEstimate  float32 // Estimated DC offset for I component
	qDcEstimate  float32 // Estimated DC offset for Q component
	batchSize    int     // Batch size for processing
}

// NewDCBlocking creates a new DC blocking filter.
// cutoffFreq: cutoff frequency in Hz
// sampleRate: sample rate in Hz
func NewDCBlocking(cutoffFreq, sampleRate float32) DCBlocking {
	return NewDCBlockingWithBatchSize(cutoffFreq, sampleRate, 16)
}

// NewDCBlockingWithBatchSize creates a new DC blocking filter with custom batch size.
func NewDCBlockingWithBatchSize(cutoffFreq, sampleRate float32, batchSize int) DCBlocking {
	if cutoffFreq <= 0 || sampleRate <= 0 {
		panic("cutoffFreq and sampleRate must be positive")
	}
	if batchSize <= 0 {
		batchSize = 16
	}

	// IIR high-pass filter coefficient
	// alpha = exp(-2*π*cutoff/sampleRate)
	alpha := math32.Exp(-2 * math32.Pi * cutoffFreq / sampleRate)

	return NewDCBlockingAlphaWithBatchSize(alpha, batchSize)
}

// NewDCBlockingAlpha creates a DC blocking filter with direct alpha parameter.
// alpha: filter coefficient (0.995 to 0.999 for typical applications)
func NewDCBlockingAlpha(alpha float32) DCBlocking {
	return NewDCBlockingAlphaWithBatchSize(alpha, 16)
}

// NewDCBlockingAlphaWithBatchSize creates a DC blocking filter with custom batch size.
func NewDCBlockingAlphaWithBatchSize(alpha float32, batchSize int) DCBlocking {
	if alpha <= 0 || alpha >= 1 {
		panic("alpha must be in (0, 1)")
	}
	if batchSize <= 0 {
		batchSize = 16
	}
	return DCBlocking{
		alpha:     alpha,
		batchSize: batchSize,
	}
}

// Process implements Processor.
// Removes DC offset from samples.
func (d *DCBlocking) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	return processInPlace(dst, src, d.batchSize, func(items []complex64) {
		for idx := range items {
			sample := items[idx]
			iVal := float32(real(sample))
			qVal := float32(imag(sample))

			// Update DC estimates using exponential moving average
			d.iDcEstimate = d.alpha*d.iDcEstimate + (1-d.alpha)*iVal
			d.qDcEstimate = d.alpha*d.qDcEstimate + (1-d.alpha)*qVal

			// Subtract DC estimate (high-pass filter)
			iFiltered := iVal - d.iDcEstimate
			qFiltered := qVal - d.qDcEstimate

			items[idx] = complex64(complex(iFiltered, qFiltered))
		}
	})
}

// Reset clears the internal state.
func (d *DCBlocking) Reset() {
	d.iDcEstimate = 0
	d.qDcEstimate = 0
}

// DCOffset returns the current estimated DC offset for I and Q components.
func (d *DCBlocking) DCOffset() (iOffset, qOffset float32) {
	return d.iDcEstimate, d.qDcEstimate
}

// SetAlpha sets the filter coefficient.
func (d *DCBlocking) SetAlpha(alpha float32) {
	if alpha <= 0 || alpha >= 1 {
		panic("alpha must be in (0, 1)")
	}
	d.alpha = alpha
}

// Alpha returns the current filter coefficient.
func (d *DCBlocking) Alpha() float32 {
	return d.alpha
}


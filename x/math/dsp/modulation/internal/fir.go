package internal

import (
	"github.com/itohio/EasyRobot/x/buffer"
	"github.com/itohio/EasyRobot/x/math/filter/fir"
)

var _ Processor = (*FIR)(nil)

// FIR implements a Finite Impulse Response filter for complex64 signals.
// It applies separate FIR filters to I and Q components.
type FIR struct {
	// Separate FIR filters for I and Q components
	iFIR *fir.FIR
	qFIR *fir.FIR

	// Batch size for processing when dst doesn't implement Viewer
	batchSize int
}

// NewFIR creates a new complex64 FIR filter processor.
// The coefficients are applied to both I and Q components.
func NewFIR(coeffs ...float32) (FIR, error) {
	return NewFIRWithBatchSize(16, coeffs...)
}

// NewFIRWithBatchSize creates a new complex64 FIR filter processor with custom batch size.
func NewFIRWithBatchSize(batchSize int, coeffs ...float32) (FIR, error) {
	if len(coeffs) < 2 {
		return FIR{}, &ConfigError{Message: "FIR filter must have at least 2 coefficients"}
	}
	if batchSize <= 0 {
		batchSize = 16
	}

	return FIR{
		iFIR:      fir.New(coeffs...),
		qFIR:      fir.New(coeffs...),
		batchSize: batchSize,
	}, nil
}

// Process implements Processor.
// Performs filtering on complex64 samples.
func (f *FIR) Process(dst buffer.Writer[complex64], src buffer.Reader[complex64]) int {
	return processInPlace(dst, src, f.batchSize, func(items []complex64) {
		for i := range items {
			sample := items[i]
			// Apply FIR filter to I component
			iFiltered := f.iFIR.Process(float32(real(sample)))
			// Apply FIR filter to Q component
			qFiltered := f.qFIR.Process(float32(imag(sample)))
			items[i] = complex64(complex(iFiltered, qFiltered))
		}
	})
}

// Reset clears the internal state.
func (f *FIR) Reset() {
	if f.iFIR != nil {
		f.iFIR.Reset()
	}
	if f.qFIR != nil {
		f.qFIR.Reset()
	}
}

// ConfigError represents a configuration error.
type ConfigError struct {
	Message string
}

func (e *ConfigError) Error() string {
	return "internal: " + e.Message
}


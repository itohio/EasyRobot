package fir

import (
	"github.com/chewxy/math32"
)

// BandSpec defines a frequency band specification for filter design.
type BandSpec struct {
	StartFreq float32 // Start frequency in Hz
	EndFreq   float32 // End frequency in Hz
	Gain      float32 // Desired gain (1.0 = pass, 0.0 = stop)
	Weight    float32 // Relative importance weight
}

// FIR implements a Finite Impulse Response digital filter.
// It uses circular buffers for efficient processing and maintains
// internal state for continuous filtering operations.
type FIR struct {
	// Filter coefficients (impulse response)
	coeffs []float32

	// Internal circular buffer for input history
	buffer []float32
	bufIdx int

	// Filter order (length of coeffs - 1)
	order int
}

// New creates a new FIR filter with the specified coefficients.
func New(coeffs ...float32) *FIR {
	if len(coeffs) < 2 {
		panic("FIR filter must have at least 2 coefficients")
	}

	order := len(coeffs) - 1
	f := &FIR{
		order:  order,
		coeffs: make([]float32, len(coeffs)),
		buffer: make([]float32, len(coeffs)),
		bufIdx: 0,
	}

	copy(f.coeffs, coeffs)
	return f
}

// Reset resets the internal filter state.
func (f *FIR) Reset() {
	f.bufIdx = 0
	for i := range f.buffer {
		f.buffer[i] = 0
	}
}

// Process processes a single input sample and returns the filtered output.
// This is the core filtering operation using circular buffer convolution.
func (f *FIR) Process(input float32) float32 {
	// Store input in circular buffer
	f.buffer[f.bufIdx] = input

	// Compute output using circular convolution
	var output float32
	for i, coeff := range f.coeffs {
		// Calculate buffer index with wraparound
		bufPos := f.bufIdx - i
		if bufPos < 0 {
			bufPos += len(f.buffer)
		}
		output += coeff * f.buffer[bufPos]
	}

	// Update buffer index
	f.bufIdx = (f.bufIdx + 1) % len(f.buffer)

	return output
}

// ProcessBuffer processes a buffer of input samples and stores results in output buffer.
// Both input and output must be the same length.
func (f *FIR) ProcessBuffer(input, output []float32) {
	if len(input) != len(output) {
		panic("input and output buffers must be the same length")
	}

	for i, in := range input {
		output[i] = f.Process(in)
	}
}

// GetCoeffs returns a copy of the filter coefficients.
func (f *FIR) GetCoeffs() []float32 {
	coeffs := make([]float32, len(f.coeffs))
	copy(coeffs, f.coeffs)
	return coeffs
}

// SetCoeffs sets new filter coefficients.
func (f *FIR) SetCoeffs(coeffs []float32) {
	if len(coeffs) != len(f.coeffs) {
		panic("coefficient array length must match filter order+1")
	}
	copy(f.coeffs, coeffs)
}

// sinc computes the sinc function: sin(pi*x)/(pi*x)
func sinc(x float32) float32 {
	if x == 0 {
		return 1.0
	}
	return math32.Sin(math32.Pi*x) / (math32.Pi * x)
}

// hamming computes the Hamming window function.
func hamming(n, N int) float32 {
	return 0.54 - 0.46*math32.Cos(2*math32.Pi*float32(n)/float32(N-1))
}

// NewLowPass creates a low-pass FIR filter using windowed sinc method.
func NewLowPass(order int, cutoffHz, sampleRate float32) *FIR {
	if order < 1 {
		panic("filter order must be >= 1")
	}
	if cutoffHz <= 0 || cutoffHz >= sampleRate/2 {
		panic("cutoff frequency must be between 0 and Nyquist frequency")
	}

	// Normalize cutoff frequency
	normalizedCutoff := cutoffHz / sampleRate

	// Calculate windowed sinc coefficients
	halfOrder := order / 2
	coeffs := make([]float32, order+1)
	for i := 0; i <= order; i++ {
		n := i - halfOrder
		// Sinc response
		coeffs[i] = 2 * normalizedCutoff * sinc(2*normalizedCutoff*float32(n))
		// Apply Hamming window
		coeffs[i] *= hamming(i, order+1)
	}

	// Create filter
	f := New(coeffs...)

	return f
}

// NewHighPass creates a high-pass FIR filter using windowed sinc method.
func NewHighPass(order int, cutoffHz, sampleRate float32) *FIR {
	if order < 1 {
		panic("filter order must be >= 1")
	}
	if cutoffHz <= 0 || cutoffHz >= sampleRate/2 {
		panic("cutoff frequency must be between 0 and Nyquist frequency")
	}

	// Normalize cutoff frequency
	normalizedCutoff := cutoffHz / sampleRate

	// Calculate windowed sinc coefficients
	halfOrder := order / 2
	coeffs := make([]float32, order+1)
	for i := 0; i <= order; i++ {
		n := i - halfOrder
		if n == 0 {
			// Special case for DC component
			coeffs[i] = 1.0 - 2*normalizedCutoff
		} else {
			// Sinc response
			coeffs[i] = -2 * normalizedCutoff * sinc(2*normalizedCutoff*float32(n))
		}
		// Apply Hamming window
		coeffs[i] *= hamming(i, order+1)
	}

	// Create filter
	f := New(coeffs...)

	return f
}

// NewGeneralBand creates a general band FIR filter with arbitrary frequency response.
// This is a simplified implementation - for production use, consider Remez algorithm.
func NewGeneralBand(order int, bands []BandSpec, sampleRate float32) *FIR {
	if order < 1 {
		panic("filter order must be >= 1")
	}
	if len(bands) == 0 {
		panic("at least one band specification required")
	}

	// For now, implement as a simple approximation
	// In a full implementation, this would use frequency sampling or Remez algorithm
	// Here we use a basic approach that combines multiple sinc responses

	halfOrder := order / 2
	coeffs := make([]float32, order+1)
	for i := 0; i <= order; i++ {
		n := i - halfOrder
		var sum float32

		for _, band := range bands {
			if band.StartFreq >= band.EndFreq {
				continue
			}

			// Normalize frequencies
			f1 := band.StartFreq / sampleRate
			f2 := band.EndFreq / sampleRate

			// Simple band response approximation
			if n == 0 {
				sum += band.Gain * (f2 - f1) * 2
			} else {
				// Integrate sinc over the band
				sum += band.Gain * 2 * (sinc(2*f2*float32(n)) - sinc(2*f1*float32(n)))
			}
		}

		coeffs[i] = sum * hamming(i, order+1)
	}

	// Create filter
	f := New(coeffs...)

	return f
}

// NewBandPass creates a band-pass FIR filter.
func NewBandPass(order int, lowCutoffHz, highCutoffHz, sampleRate float32) *FIR {
	if lowCutoffHz >= highCutoffHz {
		panic("low cutoff must be less than high cutoff")
	}

	// Create band-pass as combination of low-pass and high-pass
	bands := []BandSpec{
		{StartFreq: 0, EndFreq: lowCutoffHz, Gain: 0.0, Weight: 1.0},
		{StartFreq: lowCutoffHz, EndFreq: highCutoffHz, Gain: 1.0, Weight: 1.0},
		{StartFreq: highCutoffHz, EndFreq: sampleRate / 2, Gain: 0.0, Weight: 1.0},
	}

	return NewGeneralBand(order, bands, sampleRate)
}

// NewBandStop creates a band-stop FIR filter.
func NewBandStop(order int, lowCutoffHz, highCutoffHz, sampleRate float32) *FIR {
	if lowCutoffHz >= highCutoffHz {
		panic("low cutoff must be less than high cutoff")
	}

	// Create band-stop as combination of low-pass and high-pass
	bands := []BandSpec{
		{StartFreq: 0, EndFreq: lowCutoffHz, Gain: 1.0, Weight: 1.0},
		{StartFreq: lowCutoffHz, EndFreq: highCutoffHz, Gain: 0.0, Weight: 1.0},
		{StartFreq: highCutoffHz, EndFreq: sampleRate / 2, Gain: 1.0, Weight: 1.0},
	}

	return NewGeneralBand(order, bands, sampleRate)
}

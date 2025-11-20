package iir

import (
	"github.com/chewxy/math32"
)

// IIR implements an Infinite Impulse Response digital filter.
// It uses the direct form II implementation for efficiency and maintains
// internal buffers for input and output history.
type IIR struct {
	// Feedforward coefficients (numerator)
	b []float32

	// Feedback coefficients (denominator, a[0] = 1.0)
	a []float32

	// Internal buffers for input/output history (Direct Form II)
	w []float32 // Delay line for intermediate values

	// Filter order
	order int
}

// New creates a new IIR filter with the specified coefficients.
// Coefficients should be provided as [b0, b1, ..., bn, a1, a2, ..., am]
// where a0 is always 1.0 (normalized).
func New(coeffs ...float32) *IIR {
	if len(coeffs) < 3 {
		panic("IIR filter must have at least 3 coefficients [b0, b1, a1]")
	}

	// Assume equal number of b and a coefficients for now
	// In practice, this might need adjustment based on the actual filter design
	totalCoeffs := len(coeffs)
	if totalCoeffs%2 != 1 {
		panic("IIR filter coefficients should have odd length: [b0,b1,...,bn,a1,a2,...,am]")
	}

	// Split coefficients: first half+1 are b coeffs, rest are a coeffs (excluding a0=1.0)
	numBCoeffs := (totalCoeffs + 1) / 2
	numACoeffs := numBCoeffs // a has same number of coeffs as b

	b := make([]float32, numBCoeffs)
	a := make([]float32, numACoeffs)

	// Copy b coefficients
	copy(b, coeffs[:numBCoeffs])

	// Set a[0] = 1.0 (normalized)
	a[0] = 1.0

	// Copy a coefficients (starting from a1)
	copy(a[1:], coeffs[numBCoeffs:])

	order := len(b) - 1
	i := &IIR{
		order: order,
		b:     b,
		a:     a,
		w:     make([]float32, order),
	}

	return i
}

// Reset resets the internal filter state.
func (i *IIR) Reset() {
	for j := range i.w {
		i.w[j] = 0
	}
}

// Process processes a single input sample and returns the filtered output.
// Uses Direct Form II implementation for efficiency.
func (i *IIR) Process(input float32) float32 {
	// Direct Form II implementation
	// w[n] = x[n] - Σ(k=1 to N) a[k]*w[n-k]
	// y[n] = Σ(k=0 to N) b[k]*w[n-k]

	// Compute new w[0]
	w0 := input
	for k := 1; k <= i.order; k++ {
		w0 -= i.a[k] * i.w[k-1]
	}

	// Compute output
	output := i.b[0] * w0
	for k := 1; k <= i.order; k++ {
		output += i.b[k] * i.w[k-1]
	}

	// Shift delay line
	for k := i.order - 1; k > 0; k-- {
		i.w[k] = i.w[k-1]
	}
	i.w[0] = w0

	return output
}

// ProcessBuffer processes a buffer of input samples and stores results in output buffer.
// Both input and output must be the same length.
func (i *IIR) ProcessBuffer(input, output []float32) {
	if len(input) != len(output) {
		panic("input and output buffers must be the same length")
	}

	for j, in := range input {
		output[j] = i.Process(in)
	}
}

// GetCoeffs returns copies of the filter coefficients.
func (i *IIR) GetCoeffs() (b, a []float32) {
	bCoeffs := make([]float32, len(i.b))
	aCoeffs := make([]float32, len(i.a))
	copy(bCoeffs, i.b)
	copy(aCoeffs, i.a)
	return bCoeffs, aCoeffs
}

// SetCoeffs sets new filter coefficients.
func (i *IIR) SetCoeffs(b, a []float32) {
	if len(b) != len(i.b) || len(a) != len(i.a) {
		panic("coefficient array lengths must match filter order+1")
	}
	if a[0] != 1.0 {
		panic("a[0] must be 1.0 (normalized)")
	}
	copy(i.b, b)
	copy(i.a, a)
}

// ButterworthLowPass creates a Butterworth low-pass IIR filter.
func NewButterworthLowPass(order int, cutoffHz, sampleRate float32) *IIR {
	if order < 1 {
		panic("filter order must be >= 1")
	}
	if cutoffHz <= 0 || cutoffHz >= sampleRate/2 {
		panic("cutoff frequency must be between 0 and Nyquist frequency")
	}

	// Simplified second-order Butterworth implementation
	// Using standard coefficients for testing

	// Pre-warp cutoff frequency
	wc := 2.0 * math32.Pi * cutoffHz
	k := wc / math32.Tan(wc/(2.0*sampleRate))

	// Second order Butterworth analog: poles at -0.707 ± j*0.707
	poleReal := -0.7071067811865476 * k
	poleImag := 0.7071067811865476 * k

	// Bilinear transform
	T := 2.0 * sampleRate // Sample period (normalized)
	denom := T*T + 2.0*poleReal*T + poleReal*poleReal + poleImag*poleImag
	if denom == 0 {
		denom = 1e-10
	}

	b0 := T * T / denom
	b1 := 2.0 * b0
	b2 := b0
	a1 := (2.0*T*T - 2.0*poleReal*poleReal - 2.0*poleImag*poleImag) / denom
	a2 := (T*T - 2.0*poleReal*T + poleReal*poleReal + poleImag*poleImag) / denom

	b := []float32{b0, b1, b2}
	a := []float32{1.0, a1, a2}

	return New(b[0], b[1], b[2], a[1], a[2])
}

// ButterworthHighPass creates a Butterworth high-pass IIR filter.
func NewButterworthHighPass(order int, cutoffHz, sampleRate float32) *IIR {
	if order < 1 {
		panic("filter order must be >= 1")
	}
	if cutoffHz <= 0 || cutoffHz >= sampleRate/2 {
		panic("cutoff frequency must be between 0 and Nyquist frequency")
	}

	// Simple first-order high-pass for now
	// H(z) = (1 - z^-1) / (1 - α*z^-1) where α = (tan(π*fc/fs) - 1) / (tan(π*fc/fs) + 1)
	fc := cutoffHz / sampleRate
	alpha := (math32.Tan(math32.Pi*fc) - 1.0) / (math32.Tan(math32.Pi*fc) + 1.0)

	b := []float32{(1.0 + alpha) / 2.0, -(1.0 + alpha) / 2.0}
	a := []float32{1.0, -alpha}

	return New(b[0], b[1], a[1])
}

// ButterworthBandPass creates a Butterworth band-pass IIR filter.
func NewButterworthBandPass(order int, lowCutoffHz, highCutoffHz, sampleRate float32) *IIR {
	if order < 2 || order%2 != 0 {
		panic("band-pass filter order must be even and >= 2")
	}
	if lowCutoffHz >= highCutoffHz {
		panic("low cutoff must be less than high cutoff")
	}

	// Simple second-order band-pass
	center := math32.Sqrt(lowCutoffHz * highCutoffHz)
	bw := highCutoffHz - lowCutoffHz

	f0 := center / sampleRate
	q := center / bw

	alpha := math32.Sin(2.0*math32.Pi*f0) / (2.0 * q)
	cosW0 := math32.Cos(2.0 * math32.Pi * f0)

	b0 := alpha
	b1 := float32(0.0)
	b2 := -alpha
	a0 := 1.0 + alpha
	a1 := -2.0 * cosW0
	a2 := 1.0 - alpha

	b := []float32{b0 / a0, b1 / a0, b2 / a0}
	a := []float32{1.0, a1 / a0, a2 / a0}

	return New(b[0], b[1], b[2], a[1], a[2])
}

// ButterworthBandStop creates a Butterworth band-stop IIR filter.
func NewButterworthBandStop(order int, lowCutoffHz, highCutoffHz, sampleRate float32) *IIR {
	if order < 2 || order%2 != 0 {
		panic("band-stop filter order must be even and >= 2")
	}
	if lowCutoffHz >= highCutoffHz {
		panic("low cutoff must be less than high cutoff")
	}

	// Simple second-order band-stop
	center := math32.Sqrt(lowCutoffHz * highCutoffHz)
	bw := highCutoffHz - lowCutoffHz

	f0 := center / sampleRate
	q := center / bw

	alpha := math32.Sin(2.0*math32.Pi*f0) / (2.0 * q)
	cosW0 := math32.Cos(2.0 * math32.Pi * f0)

	b0 := float32(1.0)
	b1 := -2.0 * cosW0
	b2 := float32(1.0)
	a0 := 1.0 + alpha
	a1 := -2.0 * cosW0
	a2 := 1.0 - alpha

	b := []float32{b0 / a0, b1 / a0, b2 / a0}
	a := []float32{1.0, a1 / a0, a2 / a0}

	return New(b[0], b[1], b[2], a[1], a[2])
}

// ChebyshevILowPass creates a Chebyshev I low-pass IIR filter.
// For now, falls back to Butterworth (Chebyshev implementation is complex)
func NewChebyshevLowPass(order int, cutoffHz, sampleRate, rippleDb float32) *IIR {
	// TODO: Implement proper Chebyshev I design
	return NewButterworthLowPass(order, cutoffHz, sampleRate)
}

// ChebyshevIHighPass creates a Chebyshev I high-pass IIR filter.
// For now, falls back to Butterworth (Chebyshev implementation is complex)
func NewChebyshevHighPass(order int, cutoffHz, sampleRate, rippleDb float32) *IIR {
	// TODO: Implement proper Chebyshev I design
	return NewButterworthHighPass(order, cutoffHz, sampleRate)
}

// ChebyshevIBandPass creates a Chebyshev I band-pass IIR filter.
// For now, falls back to Butterworth (Chebyshev implementation is complex)
func NewChebyshevBandPass(order int, lowCutoffHz, highCutoffHz, sampleRate, rippleDb float32) *IIR {
	// TODO: Implement proper Chebyshev I design
	return NewButterworthBandPass(order, lowCutoffHz, highCutoffHz, sampleRate)
}

// ChebyshevIBandStop creates a Chebyshev I band-stop IIR filter.
// For now, falls back to Butterworth (Chebyshev implementation is complex)
func NewChebyshevBandStop(order int, lowCutoffHz, highCutoffHz, sampleRate, rippleDb float32) *IIR {
	// TODO: Implement proper Chebyshev I design
	return NewButterworthBandStop(order, lowCutoffHz, highCutoffHz, sampleRate)
}

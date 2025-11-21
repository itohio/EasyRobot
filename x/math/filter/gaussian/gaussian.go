package gaussian

// Gaussian filter for smoothing signals

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Filter implements a Gaussian smoothing filter.
type Filter struct {
	sigma        float32
	windowSize   int
	coefficients vec.Vector
	buffer       vec.Vector
	index        int
	initialized  bool
}

// New creates a new Gaussian filter.
// sigma controls the smoothing amount (larger = more smoothing).
// windowSize must be odd (will be adjusted if even).
func New(sigma float32, windowSize int) *Filter {
	if sigma <= 0 {
		panic("Gaussian sigma must be positive")
	}
	if windowSize <= 0 {
		panic("Gaussian window size must be positive")
	}

	// Ensure window size is odd
	if windowSize%2 == 0 {
		windowSize++
	}

	coeffs := computeCoefficients(sigma, windowSize)
	return &Filter{
		sigma:        sigma,
		windowSize:   windowSize,
		coefficients: coeffs,
		buffer:       vec.New(windowSize),
		index:        0,
		initialized:  false,
	}
}

// Reset resets the filter state.
func (gf *Filter) Reset() {
	for i := range gf.buffer {
		gf.buffer[i] = 0
	}
	gf.index = 0
	gf.initialized = false
}

// Process processes a single sample and returns the filtered value.
func (gf *Filter) Process(sample float32) float32 {
	gf.buffer[gf.index] = sample
	gf.index = (gf.index + 1) % gf.windowSize
	gf.initialized = true

	// Need at least half the window size for valid output
	halfWindow := gf.windowSize / 2
	if !gf.initialized || gf.index < halfWindow {
		return sample
	}

	// Apply filter: weighted average using Gaussian coefficients
	result := float32(0)
	sum := float32(0)
	coeffIdx := 0
	for i := 0; i < gf.windowSize; i++ {
		bufIdx := (gf.index - halfWindow + i) % gf.windowSize
		if bufIdx < 0 {
			bufIdx += gf.windowSize
		}
		coeff := gf.coefficients[coeffIdx]
		result += coeff * gf.buffer[bufIdx]
		sum += coeff
		coeffIdx++
	}

	// Normalize by sum of coefficients
	if sum > 0 {
		result /= sum
	}
	return result
}

// ProcessBuffer processes an entire buffer of samples.
// Returns a new vector with filtered values.
func (gf *Filter) ProcessBuffer(input vecTypes.Vector) vec.Vector {
	inputVec := input.View().(vec.Vector)
	output := vec.New(len(inputVec))

	// Reset filter state
	gf.Reset()

	// Process each sample
	for i := range inputVec {
		output[i] = gf.Process(inputVec[i])
	}

	return output
}

// Gaussian applies Gaussian filtering to a signal vector.
// This is a convenience function that creates a filter and processes the entire signal.
func Gaussian(signal vecTypes.Vector, sigma float32, windowSize int) vec.Vector {
	filter := New(sigma, windowSize)
	return filter.ProcessBuffer(signal)
}

// computeCoefficients calculates Gaussian filter coefficients.
// Uses 1D Gaussian kernel: G(x) = exp(-x^2 / (2 * sigma^2))
func computeCoefficients(sigma float32, windowSize int) vec.Vector {
	halfWindow := windowSize / 2
	coefficients := vec.New(windowSize)
	sum := float32(0)

	// Two sigma squared
	twoSigmaSq := 2.0 * sigma * sigma

	// Compute Gaussian kernel
	for i := 0; i < windowSize; i++ {
		x := float32(i - halfWindow)
		coeff := math32.Exp(-(x * x) / twoSigmaSq)
		coefficients[i] = coeff
		sum += coeff
	}

	// Normalize coefficients so they sum to 1
	for i := range coefficients {
		coefficients[i] /= sum
	}

	return coefficients
}

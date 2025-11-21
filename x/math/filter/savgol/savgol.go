package savgol

// Savitzky-Golay filter for smoothing while preserving peak shape

import (
	"fmt"

	"github.com/itohio/EasyRobot/x/math/mat"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

// Filter implements a Savitzky-Golay filter for smoothing signals.
// This filter preserves peak shape while reducing noise.
type Filter struct {
	windowSize      int
	polynomialOrder int
	coefficients    vec.Vector
	buffer          vec.Vector
	index           int
	initialized     bool
}

// New creates a new Savitzky-Golay filter.
// windowSize must be odd and greater than polynomialOrder.
func New(windowSize, polynomialOrder int) *Filter {
	if windowSize <= 0 || windowSize%2 == 0 {
		panic("Savitzky-Golay window size must be positive and odd")
	}
	if polynomialOrder < 0 || polynomialOrder >= windowSize {
		panic("Savitzky-Golay polynomial order must be non-negative and less than window size")
	}

	coeffs := computeCoefficients(windowSize, polynomialOrder)
	return &Filter{
		windowSize:      windowSize,
		polynomialOrder: polynomialOrder,
		coefficients:    coeffs,
		buffer:          vec.New(windowSize),
		index:           0,
		initialized:     false,
	}
}

// Reset resets the filter state.
func (sg *Filter) Reset() {
	for i := range sg.buffer {
		sg.buffer[i] = 0
	}
	sg.index = 0
	sg.initialized = false
}

// Process processes a single sample and returns the filtered value.
// The filter needs at least (windowSize+1)/2 samples before producing valid output.
func (sg *Filter) Process(sample float32) float32 {
	sg.buffer[sg.index] = sample
	sg.index = (sg.index + 1) % sg.windowSize
	sg.initialized = true

	// Need at least half the window size for valid output
	halfWindow := sg.windowSize / 2

	// Check if we have enough samples for valid output
	// We need to track how many samples we've seen
	if !sg.initialized {
		return sample
	}

	// For simplicity, only produce valid output after full window is filled
	// This could be optimized to use boundary handling for earlier output
	if sg.index < halfWindow {
		return sample // Not enough samples yet
	}

	// Apply filter: dot product of coefficients with window centered at current position
	result := float32(0)
	for i := 0; i < sg.windowSize; i++ {
		bufIdx := (sg.index - halfWindow + i + sg.windowSize) % sg.windowSize
		result += sg.coefficients[i] * sg.buffer[bufIdx]
	}

	return result
}

// ProcessBuffer processes an entire buffer of samples.
// Returns a new vector with filtered values.
func (sg *Filter) ProcessBuffer(input vecTypes.Vector) vec.Vector {
	inputVec := input.View().(vec.Vector)
	output := vec.New(len(inputVec))

	// Reset filter state
	sg.Reset()

	// Process each sample
	for i := range inputVec {
		output[i] = sg.Process(inputVec[i])
	}

	return output
}

// SavitzkyGolay applies Savitzky-Golay filtering to a signal vector.
// This is a convenience function that creates a filter and processes the entire signal.
func SavitzkyGolay(signal vecTypes.Vector, windowSize, polynomialOrder int) vec.Vector {
	filter := New(windowSize, polynomialOrder)
	return filter.ProcessBuffer(signal)
}

// computeCoefficients calculates the Savitzky-Golay filter coefficients.
// Uses Gram-Schmidt orthogonalization to compute the least-squares fit coefficients.
func computeCoefficients(windowSize, polynomialOrder int) vec.Vector {
	halfWindow := windowSize / 2

	// Build Vandermonde matrix A: rows are positions, columns are polynomial powers
	// A[i][j] = i^j where i ranges from -halfWindow to +halfWindow
	positions := vec.New(windowSize)
	for i := 0; i < windowSize; i++ {
		positions[i] = float32(i - halfWindow)
	}

	// Build design matrix (Vandermonde matrix)
	designMatrix := mat.New(windowSize, polynomialOrder+1)
	for i := 0; i < windowSize; i++ {
		x := positions[i]
		designMatrix[i][0] = 1.0 // x^0 = 1
		for j := 1; j <= polynomialOrder; j++ {
			designMatrix[i][j] = designMatrix[i][j-1] * x // x^j = x^(j-1) * x
		}
	}

	// Solve least squares: (A^T * A)^(-1) * A^T
	// For Savitzky-Golay, we want coefficients for the center point

	// Compute (A^T * A)
	ata := mat.New(polynomialOrder+1, polynomialOrder+1)
	for i := 0; i <= polynomialOrder; i++ {
		for j := 0; j <= polynomialOrder; j++ {
			sum := float32(0)
			for k := 0; k < windowSize; k++ {
				sum += designMatrix[k][i] * designMatrix[k][j]
			}
			ata[i][j] = sum
		}
	}

	// Invert (A^T * A) using pseudo-inverse
	invAta := mat.New(polynomialOrder+1, polynomialOrder+1)
	if err := ata.PseudoInverse(invAta); err != nil {
		panic(fmt.Sprintf("Savitzky-Golay: failed to compute pseudo-inverse: %v", err))
	}

	// Compute A^T (transpose of design matrix)
	at := mat.New(polynomialOrder+1, windowSize)
	for i := 0; i <= polynomialOrder; i++ {
		for j := 0; j < windowSize; j++ {
			at[i][j] = designMatrix[j][i]
		}
	}

	// Compute inv(A^T * A) * A^T
	// This gives us the projection matrix that maps data to polynomial coefficients
	invAtaAt := mat.New(polynomialOrder+1, windowSize)
	for i := 0; i <= polynomialOrder; i++ {
		for j := 0; j < windowSize; j++ {
			sum := float32(0)
			for k := 0; k <= polynomialOrder; k++ {
				sum += invAta[i][k] * at[k][j]
			}
			invAtaAt[i][j] = sum
		}
	}

	// For Savitzky-Golay, we want the filter coefficients for the center point (position 0).
	// The center point in polynomial space is [1, 0, 0, ..., 0]^T (constant term = 1, all derivatives = 0).
	// So we compute: A * inv(A^T * A) * A^T * [1, 0, 0, ..., 0]^T
	// But we only need the first row of invAtaAt (which corresponds to constant term coefficient)
	// and multiply it by the design matrix to get the filter coefficients.

	// Extract the first row of invAtaAt (polynomial coefficients for constant term at center)
	polyCoeffs := vec.New(polynomialOrder + 1)
	for i := 0; i <= polynomialOrder; i++ {
		// The center point is at position 0 in the window (after shifting by -halfWindow)
		// For the constant term (x^0 = 1), we use the first row of invAtaAt
		polyCoeffs[i] = invAtaAt[i][halfWindow]
	}

	// Now compute A * polyCoeffs to get the filter coefficients
	// This gives us the fitted values at each position in the window
	coefficients := vec.New(windowSize)
	for i := 0; i < windowSize; i++ {
		sum := float32(0)
		for j := 0; j <= polynomialOrder; j++ {
			sum += designMatrix[i][j] * polyCoeffs[j]
		}
		coefficients[i] = sum
	}

	return coefficients
}

package rs

import (
	"fmt"
	"math"
)

// CalculateRSParams calculates optimal Reed-Solomon parameters for given requirements.
// Parameters:
//   - dataLen: Expected data length in bytes
//   - maxErrors: Maximum number of errors to correct per codeword
//
// Returns: (n, k, t) where:
//   - n: Total codeword length
//   - k: Data symbols per codeword
//   - t: Maximum correctable errors
func CalculateRSParams(dataLen, maxErrors int) (n, k, t int, err error) {
	if dataLen <= 0 {
		return 0, 0, 0, fmt.Errorf("dataLen must be positive, got %d", dataLen)
	}
	if maxErrors <= 0 {
		return 0, 0, 0, fmt.Errorf("maxErrors must be positive, got %d", maxErrors)
	}

	// For byte-level RS, n must be <= 255
	// We want: t = maxErrors, and t = (n-k)/2
	// So: n - k = 2*t, and we want to minimize overhead
	// Choose k to be close to dataLen, but ensure n <= 255

	// Start with a reasonable k value
	k = dataLen
	if k > 200 {
		k = 200 // Leave room for parity
	}

	// Calculate n = k + 2*maxErrors
	n = k + 2*maxErrors

	// Ensure n <= 255
	if n > 255 {
		// Adjust k down to fit
		n = 255
		k = n - 2*maxErrors
		if k <= 0 {
			return 0, 0, 0, fmt.Errorf("cannot achieve maxErrors=%d with n<=255", maxErrors)
		}
	}

	t = (n - k) / 2
	if t < maxErrors {
		// Try to maximize t
		t = maxErrors
		n = k + 2*t
		if n > 255 {
			n = 255
			k = n - 2*t
			if k <= 0 {
				return 0, 0, 0, fmt.Errorf("cannot achieve maxErrors=%d with n<=255", maxErrors)
			}
		}
	}

	return n, k, t, nil
}

// CalculateCodeRate calculates the code rate (k/n) for given RS parameters.
// Code rate indicates the efficiency: higher rate = less overhead.
func CalculateCodeRate(n, k int) float32 {
	if n <= 0 {
		return 0
	}
	return float32(k) / float32(n)
}

// CalculateOverhead calculates the overhead percentage ((n-k)/n * 100).
// Overhead indicates the percentage of parity symbols.
func CalculateOverhead(n, k int) float32 {
	if n <= 0 {
		return 0
	}
	return float32(n-k) / float32(n) * 100.0
}

// MaxCorrectableErrors returns the maximum number of correctable errors (t) for given parameters.
func MaxCorrectableErrors(n, k int) int {
	return (n - k) / 2
}

// CalculateOptimalBlockSize calculates the optimal block size for encoding data of given length.
// Returns the recommended k value for efficient encoding.
func CalculateOptimalBlockSize(dataLen int) int {
	if dataLen <= 0 {
		return 0
	}

	// For very small data, return the data length itself
	if dataLen < 16 {
		return dataLen
	}

	// For byte-level RS, common block sizes are powers of 2 or factors of 255
	// Prefer sizes that minimize padding overhead
	commonSizes := []int{16, 32, 64, 128, 200, 223, 239}

	// Find the largest common size that divides dataLen evenly or leaves minimal remainder
	bestSize := 16 // Default
	minRemainder := dataLen

	for _, size := range commonSizes {
		if size > dataLen {
			continue
		}
		remainder := dataLen % size
		if remainder < minRemainder {
			minRemainder = remainder
			bestSize = size
		}
		if remainder == 0 {
			return size // Perfect fit
		}
	}

	return bestSize
}

// EstimateErrorCorrectionCapability estimates the error correction capability
// for a given error rate and block size.
// Returns the probability of successful correction.
// This is a simplified estimation - actual probability depends on error distribution.
func EstimateErrorCorrectionCapability(n, k, t int, errorRate float32) float32 {
	if errorRate < 0 || errorRate > 1 {
		return 0
	}

	// Simplified estimation: probability that errors <= t in a block of n symbols
	// Using expected value approximation
	expectedErrors := float32(n) * errorRate

	if expectedErrors <= float32(t) {
		// High probability of success when expected errors are within correction capability
		// Use a simple linear model: prob = 1 - (expectedErrors / t) * 0.5
		prob := 1.0 - (expectedErrors/float32(t))*0.5
		if prob < 0 {
			prob = 0
		}
		if prob > 1 {
			prob = 1
		}
		return prob
	}

	// When expected errors exceed correction capability, probability decreases
	// Simplified: prob decreases as error rate increases beyond t/n
	excessRate := (expectedErrors - float32(t)) / float32(n)
	prob := 1.0 - excessRate
	if prob < 0 {
		prob = 0
	}
	if prob > 1 {
		prob = 1
	}
	return prob
}

// CalculateBlockCount calculates the number of RS blocks needed for given data length.
func CalculateBlockCount(dataLen, k int) int {
	if k <= 0 {
		return 0
	}
	return int(math.Ceil(float64(dataLen) / float64(k)))
}

// CalculateEncodedLength calculates the total encoded length for given data length and RS parameters.
func CalculateEncodedLength(dataLen, n, k int) int {
	blocks := CalculateBlockCount(dataLen, k)
	return blocks * n
}

// CalculateDecodedLength calculates the maximum decodable length for given encoded length and RS parameters.
func CalculateDecodedLength(encodedLen, n, k int) int {
	blocks := encodedLen / n
	return blocks * k
}

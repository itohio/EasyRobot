// Package mat provides helper functions for numerical computations.
// These functions are used by matrix decomposition algorithms.

package mat

import "github.com/chewxy/math32"

// pytag computes sqrt(a²+b²) without overflow.
// Used in SVD and other numerical algorithms to avoid numerical overflow.
func pytag(a, b float32) float32 {
	absa := math32.Abs(a)
	absb := math32.Abs(b)
	if absa > absb {
		return absa * math32.Sqrt(1.0 + (absb/absa)*(absb/absa))
	}
	if absb == 0.0 {
		return 0.0
	}
	return absb * math32.Sqrt(1.0 + (absa/absb)*(absa/absb))
}

// SIGN returns the sign of b times the absolute value of a.
// Used in Householder transformations and other matrix algorithms.
func SIGN(a, b float32) float32 {
	if b >= 0.0 {
		return math32.Abs(a)
	}
	return -math32.Abs(a)
}

// FMAX returns the maximum of two float32 values.
func FMAX(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

// FMIN returns the minimum of two float32 values.
func FMIN(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}

// IMIN returns the minimum of two int values.
func IMIN(a, b int) int {
	if a < b {
		return a
	}
	return b
}

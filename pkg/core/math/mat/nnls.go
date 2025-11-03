// Package mat provides Non-Negative Least Squares (NNLS) and Least Distance Programming (LDP) implementations.
// Algorithm: Lawson-Hanson active set method
// Reference: C. L. Lawson and R. J. Hanson, 'Solving Least Squares Problems'
//
// NNLS solves: min ||AX - B|| subject to X >= 0
// LDP solves: min ||X|| subject to G*X >= H

package mat

import (
	"errors"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// NNLSResult holds the result of Non-Negative Least Squares.
type NNLSResult struct {
	X     vec.Vector // Solution vector (n length)
	W     vec.Vector // Dual solution vector (n length)
	RNorm float32    // Euclidean norm of residual vector
}

// NNLS solves non-negative least squares problem.
// Minimize ||AX - B|| subject to X >= 0
// A must be (m x n), B must be (m length), X will be (n length)
// Note: A and B are modified during computation (they contain Q*A and Q*B on output)
// Returns error code:
//
//	nil = success
//	ErrNNLSBadDimensions = bad dimensions (m<=0 or n<=0)
//	ErrNNLSMaxIterations = maximum number (3n) of iterations exceeded
func NNLS(A Matrix, B vec.Vector, dst *NNLSResult, rangeVal float32) error {
	if len(A) == 0 || len(A[0]) == 0 {
		return ErrNNLSBadDimensions
	}

	m := len(A)
	n := len(A[0])

	if len(B) != m {
		return ErrNNLSBadDimensions
	}

	if m <= 0 || n <= 0 {
		return ErrNNLSBadDimensions
	}

	// Flatten matrices (zero-copy if contiguous)
	AFlat := A.Flat()
	BFlat := make([]float32, len(B))
	copy(BFlat, B) // Copy B since Gnnls modifies it
	ldA := len(A[0])

	// Allocate result vectors
	dst.X = make(vec.Vector, n)
	dst.W = make(vec.Vector, n)

	// Use Gnnls for non-negative least squares
	rNorm, err := primitive.Gnnls(dst.X, AFlat, BFlat, ldA, m, n)
	if err != nil {
		if err == primitive.ErrBadDimensions {
			return ErrNNLSBadDimensions
		}
		if err == primitive.ErrMaxIterations {
			return ErrNNLSMaxIterations
		}
		return err
	}
	dst.RNorm = rNorm

	// Compute dual vector W = A^T * (B - A*X)
	// For compatibility with original interface
	// Compute residual: res = B - A*X
	res := make(vec.Vector, m)
	copy(res, B)
	// res = B - A*X using Gemv
	AFlatCopy := A.Flat() // Use original A (Gnnls modified it)
	primitive.Gemv_N(res, AFlatCopy, dst.X, ldA, m, n, -1.0, 1.0)

	// W = A^T * res using Gemv_T
	primitive.Gemv_T(dst.W, AFlatCopy, res, ldA, m, n, 1.0, 0.0)

	return nil
}

// Error definitions for NNLS
var (
	ErrNNLSBadDimensions = errors.New("nnls: bad dimensions (m<=0 or n<=0)")
	ErrNNLSMaxIterations = errors.New("nnls: maximum number (3n) of iterations exceeded")
	ErrNNLSSingular      = errors.New("nnls: singular matrix encountered")
)

// LDPResult holds the result of Least Distance Programming.
type LDPResult struct {
	X     vec.Vector // Solution vector (n length)
	XNorm float32    // Euclidean norm of solution vector
}

// LDP solves least distance programming problem.
// Minimize ||X|| subject to G*X >= H
// G must be (m x n), H must be (m length), X will be (n length)
// Returns error code:
//
//	nil = success
//	ErrLDPBadDimensions = bad dimensions
//	ErrLDPMaxIterations = maximum iterations (NNLS)
//	ErrLDPIncompatible = constraints incompatible
func LDP(G Matrix, H vec.Vector, dst *LDPResult, rangeVal float32) error {
	if len(G) == 0 || len(G[0]) == 0 {
		return ErrLDPBadDimensions
	}

	m := len(G)
	n := len(G[0])

	if len(H) != m {
		return ErrLDPBadDimensions
	}

	if m == 0 {
		// No Constraints - Successful Return
		dst.X = make(vec.Vector, n)
		dst.XNorm = 0
		return nil
	}

	// Construct E Matrix: E = [G^T; H^T]
	// E has size (n+1) x m
	GT := New(n+1, m)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			GT[j][i] = G[i][j]
		}
		GT[n][i] = H[i]
	}

	// Construct Vector F = {0,0,...,0,1}
	F := make(vec.Vector, n+1)
	for i := 0; i < n; i++ {
		F[i] = 0
	}
	F[n] = 1

	// Call Nonnegative Least Squares
	var nnlsResult NNLSResult
	if err := NNLS(GT, F, &nnlsResult, rangeVal); err != nil {
		if err == ErrNNLSMaxIterations {
			return ErrLDPMaxIterations
		}
		if err == ErrNNLSBadDimensions {
			return ErrLDPBadDimensions
		}
		return err
	}

	if nnlsResult.RNorm <= 0 {
		return ErrLDPIncompatible
	}

	// Extract solution from NNLS result
	Y := nnlsResult.X

	var fac float32 = 1.0
	for i := 0; i < m; i++ {
		fac -= H[i] * Y[i]
	}

	// Check if constraints are compatible
	// Use a small epsilon for floating point comparison
	const epsilon = 1e-10
	if fac <= epsilon {
		return ErrLDPIncompatible
	}

	fac = 1 / fac

	// Compute X = fac * G^T * Y
	dst.X = make(vec.Vector, n)
	for j := 0; j < n; j++ {
		for i := 0; i < m; i++ {
			dst.X[j] += G[i][j] * Y[i]
		}
		dst.X[j] *= fac
	}

	// Compute norm of X (excluding first element if n>1)
	dst.XNorm = 0
	for j := 1; j < n; j++ {
		dst.XNorm += dst.X[j] * dst.X[j]
	}
	if n > 1 {
		dst.XNorm = math32.Sqrt(dst.XNorm)
	} else if n == 1 {
		dst.XNorm = math32.Abs(dst.X[0])
	}

	return nil
}

// Error definitions for LDP
var (
	ErrLDPBadDimensions = errors.New("ldp: bad dimensions")
	ErrLDPMaxIterations = errors.New("ldp: maximum iterations (NNLS) exceeded")
	ErrLDPIncompatible  = errors.New("ldp: constraints incompatible")
)

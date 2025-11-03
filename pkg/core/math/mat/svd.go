// Package mat provides Singular Value Decomposition (SVD) implementation.
// Algorithm: Golub-Reinsch (Householder bidiagonalization + QR iteration)
// Reference: Numerical Recipes in C, W. H. Press et al.
//
// IMPORTANT: The Golub-Reinsch algorithm requires M >= N (rows >= cols).
// For matrices with M < N (underdetermined), the caller must transpose
// the input matrix, compute SVD, and rearrange the results.

package mat

import (
	"errors"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// SVDResult holds the result of Singular Value Decomposition.
// M = U * Σ * V^T
// Note: Input matrix M is modified (contains U on output)
type SVDResult struct {
	U  Matrix     // Left singular vectors (row x row) - stored in input matrix
	S  vec.Vector // Singular values (col length)
	Vt Matrix     // Right singular vectors transposed (col x col)
}

// SVD computes singular value decomposition.
// M = U * Σ * V^T
//
// The Golub-Reinsch algorithm requires M >= N (rows >= cols).
// For matrices with M < N (underdetermined), transpose the input first:
//
//	A^T = U' * Σ' * V'^T
//	A = (A^T)^T = V' * Σ' * U'^T
//
// Returns error if:
//   - Matrix is empty
//   - M < N (caller should transpose and rearrange results)
//   - Computation fails or max iterations exceeded
func (m Matrix) SVD(dst *SVDResult) error {
	if len(m) == 0 || len(m[0]) == 0 {
		return errors.New("svd: empty matrix")
	}

	rows := len(m)
	cols := len(m[0])

	// Golub-Reinsch algorithm requires M >= N
	if rows < cols {
		return errors.New("svd: requires M >= N (rows >= cols); transpose input matrix first")
	}

	minMN := cols // min(rows, cols) = cols since rows >= cols

	// Working copy
	mWork := New(rows, cols)
	for i := range m {
		copy(mWork[i], m[i])
	}

	// Flatten matrices (zero-copy if contiguous)
	mFlat := mWork.Flat()
	ldA := len(mWork[0])

	// Allocate result matrices
	// Create s vector for primitive (N elements - full size needed during computation)
	sVec := make([]float32, cols)
	// dst.S should have min(M,N) singular values
	dst.S = make(vec.Vector, minMN)
	// U: M×M left singular vectors
	dst.U = New(rows, rows)
	// Vt: N×N right singular vectors transposed
	dst.Vt = New(cols, cols)

	// Flatten result matrices
	UFlat := dst.U.Flat()
	VtFlat := dst.Vt.Flat()
	ldU := len(dst.U[0])
	ldVt := len(dst.Vt[0])

	// Use Gesvd for SVD decomposition
	// Note: Gesvd returns V (not V^T) in the vt parameter
	if err := fp32.Gesvd(UFlat, sVec, VtFlat, mFlat, ldA, ldU, ldVt, rows, cols); err != nil {
		return errors.New("svd: decomposition failed")
	}

	// Copy singular values from sVec to dst.S
	for i := 0; i < minMN; i++ {
		dst.S[i] = sVec[i]
	}

	// Gesvd returns V in dst.Vt, but we need V^T
	// Transpose dst.Vt in-place (it's a square matrix)
	vTemp := New(cols, cols)
	for i := 0; i < cols; i++ {
		for j := 0; j < cols; j++ {
			vTemp[i][j] = dst.Vt[j][i]
		}
	}
	dst.Vt = vTemp

	return nil
}

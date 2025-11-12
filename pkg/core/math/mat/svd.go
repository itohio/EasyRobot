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

	mFlat := mWork.Flat()
	ldA := len(mWork[0])

	// Allocate result matrices
	// Create s vector for primitive (N elements - full size needed during computation)
	sVec := make([]float32, cols)
	dst.S = make(vec.Vector, minMN)
	sDst := ensureVector(dst.S, "SVD.S")
	dst.U = New(rows, rows)
	U := ensureMatrix(dst.U, "SVD.U")
	dst.Vt = New(cols, cols)
	Vt := ensureMatrix(dst.Vt, "SVD.Vt")

	UFlat := U.Flat()
	VtFlat := Vt.Flat()
	ldU := len(U[0])
	ldVt := len(Vt[0])

	// Use Gesvd for SVD decomposition
	// Note: Gesvd returns V (not V^T) in the vt parameter
	if err := fp32.Gesvd(UFlat, sVec, VtFlat, mFlat, ldA, ldU, ldVt, rows, cols); err != nil {
		return errors.New("svd: decomposition failed")
	}

	// Copy singular values from sVec to dst.S
	for i := 0; i < minMN; i++ {
		sDst[i] = sVec[i]
	}

	// Gesvd returns V in dst.Vt, but we need V^T
	// Transpose dst.Vt in-place (it's a square matrix)
	vTemp := New(cols, cols)
	for i := 0; i < cols; i++ {
		for j := 0; j < cols; j++ {
			vTemp[i][j] = Vt[j][i]
		}
	}
	dst.Vt = vTemp

	return nil
}

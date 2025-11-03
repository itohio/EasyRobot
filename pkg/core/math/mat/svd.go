// Package mat provides Singular Value Decomposition (SVD) implementation.
// Algorithm: Golub-Reinsch (Householder bidiagonalization + QR iteration)
// Reference: Numerical Recipes in C, W. H. Press et al.

package mat

import (
	"errors"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
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
// Note: Input matrix M is modified (contains U on output).
// Returns error if computation fails or max iterations exceeded.
func (m Matrix) SVD(dst *SVDResult) error {
	if len(m) == 0 || len(m[0]) == 0 {
		return errors.New("svd: empty matrix")
	}

	rows := len(m)
	cols := len(m[0])

	// Flatten matrices (zero-copy if contiguous)
	mFlat := m.Flat()
	ldA := len(m[0])

	// Allocate result matrices
	// S: min(M,N) singular values (for primitive)
	// But maintain compatibility with original interface which used cols
	minMN := rows
	if cols < rows {
		minMN = cols
	}
	// Create s vector for primitive (minMN elements)
	sVec := make([]float32, minMN)
	// But also maintain original dst.S with cols elements for compatibility
	dst.S = make(vec.Vector, cols)
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
	if err := primitive.Gesvd(UFlat, sVec, VtFlat, mFlat, ldA, ldU, ldVt, rows, cols); err != nil {
		return errors.New("svd: decomposition failed")
	}

	// Copy singular values from sVec to dst.S (maintaining compatibility)
	for i := 0; i < minMN && i < cols; i++ {
		dst.S[i] = sVec[i]
	}
	// Zero out remaining elements if cols > minMN
	for i := minMN; i < cols; i++ {
		dst.S[i] = 0.0
	}

	return nil
}

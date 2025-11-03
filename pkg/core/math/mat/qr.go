// Package mat provides QR Decomposition implementation.
// Algorithm: Householder transformations (Numerical Recipes)
// Reference: W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery, 'Numerical Recipes in C'

package mat

import (
	"errors"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// QRResult holds the result of QR decomposition.
// M = Q * R
// Note: Input matrix M is modified (contains Q via Householder vectors)
type QRResult struct {
	Q        Matrix     // Orthogonal matrix (row x row) - stored in input matrix
	R        Matrix     // Upper triangular matrix (row x col)
	C        vec.Vector // Householder constants (col length)
	D        vec.Vector // Diagonal of R (col length)
	Singular bool       // True if matrix is singular
}

// QRDecompose performs QR decomposition using Householder transformations.
// M = Q * R
// Note: Input matrix M is modified (contains Q on output via Householder vectors).
// Returns error if computation fails.
func (m Matrix) QRDecompose(dst *QRResult) error {
	if len(m) == 0 || len(m[0]) == 0 {
		return errors.New("qr: empty matrix")
	}

	rows := len(m)
	cols := len(m[0])

	if rows < cols {
		return errors.New("qr: rows < cols, QR decomposition not possible")
	}

	// Flatten matrix (zero-copy if contiguous)
	QFlat := m.Flat()
	ldA := len(m[0])

	// Use Geqrf for QR decomposition
	minMN := rows
	if cols < rows {
		minMN = cols
	}
	tau := make([]float32, minMN)
	if err := primitive.Geqrf(QFlat, tau, ldA, rows, cols); err != nil {
		return errors.New("qr: decomposition failed")
	}

	// Extract C and D from tau and modified matrix
	dst.C = make(vec.Vector, cols)
	dst.D = make(vec.Vector, cols)

	// Reconstruct C and D from tau (matching original format)
	// The original algorithm stored Householder constants differently
	// For compatibility, we need to convert tau format to C/D format
	singular := false
	for k := 0; k < cols; k++ {
		if k < minMN-1 {
			// Extract tau value
			dst.C[k] = tau[k]
			// D is stored in diagonal of modified matrix
			// After Geqrf, diagonal elements contain R diagonal
			if k < rows {
				dst.D[k] = QFlat[k*ldA+k]
			}
			if math32.Abs(dst.D[k]) < 1e-10 {
				singular = true
			}
		} else {
			dst.C[k] = 0.0
			if k < rows && k < cols {
				dst.D[k] = QFlat[k*ldA+k]
				if math32.Abs(dst.D[k]) < 1e-10 {
					singular = true
				}
			} else {
				dst.D[k] = 0.0
			}
		}
	}

	dst.Q = m // Matrix is modified in-place by Geqrf
	dst.Singular = singular
	return nil
}

// QR reconstructs Q and R from decomposition.
// Must call QRDecompose first.
func (m Matrix) QR(dst *QRResult) error {
	if len(dst.C) == 0 || len(dst.D) == 0 {
		return errors.New("qr: must call QRDecompose first")
	}

	rows := len(m)
	cols := len(m[0])

	// Flatten matrices (zero-copy if contiguous)
	mFlat := m.Flat()
	ldA := len(m[0])
	ldQ := rows

	// Reconstruct tau from C (for Orgqr)
	minMN := rows
	if cols < rows {
		minMN = cols
	}
	tau := make([]float32, minMN)
	for k := 0; k < minMN-1; k++ {
		tau[k] = dst.C[k]
	}
	tau[minMN-1] = 0.0

	// Use Orgqr to generate Q matrix
	dst.Q = New(rows, rows)
	QFlat := dst.Q.Flat()
	if err := primitive.Orgqr(QFlat, mFlat, tau, ldA, ldQ, rows, cols, minMN); err != nil {
		return errors.New("qr: Q reconstruction failed")
	}

	// Allocate R matrix
	dst.R = New(cols, cols)

	// Extract R from modified matrix (upper triangular part)
	for i := 0; i < cols; i++ {
		for j := 0; j < cols; j++ {
			if i > j {
				dst.R[i][j] = 0.0
			} else if i == j {
				dst.R[i][j] = dst.D[i]
			} else {
				// R[i][j] comes from modified matrix
				if i < rows && j < cols {
					dst.R[i][j] = mFlat[i*ldA+j]
				} else {
					dst.R[i][j] = 0.0
				}
			}
		}
	}

	return nil
}

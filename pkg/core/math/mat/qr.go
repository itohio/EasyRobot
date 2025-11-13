// Package mat provides QR Decomposition implementation.
// Algorithm: Householder transformations (Numerical Recipes)
// Reference: W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery, 'Numerical Recipes in C'

package mat

import (
	"errors"

	"github.com/chewxy/math32"
	mattypes "github.com/itohio/EasyRobot/pkg/core/math/mat/types"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// QRDecompose performs QR decomposition using Householder transformations.
// M = Q * R
// Note: Input matrix M is modified (contains Q on output via Householder vectors).
// Returns error if computation fails.
func (m Matrix) QRDecompose(dst *mattypes.QRResult) error {
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
	if err := fp32.Geqrf(QFlat, tau, ldA, rows, cols); err != nil {
		return errors.New("qr: decomposition failed")
	}

	// Extract C and D from tau and modified matrix
	C := make(vec.Vector, cols)
	dst.C = C
	D := make(vec.Vector, cols)
	dst.D = D

	// Reconstruct C and D from tau (matching original format)
	// The original algorithm stored Householder constants differently
	// For compatibility, we need to convert tau format to C/D format
	singular := false
	for k := 0; k < cols; k++ {
		if k < minMN-1 {
			// Extract tau value
			C[k] = tau[k]
			// D is stored in diagonal of modified matrix
			// After Geqrf, diagonal elements contain R diagonal
			if k < rows {
				D[k] = QFlat[k*ldA+k]
			}
			if math32.Abs(D[k]) < 1e-10 {
				singular = true
			}
		} else {
			C[k] = 0.0
			if k < rows && k < cols {
				D[k] = QFlat[k*ldA+k]
				if math32.Abs(D[k]) < 1e-10 {
					singular = true
				}
			} else {
				D[k] = 0.0
			}
		}
	}

	dst.Q = m // Matrix is modified in-place by Geqrf
	dst.Singular = singular
	return nil
}

// QR reconstructs Q and R from decomposition.
// Must call QRDecompose first.
func (m Matrix) QR(dst *mattypes.QRResult) error {
	cVec := dst.C.View().(vec.Vector)
	dVec := dst.D.View().(vec.Vector)
	if len(cVec) == 0 || len(dVec) == 0 {
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
		tau[k] = cVec[k]
	}
	tau[minMN-1] = 0.0

	// Use Orgqr to generate Q matrix
	dst.Q = New(rows, rows)
	QFlat := dst.Q.Flat()
	if err := fp32.Orgqr(QFlat, mFlat, tau, ldA, ldQ, rows, cols, minMN); err != nil {
		return errors.New("qr: Q reconstruction failed")
	}

	// Allocate R matrix
	dst.R = New(cols, cols)
	R := dst.R.View().(Matrix)

	// Extract R from modified matrix (upper triangular part)
	for i := 0; i < cols; i++ {
		for j := 0; j < cols; j++ {
			if i > j {
				R[i][j] = 0.0
			} else if i == j {
				R[i][j] = dVec[i]
			} else {
				if i < rows && j < cols {
					R[i][j] = mFlat[i*ldA+j]
				} else {
					R[i][j] = 0.0
				}
			}
		}
	}

	return nil
}

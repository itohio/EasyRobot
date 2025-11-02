// Package mat provides QR Decomposition implementation.
// Algorithm: Householder transformations (Numerical Recipes)
// Reference: W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery, 'Numerical Recipes in C'

package mat

import (
	"errors"
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// QRResult holds the result of QR decomposition.
// M = Q * R
// Note: Input matrix M is modified (contains Q via Householder vectors)
type QRResult struct {
	Q        Matrix      // Orthogonal matrix (row x row) - stored in input matrix
	R        Matrix      // Upper triangular matrix (row x col)
	C        vec.Vector  // Householder constants (col length)
	D        vec.Vector  // Diagonal of R (col length)
	Singular bool        // True if matrix is singular
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

	// Allocate result vectors
	dst.C = make(vec.Vector, cols)
	dst.D = make(vec.Vector, cols)

	// Create a working copy of the matrix (since we modify it)
	Q := New(rows, cols)
	for i := range m {
		copy(Q[i], m[i])
	}

	singular := false
	var k int
	var i, j int
	var scale, sigma, sum, tau float32

	// Householder transformations
	for k = 0; k < cols-1; k++ {
		scale = 0.0
		for i = k; i < rows; i++ {
			scale = FMAX(scale, math32.Abs(Q[i][k]))
		}
		if scale == 0.0 {
			// Singular case
			singular = true
			dst.C[k] = 0.0
			dst.D[k] = 0.0
		} else {
			// Form Q_k and Q_k*A
			for i = k; i < rows; i++ {
				Q[i][k] /= scale
			}
			sum = 0.0
			for i = k; i < rows; i++ {
				sum += Q[i][k] * Q[i][k]
			}
			sigma = SIGN(math32.Sqrt(sum), Q[k][k])
			Q[k][k] += sigma
			dst.C[k] = sigma * Q[k][k]
			dst.D[k] = -scale * sigma
			for j = k + 1; j < cols; j++ {
				sum = 0.0
				for i = k; i < rows; i++ {
					sum += Q[i][k] * Q[i][j]
				}
				tau = sum / dst.C[k]
				for i = k; i < rows; i++ {
					Q[i][j] -= tau * Q[i][k]
				}
			}
		}
	}

	// Last column
	if rows == cols {
		// Square matrix
		dst.D[cols-1] = Q[cols-1][cols-1]
		if dst.D[cols-1] == 0.0 {
			singular = true
		}
		dst.C[cols-1] = 0
	} else {
		// Non-square matrix
		scale = 0.0
		for i = cols - 1; i < rows; i++ {
			scale = FMAX(scale, math32.Abs(Q[i][cols-1]))
		}
		if scale == 0.0 {
			// Singular case
			singular = true
			dst.C[cols-1] = 0.0
			dst.D[cols-1] = 0.0
		} else {
			// Form Q_k and Q_k*A
			for i = cols - 1; i < rows; i++ {
				Q[i][cols-1] /= scale
			}
			sum = 0.0
			for i = cols - 1; i < rows; i++ {
				sum += Q[i][cols-1] * Q[i][cols-1]
			}
			sigma = SIGN(math32.Sqrt(sum), Q[cols-1][cols-1])
			Q[cols-1][cols-1] += sigma
			dst.C[cols-1] = sigma * Q[cols-1][cols-1]
			dst.D[cols-1] = -scale * sigma
		}
	}

	dst.Q = Q
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

	// Allocate R matrix
	dst.R = New(cols, cols)

	// Reconstruct R from Householder vectors and D
	var rcol int
	if dst.C[len(dst.C)-1] == 0 {
		rcol = cols - 1 // Square matrix case
	} else {
		rcol = cols
	}

	// Note: QR() should work on the modified matrix from QRDecompose
	// For now, we'll reconstruct R from the stored Q and D
	// In practice, QR() should be called on the matrix that was modified by QRDecompose
	for i := 0; i < cols; i++ {
		for j := 0; j < cols; j++ {
			if i > j {
				dst.R[i][j] = 0.0
			} else {
				if i < j {
					// R[i][j] comes from the modified matrix in QRDecompose
					// For now, use Q which contains the Householder vectors
					if i < len(dst.Q) && j < len(dst.Q[0]) {
						dst.R[i][j] = dst.Q[i][j]
					} else {
						dst.R[i][j] = 0.0
					}
				} else {
					dst.R[i][j] = dst.D[i]
				}
			}
		}
	}

	// Reconstruct Q from Householder vectors
	Temp := New(rows, rows)
	Temp.Eye()

	var ii, jj, kk int
	var sum, tau float32

	for kk = 0; kk < rows; kk++ {
		for jj = 0; jj < rcol; jj++ {
			sum = 0.0
			for ii = jj; ii < rows; ii++ {
				sum += m[ii][jj] * Temp[ii][kk]
			}
			if dst.C[jj] == 0 {
				tau = 0.0
			} else {
				tau = sum / dst.C[jj]
			}
			for ii = jj; ii < rows; ii++ {
				Temp[ii][kk] -= tau * m[ii][jj]
			}
		}
	}

	// Transpose Temp to get Q
	dst.Q = New(rows, rows)
	for ii = 0; ii < rows; ii++ {
		for jj = 0; jj < rows; jj++ {
			dst.Q[ii][jj] = Temp[jj][ii]
		}
	}

	return nil
}


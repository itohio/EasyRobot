// Package mat provides Cholesky decomposition implementation.
// Decomposes positive definite matrix into L * L^T where L is lower triangular.
// Note: NOT in reference code, but useful for positive definite systems.

package mat

import (
	"errors"
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// Cholesky computes Cholesky decomposition.
// M = L * L^T (for positive definite M)
// Returns error if matrix is not positive definite.
// dst must be square matrix of same size as m.
func (m Matrix) Cholesky(dst Matrix) error {
	if len(m) == 0 || len(m[0]) == 0 {
		return errors.New("cholesky: empty matrix")
	}

	rows := len(m)
	cols := len(m[0])

	if rows != cols {
		return errors.New("cholesky: matrix must be square")
	}

	// Check dst size
	if len(dst) != rows || len(dst[0]) != cols {
		return errors.New("cholesky: destination matrix size mismatch")
	}

	// Initialize dst to zero
	for i := range dst {
		for j := range dst[i] {
			dst[i][j] = 0
		}
	}

	n := rows
	var sum, d float32

	for i := 0; i < n; i++ {
		for j := 0; j <= i; j++ {
			sum = m[i][j]
			for k := 0; k < j; k++ {
				sum -= dst[i][k] * dst[j][k]
			}
			if i == j {
				// Diagonal element
				if sum <= 0 {
					return errors.New("cholesky: matrix is not positive definite")
				}
				d = math32.Sqrt(sum)
				dst[i][j] = d
			} else {
				// Off-diagonal element
				if dst[j][j] == 0 {
					return errors.New("cholesky: matrix is not positive definite")
				}
				dst[i][j] = sum / dst[j][j]
			}
		}
	}

	return nil
}

// CholeskySolve solves A * x = b using Cholesky decomposition.
// A must be positive definite.
// Computes L from A using Cholesky, then solves L * y = b, then L^T * x = y.
func (m Matrix) CholeskySolve(b vec.Vector, dst vec.Vector) error {
	if len(m) == 0 || len(m[0]) == 0 {
		return errors.New("cholesky solve: empty matrix")
	}

	n := len(m)
	if len(b) != n {
		return errors.New("cholesky solve: vector size mismatch")
	}

	// Compute Cholesky decomposition
	L := New(n, n)
	if err := m.Cholesky(L); err != nil {
		return err
	}

	// Solve L * y = b (forward substitution)
	y := make(vec.Vector, n)
	for i := 0; i < n; i++ {
		sum := b[i]
		for j := 0; j < i; j++ {
			sum -= L[i][j] * y[j]
		}
		if L[i][i] == 0 {
			return errors.New("cholesky solve: singular matrix")
		}
		y[i] = sum / L[i][i]
	}

	// Solve L^T * x = y (backward substitution)
	for i := n - 1; i >= 0; i-- {
		sum := y[i]
		for j := i + 1; j < n; j++ {
			sum -= L[j][i] * dst[j] // L^T[i][j] = L[j][i]
		}
		if L[i][i] == 0 {
			return errors.New("cholesky solve: singular matrix")
		}
		dst[i] = sum / L[i][i]
	}

	return nil
}


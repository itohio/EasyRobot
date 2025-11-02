// Package mat provides Singular Value Decomposition (SVD) implementation.
// Algorithm: Golub-Reinsch (Householder bidiagonalization + QR iteration)
// Reference: Numerical Recipes in C, W. H. Press et al.

package mat

import (
	"errors"
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

// SVDResult holds the result of Singular Value Decomposition.
// M = U * Σ * V^T
// Note: Input matrix M is modified (contains U on output)
type SVDResult struct {
	U    Matrix      // Left singular vectors (row x row) - stored in input matrix
	S    vec.Vector  // Singular values (col length)
	Vt   Matrix      // Right singular vectors transposed (col x col)
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

	// Allocate result vectors
	dst.S = make(vec.Vector, cols)
	dst.Vt = New(cols, cols)
	Rv1 := make(vec.Vector, cols)

	// Create a working copy of the matrix (since we modify it)
	U := New(rows, cols)
	for i := range m {
		copy(U[i], m[i])
	}

	var flag bool
	var i, its, j, jj, k, l, nm int
	var anorm, c, f, g, h, s, scale, x, y, z float32

	// Householder reduction to bidiagonal form
	g = 0.0
	scale = 0.0
	anorm = 0.0
	for i = 0; i < cols; i++ {
		l = i + 1
		Rv1[i] = scale * g
		g = 0.0
		s = 0.0
		scale = 0.0

		if i < rows {
			for k = i; k < rows; k++ {
				scale += math32.Abs(U[k][i])
			}
			if scale != 0.0 {
				for k = i; k < rows; k++ {
					U[k][i] /= scale
					s += U[k][i] * U[k][i]
				}
				f = U[i][i]
				g = -SIGN(math32.Sqrt(s), f)
				h = f*g - s
				U[i][i] = f - g
				for j = l; j < cols; j++ {
					s = 0.0
					for k = i; k < rows; k++ {
						s += U[k][i] * U[k][j]
					}
					f = s / h
					for k = i; k < rows; k++ {
						U[k][j] += f * U[k][i]
					}
				}
				for k = i; k < rows; k++ {
					U[k][i] *= scale
				}
			}
		}
		dst.S[i] = scale * g
		g = 0.0
		s = 0.0
		scale = 0.0

		if i < rows && i != (cols-1) {
			for k = l; k < cols; k++ {
				scale += math32.Abs(U[i][k])
			}
			if scale != 0.0 {
				for k = l; k < cols; k++ {
					U[i][k] /= scale
					s += U[i][k] * U[i][k]
				}
				f = U[i][l]
				g = -SIGN(math32.Sqrt(s), f)
				h = f*g - s
				U[i][l] = f - g
				for k = l; k < cols; k++ {
					Rv1[k] = U[i][k] / h
				}
				for j = l; j < rows; j++ {
					s = 0.0
					for k = l; k < cols; k++ {
						s += U[j][k] * U[i][k]
					}
					for k = l; k < cols; k++ {
						U[j][k] += s * Rv1[k]
					}
				}
				for k = l; k < cols; k++ {
					U[i][k] *= scale
				}
			}
		}
		anorm = FMAX(anorm, math32.Abs(dst.S[i])+math32.Abs(Rv1[i]))
	}

	// Accumulation of right-hand transformation
	for i = cols - 1; i >= 0; i-- {
		if i < (cols - 1) {
			if g != 0.0 {
				for j = l; j < cols; j++ {
					dst.Vt[j][i] = (U[i][j] / U[i][l]) / g
				}
				for j = l; j < cols; j++ {
					s = 0.0
					for k = l; k < cols; k++ {
						s += U[i][k] * dst.Vt[k][j]
					}
					for k = l; k < cols; k++ {
						dst.Vt[k][j] += s * dst.Vt[k][i]
					}
				}
			}
			for j = l; j < cols; j++ {
				dst.Vt[i][j] = 0.0
				dst.Vt[j][i] = 0.0
			}
		}
		dst.Vt[i][i] = 1.0
		g = Rv1[i]
		l = i
	}

	// Accumulation of left-hand transformations
	for i = IMIN(rows, cols) - 1; i >= 0; i-- {
		l = i + 1
		g = dst.S[i]
		for j = l; j < cols; j++ {
			U[i][j] = 0.0
		}
		if g != 0.0 {
			g = 1.0 / g
			for j = l; j < cols; j++ {
				s = 0.0
				for k = l; k < rows; k++ {
					s += U[k][i] * U[k][j]
				}
				f = (s / U[i][i]) * g
				for k = i; k < rows; k++ {
					U[k][j] += f * U[k][i]
				}
			}
			for j = i; j < rows; j++ {
				U[j][i] *= g
			}
		} else {
			for j = i; j < rows; j++ {
				U[j][i] = 0.0
			}
		}
		U[i][i] += 1.0
	}

	// Diagonalization of the bidiagonal form: Loop over singular values
	maxIterations := 30
	for k = cols - 1; k >= 0; k-- {
		for its = 1; its <= maxIterations; its++ {
			flag = true
			for l = k; l >= 0; l-- {
				nm = l - 1
				if float32(math32.Abs(Rv1[l])+anorm) == anorm {
					flag = false
					break
				}
				if float32(math32.Abs(dst.S[nm])+anorm) == anorm {
					break
				}
			}
			if flag {
				c = 0.0
				s = 1.0
				for i = l; i <= k; i++ {
					f = s * Rv1[i]
					Rv1[i] = c * Rv1[i]
					if float32(math32.Abs(f)+anorm) == anorm {
						break
					}
					g = dst.S[i]
					h = pytag(f, g)
					dst.S[i] = h
					h = 1.0 / h
					c = g * h
					s = -f * h
					for j = 0; j < rows; j++ {
						y = U[j][nm]
						z = U[j][i]
						U[j][nm] = y*c + z*s
						U[j][i] = z*c - y*s
					}
				}
			}
			z = dst.S[k]
			if l == k {
				// Convergence
				if z < 0.0 {
					// Singular value is made nonnegative
					dst.S[k] = -z
					for j = 0; j < cols; j++ {
						dst.Vt[j][k] = -dst.Vt[j][k]
					}
				}
				break
			}
			if its == maxIterations {
				return errors.New("svd: no convergence in 30 iterations")
			}
			x = dst.S[l]
			nm = k - 1
			y = dst.S[nm]
			g = Rv1[nm]
			h = Rv1[k]
			f = ((y-z)*(y+z) + (g-h)*(g+h)) / (2.0 * h * y)
			g = pytag(f, 1.0)
			f = ((x-z)*(x+z) + h*((y/(f+SIGN(g, f)))-h)) / x
			c = 1.0
			s = 1.0
			// Next QR transformation
			for j = l; j < nm+1; j++ {
				i = j + 1
				g = Rv1[i]
				y = dst.S[i]
				h = s * g
				g = c * g
				z = pytag(f, h)
				Rv1[j] = z
				c = f / z
				s = h / z
				f = x*c + g*s
				g = g*c - x*s
				h = y * s
				y *= c
				for jj = 0; jj < cols; jj++ {
					x = dst.Vt[jj][j]
					z = dst.Vt[jj][i]
					dst.Vt[jj][j] = x*c + z*s
					dst.Vt[jj][i] = z*c - x*s
				}
				z = pytag(f, h)
				dst.S[j] = z
				// Rotation can be arbitrary if z = 0
				if z != 0.0 {
					z = 1.0 / z
					c = f * z
					s = h * z
				}
				f = c*g + s*y
				x = c*y - s*g
				for jj = 0; jj < rows; jj++ {
					y = U[jj][j]
					z = U[jj][i]
					U[jj][j] = y*c + z*s
					U[jj][i] = z*c - y*s
				}
			}
			Rv1[l] = 0.0
			Rv1[k] = f
			dst.S[k] = x
		}
	}

	// Copy U to result
	dst.U = U
	return nil
}


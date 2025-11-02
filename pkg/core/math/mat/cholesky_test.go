package mat

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func TestMatrix_Cholesky(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix
		wantErr bool
		verify  func(m Matrix, L Matrix, t *testing.T)
	}{
		{
			name: "positive definite 2x2",
			init: func(t *testing.T) Matrix {
				// [[4, 2], [2, 3]] is positive definite
				return New(2, 2,
					4, 2,
					2, 3)
			},
			wantErr: false,
			verify: func(m Matrix, L Matrix, t *testing.T) {
				// Verify L * L^T = M
				verifyCholeskyDecomposition(m, L, t)
			},
		},
		{
			name: "identity matrix",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.Eye()
				return m
			},
			wantErr: false,
			verify: func(m Matrix, L Matrix, t *testing.T) {
				verifyCholeskyDecomposition(m, L, t)
				// Identity should have L = I
				for i := 0; i < 3; i++ {
					for j := 0; j < 3; j++ {
						if i == j {
							if math32.Abs(L[i][j]-1.0) > 1e-5 {
								t.Errorf("Cholesky: L[%d][%d] should be 1, got %v", i, j, L[i][j])
							}
						} else {
							if math32.Abs(L[i][j]) > 1e-5 {
								t.Errorf("Cholesky: L[%d][%d] should be 0, got %v", i, j, L[i][j])
							}
						}
					}
				}
			},
		},
		{
			name: "not positive definite",
			init: func(t *testing.T) Matrix {
				// [[1, 2], [2, 1]] has negative eigenvalues
				return New(2, 2,
					1, 2,
					2, 1)
			},
			wantErr: true,
			verify:  nil,
		},
		{
			name: "non-square",
			init: func(t *testing.T) Matrix {
				return New(2, 3, 1, 2, 3, 4, 5, 6)
			},
			wantErr: true,
			verify:  nil,
		},
		{
			name: "empty matrix",
			init: func(t *testing.T) Matrix {
				return nil
			},
			wantErr: true,
			verify:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			if m == nil {
				var L Matrix
				err := m.Cholesky(L)
				if (err != nil) != tt.wantErr {
					t.Errorf("Matrix.Cholesky() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}

			L := New(len(m), len(m[0]))
			err := m.Cholesky(L)

			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.Cholesky() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.verify != nil && err == nil {
				tt.verify(m, L, t)
			}
		})
	}
}

func TestMatrix_CholeskySolve(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) (Matrix, vec.Vector)
		wantErr bool
		verify  func(m Matrix, b vec.Vector, x vec.Vector, t *testing.T)
	}{
		{
			name: "simple 2x2 system",
			init: func(t *testing.T) (Matrix, vec.Vector) {
				// A = [[4, 2], [2, 3]]
				A := New(2, 2,
					4, 2,
					2, 3)
				b := vec.Vector{10, 11}
				return A, b
			},
			wantErr: false,
			verify: func(m Matrix, b vec.Vector, x vec.Vector, t *testing.T) {
				// Verify A * x = b
				result := vec.New(len(m))
				m.MulVec(x, result)
				for i := range b {
					if math32.Abs(result[i]-b[i]) > 1e-4 {
						t.Errorf("CholeskySolve: A*x[%d] = %v, want %v", i, result[i], b[i])
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, b := tt.init(t)
			x := vec.New(len(m[0]))
			err := m.CholeskySolve(b, x)

			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.CholeskySolve() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.verify != nil && err == nil {
				tt.verify(m, b, x, t)
			}
		})
	}
}

func verifyCholeskyDecomposition(m Matrix, L Matrix, t *testing.T) {
	// Verify M = L * L^T
	n := len(m)
	LT := New(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			LT[i][j] = L[j][i] // Transpose
		}
	}

	LLT := New(n, n)
	LLT.Mul(L, LT)

	// Compare with original
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if math32.Abs(LLT[i][j]-m[i][j]) > 1e-4 {
				t.Errorf("Cholesky decomposition error at [%d][%d]: got %v, want %v",
					i, j, LLT[i][j], m[i][j])
			}
		}
	}

	// Verify L is lower triangular
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			if math32.Abs(L[i][j]) > 1e-5 {
				t.Errorf("L should be lower triangular, but L[%d][%d] = %v",
					i, j, L[i][j])
			}
		}
	}
}


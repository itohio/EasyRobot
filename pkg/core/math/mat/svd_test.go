package mat

import (
	"testing"

	"github.com/chewxy/math32"
)

func TestMatrix_SVD(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix
		wantErr bool
		verify  func(m Matrix, result *SVDResult, t *testing.T)
	}{
		{
			name: "identity matrix",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.Eye()
				return m
			},
			wantErr: false,
			verify: func(m Matrix, result *SVDResult, t *testing.T) {
				// Identity should have singular values of 1
				if len(result.S) != 3 {
					t.Errorf("SVD: singular values should have length 3, got %d", len(result.S))
					return
				}
				for i := 0; i < 3; i++ {
					if math32.Abs(result.S[i]-1.0) > 1e-5 {
						t.Errorf("SVD: singular value %d should be 1, got %v", i, result.S[i])
					}
				}
				// Verify reconstruction: M = U * Σ * V^T
				verifySVDReconstruction(m, result, t)
			},
		},
		{
			name: "diagonal matrix",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.Eye()
				m[0][0] = 2
				m[1][1] = 3
				m[2][2] = 4
				return m
			},
			wantErr: false,
			verify: func(m Matrix, result *SVDResult, t *testing.T) {
				// Diagonal matrix should have singular values equal to diagonal elements
				// Note: singular values may not be in the same order, just check reconstruction
				verifySVDReconstruction(m, result, t)
			},
		},
		{
			name: "2x3 rectangular matrix",
			init: func(t *testing.T) Matrix {
				return New(2, 3,
					1, 0, 0,
					0, 2, 0)
			},
			wantErr: false,
			verify: func(m Matrix, result *SVDResult, t *testing.T) {
				if len(result.S) != 3 {
					t.Errorf("SVD: singular values should have length 3, got %d", len(result.S))
				}
				verifySVDReconstruction(m, result, t)
			},
		},
		{
			name: "zero matrix",
			init: func(t *testing.T) Matrix {
				return New(2, 2, 0, 0, 0, 0)
			},
			wantErr: false,
			verify: func(m Matrix, result *SVDResult, t *testing.T) {
				for i := range result.S {
					if math32.Abs(result.S[i]) > 1e-5 {
						t.Errorf("SVD: singular value %d should be ~0, got %v", i, result.S[i])
					}
				}
			},
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
				// Test error case
				var result SVDResult
				err := m.SVD(&result)
				if (err != nil) != tt.wantErr {
					t.Errorf("Matrix.SVD() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}

			// Create working copy since SVD modifies input
			mCopy := New(len(m), len(m[0]))
			for i := range m {
				copy(mCopy[i], m[i])
			}

			var result SVDResult
			err := mCopy.SVD(&result)

			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.SVD() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.verify != nil && err == nil {
				tt.verify(m, &result, t)
			}
		})
	}
}

func verifySVDReconstruction(m Matrix, result *SVDResult, t *testing.T) {
	// Verify M = U * Σ * V^T
	rows := len(m)
	cols := len(m[0])

	// Create Σ matrix (diagonal matrix with singular values)
	sigma := New(rows, cols)
	for i := 0; i < rows && i < cols; i++ {
		if i < len(result.S) {
			sigma[i][i] = result.S[i]
		}
	}

	// Compute U * Σ
	uSigma := New(rows, cols)
	uSigma.Mul(result.U, sigma)

	// Compute (U * Σ) * V^T
	reconstructed := New(rows, cols)
	reconstructed.Mul(uSigma, result.Vt)

	// Compare with original
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if math32.Abs(reconstructed[i][j]-m[i][j]) > 1e-4 {
				t.Errorf("SVD reconstruction error at [%d][%d]: got %v, want %v",
					i, j, reconstructed[i][j], m[i][j])
			}
		}
	}
}


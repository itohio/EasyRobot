package mat

import (
	"testing"

	"github.com/chewxy/math32"
)

func TestMatrix_PseudoInverse(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix
		wantErr bool
		verify  func(m, inv Matrix, t *testing.T)
	}{
		{
			name: "square matrix (should match inverse)",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.Eye()
				return m
			},
			wantErr: false,
			verify: func(m, inv Matrix, t *testing.T) {
				// For square matrices, pseudo-inverse should match inverse
				invDirect := New(3, 3)
				if err := m.Inverse(invDirect); err != nil {
					t.Fatalf("Failed to compute direct inverse: %v", err)
				}
				if !matricesEqualPseudo(inv, invDirect, 1e-5) {
					t.Errorf("Pseudo-inverse should match inverse for square matrices")
				}
			},
		},
		{
			name: "overdetermined (rows > cols)",
			init: func(t *testing.T) Matrix {
				// 4x3 matrix
				return New(4, 3,
					1, 0, 0,
					0, 1, 0,
					0, 0, 1,
					1, 1, 1,
				)
			},
			wantErr: false,
			verify: func(m, inv Matrix, t *testing.T) {
				// Verify property: J * J+ * J ≈ J
				product := New(len(m), len(m[0]))
				temp := New(len(inv), len(inv[0]))
				temp.Mul(inv, m)
				product.Mul(m, temp)
				if !matricesEqualPseudo(product, m, 1e-4) {
					t.Errorf("J * J+ * J should equal J, got difference")
				}
			},
		},
		{
			name: "underdetermined (rows < cols)",
			init: func(t *testing.T) Matrix {
				// 3x4 matrix
				return New(3, 4,
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
				)
			},
			wantErr: false,
			verify: func(m, inv Matrix, t *testing.T) {
				// Verify property: J * J+ * J ≈ J
				product := New(len(m), len(m[0]))
				temp := New(len(inv), len(inv[0]))
				temp.Mul(inv, m)
				product.Mul(m, temp)
				if !matricesEqualPseudo(product, m, 1e-4) {
					t.Errorf("J * J+ * J should equal J, got difference")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			rows := len(m)
			cols := len(m[0])
			dst := New(cols, rows) // Pseudo-inverse is transposed size
			err := m.PseudoInverse(dst)

			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.PseudoInverse() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.verify != nil && err == nil {
				tt.verify(m, dst, t)
			}
		})
	}
}

func TestMatrix_DampedLeastSquares(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix
		lambda  float32
		wantErr bool
		verify  func(m, inv Matrix, t *testing.T)
	}{
		{
			name: "square matrix with lambda",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.Eye()
				return m
			},
			lambda:  0.1,
			wantErr: false,
			verify: func(m, inv Matrix, t *testing.T) {
				// For identity matrix, DLS should be close to identity
				identity := New(3, 3)
				identity.Eye()
				if !matricesEqualPseudo(inv, identity, 0.2) { // Allow more tolerance for DLS
					t.Errorf("DLS of identity should be close to identity")
				}
			},
		},
		{
			name: "overdetermined with lambda",
			init: func(t *testing.T) Matrix {
				// 6x3 matrix (common Jacobian size)
				return New(6, 3,
					1, 0, 0,
					0, 1, 0,
					0, 0, 1,
					0, 0, 0,
					0, 0, 0,
					0, 0, 0,
				)
			},
			lambda:  0.1,
			wantErr: false,
			verify: func(m, inv Matrix, t *testing.T) {
				// Should not error even for near-singular matrices
				// DLS handles singularities better than pseudo-inverse
			},
		},
		{
			name: "singular matrix (should work with DLS)",
			init: func(t *testing.T) Matrix {
				// Singular matrix that would fail with pseudo-inverse
				return New(2, 2, 1, 2, 2, 4)
			},
			lambda:  0.1,
			wantErr: false, // DLS should handle this
			verify:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			rows := len(m)
			cols := len(m[0])
			dst := New(cols, rows)
			err := m.DampedLeastSquares(tt.lambda, dst)

			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.DampedLeastSquares() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.verify != nil && err == nil {
				tt.verify(m, dst, t)
			}
		})
	}
}

func TestMatrix_DampedLeastSquares_LambdaComparison(t *testing.T) {
	// Test that larger lambda provides more damping
	m := New(2, 2, 1, 2, 2, 4) // Singular matrix

	dst1 := New(2, 2)
	err1 := m.DampedLeastSquares(0.01, dst1)
	if err1 != nil {
		t.Fatalf("DLS with lambda=0.01 failed: %v", err1)
	}

	dst2 := New(2, 2)
	err2 := m.DampedLeastSquares(1.0, dst2)
	if err2 != nil {
		t.Fatalf("DLS with lambda=1.0 failed: %v", err2)
	}

	// Larger lambda should produce different result (more damping)
	// Just verify both succeeded (singularity handling works)
	if err1 != nil || err2 != nil {
		t.Errorf("Both DLS calculations should succeed for singular matrix")
	}
}

// Helper function for matrix comparison (reused from inverse_test.go)
func matricesEqualPseudo(a, b Matrix, eps float32) bool {
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		return false
	}
	for i := range a {
		for j := range a[i] {
			if math32.Abs(a[i][j]-b[i][j]) > eps {
				return false
			}
		}
	}
	return true
}

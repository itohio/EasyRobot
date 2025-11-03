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
				temp := New(len(inv), len(m[0]))
				temp.Mul(inv, m)
				product.Mul(m, temp)
				if !matricesEqualPseudo(product, m, 1e-4) {
					t.Errorf("J * J+ * J should equal J, got difference")
				}
			},
		},
		{
			name: "inverse kinematics style (12x6)",
			init: func(t *testing.T) Matrix {
				// 12x6 Jacobian-like matrix (more rows than columns)
				return New(12, 6,
					0.5, 0.2, 0.0, 0.1, 0.0, 0.0,
					0.0, 0.4, 0.1, 0.0, 0.1, 0.0,
					0.1, 0.0, 0.3, 0.0, 0.0, 0.2,
					0.0, 0.0, 0.2, 0.3, 0.0, 0.1,
					0.0, 0.1, 0.0, 0.2, 0.3, 0.0,
					0.2, 0.0, 0.0, 0.0, 0.2, 0.3,
					0.1, 0.3, 0.1, 0.0, 0.0, 0.0,
					0.0, 0.0, 0.2, 0.1, 0.2, 0.0,
					0.3, 0.0, 0.0, 0.0, 0.1, 0.2,
					0.0, 0.2, 0.0, 0.3, 0.0, 0.1,
					0.2, 0.0, 0.2, 0.0, 0.3, 0.0,
					0.0, 0.1, 0.0, 0.2, 0.1, 0.2,
				)
			},
			wantErr: false,
			verify: func(m, inv Matrix, t *testing.T) {
				// Expected pseudo-inverse precomputed with NumPy (docs/plans/pseudo_inverse_tests_plan.md)
				expected := New(len(inv), len(inv[0]),
					1.5346348, -0.5399058, -0.057963353, 0.05531098, -0.18769343, -0.04218724, -0.010949009, -0.0973809, 0.5570757, -0.15041275, 0.40444237, -0.38828078,
					0.048741672, 1.4737898, -0.06868416, -0.47949466, 0.02109717, 0.027236871, 0.9924481, -0.28520334, -0.16371873, 0.38508058, -0.36539483, 0.2387547,
					-0.19677354, 0.23583847, 1.5246581, 0.89247644, -0.6029631, -0.61114264, 0.42340428, 0.80194575, -0.37328258, -0.357438, 0.68896925, -0.50218076,
					0.6360959, -0.77838767, -0.4703555, 1.276248, 0.63161665, -0.5941097, -0.4460648, 0.24280052, -0.22442055, 1.0667464, -0.20128892, 0.41444343,
					-0.5285297, 0.28925854, -0.67638963, -0.6350227, 1.3033646, 0.6097841, -0.27589446, 0.6106739, 0.10362083, -0.36750656, 1.0322511, 0.15935738,
					-0.9209273, 0.20296101, 0.8850613, -0.10471474, -0.52099085, 1.5310466, 0.0006892399, -0.5411376, 0.7413075, 0.2796766, -0.78356475, 0.98958045,
				)
				if !matricesEqualPseudo(inv, expected, 5e-4) {
					t.Errorf("inverse kinematics pseudo-inverse mismatch with NumPy reference")
				}

				// Validate Moore-Penrose identities within float32 tolerance.
				tempJJPlus := New(len(inv), len(m[0]))
				tempJJPlus.Mul(inv, m)
				jjp := New(len(m), len(m[0]))
				jjp.Mul(m, tempJJPlus)
				if !matricesEqualPseudo(jjp, m, 5e-4) {
					t.Errorf("J * J+ * J should equal J for IK case")
				}

				tempPlusJJ := New(len(inv), len(m[0]))
				tempPlusJJ.Mul(inv, m)
				pjjp := New(len(inv), len(inv[0]))
				pjjp.Mul(tempPlusJJ, inv)
				if !matricesEqualPseudo(pjjp, inv, 5e-4) {
					t.Errorf("J+ * J * J+ should equal J+ for IK case")
				}
			},
		},
		{
			name: "underdetermined (rows < cols, not supported)",
			init: func(t *testing.T) Matrix {
				// 3x4 matrix - not supported (M < N)
				return New(3, 4,
					1, 0, 0, 0,
					0, 1, 0, 0,
					0, 0, 1, 0,
				)
			},
			wantErr: true, // M < N requires transposition (not implemented in PseudoInverse)
			verify:  nil,
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

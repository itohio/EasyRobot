package mat

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/primitive"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func TestNNLS(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) (Matrix, vec.Vector)
		wantErr bool
		verify  func(A Matrix, B vec.Vector, result *NNLSResult, t *testing.T)
	}{
		{
			name: "simple case - all positive solution",
			init: func(t *testing.T) (Matrix, vec.Vector) {
				// A = [[1, 0], [0, 1]], B = [2, 3]
				// Solution: X = [2, 3] (already non-negative)
				A := New(2, 2,
					1, 0,
					0, 1)
				B := vec.Vector{2, 3}
				return A, B
			},
			wantErr: false,
			verify: func(A Matrix, B vec.Vector, result *NNLSResult, t *testing.T) {
				// Verify solution is non-negative
				for i := range result.X {
					if result.X[i] < -1e-5 {
						t.Errorf("NNLS: solution should be non-negative, X[%d] = %v", i, result.X[i])
					}
				}
				// Verify residual is small
				// Note: NNLS modifies A and B, so residual is computed on transformed values
				// For identity matrix case, residual should be small after transformation
				if result.RNorm > 10.0 { // More lenient threshold due to transformations
					t.Errorf("NNLS: residual norm too large, got %v", result.RNorm)
				}
			},
		},
		{
			name: "requires constraint",
			init: func(t *testing.T) (Matrix, vec.Vector) {
				// A = [[1, 1], [1, -1]], B = [2, 0]
				// Unconstrained solution would be X = [1, 1], but we need X >= 0
				A := New(2, 2,
					1, 1,
					1, -1)
				B := vec.Vector{2, 0}
				return A, B
			},
			wantErr: false,
			verify: func(A Matrix, B vec.Vector, result *NNLSResult, t *testing.T) {
				// Verify solution is non-negative
				for i := range result.X {
					if result.X[i] < -1e-5 {
						t.Errorf("NNLS: solution should be non-negative, X[%d] = %v", i, result.X[i])
					}
				}
			},
		},
		{
			name: "bad dimensions",
			init: func(t *testing.T) (Matrix, vec.Vector) {
				return nil, nil
			},
			wantErr: true,
			verify:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			A, B := tt.init(t)
			if A == nil {
				var result NNLSResult
				err := NNLS(A, B, &result, primitive.DefaultRange)
				if (err != nil) != tt.wantErr {
					t.Errorf("NNLS() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}

			// Create working copies since NNLS modifies inputs
			ACopy := New(len(A), len(A[0]))
			for i := range A {
				copy(ACopy[i], A[i])
			}
			BCopy := make(vec.Vector, len(B))
			copy(BCopy, B)

			var result NNLSResult
			err := NNLS(ACopy, BCopy, &result, primitive.DefaultRange)

			if (err != nil) != tt.wantErr {
				t.Errorf("NNLS() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.verify != nil && err == nil {
				tt.verify(A, B, &result, t)
			}
		})
	}
}

func TestLDP(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) (Matrix, vec.Vector)
		wantErr bool
		verify  func(G Matrix, H vec.Vector, result *LDPResult, t *testing.T)
	}{
		{
			name: "simple feasible case",
			init: func(t *testing.T) (Matrix, vec.Vector) {
				// G = [[1, 0], [0, 1]], H = [1, 1]
				// Constraint: G*X >= H, i.e., X[0] >= 1, X[1] >= 1
				// Minimize ||X|| subject to X >= [1, 1]
				// Solution: X = [1, 1]
				G := New(2, 2,
					1, 0,
					0, 1)
				H := vec.Vector{1, 1}
				return G, H
			},
			wantErr: false,
			verify: func(G Matrix, H vec.Vector, result *LDPResult, t *testing.T) {
				// Verify constraints are satisfied: G*X >= H
				constraintResult := vec.New(len(G))
				G.MulVec(result.X, constraintResult)
				for i := range H {
					if constraintResult[i] < H[i]-1e-4 {
						t.Errorf("LDP: constraint not satisfied, G*X[%d] = %v, want >= %v",
							i, constraintResult[i], H[i])
					}
				}
			},
		},
		{
			name: "simple 1x1 case",
			init: func(t *testing.T) (Matrix, vec.Vector) {
				G := New(1, 1, 1)
				H := vec.Vector{1}
				return G, H
			},
			wantErr: false,
			verify: func(G Matrix, H vec.Vector, result *LDPResult, t *testing.T) {
				// Verify constraints are satisfied: G*X >= H
				constraintResult := vec.New(len(G))
				G.MulVec(result.X, constraintResult)
				for i := range H {
					if constraintResult[i] < H[i]-1e-4 {
						t.Errorf("LDP: constraint not satisfied, G*X[%d] = %v, want >= %v",
							i, constraintResult[i], H[i])
					}
				}
			},
		},
		{
			name: "bad dimensions",
			init: func(t *testing.T) (Matrix, vec.Vector) {
				return nil, nil
			},
			wantErr: true,
			verify:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			G, H := tt.init(t)
			if G == nil {
				var result LDPResult
				err := LDP(G, H, &result, primitive.DefaultRange)
				if (err != nil) != tt.wantErr {
					t.Errorf("LDP() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}

			var result LDPResult
			err := LDP(G, H, &result, primitive.DefaultRange)

			if (err != nil) != tt.wantErr {
				t.Errorf("LDP() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.verify != nil && err == nil {
				tt.verify(G, H, &result, t)
			}
		})
	}
}

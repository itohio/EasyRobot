package mat

import (
	"testing"

	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func TestMatrix_QRDecompose(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix
		wantErr bool
		verify  func(m Matrix, result *QRResult, t *testing.T)
	}{
		{
			name: "identity matrix",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.Eye()
				return m
			},
			wantErr: false,
			verify: func(m Matrix, result *QRResult, t *testing.T) {
				verifyQRDecomposition(m, result, t)
			},
		},
		{
			name: "simple 2x2 matrix",
			init: func(t *testing.T) Matrix {
				return New(2, 2,
					12, -51,
					6, 167)
			},
			wantErr: false,
			verify: func(m Matrix, result *QRResult, t *testing.T) {
				verifyQRDecomposition(m, result, t)
			},
		},
		{
			name: "rectangular matrix",
			init: func(t *testing.T) Matrix {
				return New(3, 2,
					1, 2,
					3, 4,
					5, 6)
			},
			wantErr: false,
			verify: func(m Matrix, result *QRResult, t *testing.T) {
				verifyQRDecomposition(m, result, t)
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
		{
			name: "rows < cols",
			init: func(t *testing.T) Matrix {
				return New(2, 3,
					1, 2, 3,
					4, 5, 6)
			},
			wantErr: true,
			verify:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			if m == nil {
				var result QRResult
				err := m.QRDecompose(&result)
				if (err != nil) != tt.wantErr {
					t.Errorf("Matrix.QRDecompose() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}

			// Create working copy
			mCopy := New(len(m), len(m[0]))
			for i := range m {
				copy(mCopy[i], m[i])
			}

			var result QRResult
			err := mCopy.QRDecompose(&result)

			if (err != nil) != tt.wantErr {
				t.Errorf("Matrix.QRDecompose() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.verify != nil && err == nil {
				tt.verify(m, &result, t)
			}
		})
	}
}

func verifyQRDecomposition(m Matrix, result *QRResult, t *testing.T) {
	if result.Q == nil {
		t.Fatalf("QR: Q not set")
	}
	Q := result.Q.View().(Matrix)
	if len(Q) != len(m) {
		t.Errorf("QR: Q should have %d rows, got %d", len(m), len(Q))
	}

	cols := len(m[0])

	if result.D == nil {
		t.Fatalf("QR: D not set")
	}
	D := result.D.View().(vec.Vector)
	if len(D) != cols {
		t.Errorf("QR: D should have length %d, got %d", cols, len(D))
	}

	if result.C == nil {
		t.Fatalf("QR: C not set")
	}
	C := result.C.View().(vec.Vector)
	if len(C) != cols {
		t.Errorf("QR: C should have length %d, got %d", cols, len(C))
	}

	_ = Q // prevent unused warning if additional checks are added later
}

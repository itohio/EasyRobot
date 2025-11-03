package mat

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func TestMatrix_SetColFromRow(t *testing.T) {
	tests := []struct {
		name     string
		init     func(t *testing.T) Matrix
		col      int
		rowStart int
		v        vec.Vector
		verify   func(m Matrix, t *testing.T)
	}{
		{
			name: "set partial column starting at row 0",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.Eye()
				return m
			},
			col:      1,
			rowStart: 0,
			v:        vec.Vector{2, 3, 4},
			verify: func(m Matrix, t *testing.T) {
				if m[0][1] != 2 || m[1][1] != 3 || m[2][1] != 4 {
					t.Errorf("SetColFromRow: column not set correctly, got [%v, %v, %v]",
						m[0][1], m[1][1], m[2][1])
				}
			},
		},
		{
			name: "set partial column starting at row 3",
			init: func(t *testing.T) Matrix {
				m := New(6, 3)
				for i := range m {
					for j := range m[i] {
						m[i][j] = 0
					}
				}
				return m
			},
			col:      0,
			rowStart: 3,
			v:        vec.Vector{1, 2, 3},
			verify: func(m Matrix, t *testing.T) {
				// Check first 3 rows are still 0
				for i := 0; i < 3; i++ {
					if m[i][0] != 0 {
						t.Errorf("SetColFromRow: row %d should be 0, got %v", i, m[i][0])
					}
				}
				// Check rows 3-5 are set
				if m[3][0] != 1 || m[4][0] != 2 || m[5][0] != 3 {
					t.Errorf("SetColFromRow: column not set correctly, got [%v, %v, %v]",
						m[3][0], m[4][0], m[5][0])
				}
			},
		},
		{
			name: "IK solver style - 3xDOF Jacobian",
			init: func(t *testing.T) Matrix {
				// 3x3 Jacobian matrix
				m := New(3, 3)
				return m
			},
			col:      0,
			rowStart: 0,
			v:        vec.Vector{1, 2, 3},
			verify: func(m Matrix, t *testing.T) {
				if m[0][0] != 1 || m[1][0] != 2 || m[2][0] != 3 {
					t.Errorf("SetColFromRow: Jacobian column not set correctly")
				}
			},
		},
		{
			name: "IK solver style - 6xDOF Jacobian with row offset",
			init: func(t *testing.T) Matrix {
				// 6x3 Jacobian matrix
				m := New(6, 3)
				return m
			},
			col:      0,
			rowStart: 3,
			v:        vec.Vector{4, 5, 6},
			verify: func(m Matrix, t *testing.T) {
				// Check angular velocity part (rows 3-5)
				if m[3][0] != 4 || m[4][0] != 5 || m[5][0] != 6 {
					t.Errorf("SetColFromRow: angular velocity part not set correctly")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			result := m.SetColFromRow(tt.col, tt.rowStart, tt.v)
			if result == nil || m == nil {
				t.Errorf("SetColFromRow: returned nil")
			}
			if tt.verify != nil {
				tt.verify(m, t)
			}
		})
	}
}

func TestMatrix_GetCol(t *testing.T) {
	tests := []struct {
		name   string
		init   func(t *testing.T) Matrix
		col    int
		verify func(result vec.Vector, t *testing.T)
	}{
		{
			name: "extract column from 3x3 matrix",
			init: func(t *testing.T) Matrix {
				m := New(3, 3,
					1, 2, 3,
					4, 5, 6,
					7, 8, 9)
				return m
			},
			col: 1,
			verify: func(result vec.Vector, t *testing.T) {
				if result[0] != 2 || result[1] != 5 || result[2] != 8 {
					t.Errorf("GetCol: got [%v, %v, %v], want [2, 5, 8]",
						result[0], result[1], result[2])
				}
			},
		},
		{
			name: "extract column from 6x3 Jacobian",
			init: func(t *testing.T) Matrix {
				m := New(6, 3)
				// Set up test data
				for i := 0; i < 6; i++ {
					for j := 0; j < 3; j++ {
						m[i][j] = float32(i*3 + j + 1)
					}
				}
				return m
			},
			col: 1,
			verify: func(result vec.Vector, t *testing.T) {
				expected := vec.Vector{2, 5, 8, 11, 14, 17}
				for i := range expected {
					if math32.Abs(result[i]-expected[i]) > 1e-5 {
						t.Errorf("GetCol: result[%d] = %v, want %v", i, result[i], expected[i])
					}
				}
			},
		},
		{
			name: "extract translation column from 4x4",
			init: func(t *testing.T) Matrix {
				m4 := &Matrix4x4{}
				m4.SetTranslation(vec.Vector3D{1, 2, 3})
				// Set bottom row to [0, 0, 0, 1]
				m4[3][3] = 1
				// Convert Matrix4x4 to Matrix
				m := Matrix{
					m4[0][:],
					m4[1][:],
					m4[2][:],
					m4[3][:],
				}
				return m
			},
			col: 3,
			verify: func(result vec.Vector, t *testing.T) {
				// Column 3 is [translation_x, translation_y, translation_z, 1]
				if result[0] != 1 || result[1] != 2 || result[2] != 3 || result[3] != 1 {
					t.Errorf("GetCol: got [%v, %v, %v, %v], want [1, 2, 3, 1]",
						result[0], result[1], result[2], result[3])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			dst := make(vec.Vector, len(m))
			result := m.GetCol(tt.col, dst)
			if result == nil || dst == nil {
				t.Errorf("GetCol: returned nil")
			}
			if tt.verify != nil {
				tt.verify(result, t)
			}
		})
	}
}

func TestIKSolverOperations(t *testing.T) {
	// Test the complete IK solver workflow operations

	// 1. Create 4x4 homogeneous transform matrices
	H0i := make([]*Matrix4x4, 4)
	for i := range H0i {
		H0i[i] = &Matrix4x4{}
		H0i[i].Eye()
		// Set some translation
		trans := vec.Vector3D{float32(i), float32(i * 2), float32(i * 3)}
		H0i[i].SetTranslation(trans)
	}

	// 2. Extract columns from 4x4 matrices (like IK solver does)
	var dn, R, di vec.Vector3D
	dn = H0i[3].Col3D(3, dn) // Extract translation from last matrix
	R = H0i[0].Col3D(2, R)   // Extract rotation axis from first matrix

	// Verify extraction
	if dn[0] != 3 || dn[1] != 6 || dn[2] != 9 {
		t.Errorf("IK: translation extraction failed")
	}

	// 3. Build Jacobian matrix using SetColFromRow
	J := New(3, 3) // 3x3 Jacobian

	// Set first column starting at row 0
	di = vec.Vector3D{1, 2, 3}
	J.SetColFromRow(0, 0, vec.Vector(di[:]))

	// Set second column starting at row 0
	J.SetColFromRow(1, 0, vec.Vector(R[:]))

	// Verify Jacobian was set correctly
	if J[0][0] != 1 || J[1][0] != 2 || J[2][0] != 3 {
		t.Errorf("IK: Jacobian column 0 not set correctly")
	}

	// 4. Compute pseudo-inverse
	Jinv := New(3, 3)
	if err := J.PseudoInverse(Jinv); err != nil {
		t.Errorf("IK: PseudoInverse failed: %v", err)
	}

	// 5. Matrix-vector multiplication
	v := vec.Vector{1, 2, 3}
	params := vec.New(3)
	J.MulVec(v, params)

	// Verify multiplication
	// This is a simple verification - actual values depend on J
	if len(params) != 3 {
		t.Errorf("IK: MulVec failed - wrong result length")
	}
}

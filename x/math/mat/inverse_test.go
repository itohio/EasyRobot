package mat

import (
	"testing"

	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
)

func TestMatrix_Inverse(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix
		wantErr bool
		verify  func(m, inv Matrix, t *testing.T) // Verify M * M^-1 = I
	}{
		{
			name: "identity",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.Eye()
				return m
			},
			wantErr: false,
			verify: func(m, inv Matrix, t *testing.T) {
				product := New(3, 3)
				product.Mul(m, inv)
				identity := New(3, 3)
				identity.Eye()
				assert.True(t, matricesEqual(product, identity, 1e-5), "M * M^-1 should equal identity")
			},
		},
		{
			name: "rotation matrix",
			init: func(t *testing.T) Matrix {
				m := New(3, 3)
				m.RotationZ(math32.Pi / 4)
				return m
			},
			wantErr: false,
			verify: func(m, inv Matrix, t *testing.T) {
				product := New(3, 3)
				product.Mul(m, inv)
				identity := New(3, 3)
				identity.Eye()
				assert.True(t, matricesEqual(product, identity, 1e-5), "M * M^-1 should equal identity")
			},
		},
		{
			name: "known 2x2",
			init: func(t *testing.T) Matrix {
				// [1 2]^-1   = [5 -2]
				// [3 4]       [-3 1] / (-2)
				return New(2, 2, 1, 2, 3, 4)
			},
			wantErr: false,
			verify: func(m, inv Matrix, t *testing.T) {
				product := New(2, 2)
				product.Mul(m, inv)
				identity := New(2, 2)
				identity.Eye()
				assert.True(t, matricesEqual(product, identity, 1e-5), "M * M^-1 should equal identity")
			},
		},
		{
			name: "singular matrix",
			init: func(t *testing.T) Matrix {
				// Matrix with zero determinant
				return New(2, 2, 1, 2, 2, 4)
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
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			dst := New(len(m), len(m[0]))
			err := m.Inverse(dst)

			if tt.wantErr {
				assert.Error(t, err, "Matrix.Inverse() should return error")
			} else {
				assert.NoError(t, err, "Matrix.Inverse() should not return error")
			}

			if tt.verify != nil && err == nil {
				tt.verify(m, dst, t)
			}
		})
	}
}

func TestMatrix2x2_Inverse(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix2x2
		wantErr bool
		verify  func(m, inv *Matrix2x2, t *testing.T)
	}{
		{
			name: "identity",
			init: func(t *testing.T) *Matrix2x2 {
				mat := Matrix2x2{}.Eye().(Matrix2x2)
				return &mat
			},
			wantErr: false,
			verify: func(m, inv *Matrix2x2, t *testing.T) {
				product := &Matrix2x2{}
				product.Mul(m, inv)
				identity := &Matrix2x2{}
				identity.Eye()
				assert.True(t, matrices2x2Equal(product, identity, 1e-5), "M * M^-1 should equal identity")
			},
		},
		{
			name: "known inverse",
			init: func(t *testing.T) *Matrix2x2 {
				// [1 2]^-1   = [-2  1 ]
				// [3 4]       [ 1.5 -0.5]  (after dividing by det = -2)
				m := New2x2(1, 2, 3, 4)
				return &m
			},
			wantErr: false,
			verify: func(m, inv *Matrix2x2, t *testing.T) {
				product := &Matrix2x2{}
				product.Mul(m, inv)
				identity := &Matrix2x2{}
				identity.Eye()
				assert.True(t, matrices2x2Equal(product, identity, 1e-5), "M * M^-1 should equal identity")
			},
		},
		{
			name: "singular",
			init: func(t *testing.T) *Matrix2x2 {
				m := New2x2(1, 2, 2, 4)
				return &m
			},
			wantErr: true,
			verify:  nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			var dst Matrix2x2
			err := m.Inverse(&dst)

			if tt.wantErr {
				assert.Error(t, err, "Matrix2x2.Inverse() should return error")
			} else {
				assert.NoError(t, err, "Matrix2x2.Inverse() should not return error")
			}

			if tt.verify != nil && err == nil {
				tt.verify(m, &dst, t)
			}
		})
	}
}

func TestMatrix3x3_Inverse(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix3x3
		wantErr bool
		verify  func(m, inv *Matrix3x3, t *testing.T)
	}{
		{
			name: "identity",
			init: func(t *testing.T) *Matrix3x3 {
				mat := Matrix3x3{}.Eye().(Matrix3x3)
				return &mat
			},
			wantErr: false,
			verify: func(m, inv *Matrix3x3, t *testing.T) {
				product := &Matrix3x3{}
				product.Mul(m, inv)
				identity := &Matrix3x3{}
				identity.Eye()
				assert.True(t, matrices3x3Equal(product, identity, 1e-5), "M * M^-1 should equal identity")
			},
		},
		{
			name: "rotation matrix",
			init: func(t *testing.T) *Matrix3x3 {
				mat := Matrix3x3{}.RotationZ(math32.Pi / 4).(Matrix3x3)
				return &mat
			},
			wantErr: false,
			verify: func(m, inv *Matrix3x3, t *testing.T) {
				product := &Matrix3x3{}
				product.Mul(m, inv)
				identity := &Matrix3x3{}
				identity.Eye()
				assert.True(t, matrices3x3Equal(product, identity, 1e-5), "M * M^-1 should equal identity")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			var dst Matrix3x3
			err := m.Inverse(&dst)

			if tt.wantErr {
				assert.Error(t, err, "Matrix3x3.Inverse() should return error")
			} else {
				assert.NoError(t, err, "Matrix3x3.Inverse() should not return error")
			}

			if tt.verify != nil && err == nil {
				tt.verify(m, &dst, t)
			}
		})
	}
}

func TestMatrix4x4_Inverse(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) Matrix4x4
		wantErr bool
		verify  func(m, inv Matrix4x4, t *testing.T)
	}{
		{
			name: "identity",
			init: func(t *testing.T) Matrix4x4 {
				mat := Matrix4x4{}.Eye().(Matrix4x4)
				return mat
			},
			wantErr: false,
			verify: func(m, inv Matrix4x4, t *testing.T) {
				product := Matrix4x4{}.Mul(m, inv)
				identity := Matrix4x4{}.Eye()
				matrices4x4Equal(t, product, identity, 1e-5, "M * M^-1 should equal identity")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			var dst Matrix4x4
			err := m.Inverse(&dst)

			if tt.wantErr {
				assert.Error(t, err, "Matrix4x4.Inverse() should return error")
			} else {
				assert.NoError(t, err, "Matrix4x4.Inverse() should not return error")
			}

			if tt.verify != nil && err == nil {
				tt.verify(m, dst, t)
			}
		})
	}
}

// Helper functions for matrix comparison
func matricesEqual(a, b Matrix, eps float32) bool {
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

func matrices2x2Equal(a, b *Matrix2x2, eps float32) bool {
	for i := range a {
		for j := range a[i] {
			if math32.Abs(a[i][j]-b[i][j]) > eps {
				return false
			}
		}
	}
	return true
}

func matrices3x3Equal(a, b *Matrix3x3, eps float32) bool {
	for i := range a {
		for j := range a[i] {
			if math32.Abs(a[i][j]-b[i][j]) > eps {
				return false
			}
		}
	}
	return true
}

func matrices4x4Equal(t *testing.T, aM, bM matTypes.Matrix, eps float32, msg string) bool {
	a := aM.View().(Matrix)
	b := bM.View().(Matrix)
	for i := range a {
		for j := range a[i] {
			assert.InDelta(t, a[i][j], b[i][j], float64(eps), msg)
		}
	}
	t.Logf("%s: passed", msg)
	return true
}

func TestMatrix4x4_InverseNeedsPivoting(t *testing.T) {
	tests := []struct {
		name string
		m    Matrix4x4
	}{
		{"pose rot z 90deg", Matrix4x4{{0, -1, 0, 1}, {1, 0, 0, 2}, {0, 0, 1, 3}, {0, 0, 0, 1}}},
		{"pose rot x 90deg", Matrix4x4{{1, 0, 0, 0.5}, {0, 0, -1, -2}, {0, 1, 0, 0.25}, {0, 0, 0, 1}}},
		{"permutation", Matrix4x4{{0, 0, 1, 0}, {0, 0, 0, 2}, {3, 0, 0, 0}, {0, 4, 0, 0}}},
		{"general dense", Matrix4x4{{2, 1, -1, 3}, {4, -6, 0, 1}, {-2, 7, 2, 0}, {1, 1, 1, 5}}},
		{"small leading pivot", Matrix4x4{{1e-6, 1, 2, 3}, {1, 2, 3, 4}, {2, 3, 5, 7}, {3, 1, 4, 1}}},
	}
	eye := Matrix4x4{}.Eye().(Matrix4x4)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var inv Matrix4x4
			if err := tt.m.Inverse(&inv); err != nil {
				t.Fatalf("Inverse: %v", err)
			}
			prod := Matrix4x4{}.Mul(tt.m, inv).(Matrix4x4)
			for i := range 4 {
				for j := range 4 {
					if d := math32.Abs(prod[i][j] - eye[i][j]); d > 1e-4 || d != d {
						t.Fatalf("M*M^-1 [%d][%d] = %v, want %v\nprod %v", i, j, prod[i][j], eye[i][j], prod)
					}
				}
			}
		})
	}
}

// Review K11: Matrix4x4.Inverse reports ErrSingular for rank-deficient input.
func TestMatrix4x4_InverseSingular(t *testing.T) {
	tests := []struct {
		name string
		m    Matrix4x4
	}{
		{"zero", Matrix4x4{}},
		{"rank 2", Matrix4x4{{1, 2, 3, 4}, {2, 4, 6, 8}, {0, 1, 0, 1}, {1, 3, 3, 5}}},
		{"rank 3 (row 3 = row 0 + row 1)", Matrix4x4{{2, 1, -1, 3}, {4, -6, 0, 1}, {-2, 7, 2, 0}, {6, -5, -1, 4}}},
		{"rank 3 pose with zero rotation column", Matrix4x4{{0, 0, 0, 1}, {0, 1, 0, 2}, {0, 0, 1, 3}, {0, 0, 0, 1}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var inv Matrix4x4
			assert.ErrorIs(t, tt.m.Inverse(&inv), ErrSingular)
		})
	}
}

// Review K4: the homogeneous-transform helpers compute in place without
// allocating.
func TestHomogenous_NoAllocations(t *testing.T) {
	var h, inv Matrix4x4
	euler := vec.Vector3D{0.1, -0.4, 0.7}
	trans := vec.Vector3D{1, -2, 3}
	q := vec.Quaternion{0.1, 0.2, 0.3, 0.927}
	tests := []struct {
		name string
		fn   func()
	}{
		{"HomogenousFromEuler", func() { h.HomogenousFromEuler(euler, trans) }},
		{"HomogenousFromQuaternion", func() { h.HomogenousFromQuaternion(q, trans) }},
		{"HomogenousInverse", func() { h.HomogenousInverse(&inv) }},
	}
	for _, tt := range tests {
		assert.Zero(t, testing.AllocsPerRun(100, tt.fn), tt.name)
	}
}

// The allocation-free homogeneous helpers match the interface-based
// composition for a general (non-axis-aligned) rotation, and the inverse is
// exact (also in place).
func TestHomogenous_MatchesComposition(t *testing.T) {
	euler := vec.Vector3D{0.3, -0.7, 1.1}
	trans := vec.Vector3D{0.5, -1.5, 2}
	q := vec.Quaternion{0.1, -0.3, 0.2, 0.927}
	n := math32.Sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
	for i := range q {
		q[i] /= n
	}

	rz := Matrix3x3{}.RotationZ(euler[2])
	ry := Matrix3x3{}.RotationY(euler[1])
	rx := Matrix3x3{}.RotationX(euler[0])
	wantEuler := Matrix3x3{}.Mul(Matrix3x3{}.Mul(rz, ry), rx).(Matrix3x3)
	wantQuat := Matrix3x3{}.Orientation(q).(Matrix3x3)

	var he, hq Matrix4x4
	he.HomogenousFromEuler(euler, trans)
	hq.HomogenousFromQuaternion(q, trans)
	for i := range 3 {
		for j := range 3 {
			assert.InDelta(t, wantEuler[i][j], he[i][j], 1e-6, "euler R[%d][%d]", i, j)
			assert.InDelta(t, wantQuat[i][j], hq[i][j], 1e-6, "quaternion R[%d][%d]", i, j)
		}
		assert.Equal(t, trans[i], he[i][3])
		assert.Equal(t, trans[i], hq[i][3])
	}
	assert.Equal(t, [4]float32{0, 0, 0, 1}, [4]float32(he[3]))

	var inv Matrix4x4
	he.HomogenousInverse(&inv)
	inPlace := he
	inPlace.HomogenousInverse(&inPlace)
	assert.Equal(t, inv, inPlace, "dst may alias m")
	matrices4x4Equal(t, Matrix4x4{}.Mul(he, inv), Matrix4x4{}.Eye(), 1e-5, "H * H^-1 should equal identity")
}

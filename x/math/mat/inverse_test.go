package mat

import (
	"testing"

	"github.com/chewxy/math32"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
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

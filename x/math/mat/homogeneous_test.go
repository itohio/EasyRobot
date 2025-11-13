package mat

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/vec"
	"github.com/stretchr/testify/assert"
)

func TestMatrix4x4_Homogenous(t *testing.T) {
	tests := []struct {
		name   string
		rot    Matrix3x3
		trans  vec.Vector3D
		verify func(m *Matrix4x4, t *testing.T)
	}{
		{
			name:  "identity rotation and zero translation",
			rot:   FromDiagonal3x3(1, 1, 1),
			trans: vec.Vector3D{0, 0, 0},
			verify: func(m *Matrix4x4, t *testing.T) {
				// Should be identity
				for i := 0; i < 4; i++ {
					for j := 0; j < 4; j++ {
						expected := float32(0)
						if i == j {
							expected = 1
						}
						assert.InDelta(t, expected, m[i][j], 1e-5, "Homogenous: [%d][%d]", i, j)
					}
				}
			},
		},
		{
			name:  "with translation",
			rot:   FromDiagonal3x3(1, 1, 1),
			trans: vec.Vector3D{1, 2, 3},
			verify: func(m *Matrix4x4, t *testing.T) {
				// Check translation
				assert.InDelta(t, 1.0, m[0][3], 1e-5, "Homogenous: translation[0]")
				assert.InDelta(t, 2.0, m[1][3], 1e-5, "Homogenous: translation[1]")
				assert.InDelta(t, 3.0, m[2][3], 1e-5, "Homogenous: translation[2]")
				// Check bottom row
				assert.InDelta(t, 1.0, m[3][3], 1e-5, "Homogenous: bottom row should be [0,0,0,1]")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix4x4{}
			rot := tt.rot
			result := m.Homogenous(&rot, tt.trans)
			assert.Equal(t, m, result, "Homogenous: should return receiver")
			if tt.verify != nil {
				tt.verify(m, t)
			}
		})
	}
}

func TestMatrix4x4_HomogenousInverse(t *testing.T) {
	tests := []struct {
		name   string
		init   func(t *testing.T) Matrix4x4
		verify func(m, inv Matrix4x4, t *testing.T)
	}{
		{
			name: "identity",
			init: func(t *testing.T) Matrix4x4 {
				m := FromDiagonal4x4(1, 1, 1, 1)
				return m
			},
			verify: func(m, inv Matrix4x4, t *testing.T) {
				product := Matrix4x4{}.Mul(m, inv)
				assert.Equal(t, m.Rows(), product.Rows())
				assert.Equal(t, m.Cols(), product.Cols())
			},
		},
		{
			name: "rotation and translation",
			init: func(t *testing.T) Matrix4x4 {
				m := Matrix4x4{}
				angle := math32.Pi / 4
				c := math32.Cos(angle)
				s := math32.Sin(angle)
				rot := Matrix3x3{
					[3]float32{c, -s, 0},
					[3]float32{s, c, 0},
					[3]float32{0, 0, 1},
				}
				trans := vec.Vector3D{1, 2, 3}
				m.Homogenous(&rot, trans)
				return m
			},
			verify: func(m, inv Matrix4x4, t *testing.T) {
				product := Matrix4x4{}.Mul(m, inv)
				assert.Equal(t, m.Rows(), product.Rows())
				assert.Equal(t, m.Cols(), product.Cols())
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			inv := Matrix4x4{}
			m.HomogenousInverse(&inv)
			if tt.verify != nil {
				tt.verify(m, inv, t)
			}
		})
	}
}

func TestMatrix4x4_SetRotation(t *testing.T) {
	tests := []struct {
		name   string
		rot    Matrix3x3
		verify func(m *Matrix4x4, t *testing.T)
	}{
		{
			name: "set identity rotation",
			rot:  FromDiagonal3x3(1, 1, 1),
			verify: func(m *Matrix4x4, t *testing.T) {
				// Check top-left 3x3
				for i := 0; i < 3; i++ {
					for j := 0; j < 3; j++ {
						expected := float32(0)
						if i == j {
							expected = 1
						}
						assert.InDelta(t, expected, m[i][j], 1e-5, "SetRotation: [%d][%d]", i, j)
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix4x4{}
			rot := tt.rot
			result := m.SetRotation(&rot)
			assert.Equal(t, m, result, "SetRotation: should return receiver")
			if tt.verify != nil {
				tt.verify(m, t)
			}
		})
	}
}

func TestMatrix4x4_SetTranslation(t *testing.T) {
	tests := []struct {
		name   string
		trans  vec.Vector3D
		verify func(m *Matrix4x4, t *testing.T)
	}{
		{
			name:  "set translation",
			trans: vec.Vector3D{1, 2, 3},
			verify: func(m *Matrix4x4, t *testing.T) {
				assert.InDelta(t, 1.0, m[0][3], 1e-5, "SetTranslation: [0][3]")
				assert.InDelta(t, 2.0, m[1][3], 1e-5, "SetTranslation: [1][3]")
				assert.InDelta(t, 3.0, m[2][3], 1e-5, "SetTranslation: [2][3]")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix4x4{}
			result := m.SetTranslation(tt.trans)
			assert.Equal(t, m, result, "SetTranslation: should return receiver")
			if tt.verify != nil {
				tt.verify(m, t)
			}
		})
	}
}

func TestMatrix4x4_GetTranslation(t *testing.T) {
	m := &Matrix4x4{}
	m.SetTranslation(vec.Vector3D{1, 2, 3})

	dst := vec.Vector3D{}
	result := m.GetTranslation(dst)

	// Verify result matches expected
	assert.Equal(t, vec.Vector3D{1, 2, 3}, result, "GetTranslation")
}

func TestMatrix4x4_GetRotation(t *testing.T) {
	rot := &Matrix3x3{}
	rot.RotationZ(math32.Pi / 4)

	m := &Matrix4x4{}
	m.SetRotation(rot)

	dst := &Matrix3x3{}
	result := m.GetRotation(dst)

	assert.Equal(t, dst, result, "GetRotation: should return destination")

	// Check rotation matches
	assert.True(t, matrices3x3Equal(dst, rot, 1e-5), "GetRotation: rotation should match")
}

func TestMatrix4x4_Col3D(t *testing.T) {
	m := &Matrix4x4{}
	m.SetTranslation(vec.Vector3D{1, 2, 3})

	dst := vec.Vector3D{}
	result := m.Col3D(3, dst)

	// Verify result matches expected
	assert.Equal(t, vec.Vector3D{1, 2, 3}, result, "Col3D")
}

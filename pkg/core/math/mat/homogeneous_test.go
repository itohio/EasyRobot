package mat

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
)

func TestMatrix4x4_Homogenous(t *testing.T) {
	tests := []struct {
		name    string
		rot     *Matrix3x3
		trans   vec.Vector3D
		verify  func(m *Matrix4x4, t *testing.T)
	}{
		{
			name: "identity rotation and zero translation",
			rot: func() *Matrix3x3 {
				r := &Matrix3x3{}
				r.Eye()
				return r
			}(),
			trans: vec.Vector3D{0, 0, 0},
			verify: func(m *Matrix4x4, t *testing.T) {
				// Should be identity
				for i := 0; i < 4; i++ {
					for j := 0; j < 4; j++ {
						expected := float32(0)
						if i == j {
							expected = 1
						}
						if math32.Abs(m[i][j]-expected) > 1e-5 {
							t.Errorf("Homogenous: [%d][%d] = %v, want %v", i, j, m[i][j], expected)
						}
					}
				}
			},
		},
		{
			name: "with translation",
			rot: func() *Matrix3x3 {
				r := &Matrix3x3{}
				r.Eye()
				return r
			}(),
			trans: vec.Vector3D{1, 2, 3},
			verify: func(m *Matrix4x4, t *testing.T) {
				// Check translation
				if math32.Abs(m[0][3]-1) > 1e-5 {
					t.Errorf("Homogenous: translation[0] = %v, want 1", m[0][3])
				}
				if math32.Abs(m[1][3]-2) > 1e-5 {
					t.Errorf("Homogenous: translation[1] = %v, want 2", m[1][3])
				}
				if math32.Abs(m[2][3]-3) > 1e-5 {
					t.Errorf("Homogenous: translation[2] = %v, want 3", m[2][3])
				}
				// Check bottom row
				if math32.Abs(m[3][3]-1) > 1e-5 {
					t.Errorf("Homogenous: bottom row should be [0,0,0,1]")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix4x4{}
			result := m.Homogenous(tt.rot, tt.trans)
			if result != m {
				t.Errorf("Homogenous: should return receiver")
			}
			if tt.verify != nil {
				tt.verify(m, t)
			}
		})
	}
}

func TestMatrix4x4_HomogenousInverse(t *testing.T) {
	tests := []struct {
		name    string
		init    func(t *testing.T) *Matrix4x4
		verify  func(m, inv *Matrix4x4, t *testing.T)
	}{
		{
			name: "identity",
			init: func(t *testing.T) *Matrix4x4 {
				m := &Matrix4x4{}
				m.Eye()
				return m
			},
			verify: func(m, inv *Matrix4x4, t *testing.T) {
				// M * M^-1 should be identity
				product := &Matrix4x4{}
				product.Mul(*m, *inv)
				identity := &Matrix4x4{}
				identity.Eye()
				for i := 0; i < 4; i++ {
					for j := 0; j < 4; j++ {
						expected := float32(0)
						if i == j {
							expected = 1
						}
						if math32.Abs(product[i][j]-expected) > 1e-4 {
							t.Errorf("HomogenousInverse: product[%d][%d] = %v, want %v",
								i, j, product[i][j], expected)
						}
					}
				}
			},
		},
		{
			name: "rotation and translation",
			init: func(t *testing.T) *Matrix4x4 {
				m := &Matrix4x4{}
				rot := &Matrix3x3{}
				rot.RotationZ(math32.Pi / 4)
				trans := vec.Vector3D{1, 2, 3}
				m.Homogenous(rot, trans)
				return m
			},
			verify: func(m, inv *Matrix4x4, t *testing.T) {
				// M * M^-1 should be identity
				product := &Matrix4x4{}
				product.Mul(*m, *inv)
				identity := &Matrix4x4{}
				identity.Eye()
				for i := 0; i < 4; i++ {
					for j := 0; j < 4; j++ {
						expected := float32(0)
						if i == j {
							expected = 1
						}
						if math32.Abs(product[i][j]-expected) > 1e-4 {
							t.Errorf("HomogenousInverse: product[%d][%d] = %v, want %v",
								i, j, product[i][j], expected)
						}
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.init(t)
			inv := &Matrix4x4{}
			result := m.HomogenousInverse(inv)
			if result != inv {
				t.Errorf("HomogenousInverse: should return destination")
			}
			if tt.verify != nil {
				tt.verify(m, inv, t)
			}
		})
	}
}

func TestMatrix4x4_SetRotation(t *testing.T) {
	tests := []struct {
		name    string
		rot     *Matrix3x3
		verify  func(m *Matrix4x4, t *testing.T)
	}{
		{
			name: "set identity rotation",
			rot: func() *Matrix3x3 {
				r := &Matrix3x3{}
				r.Eye()
				return r
			}(),
			verify: func(m *Matrix4x4, t *testing.T) {
				// Check top-left 3x3
				for i := 0; i < 3; i++ {
					for j := 0; j < 3; j++ {
						expected := float32(0)
						if i == j {
							expected = 1
						}
						if math32.Abs(m[i][j]-expected) > 1e-5 {
							t.Errorf("SetRotation: [%d][%d] = %v, want %v", i, j, m[i][j], expected)
						}
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix4x4{}
			result := m.SetRotation(tt.rot)
			if result != m {
				t.Errorf("SetRotation: should return receiver")
			}
			if tt.verify != nil {
				tt.verify(m, t)
			}
		})
	}
}

func TestMatrix4x4_SetTranslation(t *testing.T) {
	tests := []struct {
		name    string
		trans   vec.Vector3D
		verify  func(m *Matrix4x4, t *testing.T)
	}{
		{
			name:  "set translation",
			trans: vec.Vector3D{1, 2, 3},
			verify: func(m *Matrix4x4, t *testing.T) {
				if math32.Abs(m[0][3]-1) > 1e-5 {
					t.Errorf("SetTranslation: [0][3] = %v, want 1", m[0][3])
				}
				if math32.Abs(m[1][3]-2) > 1e-5 {
					t.Errorf("SetTranslation: [1][3] = %v, want 2", m[1][3])
				}
				if math32.Abs(m[2][3]-3) > 1e-5 {
					t.Errorf("SetTranslation: [2][3] = %v, want 3", m[2][3])
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := &Matrix4x4{}
			result := m.SetTranslation(tt.trans)
			if result != m {
				t.Errorf("SetTranslation: should return receiver")
			}
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
	if result[0] != 1 || result[1] != 2 || result[2] != 3 {
		t.Errorf("GetTranslation: got %v, want [1, 2, 3]", result)
	}
}

func TestMatrix4x4_GetRotation(t *testing.T) {
	rot := &Matrix3x3{}
	rot.RotationZ(math32.Pi / 4)

	m := &Matrix4x4{}
	m.SetRotation(rot)

	dst := &Matrix3x3{}
	result := m.GetRotation(dst)

	if result != dst {
		t.Errorf("GetRotation: should return destination")
	}

	// Check rotation matches
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if math32.Abs(dst[i][j]-rot[i][j]) > 1e-5 {
				t.Errorf("GetRotation: [%d][%d] = %v, want %v", i, j, dst[i][j], rot[i][j])
			}
		}
	}
}

func TestMatrix4x4_Col3D(t *testing.T) {
	m := &Matrix4x4{}
	m.SetTranslation(vec.Vector3D{1, 2, 3})

	dst := vec.Vector3D{}
	result := m.Col3D(3, dst)

	// Verify result matches expected
	if result[0] != 1 || result[1] != 2 || result[2] != 3 {
		t.Errorf("Col3D: got %v, want [1, 2, 3]", result)
	}
}


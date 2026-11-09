// Package mat provides homogeneous transform matrix operations for robotics.
// Homogeneous matrices are 4x4 matrices representing 3D transformations:
// [R t] where R is 3x3 rotation, t is 3x1 translation
// [0 1]
// Reference: matrices.h::homogenous(), homogenous_inverse(), set_translation(), set_rotation()

package mat

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/vec"
)

// Homogenous creates homogeneous transform matrix from rotation and translation.
// M = [R t] where R is 3x3 rotation matrix, t is 3D translation vector
//
//	[0 1]
//
// Matrix must be 4x4.
func (m *Matrix4x4) Homogenous(rot *Matrix3x3, trans vec.Vector3D) *Matrix4x4 {
	m.SetRotation(rot)
	m.SetTranslation(trans)
	// Set bottom row to [0, 0, 0, 1]
	for j := 0; j < 3; j++ {
		m[3][j] = 0
	}
	m[3][3] = 1.0
	return m
}

// HomogenousFromQuaternion creates homogeneous transform from quaternion rotation and translation.
// M = [R(q) t] where R(q) is rotation from quaternion (same as Matrix3x3.Orientation)
//
//	[0    1]
//
// It computes in place and does not allocate.
func (m *Matrix4x4) HomogenousFromQuaternion(rot vec.Quaternion, trans vec.Vector3D) *Matrix4x4 {
	x, y, z, w := rot[0], rot[1], rot[2], rot[3]
	xx, yy, zz := x*x, y*y, z*z
	xy, xz, yz := x*y, x*z, y*z
	wx, wy, wz := w*x, w*y, w*z
	r := Matrix3x3{
		{1 - 2*(yy+zz), 2 * (xy - wz), 2 * (xz + wy)},
		{2 * (xy + wz), 1 - 2*(xx+zz), 2 * (yz - wx)},
		{2 * (xz - wy), 2 * (yz + wx), 1 - 2*(xx+yy)},
	}
	return m.Homogenous(&r, trans)
}

// HomogenousFromEuler creates homogeneous transform from Euler angles and translation.
// M = [R(euler) t] where R(euler) is rotation from Euler angles
//
//	[0         1]
//
// It computes in place and does not allocate.
func (m *Matrix4x4) HomogenousFromEuler(rot vec.Vector3D, trans vec.Vector3D) *Matrix4x4 {
	// R = Rz(yaw) * Ry(pitch) * Rx(roll) (ZYX convention), expanded.
	cx, sx := math32.Cos(rot[0]), math32.Sin(rot[0])
	cy, sy := math32.Cos(rot[1]), math32.Sin(rot[1])
	cz, sz := math32.Cos(rot[2]), math32.Sin(rot[2])
	r := Matrix3x3{
		{cz * cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx},
		{sz * cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx},
		{-sy, cy * sx, cy * cx},
	}
	return m.Homogenous(&r, trans)
}

// HomogenousInverse computes inverse of homogeneous transform matrix.
// For H = [R t], computes H^-1 = [R^T -R^T*t]
//
//	[0 1]                   [0       1  ]
//
// Uses efficient formula instead of full matrix inverse. dst may be m; it
// does not allocate.
func (m *Matrix4x4) HomogenousInverse(dst *Matrix4x4) {
	var rotT Matrix3x3
	for i := range 3 {
		for j := range 3 {
			rotT[i][j] = m[j][i]
		}
	}
	tx, ty, tz := m[0][3], m[1][3], m[2][3]
	var negRotTt vec.Vector3D
	for i := range 3 {
		negRotTt[i] = -(rotT[i][0]*tx + rotT[i][1]*ty + rotT[i][2]*tz)
	}

	// Construct inverse
	dst.SetRotation(&rotT)
	dst.SetTranslation(negRotTt)
	// Set bottom row to [0, 0, 0, 1]
	for j := 0; j < 3; j++ {
		dst[3][j] = 0
	}
	dst[3][3] = 1.0
}

// SetRotation sets the 3x3 rotation submatrix of 4x4 homogeneous matrix.
// Sets top-left 3x3 block to rotation matrix R.
func (m *Matrix4x4) SetRotation(rot *Matrix3x3) *Matrix4x4 {
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			m[i][j] = rot[i][j]
		}
		m[i][3] = 0 // Clear translation column elements in rotation rows
	}
	return m
}

// SetTranslation sets the translation vector (4th column, first 3 rows) of 4x4 homogeneous matrix.
// Sets translation t in [R t]
//
//	[0 1]
func (m *Matrix4x4) SetTranslation(trans vec.Vector3D) *Matrix4x4 {
	m[0][3] = trans[0]
	m[1][3] = trans[1]
	m[2][3] = trans[2]
	return m
}

// GetTranslation extracts translation vector from 4x4 homogeneous matrix.
// Returns translation t from [R t]
//
//	[0 1]
func (m *Matrix4x4) GetTranslation(dst vec.Vector3D) vec.Vector3D {
	dst[0] = m[0][3]
	dst[1] = m[1][3]
	dst[2] = m[2][3]
	return dst
}

// GetRotation extracts 3x3 rotation submatrix from 4x4 homogeneous matrix.
// Returns rotation R from [R t]
//
//	[0 1]
func (m *Matrix4x4) GetRotation(dst *Matrix3x3) *Matrix3x3 {
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			dst[i][j] = m[i][j]
		}
	}
	return dst
}

// Col3D extracts first 3 elements of a column as 3D vector.
// Useful for extracting translation from column 3 or rotation axis from columns 0-2.
func (m *Matrix4x4) Col3D(col int, dst vec.Vector3D) vec.Vector3D {
	if col < 0 || col >= 4 {
		return dst
	}
	dst[0] = m[0][col]
	dst[1] = m[1][col]
	dst[2] = m[2][col]
	return dst
}

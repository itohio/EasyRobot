// Package mat provides homogeneous transform matrix operations for robotics.
// Homogeneous matrices are 4x4 matrices representing 3D transformations:
// [R t] where R is 3x3 rotation, t is 3x1 translation
// [0 1]
// Reference: matrices.h::homogenous(), homogenous_inverse(), set_translation(), set_rotation()

package mat

import (
	"github.com/itohio/EasyRobot/pkg/core/math/vec"
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
// M = [R(q) t] where R(q) is rotation from quaternion
//
//	[0    1]
func (m *Matrix4x4) HomogenousFromQuaternion(rot vec.Quaternion, trans vec.Vector3D) *Matrix4x4 {
	var rot3x3 Matrix3x3
	rot3x3.Orientation(&rot)
	return m.Homogenous(&rot3x3, trans)
}

// HomogenousFromEuler creates homogeneous transform from Euler angles and translation.
// M = [R(euler) t] where R(euler) is rotation from Euler angles
//
//	[0         1]
func (m *Matrix4x4) HomogenousFromEuler(rot vec.Vector3D, trans vec.Vector3D) *Matrix4x4 {
	var rot3x3 Matrix3x3
	// Create rotation matrix from Euler angles (ZYX convention)
	var rz Matrix3x3
	rz.RotationZ(rot[2])
	var ry Matrix3x3
	ry.RotationY(rot[1])
	var rx Matrix3x3
	rx.RotationX(rot[0])
	// R = Rz * Ry * Rx
	var temp1 Matrix3x3
	temp1.Mul(&rz, &ry)
	var temp2 Matrix3x3
	temp2.Mul(&temp1, &rx)
	rot3x3 = temp2
	return m.Homogenous(&rot3x3, trans)
}

// HomogenousInverse computes inverse of homogeneous transform matrix.
// For H = [R t], computes H^-1 = [R^T -R^T*t]
//
//	[0 1]                   [0       1  ]
//
// Uses efficient formula instead of full matrix inverse.
func (m *Matrix4x4) HomogenousInverse(dst *Matrix4x4) {
	// Extract rotation and translation
	var rot Matrix3x3
	var trans vec.Vector3D
	m.GetRotation(&rot)
	trans = m.GetTranslation(trans)

	// Transpose rotation
	var rotT Matrix3x3
	rotT.Transpose(&rot)

	// Compute -R^T * t
	var negRotTt vec.Vector3D
	product := rotT.MulVec(trans, nil).(vec.Vector3D)
	negRotTt[0] = -product[0]
	negRotTt[1] = -product[1]
	negRotTt[2] = -product[2]

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

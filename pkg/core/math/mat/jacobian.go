package mat

// JacobianColumn represents a single column of geometric Jacobian matrix
// containing linear and angular velocity components.
type JacobianColumn struct {
	Linear  [3]float32 // Linear velocity component (3D)
	Angular [3]float32 // Angular velocity component (3D)
}

// CalculateJacobianColumn calculates one column of geometric Jacobian matrix.
// For revolute joint:
//   Linear = Z_i × (p_ee - p_i)
//   Angular = Z_i
// For prismatic joint:
//   Linear = Z_i
//   Angular = [0, 0, 0]
// where:
//   Z_i = joint rotation/translation axis (unit vector)
//   p_i = joint position
//   p_ee = end-effector position
func CalculateJacobianColumn(
	jointPos [3]float32, // Joint position
	jointAxis [3]float32, // Joint axis (Z for revolute, translation for prismatic)
	eePos [3]float32, // End-effector position
	isRevolute bool, // Joint type (revolute vs prismatic)
) JacobianColumn {
	var col JacobianColumn

	if isRevolute {
		// Linear: Z_i × (p_ee - p_i)
		r := [3]float32{
			eePos[0] - jointPos[0],
			eePos[1] - jointPos[1],
			eePos[2] - jointPos[2],
		}

		// Cross product: jointAxis × r
		col.Linear[0] = jointAxis[1]*r[2] - jointAxis[2]*r[1]
		col.Linear[1] = jointAxis[2]*r[0] - jointAxis[0]*r[2]
		col.Linear[2] = jointAxis[0]*r[1] - jointAxis[1]*r[0]

		// Angular: Z_i
		col.Angular[0] = jointAxis[0]
		col.Angular[1] = jointAxis[1]
		col.Angular[2] = jointAxis[2]
	} else {
		// Prismatic: Linear = Z_i, Angular = 0
		col.Linear[0] = jointAxis[0]
		col.Linear[1] = jointAxis[1]
		col.Linear[2] = jointAxis[2]

		col.Angular[0] = 0
		col.Angular[1] = 0
		col.Angular[2] = 0
	}

	return col
}

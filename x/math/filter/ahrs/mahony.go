package ahrs

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

type MahonyAHRS struct {
	Options
	q            vec.Quaternion
	eInt         vec.Vector3D
	SamplePeriod float32

	// Filter interface - fixed matrix for input (3x3: accel, gyro, mag rows)
	inputMatrix mat.Matrix3x3
	// Output is quaternion (fixed [4]float32)
	outputQuat vec.Quaternion
}

func NewMahony(opts ...Option) *MahonyAHRS {
	m := &MahonyAHRS{
		Options:    defaultOptions(),
		q:          vec.Quaternion{1, 0, 0, 0},
		eInt:       vec.Vector3D{0, 0, 0},
		outputQuat: vec.Quaternion{1, 0, 0, 0},
	}
	applyOptions(&m.Options, opts...)
	return m
}

// Reset resets the filter state to identity quaternion.
func (m *MahonyAHRS) Reset() {
	m.q = vec.Quaternion{1, 0, 0, 0}
	m.eInt = vec.Vector3D{0, 0, 0}
	m.outputQuat = vec.Quaternion{1, 0, 0, 0}
}

// asMatrix3x3 casts matTypes.Matrix to Matrix3x3 for direct access.
func asMatrix3x3Mahony(arg matTypes.Matrix, op string) mat.Matrix3x3 {
	switch v := arg.(type) {
	case mat.Matrix3x3:
		return v
	case *mat.Matrix3x3:
		return *v
	case mat.Matrix:
		// Matrix is [][]float32, extract 3x3
		if len(v) < 3 || len(v[0]) < 3 {
			panic(fmt.Sprintf("ahrs.%s: matrix must be at least 3x3", op))
		}
		return mat.Matrix3x3{
			{v[0][0], v[0][1], v[0][2]},
			{v[1][0], v[1][1], v[1][2]},
			{v[2][0], v[2][1], v[2][2]},
		}
	default:
		panic(fmt.Sprintf("ahrs.%s: unsupported matrix type %T", op, arg))
	}
}

// Update implements the Filter interface.
// Input matrix format: 3x3 matrix where rows are [accel, gyro, mag]
//
//	Row 0: [ax, ay, az] - accelerometer
//	Row 1: [gx, gy, gz] - gyroscope
//	Row 2: [mx, my, mz] - magnetometer
//
// Output: quaternion [qw, qx, qy, qz]
func (m *MahonyAHRS) Update(timestep float32, input matTypes.Matrix) {
	if input == nil {
		return
	}

	// Cast to Matrix3x3 for direct access
	m.inputMatrix = asMatrix3x3Mahony(input, "Update")
	m.SamplePeriod = timestep

	// Perform calculation using direct matrix access
	m.calculate()

	// Update output quaternion
	m.outputQuat = m.q
}

// Input returns the input matrix.
func (m *MahonyAHRS) Input() matTypes.Matrix {
	return m.inputMatrix
}

// Output returns the estimated quaternion [qw, qx, qy, qz].
func (m *MahonyAHRS) Output() vec.Quaternion {
	return m.outputQuat
}

// calculate performs the filter calculation (internal method).
// Uses direct matrix access - no vector copies.
func (m *MahonyAHRS) calculate() {
	if !(m.Options.HasAccelerator && m.Options.HasGyroscope) {
		panic("ahrs: accelerometer and gyroscope are required")
	}
	if !m.Options.HasMagnetometer {
		m.calculateWOMag()
		return
	}

	q1, q2, q3, q4 := m.q[0], m.q[1], m.q[2], m.q[3] // short name local variable for readability

	// Auxiliary variables to avoid repeated arithmetic
	q1q1 := q1 * q1
	q1q2 := q1 * q2
	q1q3 := q1 * q3
	q1q4 := q1 * q4
	q2q2 := q2 * q2
	q2q3 := q2 * q3
	q2q4 := q2 * q4
	q3q3 := q3 * q3
	q3q4 := q3 * q4
	q4q4 := q4 * q4

	// Normalise accelerometer measurement - direct matrix access
	ax := m.inputMatrix[0][0]
	ay := m.inputMatrix[0][1]
	az := m.inputMatrix[0][2]
	accelMag := math32.Sqrt(ax*ax + ay*ay + az*az)
	if accelMag > 0 {
		ax /= accelMag
		ay /= accelMag
		az /= accelMag
	}

	// Normalise magnetometer measurement - direct matrix access
	mx := m.inputMatrix[2][0]
	my := m.inputMatrix[2][1]
	mz := m.inputMatrix[2][2]
	magMag := math32.Sqrt(mx*mx + my*my + mz*mz)
	if magMag > 0 {
		mx /= magMag
		my /= magMag
		mz /= magMag
	}

	// Reference direction of Earth's magnetic field
	hx := 2*mx*(0.5-q3q3-q4q4) + 2*my*(q2q3-q1q4) + 2*mz*(q2q4+q1q3)
	hy := 2*mx*(q2q3+q1q4) + 2*my*(0.5-q2q2-q4q4) + 2*mz*(q3q4-q1q2)
	bx := math32.Sqrt((hx * hx) + (hy * hy))
	bz := 2*mx*(q2q4-q1q3) + 2*my*(q3q4+q1q2) + 2*mz*(0.5-q2q2-q3q3)

	// Estimated direction of gravity and magnetic field
	v := vec.Vector3D{
		2 * (q2q4 - q1q3),
		2 * (q1q2 + q3q4),
		q1q1 - q2q2 - q3q3 + q4q4,
	}
	w := vec.Vector3D{
		2*bx*(0.5-q3q3-q4q4) + 2*bz*(q2q4-q1q3),
		2*bx*(q2q3-q1q4) + 2*bz*(q1q2+q3q4),
		2*bx*(q1q3+q2q4) + 2*bz*(0.5-q2q2-q3q3),
	}

	// Error is cross product between estimated direction and measured direction of gravity
	e := vec.Vector3D{
		(ay*v[2] - az*v[1]) + (my*w[2] - mz*w[1]),
		(az*v[0] - ax*v[2]) + (mz*w[0] - mx*w[2]),
		(ax*v[1] - ay*v[0]) + (mx*w[1] - my*w[0]),
	}
	if m.GainI > 0 {
		m.eInt[0] += e[0] // accumulate integral error
		m.eInt[1] += e[1]
		m.eInt[2] += e[2]
	} else {
		m.eInt = vec.Vector3D{} // prevent integral wind up
	}

	// Apply feedback terms - direct matrix access for gyro
	gx := m.inputMatrix[1][0]
	gy := m.inputMatrix[1][1]
	gz := m.inputMatrix[1][2]
	g := vec.Vector3D{
		gx + m.GainP*e[0] + m.GainI*m.eInt[0],
		gy + m.GainP*e[1] + m.GainI*m.eInt[1],
		gz + m.GainP*e[2] + m.GainI*m.eInt[2],
	}

	// Integrate rate of change of quaternion
	m.q[0] = q1 + (-q2*g[0]-q3*g[1]-q4*g[2])*(0.5*m.SamplePeriod)
	m.q[1] = q2 + (q1*g[0]+q3*g[2]-q4*g[1])*(0.5*m.SamplePeriod)
	m.q[2] = q3 + (q1*g[1]-q2*g[2]+q4*g[0])*(0.5*m.SamplePeriod)
	m.q[3] = q4 + (q1*g[2]+q2*g[1]-q3*g[0])*(0.5*m.SamplePeriod)

	// Normalize quaternion
	qMag := math32.Sqrt(m.q[0]*m.q[0] + m.q[1]*m.q[1] + m.q[2]*m.q[2] + m.q[3]*m.q[3])
	if qMag > 0 {
		m.q[0] /= qMag
		m.q[1] /= qMag
		m.q[2] /= qMag
		m.q[3] /= qMag
	}
}

func (m *MahonyAHRS) calculateWOMag() {
	q1, q2, q3, q4 := m.q[0], m.q[1], m.q[2], m.q[3] // short name local variable for readability

	// Normalise accelerometer measurement - direct matrix access
	ax := m.inputMatrix[0][0]
	ay := m.inputMatrix[0][1]
	az := m.inputMatrix[0][2]
	accelMag := math32.Sqrt(ax*ax + ay*ay + az*az)
	if accelMag > 0 {
		ax /= accelMag
		ay /= accelMag
		az /= accelMag
	}

	// Estimated direction of gravity
	v := vec.Vector3D{
		2.0 * (q2*q4 - q1*q3),
		2.0 * (q1*q2 + q3*q4),
		q1*q1 - q2*q2 - q3*q3 + q4*q4,
	}

	// Error is cross product between estimated direction and measured direction of gravity
	e := vec.Vector3D{
		v[1]*az - v[2]*ay,
		v[2]*ax - v[0]*az,
		v[0]*ay - v[1]*ax,
	}
	if m.GainI > 0 {
		m.eInt[0] += e[0] // accumulate integral error
		m.eInt[1] += e[1]
		m.eInt[2] += e[2]
	} else {
		m.eInt = vec.Vector3D{} // prevent integral wind up
	}

	// Apply feedback terms - direct matrix access for gyro
	gx := m.inputMatrix[1][0]
	gy := m.inputMatrix[1][1]
	gz := m.inputMatrix[1][2]
	g := vec.Vector3D{
		gx + m.GainP*e[0] + m.GainI*m.eInt[0],
		gy + m.GainP*e[1] + m.GainI*m.eInt[1],
		gz + m.GainP*e[2] + m.GainI*m.eInt[2],
	}

	// Integrate rate of change of quaternion
	m.q[0] = q1 + (-q2*g[0]-q3*g[1]-q4*g[2])*(0.5*m.SamplePeriod)
	m.q[1] = q2 + (q1*g[0]+q3*g[2]-q4*g[1])*(0.5*m.SamplePeriod)
	m.q[2] = q3 + (q1*g[1]-q2*g[2]+q4*g[0])*(0.5*m.SamplePeriod)
	m.q[3] = q4 + (q1*g[2]+q2*g[1]-q3*g[0])*(0.5*m.SamplePeriod)

	// Normalize quaternion
	qMag := math32.Sqrt(m.q[0]*m.q[0] + m.q[1]*m.q[1] + m.q[2]*m.q[2] + m.q[3]*m.q[3])
	if qMag > 0 {
		m.q[0] /= qMag
		m.q[1] /= qMag
		m.q[2] /= qMag
		m.q[3] /= qMag
	}
}

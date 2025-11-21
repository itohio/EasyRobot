package ahrs

import (
	"fmt"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/mat"
	matTypes "github.com/itohio/EasyRobot/x/math/mat/types"
	"github.com/itohio/EasyRobot/x/math/vec"
)

type MadgwickAHRS struct {
	Options
	q            vec.Quaternion
	SamplePeriod float32

	// Filter interface - fixed matrix for input (3x3: accel, gyro, mag rows)
	inputMatrix mat.Matrix3x3
	// Output is quaternion (fixed [4]float32)
	outputQuat vec.Quaternion
}

func NewMadgwick(opts ...Option) *MadgwickAHRS {
	m := &MadgwickAHRS{
		Options:    defaultOptions(),
		q:          vec.Quaternion{1, 0, 0, 0},
		outputQuat: vec.Quaternion{1, 0, 0, 0},
	}
	applyOptions(&m.Options, opts...)
	return m
}

// Reset resets the filter state to identity quaternion.
func (m *MadgwickAHRS) Reset() {
	m.q = vec.Quaternion{1, 0, 0, 0}
	m.outputQuat = vec.Quaternion{1, 0, 0, 0}
}

// asMatrix3x3 casts matTypes.Matrix to Matrix3x3 for direct access.
func asMatrix3x3(arg matTypes.Matrix, op string) mat.Matrix3x3 {
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
func (m *MadgwickAHRS) Update(timestep float32, input matTypes.Matrix) {
	if input == nil {
		return
	}

	// Cast to Matrix3x3 for direct access
	m.inputMatrix = asMatrix3x3(input, "Update")
	m.SamplePeriod = timestep

	// Perform calculation using direct matrix access
	m.calculate()

	// Update output quaternion
	m.outputQuat = m.q
}

// Input returns the input matrix.
func (m *MadgwickAHRS) Input() matTypes.Matrix {
	return m.inputMatrix
}

// Output returns the estimated quaternion [qw, qx, qy, qz].
func (m *MadgwickAHRS) Output() vec.Quaternion {
	return m.outputQuat
}

// calculate performs the filter calculation (internal method).
// Uses direct matrix access - no vector copies.
func (m *MadgwickAHRS) calculate() {
	if !(m.Options.HasAccelerator && m.Options.HasGyroscope) {
		panic("ahrs: accelerometer and gyroscope are required")
	}
	if !m.Options.HasMagnetometer {
		m.calculateWOMag()
		return
	}

	q1, q2, q3, q4 := m.q[0], m.q[1], m.q[2], m.q[3] // short name local variable for readability

	// Auxiliary variables to avoid repeated arithmetic
	_2q1 := 2 * q1
	_2q2 := 2 * q2
	_2q3 := 2 * q3
	_2q4 := 2 * q4
	_2q1q3 := 2 * q1 * q3
	_2q3q4 := 2 * q3 * q4
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
	_2q1mx := 2 * q1 * mx
	_2q1my := 2 * q1 * my
	_2q1mz := 2 * q1 * mz
	_2q2mx := 2 * q2 * mx
	hx := mx*q1q1 - _2q1my*q4 + _2q1mz*q3 + mx*q2q2 + _2q2*my*q3 + _2q2*mz*q4 - mx*q3q3 - mx*q4q4
	hy := _2q1mx*q4 + my*q1q1 - _2q1mz*q2 + _2q2mx*q3 - my*q2q2 + my*q3q3 + _2q3*mz*q4 - my*q4q4
	_2bx := math32.Sqrt(hx*hx + hy*hy)
	_2bz := -_2q1mx*q3 + _2q1my*q2 + mz*q1q1 + _2q2mx*q4 - mz*q2q2 + _2q3*my*q4 - mz*q3q3 + mz*q4q4
	_4bx := 2 * _2bx
	_4bz := 2 * _2bz

	// Gradient decent algorithm corrective step
	s := vec.Quaternion{
		-_2q3*(2*q2q4-_2q1q3-ax) + _2q2*(2*q1q2+_2q3q4-ay) - _2bz*q3*(_2bx*(0.5-q3q3-q4q4)+_2bz*(q2q4-q1q3)-mx) + (-_2bx*q4+_2bz*q2)*(_2bx*(q2q3-q1q4)+_2bz*(q1q2+q3q4)-my) + _2bx*q3*(_2bx*(q1q3+q2q4)+_2bz*(0.5-q2q2-q3q3)-mz),
		_2q4*(2*q2q4-_2q1q3-ax) + _2q1*(2*q1q2+_2q3q4-ay) - 4*q2*(1-2*q2q2-2*q3q3-az) + _2bz*q4*(_2bx*(0.5-q3q3-q4q4)+_2bz*(q2q4-q1q3)-mx) + (_2bx*q3+_2bz*q1)*(_2bx*(q2q3-q1q4)+_2bz*(q1q2+q3q4)-my) + (_2bx*q4-_4bz*q2)*(_2bx*(q1q3+q2q4)+_2bz*(0.5-q2q2-q3q3)-mz),
		-_2q1*(2*q2q4-_2q1q3-ax) + _2q4*(2*q1q2+_2q3q4-ay) - 4*q3*(1-2*q2q2-2*q3q3-az) + (-_4bx*q3-_2bz*q1)*(_2bx*(0.5-q3q3-q4q4)+_2bz*(q2q4-q1q3)-mx) + (_2bx*q2+_2bz*q4)*(_2bx*(q2q3-q1q4)+_2bz*(q1q2+q3q4)-my) + (_2bx*q1-_4bz*q3)*(_2bx*(q1q3+q2q4)+_2bz*(0.5-q2q2-q3q3)-mz),
		_2q2*(2*q2q4-_2q1q3-ax) + _2q3*(2*q1q2+_2q3q4-ay) + (-_4bx*q4+_2bz*q2)*(_2bx*(0.5-q3q3-q4q4)+_2bz*(q2q4-q1q3)-mx) + (-_2bx*q1+_2bz*q3)*(_2bx*(q2q3-q1q4)+_2bz*(q1q2+q3q4)-my) + _2bx*q2*(_2bx*(q1q3+q2q4)+_2bz*(0.5-q2q2-q3q3)-mz),
	}
	sMag := math32.Sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2] + s[3]*s[3])
	if sMag > 0 {
		s[0] /= sMag
		s[1] /= sMag
		s[2] /= sMag
		s[3] /= sMag
	}

	// Compute rate of change of quaternion - direct matrix access for gyro
	gx := m.inputMatrix[1][0]
	gy := m.inputMatrix[1][1]
	gz := m.inputMatrix[1][2]
	qDot := vec.Quaternion{
		0.5 * (-q2*gx - q3*gy - q4*gz),
		0.5 * (q1*gx + q3*gz - q4*gy),
		0.5 * (q1*gy - q2*gz + q4*gx),
		0.5 * (q1*gz + q2*gy - q3*gx),
	}
	qDot[0] = qDot[0] - m.GainP*s[0]
	qDot[1] = qDot[1] - m.GainP*s[1]
	qDot[2] = qDot[2] - m.GainP*s[2]
	qDot[3] = qDot[3] - m.GainP*s[3]

	// Integrate to yield quaternion
	m.q[0] = q1 + qDot[0]*m.SamplePeriod
	m.q[1] = q2 + qDot[1]*m.SamplePeriod
	m.q[2] = q3 + qDot[2]*m.SamplePeriod
	m.q[3] = q4 + qDot[3]*m.SamplePeriod

	// Normalize quaternion
	qMag := math32.Sqrt(m.q[0]*m.q[0] + m.q[1]*m.q[1] + m.q[2]*m.q[2] + m.q[3]*m.q[3])
	if qMag > 0 {
		m.q[0] /= qMag
		m.q[1] /= qMag
		m.q[2] /= qMag
		m.q[3] /= qMag
	}
}

func (m *MadgwickAHRS) calculateWOMag() {
	q1, q2, q3, q4 := m.q[0], m.q[1], m.q[2], m.q[3] // short name local variable for readability

	// Auxiliary variables to avoid repeated arithmetic
	_2q1 := 2 * q1
	_2q2 := 2 * q2
	_2q3 := 2 * q3
	_2q4 := 2 * q4
	_4q1 := 4 * q1
	_4q2 := 4 * q2
	_4q3 := 4 * q3
	_8q2 := 8 * q2
	_8q3 := 8 * q3
	q1q1 := q1 * q1
	q2q2 := q2 * q2
	q3q3 := q3 * q3
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

	// Gradient decent algorithm corrective step
	s := vec.Quaternion{
		_4q1*q3q3 + _2q3*ax + _4q1*q2q2 - _2q2*ay,
		_4q2*q4q4 - _2q4*ax + 4*q1q1*q2 - _2q1*ay - _4q2 + _8q2*q2q2 + _8q2*q3q3 + _4q2*az,
		4*q1q1*q3 + _2q1*ax + _4q3*q4q4 - _2q4*ay - _4q3 + _8q3*q2q2 + _8q3*q3q3 + _4q3*az,
		4*q2q2*q4 - _2q2*ax + 4*q3q3*q4 - _2q3*ay,
	}
	sMag := math32.Sqrt(s[0]*s[0] + s[1]*s[1] + s[2]*s[2] + s[3]*s[3])
	if sMag > 0 {
		s[0] /= sMag
		s[1] /= sMag
		s[2] /= sMag
		s[3] /= sMag
	}

	// Compute rate of change of quaternion - direct matrix access for gyro
	gx := m.inputMatrix[1][0]
	gy := m.inputMatrix[1][1]
	gz := m.inputMatrix[1][2]
	qDot := vec.Quaternion{
		0.5 * (-q2*gx - q3*gy - q4*gz),
		0.5 * (q1*gx + q3*gz - q4*gy),
		0.5 * (q1*gy - q2*gz + q4*gx),
		0.5 * (q1*gz + q2*gy - q3*gx),
	}
	qDot[0] = qDot[0] - m.GainP*s[0]
	qDot[1] = qDot[1] - m.GainP*s[1]
	qDot[2] = qDot[2] - m.GainP*s[2]
	qDot[3] = qDot[3] - m.GainP*s[3]

	// Integrate to yield quaternion
	m.q[0] = q1 + qDot[0]*m.SamplePeriod
	m.q[1] = q2 + qDot[1]*m.SamplePeriod
	m.q[2] = q3 + qDot[2]*m.SamplePeriod
	m.q[3] = q4 + qDot[3]*m.SamplePeriod

	// Normalize quaternion
	qMag := math32.Sqrt(m.q[0]*m.q[0] + m.q[1]*m.q[1] + m.q[2]*m.q[2] + m.q[3]*m.q[3])
	if qMag > 0 {
		m.q[0] /= qMag
		m.q[1] /= qMag
		m.q[2] /= qMag
		m.q[3] /= qMag
	}
}

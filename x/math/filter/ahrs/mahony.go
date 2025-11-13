package ahrs

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/vec"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

type MahonyAHRS struct {
	Options
	accel, gyro, mag vec.Vector3D
	q                vec.Quaternion
	eInt             vec.Vector3D
	SamplePeriod     float32
}

func NewMahony(opts ...Option) AHRS {
	m := MahonyAHRS{
		Options: defaultOptions(),
		q:       vec.Quaternion{1, 0, 0, 0},
	}
	applyOptions(&m.Options, opts...)

	return &m
}

func (m *MahonyAHRS) Acceleration() vecTypes.Vector {
	return m.accel.View()
}

func (m *MahonyAHRS) Gyroscope() vecTypes.Vector {
	return m.gyro.View()
}

func (m *MahonyAHRS) Magnetometer() vecTypes.Vector {
	return m.mag.View()
}

func (m *MahonyAHRS) Orientation() vecTypes.Vector {
	return m.q.View()
}

func (m *MahonyAHRS) Reset() AHRS {
	m.q = vec.Quaternion{1, 0, 0, 0}
	m.eInt = vec.Vector3D{0, 0, 0}
	return m
}

func (m *MahonyAHRS) Update(samplePeriod float32) AHRS {
	m.SamplePeriod = samplePeriod
	return m
}

func (m *MahonyAHRS) Calculate() AHRS {
	if !(m.Options.HasAccelerator && m.Options.HasGyroscope) {
		panic(-1)
	}
	if !m.Options.HasMagnetometer {
		return m.calculateWOMag()
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

	// Normalise accelerometer measurement
	a := m.accel.NormalFast().(vec.Vector3D)

	// Normalise magnetometer measurement
	mag := m.mag.NormalFast().(vec.Vector3D)

	// Reference direction of Earth's magnetic field
	hx := 2*mag[0]*(0.5-q3q3-q4q4) + 2*mag[1]*(q2q3-q1q4) + 2*mag[2]*(q2q4+q1q3)
	hy := 2*mag[0]*(q2q3+q1q4) + 2*mag[1]*(0.5-q2q2-q4q4) + 2*mag[2]*(q3q4-q1q2)
	bx := math32.Sqrt((hx * hx) + (hy * hy))
	bz := 2*mag[0]*(q2q4-q1q3) + 2*mag[1]*(q3q4+q1q2) + 2*mag[2]*(0.5-q2q2-q3q3)

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
		(a[1]*v[2] - a[2]*v[1]) + (mag[1]*w[2] - mag[2]*w[1]),
		(a[2]*v[0] - a[0]*v[2]) + (mag[2]*w[0] - mag[0]*w[2]),
		(a[0]*v[1] - a[1]*v[0]) + (mag[0]*w[1] - mag[1]*w[0]),
	}
	if m.GainI > 0 {
		m.eInt = m.eInt.Add(e).(vec.Vector3D) // accumulate integral error
	} else {
		m.eInt = vec.Vector3D{} // prevent integral wind up
	}

	// Apply feedback terms
	g := m.gyro.MulCAdd(m.GainP, e).MulCAdd(m.GainI, m.eInt).(vec.Vector3D)

	// Integrate rate of change of quaternion
	m.q[0] = q1 + (-q2*g[0]-q3*g[1]-q4*g[2])*(0.5*m.SamplePeriod)
	m.q[1] = q2 + (q1*g[0]+q3*g[2]-q4*g[1])*(0.5*m.SamplePeriod)
	m.q[2] = q3 + (q1*g[1]-q2*g[2]+q4*g[0])*(0.5*m.SamplePeriod)
	m.q[3] = q4 + (q1*g[2]+q2*g[1]-q3*g[0])*(0.5*m.SamplePeriod)

	m.q = m.q.NormalFast().(vec.Quaternion)

	return m
}

func (m *MahonyAHRS) calculateWOMag() AHRS {
	q1, q2, q3, q4 := m.q[0], m.q[1], m.q[2], m.q[3] // short name local variable for readability

	// Normalise accelerometer measurement
	a := m.accel.NormalFast().(vec.Vector3D)

	// Estimated direction of gravity
	v := vec.Vector3D{
		2.0 * (q2*q4 - q1*q3),
		2.0 * (q1*q2 + q3*q4),
		q1*q1 - q2*q2 - q3*q3 + q4*q4,
	}

	// Error is cross product between estimated direction and measured direction of gravity
	e := v.Cross(a).(vec.Vector3D)
	if m.GainI > 0 {
		m.eInt = m.eInt.Add(e).(vec.Vector3D) // accumulate integral error
	} else {
		m.eInt = vec.Vector3D{} // prevent integral wind up
	}

	// Apply feedback terms
	g := m.gyro.MulCAdd(m.GainP, e).MulCAdd(m.GainI, m.eInt).(vec.Vector3D)

	// Integrate rate of change of quaternion
	m.q[0] = q1 + (-q2*g[0]-q3*g[1]-q4*g[2])*(0.5*m.SamplePeriod)
	m.q[1] = q2 + (q1*g[0]+q3*g[2]-q4*g[1])*(0.5*m.SamplePeriod)
	m.q[2] = q3 + (q1*g[1]-q2*g[2]+q4*g[0])*(0.5*m.SamplePeriod)
	m.q[3] = q4 + (q1*g[2]+q2*g[1]-q3*g[0])*(0.5*m.SamplePeriod)

	m.q = m.q.NormalFast().(vec.Quaternion)

	return m
}

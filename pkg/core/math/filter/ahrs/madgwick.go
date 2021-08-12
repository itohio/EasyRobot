package ahrs

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math/vec"
)

type MadgwickAHRS struct {
	Options
	accel, gyro, mag vec.Vector3D
	q                vec.Quaternion
	SamplePeriod     float32
}

func NewMadgwick(opts ...Option) AHRS {
	m := MadgwickAHRS{
		Options: defaultOptions(),
		q:       vec.Quaternion{1, 0, 0, 0},
	}
	applyOptions(&m.Options, opts...)

	return &m
}

func (m *MadgwickAHRS) Acceleration() vec.Vector {
	return m.accel[:]
}

func (m *MadgwickAHRS) Gyroscope() vec.Vector {
	return m.gyro[:]
}

func (m *MadgwickAHRS) Magnetometer() vec.Vector {
	return m.mag[:]
}

func (m *MadgwickAHRS) Orientation() vec.Vector {
	return m.q[:]
}

func (m *MadgwickAHRS) Reset() AHRS {
	m.q.Vector().FillC(0)
	m.q[0] = 1
	return m
}

func (m *MadgwickAHRS) Update(samplePeriod float32) AHRS {
	m.SamplePeriod = samplePeriod
	return m
}

func (m *MadgwickAHRS) Calculate() AHRS {
	if !(m.Options.HasAccelerator && m.Options.HasGyroscope) {
		panic(-1)
	}
	if !m.Options.HasMagnetometer {
		return m.calculateWOMag()
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

	// Normalise accelerometer measurement
	ax, ay, az := m.accel.Clone().NormalFast().XYZ()

	// Normalise magnetometer measurement
	mx, my, mz := m.mag.Clone().NormalFast().XYZ()

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
	s.NormalFast()

	// Compute rate of change of quaternion
	m.q[0] = 0.5 * (-q2*m.gyro[0] - q3*m.gyro[1] - q4*m.gyro[2])
	m.q[1] = 0.5 * (q1*m.gyro[0] + q3*m.gyro[2] - q4*m.gyro[1])
	m.q[2] = 0.5 * (q1*m.gyro[1] - q2*m.gyro[2] + q4*m.gyro[0])
	m.q[3] = 0.5 * (q1*m.gyro[2] + q2*m.gyro[1] - q3*m.gyro[0])
	m.q.MulCSub(m.GainP, s)

	// Integrate to yield quaternion
	m.q.MulCAdd(m.SamplePeriod, m.q)

	m.q.NormalFast()

	return m
}

func (m *MadgwickAHRS) calculateWOMag() AHRS {
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

	// Normalise accelerometer measurement
	a := m.accel.Clone().NormalFast()

	// Gradient decent algorithm corrective step
	s := vec.Quaternion{
		_4q1*q3q3 + _2q3*a[0] + _4q1*q2q2 - _2q2*a[1],
		_4q2*q4q4 - _2q4*a[0] + 4*q1q1*q2 - _2q1*a[1] - _4q2 + _8q2*q2q2 + _8q2*q3q3 + _4q2*a[2],
		4*q1q1*q3 + _2q1*a[0] + _4q3*q4q4 - _2q4*a[1] - _4q3 + _8q3*q2q2 + _8q3*q3q3 + _4q3*a[2],
		4*q2q2*q4 - _2q2*a[0] + 4*q3q3*q4 - _2q3*a[1],
	}
	s.NormalFast()

	// Compute rate of change of quaternion
	m.q[0] = 0.5 * (-q2*m.gyro[0] - q3*m.gyro[1] - q4*m.gyro[2])
	m.q[1] = 0.5 * (q1*m.gyro[0] + q3*m.gyro[2] - q4*m.gyro[1])
	m.q[2] = 0.5 * (q1*m.gyro[1] - q2*m.gyro[2] + q4*m.gyro[0])
	m.q[3] = 0.5 * (q1*m.gyro[2] + q2*m.gyro[1] - q3*m.gyro[0])
	m.q.MulCSub(m.GainP, s)

	// Integrate to yield quaternion
	m.q.MulCAdd(m.SamplePeriod, m.q)

	m.q.NormalFast()

	return m
}

package ahrs

import (
	"testing"

	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math/vec"
)

const (
	quatTol = 1e-4
	vecTol  = 1e-4
)

func TestMahonyCalculateWithoutMagnetometerMatchesReference(t *testing.T) {
	mah := NewMahony(WithMagnetometer(false), WithKP(0.5), WithKI(0.1)).(*MahonyAHRS)
	mah.accel = vec.Vector3D{0.3, 0.4, 0.8660254}
	mah.gyro = vec.Vector3D{0.02, -0.03, 0.04}
	mah.eInt = vec.Vector3D{0.01, -0.02, 0.005}
	mah.q = vec.Quaternion{0.9659258, 0.2588190, 0.0, 0.0}
	mah.Update(0.02)

	expectedQ, expectedEInt := mahonyReferenceStep(*mah)

	mah.Calculate()

	if !approxQuaternion(mah.q, expectedQ, quatTol) {
		t.Fatalf("mahony (no mag) quaternion mismatch:\n\tgot:  %+v\n\texp:  %+v", mah.q, expectedQ)
	}
	if !approxVector3(mah.eInt, expectedEInt, vecTol) {
		t.Fatalf("mahony (no mag) integral mismatch:\n\tgot:  %+v\n\texp:  %+v", mah.eInt, expectedEInt)
	}
}

func TestMahonyCalculateWithMagnetometerMatchesReference(t *testing.T) {
	mah := NewMahony(WithMagnetometer(true), WithKP(0.52), WithKI(0.12)).(*MahonyAHRS)
	mah.accel = vec.Vector3D{0.25, -0.45, 0.86}
	mah.gyro = vec.Vector3D{-0.015, 0.025, -0.035}
	mah.mag = vec.Vector3D{0.48, 0.12, -0.32}
	mah.eInt = vec.Vector3D{-0.02, 0.015, -0.01}
	mah.q = vec.Quaternion{0.9238795, 0.2, -0.2, 0.2}
	mah.Update(0.01)

	expectedQ, expectedEInt := mahonyReferenceStep(*mah)

	mah.Calculate()

	if !approxQuaternion(mah.q, expectedQ, quatTol) {
		t.Fatalf("mahony (with mag) quaternion mismatch:\n\tgot:  %+v\n\texp:  %+v", mah.q, expectedQ)
	}
	if !approxVector3(mah.eInt, expectedEInt, vecTol) {
		t.Fatalf("mahony (with mag) integral mismatch:\n\tgot:  %+v\n\texp:  %+v", mah.eInt, expectedEInt)
	}
}

func TestMahonyReset(t *testing.T) {
	mah := NewMahony().(*MahonyAHRS)
	mah.q = vec.Quaternion{0.8, -0.1, 0.3, 0.5}
	mah.eInt = vec.Vector3D{0.01, -0.01, 0.02}

	mah.Reset()

	if want := (vec.Quaternion{1, 0, 0, 0}); mah.q != want {
		t.Fatalf("mahony reset quaternion: got %v, want %v", mah.q, want)
	}
	if want := (vec.Vector3D{0, 0, 0}); mah.eInt != want {
		t.Fatalf("mahony reset integral: got %v, want %v", mah.eInt, want)
	}
}

func TestMadgwickCalculateWithMagnetometerMatchesReference(t *testing.T) {
	mad := NewMadgwick(WithMagnetometer(true), WithKP(0.3)).(*MadgwickAHRS)
	mad.accel = vec.Vector3D{-0.15, 0.48, 0.86}
	mad.gyro = vec.Vector3D{0.035, -0.02, 0.015}
	mad.mag = vec.Vector3D{0.51, -0.12, 0.28}
	mad.q = vec.Quaternion{0.9914449, 0.0871558, 0, 0.0871558}
	mad.Update(0.008)

	expected := madgwickReferenceStep(*mad)

	mad.Calculate()

	if !approxQuaternion(mad.q, expected, quatTol) {
		t.Fatalf("madgwick (with mag) quaternion mismatch:\n\tgot:  %+v\n\texp:  %+v", mad.q, expected)
	}
}

func TestMadgwickCalculateWithoutMagnetometerMatchesReference(t *testing.T) {
	mad := NewMadgwick(WithMagnetometer(false), WithKP(0.25)).(*MadgwickAHRS)
	mad.accel = vec.Vector3D{0.32, -0.12, 0.94}
	mad.gyro = vec.Vector3D{-0.02, 0.015, 0.03}
	mad.q = vec.Quaternion{0.9537169, 0.2297529, 0, -0.1893072}
	mad.Update(0.01)

	expected := madgwickReferenceStep(*mad)

	mad.Calculate()

	if !approxQuaternion(mad.q, expected, quatTol) {
		t.Fatalf("madgwick (no mag) quaternion mismatch:\n\tgot:  %+v\n\texp:  %+v", mad.q, expected)
	}
}

func TestMadgwickReset(t *testing.T) {
	mad := NewMadgwick().(*MadgwickAHRS)
	mad.q = vec.Quaternion{0.7, 0.1, -0.2, 0.6}

	mad.Reset()

	if want := (vec.Quaternion{1, 0, 0, 0}); mad.q != want {
		t.Fatalf("madgwick reset quaternion: got %v, want %v", mad.q, want)
	}
}

func TestUpdateSetsSamplePeriodOnly(t *testing.T) {
	cases := []struct {
		name   string
		init   func() AHRS
		sample float32
	}{
		{
			name: "mahony",
			init: func() AHRS {
				m := NewMahony().(*MahonyAHRS)
				m.q = vec.Quaternion{0.73, -0.12, 0.31, 0.59}
				m.eInt = vec.Vector3D{0.01, -0.02, 0.03}
				return m
			},
			sample: 0.0125,
		},
		{
			name: "madgwick",
			init: func() AHRS {
				m := NewMadgwick().(*MadgwickAHRS)
				m.q = vec.Quaternion{0.94, 0.18, -0.09, 0.27}
				return m
			},
			sample: 0.009,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			filter := tc.init()

			switch f := filter.(type) {
			case *MahonyAHRS:
				originalQ := f.q
				originalEInt := f.eInt

				if got := f.Update(tc.sample); got != f {
					t.Fatalf("mahony Update should return receiver, got %T", got)
				}
				if f.SamplePeriod != tc.sample {
					t.Fatalf("mahony sample period: got %f, want %f", f.SamplePeriod, tc.sample)
				}
				if f.q != originalQ {
					t.Fatalf("mahony Update mutated orientation: got %v, want %v", f.q, originalQ)
				}
				if f.eInt != originalEInt {
					t.Fatalf("mahony Update mutated integral error: got %v, want %v", f.eInt, originalEInt)
				}
			case *MadgwickAHRS:
				originalQ := f.q

				if got := f.Update(tc.sample); got != f {
					t.Fatalf("madgwick Update should return receiver, got %T", got)
				}
				if f.SamplePeriod != tc.sample {
					t.Fatalf("madgwick sample period: got %f, want %f", f.SamplePeriod, tc.sample)
				}
				if f.q != originalQ {
					t.Fatalf("madgwick Update mutated orientation: got %v, want %v", f.q, originalQ)
				}
			default:
				t.Fatalf("unexpected AHRS type %T", f)
			}
		})
	}
}

func TestCalculatePreservesInputVectors(t *testing.T) {
	accel := vec.Vector3D{0.32, -0.18, 0.92}
	gyro := vec.Vector3D{-0.02, 0.015, 0.03}
	mag := vec.Vector3D{0.48, 0.12, -0.32}

	cases := []struct {
		name string
		init func() AHRS
	}{
		{
			name: "mahony-with-mag",
			init: func() AHRS { return NewMahony(WithMagnetometer(true)) },
		},
		{
			name: "mahony-without-mag",
			init: func() AHRS { return NewMahony(WithMagnetometer(false)) },
		},
		{
			name: "madgwick-with-mag",
			init: func() AHRS { return NewMadgwick(WithMagnetometer(true)) },
		},
		{
			name: "madgwick-without-mag",
			init: func() AHRS { return NewMadgwick(WithMagnetometer(false)) },
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			filter := tc.init()

			switch f := filter.(type) {
			case *MahonyAHRS:
				f.accel = accel
				f.gyro = gyro
				f.mag = mag
				f.Update(0.01)

				savedAccel := f.accel
				savedGyro := f.gyro
				savedMag := f.mag

				f.Calculate()

				if f.accel != savedAccel {
					t.Fatalf("%s mutated accelerometer: got %v, want %v", tc.name, f.accel, savedAccel)
				}
				if f.gyro != savedGyro {
					t.Fatalf("%s mutated gyroscope: got %v, want %v", tc.name, f.gyro, savedGyro)
				}
				if f.mag != savedMag {
					t.Fatalf("%s mutated magnetometer: got %v, want %v", tc.name, f.mag, savedMag)
				}
			case *MadgwickAHRS:
				f.accel = accel
				f.gyro = gyro
				f.mag = mag
				f.Update(0.008)

				savedAccel := f.accel
				savedGyro := f.gyro
				savedMag := f.mag

				f.Calculate()

				if f.accel != savedAccel {
					t.Fatalf("%s mutated accelerometer: got %v, want %v", tc.name, f.accel, savedAccel)
				}
				if f.gyro != savedGyro {
					t.Fatalf("%s mutated gyroscope: got %v, want %v", tc.name, f.gyro, savedGyro)
				}
				if f.mag != savedMag {
					t.Fatalf("%s mutated magnetometer: got %v, want %v", tc.name, f.mag, savedMag)
				}
			default:
				t.Fatalf("unexpected AHRS type %T", f)
			}
		})
	}
}

func TestCalculatePanicsWithoutSensors(t *testing.T) {
	cases := []struct {
		name string
		init func() AHRS
	}{
		{
			name: "mahony without accelerometer",
			init: func() AHRS { return NewMahony(WithAccelerator(false)) },
		},
		{
			name: "mahony without gyroscope",
			init: func() AHRS { return NewMahony(WithGyroscope(false)) },
		},
		{
			name: "madgwick without accelerometer",
			init: func() AHRS { return NewMadgwick(WithAccelerator(false)) },
		},
		{
			name: "madgwick without gyroscope",
			init: func() AHRS { return NewMadgwick(WithGyroscope(false)) },
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			filter := tc.init()
			mustPanic(t, func() { filter.Calculate() })
		})
	}
}

func mahonyReferenceStep(m MahonyAHRS) (vec.Quaternion, vec.Vector3D) {
	q1, q2, q3, q4 := m.q[0], m.q[1], m.q[2], m.q[3]

	accel := m.accel.NormalFast().(vec.Vector3D)

	var e vec.Vector3D
	if m.HasMagnetometer {
		mag := m.mag.NormalFast().(vec.Vector3D)

		hx := 2*mag[0]*(0.5-q3*q3-q4*q4) + 2*mag[1]*(q2*q3-q1*q4) + 2*mag[2]*(q2*q4+q1*q3)
		hy := 2*mag[0]*(q2*q3+q1*q4) + 2*mag[1]*(0.5-q2*q2-q4*q4) + 2*mag[2]*(q3*q4-q1*q2)
		bx := math32.Sqrt(hx*hx + hy*hy)
		bz := 2*mag[0]*(q2*q4-q1*q3) + 2*mag[1]*(q3*q4+q1*q2) + 2*mag[2]*(0.5-q2*q2-q3*q3)

		v := vec.Vector3D{
			2 * (q2*q4 - q1*q3),
			2 * (q1*q2 + q3*q4),
			q1*q1 - q2*q2 - q3*q3 + q4*q4,
		}
		w := vec.Vector3D{
			2*bx*(0.5-q3*q3-q4*q4) + 2*bz*(q2*q4-q1*q3),
			2*bx*(q2*q3-q1*q4) + 2*bz*(q1*q2+q3*q4),
			2*bx*(q1*q3+q2*q4) + 2*bz*(0.5-q2*q2-q3*q3),
		}

		e = vec.Vector3D{
			(accel[1]*v[2] - accel[2]*v[1]) + (mag[1]*w[2] - mag[2]*w[1]),
			(accel[2]*v[0] - accel[0]*v[2]) + (mag[2]*w[0] - mag[0]*w[2]),
			(accel[0]*v[1] - accel[1]*v[0]) + (mag[0]*w[1] - mag[1]*w[0]),
		}
	} else {
		reference := vec.Vector3D{
			2 * (q2*q4 - q1*q3),
			2 * (q1*q2 + q3*q4),
			q1*q1 - q2*q2 - q3*q3 + q4*q4,
		}
		e = reference.Cross(accel).(vec.Vector3D)
	}

	var eInt vec.Vector3D
	if m.GainI > 0 {
		eInt = m.eInt.Add(e).(vec.Vector3D)
	}

	g := vec.Vector3D{m.gyro[0], m.gyro[1], m.gyro[2]}
	g = g.MulCAdd(m.GainP, e).(vec.Vector3D)
	if m.GainI > 0 {
		g = g.MulCAdd(m.GainI, eInt).(vec.Vector3D)
	}

	qNew := vec.Quaternion{
		q1 + (-q2*g[0]-q3*g[1]-q4*g[2])*(0.5*m.SamplePeriod),
		q2 + (q1*g[0]+q3*g[2]-q4*g[1])*(0.5*m.SamplePeriod),
		q3 + (q1*g[1]-q2*g[2]+q4*g[0])*(0.5*m.SamplePeriod),
		q4 + (q1*g[2]+q2*g[1]-q3*g[0])*(0.5*m.SamplePeriod),
	}
	qNew = qNew.NormalFast().(vec.Quaternion)

	return qNew, eInt
}

func madgwickReferenceStep(m MadgwickAHRS) vec.Quaternion {
	q1, q2, q3, q4 := m.q[0], m.q[1], m.q[2], m.q[3]

	ax, ay, az := m.accel.NormalFast().XYZ()

	var s vec.Quaternion
	if m.HasMagnetometer {
		mx, my, mz := m.mag.NormalFast().XYZ()

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

		s = vec.Quaternion{
			-_2q3*(2*q2q4-_2q1q3-ax) + _2q2*(2*q1q2+_2q3q4-ay) - _2bz*q3*(_2bx*(0.5-q3q3-q4q4)+_2bz*(q2q4-q1q3)-mx) + (-_2bx*q4+_2bz*q2)*(_2bx*(q2*q3-q1*q4)+_2bz*(q1q2+q3q4)-my) + _2bx*q3*(_2bx*(q1q3+q2q4)+_2bz*(0.5-q2*q2-q3*q3)-mz),
			_2q4*(2*q2q4-_2q1q3-ax) + _2q1*(2*q1q2+_2q3q4-ay) - 4*q2*(1-2*q2q2-2*q3q3-az) + _2bz*q4*(_2bx*(0.5-q3q3-q4q4)+_2bz*(q2q4-q1q3)-mx) + (_2bx*q3+_2bz*q1)*(_2bx*(q2q3-q1q4)+_2bz*(q1q2+q3q4)-my) + (_2bx*q4-_4bz*q2)*(_2bx*(q1q3+q2q4)+_2bz*(0.5-q2q2-q3q3)-mz),
			-_2q1*(2*q2q4-_2q1q3-ax) + _2q4*(2*q1q2+_2q3q4-ay) - 4*q3*(1-2*q2q2-2*q3q3-az) + (-_4bx*q3-_2bz*q1)*(_2bx*(0.5-q3q3-q4q4)+_2bz*(q2q4-q1q3)-mx) + (_2bx*q2+_2bz*q4)*(_2bx*(q2q3-q1q4)+_2bz*(q1q2+q3q4)-my) + (_2bx*q1-_4bz*q3)*(_2bx*(q1q3+q2q4)+_2bz*(0.5-q2q2-q3q3)-mz),
			_2q2*(2*q2q4-_2q1q3-ax) + _2q3*(2*q1q2+_2q3q4-ay) + (-_4bx*q4+_2bz*q2)*(_2bx*(0.5-q3q3-q4q4)+_2bz*(q2q4-q1q3)-mx) + (-_2bx*q1+_2bz*q3)*(_2bx*(q2q3-q1q4)+_2bz*(q1q2+q3q4)-my) + _2bx*q2*(_2bx*(q1q3+q2q4)+_2bz*(0.5-q2q2-q3*q3)-mz),
		}
	} else {
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

		s = vec.Quaternion{
			_4q1*q3q3 + _2q3*ax + _4q1*q2q2 - _2q2*ay,
			_4q2*q4q4 - _2q4*ax + 4*q1q1*q2 - _2q1*ay - _4q2 + _8q2*q2q2 + _8q2*q3q3 + _4q2*az,
			4*q1q1*q3 + _2q1*ax + _4q3*q4q4 - _2q4*ay - _4q3 + _8q3*q2q2 + _8q3*q3q3 + _4q3*az,
			4*q2q2*q4 - _2q2*ax + 4*q3q3*q4 - _2q3*ay,
		}
	}

	s = s.NormalFast().(vec.Quaternion)

	qDot := vec.Quaternion{
		0.5 * (-q2*m.gyro[0] - q3*m.gyro[1] - q4*m.gyro[2]),
		0.5 * (q1*m.gyro[0] + q3*m.gyro[2] - q4*m.gyro[1]),
		0.5 * (q1*m.gyro[1] - q2*m.gyro[2] + q4*m.gyro[0]),
		0.5 * (q1*m.gyro[2] + q2*m.gyro[1] - q3*m.gyro[0]),
	}
	qDot = qDot.MulCSub(m.GainP, s).(vec.Quaternion)

	qNew := vec.Quaternion{
		q1 + qDot[0]*m.SamplePeriod,
		q2 + qDot[1]*m.SamplePeriod,
		q3 + qDot[2]*m.SamplePeriod,
		q4 + qDot[3]*m.SamplePeriod,
	}

	return qNew.NormalFast().(vec.Quaternion)
}

func approxQuaternion(a, b vec.Quaternion, tol float32) bool {
	return math32.Abs(a[0]-b[0]) < tol &&
		math32.Abs(a[1]-b[1]) < tol &&
		math32.Abs(a[2]-b[2]) < tol &&
		math32.Abs(a[3]-b[3]) < tol
}

func approxVector3(a, b vec.Vector3D, tol float32) bool {
	return math32.Abs(a[0]-b[0]) < tol &&
		math32.Abs(a[1]-b[1]) < tol &&
		math32.Abs(a[2]-b[2]) < tol
}

func mustPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r == nil {
			t.Fatalf("expected panic but none occurred")
		}
	}()
	fn()
}

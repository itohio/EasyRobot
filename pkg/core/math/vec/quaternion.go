package vec

import (
	"github.com/chewxy/math32"
)

const SLERP_EPSILON = 1.0e-10

type Quaternion [4]float32

func NewQuaternion(x, y, z, w float32) Quaternion {
	return Quaternion{x, y, z, w}
}

func (v *Quaternion) Vector() Vector {
	return v[:]
}

func (v *Quaternion) Axis() Vector {
	return v[:3]
}

func (v *Quaternion) Theta() float32 {
	return v[3]
}

func (v *Quaternion) MagnitudeSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v *Quaternion) Magnitude() float32 {
	return math32.Sqrt(v.MagnitudeSqr())
}

func (v *Quaternion) DistanceSqr(v1 Quaternion) float32 {
	var sum float32

	d := v[0] - v1[0]
	sum += d * d
	d = v[1] - v1[1]
	sum += d * d
	d = v[2] - v1[2]
	sum += d * d
	d = v[3] - v1[3]
	sum += d * d
	return sum
}

func (v *Quaternion) Distance(v1 Quaternion) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v *Quaternion) Clone() *Quaternion {
	return &Quaternion{v[0], v[1], v[2], v[3]}
}

func (v *Quaternion) CopyFrom(start int, v1 Vector) *Quaternion {
	copy(v[start:], v1)
	return v
}

func (v *Quaternion) CopyTo(start int, v1 Vector) *Quaternion {
	copy(v1, v[start:])
	return v
}

func (v *Quaternion) Slice(start, end int) Vector {
	return v[start:end]
}

func (v *Quaternion) Neg() *Quaternion {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	v[3] = -v[3]
	return v
}

func (v *Quaternion) Conjugate() *Quaternion {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v *Quaternion) Roll() float32 {
	return math32.Atan2(v[3]*v[0]+v[1]*v[2], 0.5-v[0]*v[0]-v[1]*v[1])
}
func (v *Quaternion) Pitch() float32 {
	return math32.Asin(-2.0 * (v[0]*v[2] - v[3]*v[1]))
}
func (v *Quaternion) Yaw() float32 {
	return math32.Atan2(v[0]*v[1]+v[3]*v[2], 0.5-v[1]*v[1]-v[2]*v[2])
}

func (v *Quaternion) Add(v1 Quaternion) *Quaternion {
	v[0] += v1[0]
	v[1] += v1[1]
	v[2] += v1[2]
	v[3] += v1[3]
	return v
}

func (v *Quaternion) Sub(v1 Quaternion) *Quaternion {
	v[0] -= v1[0]
	v[1] -= v1[1]
	v[2] -= v1[2]
	v[3] -= v1[3]
	return v
}

func (v *Quaternion) MulC(c float32) *Quaternion {
	v[0] *= c
	v[1] *= c
	v[2] *= c
	return v
}

func (v *Quaternion) MulCAdd(c float32, v1 Quaternion) *Quaternion {
	for i := range v {
		v[i] += v1[i] * c
	}
	return v
}

func (v *Quaternion) MulCSub(c float32, v1 Quaternion) *Quaternion {
	for i := range v {
		v[i] -= v1[i] * c
	}
	return v
}

func (a *Quaternion) Product(b Quaternion) *Quaternion {
	x := a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1]
	y := a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0]
	z := a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3]
	w := a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2]
	a[0] = x
	a[1] = y
	a[2] = z
	a[3] = w
	return a
}

func (v *Quaternion) DivC(c float32) *Quaternion {
	v[0] /= c
	v[1] /= c
	v[2] /= c
	v[3] /= c
	return v
}

func (v *Quaternion) Normal() *Quaternion {
	d := v.Magnitude()
	return v.DivC(d)
}

func (v *Quaternion) Dot(v1 Quaternion) float32 {
	var sum float32
	sum += v[0] * v1[0]
	sum += v[1] * v1[1]
	sum += v[2] * v1[2]
	sum += v[3] * v1[3]
	return sum
}

func (v *Quaternion) Interpolate(v1 Quaternion, t float32) *Quaternion {
	if len(v) != len(v1) {
		panic(-1)
	}
	d := v1.Clone().Sub(*v)
	return v.MulCAdd(t, *d)
}

func (v *Quaternion) Slerp(v1 Quaternion, time, spin float32) *Quaternion {
	var (
		k1, k2       float32 // interpolation coefficions.
		angle        float32 // angle between A and B
		angleSpin    float32 // angle between A and B plus spin.
		sin_a, cos_a float32 // sine, cosine of angle
	)

	flipk2 := 0
	cos_a = v.Dot(v1)
	if cos_a < 0.0 {
		cos_a = -cos_a
		flipk2 = -1
	} else {
		flipk2 = 1
	}

	if (1.0 - cos_a) < SLERP_EPSILON {
		k1 = 1.0 - time
		k2 = time
	} else { /* normal case */
		angle = math32.Acos(cos_a)
		sin_a = math32.Sin(angle)
		angleSpin = angle + spin*math32.Pi
		k1 = math32.Sin(angle-time*angleSpin) / sin_a
		k2 = math32.Sin(time*angleSpin) / sin_a
	}
	k2 *= float32(flipk2)

	v[0] = k1*v[0] + k2*v1[0]
	v[1] = k1*v[1] + k2*v1[1]
	v[2] = k1*v[2] + k2*v1[2]
	v[3] = k1*v[3] + k2*v1[3]
	return v
}

func (v *Quaternion) SlerpLong(v1 Quaternion, time, spin float32) *Quaternion {
	var (
		k1, k2       float32 // interpolation coefficions.
		angle        float32 // angle between A and B
		angleSpin    float32 // angle between A and B plus spin.
		sin_a, cos_a float32 // sine, cosine of angle
	)

	cos_a = v.Dot(v1)

	if 1.0-math32.Abs(cos_a) < SLERP_EPSILON {
		k1 = 1.0 - time
		k2 = time
	} else { /* normal case */
		angle = math32.Acos(cos_a)
		sin_a = math32.Sin(angle)
		angleSpin = angle + spin*math32.Pi
		k1 = math32.Sin(angle-time*angleSpin) / sin_a
		k2 = math32.Sin(time*angleSpin) / sin_a
	}

	v[0] = k1*v[0] + k2*v1[0]
	v[1] = k1*v[1] + k2*v1[1]
	v[2] = k1*v[2] + k2*v1[2]
	v[3] = k1*v[3] + k2*v1[3]
	return v
}

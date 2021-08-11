package vec

import (
	"github.com/chewxy/math32"
)

const SLERP_EPSILON = 1.0e-10

type Quaternion [4]float32

func NewQuaternion(x, y, z, w float32) Quaternion {
	return Quaternion{x, y, z, w}
}

func (v Quaternion) Vector() Vector {
	return v[:]
}

func (v Quaternion) MagnitudeSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v Quaternion) Magnitude() float32 {
	return math32.Sqrt(v.MagnitudeSqr())
}

func (v Quaternion) DistanceSqr(v1 Quaternion) float32 {
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

func (v Quaternion) Distance(v1 Quaternion) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Quaternion) Clone() Quaternion {
	return Quaternion{v[0], v[1], v[2], v[3]}
}

func (v Quaternion) Neg() Quaternion {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	v[3] = -v[3]
	return v
}

func (v Quaternion) Conjugate() Quaternion {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v Quaternion) Roll() float32 {
	return math32.Atan2(v[3]*v[0]+v[1]*v[2], 0.5-v[0]*v[0]-v[1]*v[1])
}
func (v Quaternion) Pitch() float32 {
	return math32.Asin(-2.0 * (v[0]*v[2] - v[3]*v[1]))
}
func (v Quaternion) Yaw() float32 {
	return math32.Atan2(v[0]*v[1]+v[3]*v[2], 0.5-v[1]*v[1]-v[2]*v[2])
}

func (v Quaternion) Add(v1 Quaternion) Quaternion {
	v[0] += v1[0]
	v[1] += v1[1]
	v[2] += v1[2]
	v[3] += v1[3]
	return v
}

func (v Quaternion) Sub(v1 Quaternion) Quaternion {
	v[0] -= v1[0]
	v[1] -= v1[1]
	v[2] -= v1[2]
	v[3] -= v1[3]
	return v
}

func (v Quaternion) Mul(c float32) Quaternion {
	v[0] *= c
	v[1] *= c
	v[2] *= c
	v[3] *= c
	return v
}

func (a Quaternion) Product(b Quaternion) Quaternion {
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

func (v Quaternion) Div(c float32) Quaternion {
	v[0] /= c
	v[1] /= c
	v[2] /= c
	v[3] /= c
	return v
}

func (v Quaternion) Normal() Quaternion {
	d := v.Magnitude()
	return v.Div(d)
}

func (v Quaternion) Dot(v1 Quaternion) float32 {
	var sum float32
	sum += v[0] * v1[0]
	sum += v[1] * v1[1]
	sum += v[2] * v1[2]
	sum += v[3] * v1[3]
	return sum
}

func (v Quaternion) Interpolate(v1 Quaternion, t float32) Quaternion {
	var dst Quaternion
	d := v1[0] - v[0]
	dst[0] = v[0] + d*t
	d = v1[1] - v[1]
	dst[1] = v[1] + d*t
	d = v1[2] - v[2]
	dst[2] = v[2] + d*t
	d = v1[3] - v[3]
	dst[3] = v[3] + d*t
	return dst
}

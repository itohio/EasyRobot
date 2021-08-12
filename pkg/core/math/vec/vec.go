// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math"
)

type Vector []float32

func New(size int) Vector {
	return make(Vector, size)
}

func NewFrom(v ...float32) Vector {
	return v[:]
}

func (v Vector) Sum() float32 {
	var sum float32
	for _, val := range v {
		sum += val
	}
	return sum
}

func (v Vector) Slice(start, end int) Vector {
	if end < 0 {
		end = len(v)
	}
	return v[start:end]
}

func (v Vector) XY() (float32, float32) {
	return v[0], v[1]
}

func (v Vector) XYZ() (float32, float32, float32) {
	return v[0], v[1], v[2]
}

func (v Vector) XYZW() (float32, float32, float32, float32) {
	return v[0], v[1], v[2], v[3]
}

func (v Vector) SumSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v Vector) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v Vector) DistanceSqr(v1 Vector) float32 {
	return v.Clone().Sub(v1).SumSqr()
}

func (v Vector) Distance(v1 Vector) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Vector) Clone() Vector {
	if v == nil {
		return nil
	}

	clone := make(Vector, len(v))
	copy(clone, v)
	return clone
}

func (v Vector) CopyFrom(start int, v1 Vector) Vector {
	copy(v[start:], v1)
	return v
}

func (v Vector) CopyTo(start int, v1 Vector) Vector {
	copy(v1, v[start:])
	return v1
}

func (v Vector) Clamp(min, max Vector) Vector {
	for i := range v {
		v[i] = math.Clamp(v[i], min[i], max[i])
	}
	return v
}

func (v Vector) FillC(c float32) Vector {
	for i := range v {
		v[i] = c
	}
	return v
}

func (v Vector) Neg() Vector {
	for i := range v {
		v[i] = -v[i]
	}
	return v
}

func (v Vector) Add(v1 Vector) Vector {
	for i := range v {
		v[i] += v1[i]
	}
	return v
}

func (v Vector) AddC(c float32) Vector {
	for i := range v {
		v[i] += c
	}
	return v
}

func (v Vector) Sub(v1 Vector) Vector {
	for i := range v {
		v[i] -= v1[i]
	}
	return v
}

func (v Vector) SubC(c float32) Vector {
	for i := range v {
		v[i] -= c
	}
	return v
}

func (v Vector) MulC(c float32) Vector {
	for i := range v {
		v[i] *= c
	}
	return v
}

func (v Vector) MulCAdd(c float32, v1 Vector) Vector {
	for i := range v {
		v[i] += v1[i] * c
	}
	return v
}

func (v Vector) MulCSub(c float32, v1 Vector) Vector {
	for i := range v {
		v[i] -= v1[i] * c
	}
	return v
}

func (v Vector) DivC(c float32) Vector {
	for i := range v {
		v[i] /= c
	}
	return v
}

func (v Vector) DivCAdd(c float32, v1 Vector) Vector {
	for i := range v {
		v[i] += v1[i] / c
	}
	return v
}

func (v Vector) DivCSub(c float32, v1 Vector) Vector {
	for i := range v {
		v[i] -= v1[i] / c
	}
	return v
}

func (v Vector) Normal() Vector {
	d := v.Magnitude()
	return v.DivC(d)
}

func (v Vector) NormalFast() Vector {
	d := v.SumSqr()
	return v.MulC(math.FastISqrt(d))
}

func (v Vector) Axis() Vector {
	return v[:3]
}

func (v Vector) Theta() float32 {
	return v[3]
}

func (v Vector) Conjugate() Vector {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v Vector) Roll() float32 {
	return math32.Atan2(v[3]*v[0]+v[1]*v[2], 0.5-v[0]*v[0]-v[1]*v[1])
}
func (v Vector) Pitch() float32 {
	return math32.Asin(-2.0 * (v[0]*v[2] - v[3]*v[1]))
}
func (v Vector) Yaw() float32 {
	return math32.Atan2(v[0]*v[1]+v[3]*v[2], 0.5-v[1]*v[1]-v[2]*v[2])
}

func (a Vector) Product(b Quaternion) Vector {
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

func (v Vector) Slerp(v1 Vector, time, spin float32) Vector {
	const SLERP_EPSILON = 1.0e-10
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

func (v Vector) SlerpLong(v1 Vector, time, spin float32) Vector {
	const SLERP_EPSILON = 1.0e-10
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

func (v Vector) Multiply(v1 Vector) Vector {
	for i := range v {
		v[i] *= v1[i]
	}
	return v
}

func (v Vector) Dot(v1 Vector) float32 {
	var sum float32
	for i := range v {
		sum += v[i] * v1[i]
	}
	return sum
}

func (v Vector) Cross(v1 Vector) Vector {
	t := []float32{v[0], v[1], v[2]}
	v[0] = t[1]*v1[2] - t[2]*v1[1]
	v[1] = t[2]*v1[0] - t[0]*v1[2]
	v[2] = t[0]*v1[1] - t[1]*v1[0]
	return v
}

func (v Vector) Refract2D(n Vector, ni, nt float32) (Vector, bool) {
	var (
		cos_V  Vector
		sin_T  Vector
		n_mult float32
	)

	N_dot_V := n.Dot(v)

	if N_dot_V > 0.0 {
		n_mult = ni / nt
	} else {
		n_mult = nt / ni
	}

	cos_V[0] = n[0] * N_dot_V
	cos_V[1] = n[1] * N_dot_V
	sin_T[0] = (cos_V[0] - v[0]) * (n_mult)
	sin_T[1] = (cos_V[1] - v[1]) * (n_mult)
	len_sin_T := sin_T.Dot(sin_T)
	if len_sin_T >= 1.0 {
		return v, false // internal reflection
	}
	N_dot_T := math32.Sqrt(1.0 - len_sin_T)
	if N_dot_V < 0.0 {
		N_dot_T = -N_dot_T
	}
	v[0] = sin_T[0] - n[0]*N_dot_T
	v[1] = sin_T[1] - n[1]*N_dot_T

	return v, true
}

func (v Vector) Refract3D(n Vector, ni, nt float32) (Vector, bool) {
	var (
		sin_T  Vector  /* sin vect of the refracted vect */
		cos_V  Vector  /* cos vect of the incident vect */
		n_mult float32 /* ni over nt */
	)

	N_dot_V := n.Dot(v)

	if N_dot_V > 0.0 {
		n_mult = ni / nt
	} else {
		n_mult = nt / ni
	}
	cos_V[0] = n[0] * N_dot_V
	cos_V[1] = n[1] * N_dot_V
	cos_V[2] = n[2] * N_dot_V
	sin_T[0] = (cos_V[0] - v[0]) * (n_mult)
	sin_T[1] = (cos_V[1] - v[1]) * (n_mult)
	sin_T[2] = (cos_V[2] - v[2]) * (n_mult)
	len_sin_T := sin_T.Dot(sin_T)
	if len_sin_T >= 1.0 {
		return v, false // internal reflection
	}
	N_dot_T := math32.Sqrt(1.0 - len_sin_T)
	if N_dot_V < 0.0 {
		N_dot_T = -N_dot_T
	}
	v[0] = sin_T[0] - n[0]*N_dot_T
	v[1] = sin_T[1] - n[1]*N_dot_T
	v[2] = sin_T[2] - n[2]*N_dot_T

	return v, true
}

func (v Vector) Reflect(n Vector) Vector {

	N_dot_V := n.Dot(v) * 2

	return v.Neg().MulCAdd(N_dot_V, n)
}

func (v Vector) Interpolate(v1 Vector, t float32) Vector {

	d := v1.Clone().Sub(v)
	return v.MulCAdd(t, d)

}

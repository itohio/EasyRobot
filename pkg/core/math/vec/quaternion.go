// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math"
)

type Quaternion [4]float32

func (v *Quaternion) Sum() float32 {
	var sum float32
	for _, val := range v {
		sum += val
	}
	return sum
}

func (v *Quaternion) Vector() Vector {
	return v[:]
}

func (v *Quaternion) Slice(start, end int) Vector {
	if end < 0 {
		end = len(v)
	}
	return v[start:end]
}

func (v *Quaternion) XYZ() (float32, float32, float32) {
	return v[0], v[1], v[2]
}

func (v *Quaternion) XYZW() (float32, float32, float32, float32) {
	return v[0], v[1], v[2], v[3]
}

func (v *Quaternion) SumSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v *Quaternion) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v *Quaternion) DistanceSqr(v1 Quaternion) float32 {
	return v.Clone().Sub(v1).SumSqr()
}

func (v *Quaternion) Distance(v1 Quaternion) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v *Quaternion) Clone() *Quaternion {
	clone := Quaternion{}
	copy(clone[:], v[:])
	return &clone
}

func (v *Quaternion) CopyFrom(start int, v1 Vector) *Quaternion {
	copy(v[start:], v1)
	return v
}

func (v *Quaternion) CopyTo(start int, v1 Vector) Vector {
	copy(v1, v[start:])
	return v1
}

func (v *Quaternion) Clamp(min, max Quaternion) *Quaternion {
	for i := range v {
		v[i] = math.Clamp(v[i], min[i], max[i])
	}
	return v
}

func (v *Quaternion) FillC(c float32) *Quaternion {
	for i := range v {
		v[i] = c
	}
	return v
}

func (v *Quaternion) Neg() *Quaternion {
	for i := range v {
		v[i] = -v[i]
	}
	return v
}

func (v *Quaternion) Add(v1 Quaternion) *Quaternion {
	for i := range v {
		v[i] += v1[i]
	}
	return v
}

func (v *Quaternion) AddC(c float32) *Quaternion {
	for i := range v {
		v[i] += c
	}
	return v
}

func (v *Quaternion) Sub(v1 Quaternion) *Quaternion {
	for i := range v {
		v[i] -= v1[i]
	}
	return v
}

func (v *Quaternion) SubC(c float32) *Quaternion {
	for i := range v {
		v[i] -= c
	}
	return v
}

func (v *Quaternion) MulC(c float32) *Quaternion {
	for i := range v {
		v[i] *= c
	}
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

func (v *Quaternion) DivC(c float32) *Quaternion {
	for i := range v {
		v[i] /= c
	}
	return v
}

func (v *Quaternion) DivCAdd(c float32, v1 Quaternion) *Quaternion {
	for i := range v {
		v[i] += v1[i] / c
	}
	return v
}

func (v *Quaternion) DivCSub(c float32, v1 Quaternion) *Quaternion {
	for i := range v {
		v[i] -= v1[i] / c
	}
	return v
}

func (v *Quaternion) Normal() *Quaternion {
	d := v.Magnitude()
	return v.DivC(d)
}

func (v *Quaternion) NormalFast() *Quaternion {
	d := v.SumSqr()
	return v.MulC(math.FastISqrt(d))
}

func (v *Quaternion) Axis() Vector {
	return v[:3]
}

func (v *Quaternion) Theta() float32 {
	return v[3]
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

func (v *Quaternion) Slerp(v1 Quaternion, time, spin float32) *Quaternion {
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

func (v *Quaternion) SlerpLong(v1 Quaternion, time, spin float32) *Quaternion {
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

func (v *Quaternion) Multiply(v1 Quaternion) *Quaternion {
	for i := range v {
		v[i] *= v1[i]
	}
	return v
}

func (v *Quaternion) Dot(v1 Quaternion) float32 {
	var sum float32
	for i := range v {
		sum += v[i] * v1[i]
	}
	return sum
}

func (v *Quaternion) Refract(n Quaternion, ni, nt float32) (*Quaternion, bool) {
	var (
		sin_T  Quaternion /* sin vect of the refracted vect */
		cos_V  Quaternion /* cos vect of the incident vect */
		n_mult float32    /* ni over nt */
	)

	N_dot_V := n.Dot(*v)

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

func (v *Quaternion) Reflect(n Quaternion) *Quaternion {

	N_dot_V := n.Dot(*v) * 2

	return v.Neg().MulCAdd(N_dot_V, n)
}

func (v *Quaternion) Interpolate(v1 Quaternion, t float32) *Quaternion {

	d := v1.Clone().Sub(*v)
	return v.MulCAdd(t, *d)

}

// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math"
)

type Vector4D [4]float32

func (v *Vector4D) Sum() float32 {
	var sum float32
	for _, val := range v {
		sum += val
	}
	return sum
}

func (v *Vector4D) Vector() Vector {
	return v[:]
}

func (v *Vector4D) Slice(start, end int) Vector {
	if end < 0 {
		end = len(v)
	}
	return v[start:end]
}

func (v *Vector4D) XYZ() (float32, float32, float32) {
	return v[0], v[1], v[2]
}

func (v *Vector4D) XYZW() (float32, float32, float32, float32) {
	return v[0], v[1], v[2], v[3]
}

func (v *Vector4D) SumSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v *Vector4D) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v *Vector4D) DistanceSqr(v1 Vector4D) float32 {
	return v.Clone().Sub(v1).SumSqr()
}

func (v *Vector4D) Distance(v1 Vector4D) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v *Vector4D) Clone() *Vector4D {
	clone := Vector4D{}
	copy(clone[:], v[:])
	return &clone
}

func (v *Vector4D) CopyFrom(start int, v1 Vector) *Vector4D {
	copy(v[start:], v1)
	return v
}

func (v *Vector4D) CopyTo(start int, v1 Vector) Vector {
	copy(v1, v[start:])
	return v1
}

func (v *Vector4D) Clamp(min, max Vector4D) *Vector4D {
	for i := range v {
		v[i] = math.Clamp(v[i], min[i], max[i])
	}
	return v
}

func (v *Vector4D) FillC(c float32) *Vector4D {
	for i := range v {
		v[i] = c
	}
	return v
}

func (v *Vector4D) Neg() *Vector4D {
	for i := range v {
		v[i] = -v[i]
	}
	return v
}

func (v *Vector4D) Add(v1 Vector4D) *Vector4D {
	for i := range v {
		v[i] += v1[i]
	}
	return v
}

func (v *Vector4D) AddC(c float32) *Vector4D {
	for i := range v {
		v[i] += c
	}
	return v
}

func (v *Vector4D) Sub(v1 Vector4D) *Vector4D {
	for i := range v {
		v[i] -= v1[i]
	}
	return v
}

func (v *Vector4D) SubC(c float32) *Vector4D {
	for i := range v {
		v[i] -= c
	}
	return v
}

func (v *Vector4D) MulC(c float32) *Vector4D {
	for i := range v {
		v[i] *= c
	}
	return v
}

func (v *Vector4D) MulCAdd(c float32, v1 Vector4D) *Vector4D {
	for i := range v {
		v[i] += v1[i] * c
	}
	return v
}

func (v *Vector4D) MulCSub(c float32, v1 Vector4D) *Vector4D {
	for i := range v {
		v[i] -= v1[i] * c
	}
	return v
}

func (v *Vector4D) DivC(c float32) *Vector4D {
	for i := range v {
		v[i] /= c
	}
	return v
}

func (v *Vector4D) DivCAdd(c float32, v1 Vector4D) *Vector4D {
	for i := range v {
		v[i] += v1[i] / c
	}
	return v
}

func (v *Vector4D) DivCSub(c float32, v1 Vector4D) *Vector4D {
	for i := range v {
		v[i] -= v1[i] / c
	}
	return v
}

func (v *Vector4D) Normal() *Vector4D {
	d := v.Magnitude()
	return v.DivC(d)
}

func (v *Vector4D) NormalFast() *Vector4D {
	d := v.SumSqr()
	return v.MulC(math.FastISqrt(d))
}

func (v *Vector4D) Axis() Vector {
	return v[:3]
}

func (v *Vector4D) Theta() float32 {
	return v[3]
}

func (v *Vector4D) Conjugate() *Vector4D {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v *Vector4D) Roll() float32 {
	return math32.Atan2(v[3]*v[0]+v[1]*v[2], 0.5-v[0]*v[0]-v[1]*v[1])
}
func (v *Vector4D) Pitch() float32 {
	return math32.Asin(-2.0 * (v[0]*v[2] - v[3]*v[1]))
}
func (v *Vector4D) Yaw() float32 {
	return math32.Atan2(v[0]*v[1]+v[3]*v[2], 0.5-v[1]*v[1]-v[2]*v[2])
}

func (a *Vector4D) Product(b Quaternion) *Vector4D {
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

func (v *Vector4D) Slerp(v1 Vector4D, time, spin float32) *Vector4D {
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

func (v *Vector4D) SlerpLong(v1 Vector4D, time, spin float32) *Vector4D {
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

func (v *Vector4D) Multiply(v1 Vector4D) *Vector4D {
	for i := range v {
		v[i] *= v1[i]
	}
	return v
}

func (v *Vector4D) Dot(v1 Vector4D) float32 {
	var sum float32
	for i := range v {
		sum += v[i] * v1[i]
	}
	return sum
}

func (v *Vector4D) Reflect(n Vector4D) *Vector4D {

	N_dot_V := n.Dot(*v) * 2

	return v.Neg().MulCAdd(N_dot_V, n)
}

func (v *Vector4D) Interpolate(v1 Vector4D, t float32) *Vector4D {

	d := v1.Clone().Sub(*v)
	return v.MulCAdd(t, *d)

}

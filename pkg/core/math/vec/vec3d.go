// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math"
)

type Vector3D [3]float32

func (v *Vector3D) Sum() float32 {
	var sum float32
	for _, val := range v {
		sum += val
	}
	return sum
}

func (v *Vector3D) Vector() Vector {
	return v[:]
}

func (v *Vector3D) Slice(start, end int) Vector {
	if end < 0 {
		end = len(v)
	}
	return v[start:end]
}

func (v *Vector3D) XYZ() (float32, float32, float32) {
	return v[0], v[1], v[2]
}

func (v *Vector3D) SumSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v *Vector3D) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v *Vector3D) DistanceSqr(v1 Vector3D) float32 {
	return v.Clone().Sub(v1).SumSqr()
}

func (v *Vector3D) Distance(v1 Vector3D) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v *Vector3D) Clone() *Vector3D {
	clone := Vector3D{}
	copy(clone[:], v[:])
	return &clone
}

func (v *Vector3D) CopyFrom(start int, v1 Vector) *Vector3D {
	copy(v[start:], v1)
	return v
}

func (v *Vector3D) CopyTo(start int, v1 Vector) Vector {
	copy(v1, v[start:])
	return v1
}

func (v *Vector3D) Clamp(min, max Vector3D) *Vector3D {
	for i := range v {
		v[i] = math.Clamp(v[i], min[i], max[i])
	}
	return v
}

func (v *Vector3D) FillC(c float32) *Vector3D {
	for i := range v {
		v[i] = c
	}
	return v
}

func (v *Vector3D) Neg() *Vector3D {
	for i := range v {
		v[i] = -v[i]
	}
	return v
}

func (v *Vector3D) Add(v1 Vector3D) *Vector3D {
	for i := range v {
		v[i] += v1[i]
	}
	return v
}

func (v *Vector3D) AddC(c float32) *Vector3D {
	for i := range v {
		v[i] += c
	}
	return v
}

func (v *Vector3D) Sub(v1 Vector3D) *Vector3D {
	for i := range v {
		v[i] -= v1[i]
	}
	return v
}

func (v *Vector3D) SubC(c float32) *Vector3D {
	for i := range v {
		v[i] -= c
	}
	return v
}

func (v *Vector3D) MulC(c float32) *Vector3D {
	for i := range v {
		v[i] *= c
	}
	return v
}

func (v *Vector3D) MulCAdd(c float32, v1 Vector3D) *Vector3D {
	for i := range v {
		v[i] += v1[i] * c
	}
	return v
}

func (v *Vector3D) MulCSub(c float32, v1 Vector3D) *Vector3D {
	for i := range v {
		v[i] -= v1[i] * c
	}
	return v
}

func (v *Vector3D) DivC(c float32) *Vector3D {
	for i := range v {
		v[i] /= c
	}
	return v
}

func (v *Vector3D) DivCAdd(c float32, v1 Vector3D) *Vector3D {
	for i := range v {
		v[i] += v1[i] / c
	}
	return v
}

func (v *Vector3D) DivCSub(c float32, v1 Vector3D) *Vector3D {
	for i := range v {
		v[i] -= v1[i] / c
	}
	return v
}

func (v *Vector3D) Normal() *Vector3D {
	d := v.Magnitude()
	return v.DivC(d)
}

func (v *Vector3D) NormalFast() *Vector3D {
	d := v.SumSqr()
	return v.MulC(math.FastISqrt(d))
}

func (v *Vector3D) Multiply(v1 Vector3D) *Vector3D {
	for i := range v {
		v[i] *= v1[i]
	}
	return v
}

func (v *Vector3D) Dot(v1 Vector3D) float32 {
	var sum float32
	for i := range v {
		sum += v[i] * v1[i]
	}
	return sum
}

func (v *Vector3D) Cross(v1 Vector3D) *Vector3D {
	t := []float32{v[0], v[1], v[2]}
	v[0] = t[1]*v1[2] - t[2]*v1[1]
	v[1] = t[2]*v1[0] - t[0]*v1[2]
	v[2] = t[0]*v1[1] - t[1]*v1[0]
	return v
}

func (v *Vector3D) Refract(n Vector3D, ni, nt float32) (*Vector3D, bool) {
	var (
		sin_T  Vector3D /* sin vect of the refracted vect */
		cos_V  Vector3D /* cos vect of the incident vect */
		n_mult float32  /* ni over nt */
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

func (v *Vector3D) Reflect(n Vector3D) *Vector3D {

	N_dot_V := n.Dot(*v) * 2

	return v.Neg().MulCAdd(N_dot_V, n)
}

func (v *Vector3D) Interpolate(v1 Vector3D, t float32) *Vector3D {

	d := v1.Clone().Sub(*v)
	return v.MulCAdd(t, *d)

}

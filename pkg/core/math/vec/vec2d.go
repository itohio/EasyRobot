// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/foxis/EasyRobot/pkg/core/math"
)

type Vector2D [2]float32

func (v *Vector2D) Sum() float32 {
	var sum float32
	for _, val := range v {
		sum += val
	}
	return sum
}

func (v *Vector2D) Vector() Vector {
	return v[:]
}

func (v *Vector2D) Slice(start, end int) Vector {
	if end < 0 {
		end = len(v)
	}
	return v[start:end]
}

func (v *Vector2D) XY() (float32, float32) {
	return v[0], v[1]
}

func (v *Vector2D) SumSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v *Vector2D) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v *Vector2D) DistanceSqr(v1 Vector2D) float32 {
	return v.Clone().Sub(v1).SumSqr()
}

func (v *Vector2D) Distance(v1 Vector2D) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v *Vector2D) Clone() *Vector2D {
	clone := Vector2D{}
	copy(clone[:], v[:])
	return &clone
}

func (v *Vector2D) CopyFrom(start int, v1 Vector) *Vector2D {
	copy(v[start:], v1)
	return v
}

func (v *Vector2D) CopyTo(start int, v1 Vector) Vector {
	copy(v1, v[start:])
	return v1
}

func (v *Vector2D) Clamp(min, max Vector2D) *Vector2D {
	for i := range v {
		v[i] = math.Clamp(v[i], min[i], max[i])
	}
	return v
}

func (v *Vector2D) FillC(c float32) *Vector2D {
	for i := range v {
		v[i] = c
	}
	return v
}

func (v *Vector2D) Neg() *Vector2D {
	for i := range v {
		v[i] = -v[i]
	}
	return v
}

func (v *Vector2D) Add(v1 Vector2D) *Vector2D {
	for i := range v {
		v[i] += v1[i]
	}
	return v
}

func (v *Vector2D) AddC(c float32) *Vector2D {
	for i := range v {
		v[i] += c
	}
	return v
}

func (v *Vector2D) Sub(v1 Vector2D) *Vector2D {
	for i := range v {
		v[i] -= v1[i]
	}
	return v
}

func (v *Vector2D) SubC(c float32) *Vector2D {
	for i := range v {
		v[i] -= c
	}
	return v
}

func (v *Vector2D) MulC(c float32) *Vector2D {
	for i := range v {
		v[i] *= c
	}
	return v
}

func (v *Vector2D) MulCAdd(c float32, v1 Vector2D) *Vector2D {
	for i := range v {
		v[i] += v1[i] * c
	}
	return v
}

func (v *Vector2D) MulCSub(c float32, v1 Vector2D) *Vector2D {
	for i := range v {
		v[i] -= v1[i] * c
	}
	return v
}

func (v *Vector2D) DivC(c float32) *Vector2D {
	for i := range v {
		v[i] /= c
	}
	return v
}

func (v *Vector2D) DivCAdd(c float32, v1 Vector2D) *Vector2D {
	for i := range v {
		v[i] += v1[i] / c
	}
	return v
}

func (v *Vector2D) DivCSub(c float32, v1 Vector2D) *Vector2D {
	for i := range v {
		v[i] -= v1[i] / c
	}
	return v
}

func (v *Vector2D) Normal() *Vector2D {
	d := v.Magnitude()
	return v.DivC(d)
}

func (v *Vector2D) NormalFast() *Vector2D {
	d := v.SumSqr()
	return v.MulC(math.FastISqrt(d))
}

func (v *Vector2D) Multiply(v1 Vector2D) *Vector2D {
	for i := range v {
		v[i] *= v1[i]
	}
	return v
}

func (v *Vector2D) Dot(v1 Vector2D) float32 {
	var sum float32
	for i := range v {
		sum += v[i] * v1[i]
	}
	return sum
}

func (v *Vector2D) Refract(n Vector2D, ni, nt float32) (*Vector2D, bool) {
	var (
		cos_V  Vector2D
		sin_T  Vector2D
		n_mult float32
	)

	N_dot_V := n.Dot(*v)

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

func (v *Vector2D) Reflect(n Vector2D) *Vector2D {

	N_dot_V := n.Dot(*v) * 2

	return v.Neg().MulCAdd(N_dot_V, n)
}

func (v *Vector2D) Interpolate(v1 Vector2D, t float32) *Vector2D {

	d := v1.Clone().Sub(*v)
	return v.MulCAdd(t, *d)

}

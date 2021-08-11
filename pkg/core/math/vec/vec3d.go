package vec

import (
	"github.com/chewxy/math32"
)

type Vector3D [3]float32

func New3D(x, y, z float32) Vector3D {
	return Vector3D{x, y, z}
}

func (v Vector3D) Vector() Vector {
	return v[:]
}

func (v Vector3D) MagnitudeSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v Vector3D) Magnitude() float32 {
	return math32.Sqrt(v.MagnitudeSqr())
}

func (v Vector3D) DistanceSqr(v1 Vector3D) float32 {
	var sum float32

	d := v[0] - v1[0]
	sum += d * d
	d = v[1] - v1[1]
	sum += d * d
	d = v[2] - v1[2]
	sum += d * d
	return sum
}

func (v Vector3D) Distance(v1 Vector3D) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Vector3D) Clone() Vector3D {
	return Vector3D{v[0], v[1], v[2]}
}

func (v Vector3D) Neg() Vector3D {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v Vector3D) Add(v1 Vector3D) Vector3D {
	v[0] += v1[0]
	v[1] += v1[1]
	v[2] += v1[2]
	return v
}

func (v Vector3D) Sub(v1 Vector3D) Vector3D {
	v[0] -= v1[0]
	v[1] -= v1[1]
	v[2] -= v1[2]
	return v
}

func (v Vector3D) Mul(c float32) Vector3D {
	v[0] *= c
	v[1] *= c
	v[2] *= c
	return v
}

func (v Vector3D) Div(c float32) Vector3D {
	v[0] /= c
	v[1] /= c
	v[2] /= c
	return v
}

func (v Vector3D) Normal() Vector3D {
	d := v.Magnitude()
	return v.Div(d)
}

func (v Vector3D) Dot(v1 Vector3D) float32 {
	var sum float32
	sum += v[0] * v1[0]
	sum += v[1] * v1[1]
	sum += v[2] * v1[2]
	return sum
}

func (v Vector3D) Cross(v1 Vector3D) Vector3D {
	var dst Vector3D
	dst[0] = v[1]*v1[2] - v[2]*v1[1]
	dst[1] = v[2]*v1[0] - v[0]*v1[2]
	dst[2] = v[0]*v1[1] - v[1]*v1[0]
	return dst
}

func (v Vector3D) Reflect(n Vector3D) Vector3D {
	N_dot_V := n.Dot(v) * 2
	v[0] = N_dot_V*n[0] - v[0]
	v[1] = N_dot_V*n[1] - v[1]
	v[2] = N_dot_V*n[2] - v[2]

	return v
}

func (v Vector3D) Refract(n Vector3D, ni, nt float32) (Vector3D, bool) {
	var (
		sin_T  Vector3D /* sin vect of the refracted vect */
		cos_V  Vector3D /* cos vect of the incident vect */
		n_mult float32  /* ni over nt */
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

func (v Vector3D) Interpolate(v1 Vector3D, t float32) Vector3D {
	var dst Vector3D
	d := v1[0] - v[0]
	dst[0] = v[0] + d*t
	d = v1[1] - v[1]
	dst[1] = v[1] + d*t
	d = v1[2] - v[2]
	dst[2] = v[2] + d*t
	return dst
}

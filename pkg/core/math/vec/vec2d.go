package vec

import (
	"github.com/chewxy/math32"
)

type Vector2D [2]float32

func New2D(x, y float32) Vector2D {
	return Vector2D{x, y}
}

func (v Vector2D) Vector() Vector {
	return v[:]
}

func (v Vector2D) MagnitudeSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v Vector2D) Magnitude() float32 {
	return math32.Sqrt(v.MagnitudeSqr())
}

func (v Vector2D) DistanceSqr(v1 Vector2D) float32 {
	var sum float32

	d := v[0] - v1[0]
	sum += d * d
	d = v[1] - v1[1]
	sum += d * d
	return sum
}

func (v Vector2D) Distance(v1 Vector2D) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Vector2D) Clone() Vector2D {
	return Vector2D{v[0], v[1]}
}

func (v Vector2D) Neg() Vector2D {
	v[0] = -v[0]
	v[1] = -v[1]
	return v
}

func (v Vector2D) Add(v1 Vector2D) Vector2D {
	v[0] += v1[0]
	v[1] += v1[1]
	return v
}

func (v Vector2D) Sub(v1 Vector2D) Vector2D {
	v[0] -= v1[0]
	v[1] -= v1[1]
	return v
}

func (v Vector2D) Mul(c float32) Vector2D {
	v[0] *= c
	v[1] *= c
	return v
}

func (v Vector2D) Div(c float32) Vector2D {
	v[0] /= c
	v[1] /= c
	return v
}

func (v Vector2D) Normal() Vector2D {
	d := v.Magnitude()
	return v.Div(d)
}

func (v Vector2D) Dot(v1 Vector2D) float32 {
	var sum float32
	sum += v[0] * v1[0]
	sum += v[1] * v1[1]
	return sum
}

func (v Vector2D) Reflect(n Vector2D) Vector2D {
	N_dot_V := n.Dot(v) * 2
	v[0] = N_dot_V*n[0] - v[0]
	v[1] = N_dot_V*n[1] - v[1]

	return v
}

func (v Vector2D) Refract(n Vector2D, ni, nt float32) (Vector2D, bool) {
	var (
		cos_V  Vector2D
		sin_T  Vector2D
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

func (v Vector2D) Interpolate(v1 Vector2D, t float32) Vector2D {
	var dst Vector2D
	d := v1[0] - v[0]
	dst[0] = v[0] + d*t
	d = v1[1] - v[1]
	dst[1] = v[1] + d*t
	return dst
}

package vec

import (
	"github.com/chewxy/math32"
)

type Vector4D [4]float32

func New4D(x, y, z, w float32) Vector4D {
	return Vector4D{x, y, z, w}
}

func (v Vector4D) Vector() Vector {
	return v[:]
}

func (v Vector4D) MagnitudeSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v Vector4D) Magnitude() float32 {
	return math32.Sqrt(v.MagnitudeSqr())
}

func (v Vector4D) DistanceSqr(v1 Vector4D) float32 {
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

func (v Vector4D) Distance(v1 Vector4D) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Vector4D) Clone() Vector4D {
	return Vector4D{v[0], v[1], v[2]}
}

func (v Vector4D) Neg() Vector4D {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	v[3] = -v[3]
	return v
}

func (v Vector4D) Add(v1 Vector4D) Vector4D {
	v[0] += v1[0]
	v[1] += v1[1]
	v[2] += v1[2]
	v[3] += v1[3]
	return v
}

func (v Vector4D) Sub(v1 Vector4D) Vector4D {
	v[0] -= v1[0]
	v[1] -= v1[1]
	v[2] -= v1[2]
	v[3] -= v1[3]
	return v
}

func (v Vector4D) Mul(c float32) Vector4D {
	v[0] *= c
	v[1] *= c
	v[2] *= c
	v[3] *= c
	return v
}

func (v Vector4D) Div(c float32) Vector4D {
	v[0] /= c
	v[1] /= c
	v[2] /= c
	v[3] /= c
	return v
}

func (v Vector4D) Normal() Vector4D {
	d := v.Magnitude()
	return v.Div(d)
}

func (v Vector4D) Dot(v1 Vector4D) float32 {
	var sum float32
	sum += v[0] * v1[0]
	sum += v[1] * v1[1]
	sum += v[2] * v1[2]
	sum += v[3] * v1[3]
	return sum
}

func (v Vector4D) Interpolate(v1 Vector4D, t float32) Vector4D {
	var dst Vector4D
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

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

func (v Vector) MinLen(v1 Vector) int {
	N := len(v)
	N1 := len(v1)
	if N > N1 {
		return N1
	}
	return N
}

func (v Vector) MinLenMany(v1 ...Vector) int {
	N := len(v)
	for _, v := range v1 {
		n := len(v)
		if n < N {
			N = n
		}
	}
	return N
}

func (v Vector) Sum(stride int) float32 {
	var sum float32
	for _, val := range v {
		sum += val
	}
	return sum
}

func (v Vector) SumSqr(stride int) float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v Vector) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr(1))
}

func (v Vector) DistanceSqr(v1 Vector) float32 {
	return v.Clone().Sub(v1).SumSqr(1)
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

func (v Vector) Slice(start, end int) Vector {
	return v[start:end]
}

func (v Vector) Neg() Vector {
	for i := range v {
		v[i] = -v[i]
	}
	return v
}

func (v Vector) Add(v1 Vector) Vector {
	N := v.MinLen(v1)
	for i := 0; i < N; i++ {
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
	N := v.MinLen(v1)
	for i := 0; i < N; i++ {
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
	N := v.MinLen(v1)
	for i := 0; i < N; i++ {
		v[i] += v1[i] * c
	}
	return v
}

func (v Vector) MulCSub(c float32, v1 Vector) Vector {
	N := v.MinLen(v1)
	for i := 0; i < N; i++ {
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
	N := v.MinLen(v1)
	for i := 0; i < N; i++ {
		v[i] += v1[i] / c
	}
	return v
}

func (v Vector) DivCSub(c float32, v1 Vector) Vector {
	N := v.MinLen(v1)
	for i := 0; i < N; i++ {
		v[i] /= v1[i] / c
	}
	return v
}

func (v Vector) Normal() Vector {
	d := v.Magnitude()
	return v.DivC(d)
}

func (v Vector) NormalFast() Vector {
	d := v.SumSqr(1)
	return v.MulC(math.FastISqrt(d))
}

func (v Vector) Dot(v1 Vector) float32 {
	if len(v) != len(v1) {
		panic(-1)
	}
	var sum float32
	for i := range v {
		sum += v[i] * v1[i]
	}
	return sum
}

func (v Vector) Reflect(n Vector) Vector {
	if len(v) != len(n) {
		panic(-1)
	}
	N_dot_V := n.Dot(v) * 2

	return v.Neg().MulCAdd(N_dot_V, n)
}

func (v Vector) Interpolate(v1 Vector, t float32) Vector {
	if len(v) != len(v1) {
		panic(-1)
	}
	d := v1.Clone().Sub(v)
	return v.MulCAdd(t, d)
}

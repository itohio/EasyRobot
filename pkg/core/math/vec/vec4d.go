package vec

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math"
	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

var _ vecTypes.Vector = Vector4D{}

const vector4DSize = 4

type Vector4D [4]float32

func (v Vector4D) Sum() float32 {
	return v[0] + v[1] + v[2] + v[3]
}

func (v Vector4D) View() vecTypes.Vector {
	return Vector(v[:])
}

func (v Vector4D) Slice(start, end int) vecTypes.Vector {
	if end < 0 {
		end = len(v)
	}
	return Vector(v[start:end])
}

func (v Vector4D) XY() (float32, float32) {
	return v[0], v[1]
}

func (v Vector4D) XYZ() (float32, float32, float32) {
	return v[0], v[1], v[2]
}

func (v Vector4D) XYZW() (float32, float32, float32, float32) {
	return v[0], v[1], v[2], v[3]
}

func (v Vector4D) SumSqr() float32 {
	return v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]
}

func (v Vector4D) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v Vector4D) DistanceSqr(v1 vecTypes.Vector) float32 {
	other := v1.View().(Vector4D)
	d0 := v[0] - other[0]
	d1 := v[1] - other[1]
	d2 := v[2] - other[2]
	d3 := v[3] - other[3]
	return d0*d0 + d1*d1 + d2*d2 + d3*d3
}

func (v Vector4D) Distance(v1 vecTypes.Vector) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Vector4D) Clone() vecTypes.Vector {
	return v
}

func (v Vector4D) CopyFrom(start int, v1 vecTypes.Vector) vecTypes.Vector {
	src := v1.View().(Vector)
	copy(v[:], src[start:])
	return v
}

func (v Vector4D) CopyTo(start int, v1 vecTypes.Vector) vecTypes.Vector {
	dst := v1.View().(Vector)
	copy(dst, v[start:])
	return v1
}

func (v Vector4D) Clamp(min, max vecTypes.Vector) vecTypes.Vector {
	minVec := min.(Vector4D)
	maxVec := max.(Vector4D)
	v[0] = math.Clamp(v[0], minVec[0], maxVec[0])
	v[1] = math.Clamp(v[1], minVec[1], maxVec[1])
	v[2] = math.Clamp(v[2], minVec[2], maxVec[2])
	v[3] = math.Clamp(v[3], minVec[3], maxVec[3])
	return v
}

func (v Vector4D) FillC(c float32) vecTypes.Vector {
	v[0] = c
	v[1] = c
	v[2] = c
	v[3] = c
	return v
}

func (v Vector4D) Neg() vecTypes.Vector {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	v[3] = -v[3]
	return v
}

func (v Vector4D) Add(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector4D)
	v[0] += other[0]
	v[1] += other[1]
	v[2] += other[2]
	v[3] += other[3]
	return v
}

func (v Vector4D) AddC(c float32) vecTypes.Vector {
	v[0] += c
	v[1] += c
	v[2] += c
	v[3] += c
	return v
}

func (v Vector4D) Sub(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector4D)
	v[0] -= other[0]
	v[1] -= other[1]
	v[2] -= other[2]
	v[3] -= other[3]
	return v
}

func (v Vector4D) SubC(c float32) vecTypes.Vector {
	v[0] -= c
	v[1] -= c
	v[2] -= c
	v[3] -= c
	return v
}

func (v Vector4D) MulC(c float32) vecTypes.Vector {
	v[0] *= c
	v[1] *= c
	v[2] *= c
	v[3] *= c
	return v
}

func (v Vector4D) MulCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector4D)
	v[0] += other[0] * c
	v[1] += other[1] * c
	v[2] += other[2] * c
	v[3] += other[3] * c
	return v
}

func (v Vector4D) MulCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector4D)
	v[0] -= other[0] * c
	v[1] -= other[1] * c
	v[2] -= other[2] * c
	v[3] -= other[3] * c
	return v
}

func (v Vector4D) DivC(c float32) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector4D.DivC: divide by zero")
	}
	v[0] /= c
	v[1] /= c
	v[2] /= c
	v[3] /= c
	return v
}

func (v Vector4D) DivCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector4D.DivCAdd: divide by zero")
	}
	other := v1.(Vector4D)
	v[0] += other[0] / c
	v[1] += other[1] / c
	v[2] += other[2] / c
	v[3] += other[3] / c
	return v
}

func (v Vector4D) DivCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector4D.DivCSub: divide by zero")
	}
	other := v1.(Vector4D)
	v[0] -= other[0] / c
	v[1] -= other[1] / c
	v[2] -= other[2] / c
	v[3] -= other[3] / c
	return v
}

func (v Vector4D) Normal() vecTypes.Vector {
	m := v.Magnitude()
	if m == 0 {
		panic("vec.Vector4D.Normal: zero magnitude")
	}
	return v.DivC(m)
}

func (v Vector4D) NormalFast() vecTypes.Vector {
	s := v.SumSqr()
	if s == 0 {
		panic("vec.Vector4D.NormalFast: zero magnitude")
	}
	return v.MulC(math.FastISqrt(s))
}

func (v Vector4D) Axis() vecTypes.Vector {
	return Vector(v[:3])
}

func (v Vector4D) Theta() float32 {
	return v[3]
}

func (v Vector4D) Conjugate() vecTypes.Vector {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v Vector4D) Roll() float32 {
	return math32.Atan2(v[3]*v[0]+v[1]*v[2], 0.5-v[0]*v[0]-v[1]*v[1])
}
func (v Vector4D) Pitch() float32 {
	return math32.Asin(-2.0 * (v[0]*v[2] - v[3]*v[1]))
}
func (v Vector4D) Yaw() float32 {
	return math32.Atan2(v[0]*v[1]+v[3]*v[2], 0.5-v[1]*v[1]-v[2]*v[2])
}

func (a Vector4D) Product(b vecTypes.Quaternion) vecTypes.Vector {
	other := b.(Vector4D)
	x := a[3]*other[0] + a[0]*other[3] + a[1]*other[2] - a[2]*other[1]
	y := a[3]*other[1] - a[0]*other[2] + a[1]*other[3] + a[2]*other[0]
	z := a[3]*other[2] + a[0]*other[1] - a[1]*other[0] + a[2]*other[3]
	w := a[3]*other[3] - a[0]*other[0] - a[1]*other[1] - a[2]*other[2]
	a[0] = x
	a[1] = y
	a[2] = z
	a[3] = w
	return a
}

func (v Vector4D) Slerp(v1 vecTypes.Vector, time, spin float32) vecTypes.Vector {
	other := v1.(Vector4D)
	const slerpEpsilon = 1.0e-10
	var (
		k1, k2     float32
		angle      float32
		angleSpin  float32
		sinA, cosA float32
	)

	flipK2 := float32(1)
	cosA = v.Dot(v1)
	if cosA < 0 {
		cosA = -cosA
		flipK2 = -1
	}

	if (1 - cosA) < slerpEpsilon {
		k1 = 1 - time
		k2 = time
	} else {
		angle = math32.Acos(cosA)
		sinA = math32.Sin(angle)
		angleSpin = angle + spin*math32.Pi
		k1 = math32.Sin(angle-time*angleSpin) / sinA
		k2 = math32.Sin(time*angleSpin) / sinA
	}
	k2 *= flipK2

	v[0] = k1*v[0] + k2*other[0]
	v[1] = k1*v[1] + k2*other[1]
	v[2] = k1*v[2] + k2*other[2]
	v[3] = k1*v[3] + k2*other[3]
	return v
}

func (v Vector4D) SlerpLong(v1 vecTypes.Vector, time, spin float32) vecTypes.Vector {
	other := v1.(Vector4D)
	const slerpEpsilon = 1.0e-10
	var (
		k1, k2     float32
		angle      float32
		angleSpin  float32
		sinA, cosA float32
	)

	cosA = v.Dot(v1)

	if 1-math32.Abs(cosA) < slerpEpsilon {
		k1 = 1 - time
		k2 = time
	} else {
		angle = math32.Acos(cosA)
		sinA = math32.Sin(angle)
		angleSpin = angle + spin*math32.Pi
		k1 = math32.Sin(angle-time*angleSpin) / sinA
		k2 = math32.Sin(time*angleSpin) / sinA
	}

	v[0] = k1*v[0] + k2*other[0]
	v[1] = k1*v[1] + k2*other[1]
	v[2] = k1*v[2] + k2*other[2]
	v[3] = k1*v[3] + k2*other[3]
	return v
}

func (v Vector4D) Multiply(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector4D)
	v[0] *= other[0]
	v[1] *= other[1]
	v[2] *= other[2]
	v[3] *= other[3]
	return v
}

func (v Vector4D) Dot(v1 vecTypes.Vector) float32 {
	other := v1.(Vector4D)
	return v[0]*other[0] + v[1]*other[1] + v[2]*other[2] + v[3]*other[3]
}

func (v Vector4D) Cross(vecTypes.Vector) vecTypes.Vector {
	panic("vec.Vector4D.Cross: unsupported operation")
}

func (v Vector4D) Refract2D(vecTypes.Vector, float32, float32) (vecTypes.Vector, bool) {
	panic("vec.Vector4D.Refract2D: unsupported operation")
}

func (v Vector4D) Refract3D(vecTypes.Vector, float32, float32) (vecTypes.Vector, bool) {
	panic("vec.Vector4D.Refract3D: unsupported operation")
}

func (v Vector4D) Reflect(n vecTypes.Vector) vecTypes.Vector {
	nVec := n.(Vector4D)
	d := v.Dot(n) * 2
	v[0] = -v[0] + d*nVec[0]
	v[1] = -v[1] + d*nVec[1]
	v[2] = -v[2] + d*nVec[2]
	v[3] = -v[3] + d*nVec[3]
	return v
}

func (v Vector4D) Interpolate(v1 vecTypes.Vector, t float32) vecTypes.Vector {
	other := v1.(Vector4D)
	v[0] = v[0] + t*(other[0]-v[0])
	v[1] = v[1] + t*(other[1]-v[1])
	v[2] = v[2] + t*(other[2]-v[2])
	v[3] = v[3] + t*(other[3]-v[3])
	return v
}

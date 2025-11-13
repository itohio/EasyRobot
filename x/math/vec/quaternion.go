// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

var _ vecTypes.Vector = Quaternion{}

type Quaternion [4]float32

func (v Quaternion) View() vecTypes.Vector {
	return Vector(v[:])
}

func (v Quaternion) Slice(start, end int) vecTypes.Vector {
	if end < 0 {
		end = len(v)
	}
	return Vector(v[start:end])
}

func (v Quaternion) XY() (float32, float32) {
	return v[0], v[1]
}

func (v Quaternion) XYZ() (float32, float32, float32) {
	return v[0], v[1], v[2]
}

func (v Quaternion) XYZW() (float32, float32, float32, float32) {
	return v[0], v[1], v[2], v[3]
}

func (v Quaternion) SumSqr() float32 {
	return v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3]
}

func (v Quaternion) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v Quaternion) DistanceSqr(v1 vecTypes.Vector) float32 {
	other := v1.(Quaternion)
	d0 := v[0] - other[0]
	d1 := v[1] - other[1]
	d2 := v[2] - other[2]
	d3 := v[3] - other[3]
	return d0*d0 + d1*d1 + d2*d2 + d3*d3
}

func (v Quaternion) Distance(v1 vecTypes.Vector) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Quaternion) Clone() vecTypes.Vector {
	return v
}

func (v Quaternion) CopyFrom(start int, v1 vecTypes.Vector) vecTypes.Vector {
	src := v1.View().(Vector)
	copy(v[:], src[start:])
	return v
}

func (v Quaternion) CopyTo(start int, v1 vecTypes.Vector) vecTypes.Vector {
	dst := v1.View().(Vector)
	copy(dst, v[start:])
	return v1
}

func (v Quaternion) Clamp(min, max vecTypes.Vector) vecTypes.Vector {
	minVec := min.(Quaternion)
	maxVec := max.(Quaternion)
	v[0] = math.Clamp(v[0], minVec[0], maxVec[0])
	v[1] = math.Clamp(v[1], minVec[1], maxVec[1])
	v[2] = math.Clamp(v[2], minVec[2], maxVec[2])
	v[3] = math.Clamp(v[3], minVec[3], maxVec[3])
	return v
}

func (v Quaternion) FillC(c float32) vecTypes.Vector {
	v[0] = c
	v[1] = c
	v[2] = c
	v[3] = c
	return v
}

func (v Quaternion) Neg() vecTypes.Vector {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	v[3] = -v[3]
	return v
}

func (v Quaternion) Add(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Quaternion)
	v[0] += other[0]
	v[1] += other[1]
	v[2] += other[2]
	v[3] += other[3]
	return v
}

func (v Quaternion) AddC(c float32) vecTypes.Vector {
	v[0] += c
	v[1] += c
	v[2] += c
	v[3] += c
	return v
}

func (v Quaternion) Sub(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Quaternion)
	v[0] -= other[0]
	v[1] -= other[1]
	v[2] -= other[2]
	v[3] -= other[3]
	return v
}

func (v Quaternion) SubC(c float32) vecTypes.Vector {
	v[0] -= c
	v[1] -= c
	v[2] -= c
	v[3] -= c
	return v
}

func (v Quaternion) MulC(c float32) vecTypes.Vector {
	v[0] *= c
	v[1] *= c
	v[2] *= c
	v[3] *= c
	return v
}

func (v Quaternion) MulCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Quaternion)
	v[0] += other[0] * c
	v[1] += other[1] * c
	v[2] += other[2] * c
	v[3] += other[3] * c
	return v
}

func (v Quaternion) MulCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Quaternion)
	v[0] -= other[0] * c
	v[1] -= other[1] * c
	v[2] -= other[2] * c
	v[3] -= other[3] * c
	return v
}

func (v Quaternion) DivC(c float32) vecTypes.Vector {
	if c == 0 {
		panic("vec.Quaternion.DivC: divide by zero")
	}
	v[0] /= c
	v[1] /= c
	v[2] /= c
	v[3] /= c
	return v
}

func (v Quaternion) DivCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Quaternion.DivCAdd: divide by zero")
	}
	other := v1.(Quaternion)
	v[0] += other[0] / c
	v[1] += other[1] / c
	v[2] += other[2] / c
	v[3] += other[3] / c
	return v
}

func (v Quaternion) DivCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Quaternion.DivCSub: divide by zero")
	}
	other := v1.(Quaternion)
	v[0] -= other[0] / c
	v[1] -= other[1] / c
	v[2] -= other[2] / c
	v[3] -= other[3] / c
	return v
}

func (v Quaternion) Normal() vecTypes.Vector {
	m := v.Magnitude()
	if m == 0 {
		panic("vec.Quaternion.Normal: zero magnitude")
	}
	return v.DivC(m)
}

func (v Quaternion) NormalFast() vecTypes.Vector {
	s := v.SumSqr()
	if s == 0 {
		panic("vec.Quaternion.NormalFast: zero magnitude")
	}
	return v.MulC(math.FastISqrt(s))
}

func (v Quaternion) Axis() vecTypes.Vector {
	return Vector(v[:3])
}

func (v Quaternion) Theta() float32 {
	return v[3]
}

func (v Quaternion) Conjugate() vecTypes.Vector {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v Quaternion) Roll() float32 {
	return math32.Atan2(v[3]*v[0]+v[1]*v[2], 0.5-v[0]*v[0]-v[1]*v[1])
}
func (v Quaternion) Pitch() float32 {
	return math32.Asin(-2.0 * (v[0]*v[2] - v[3]*v[1]))
}
func (v Quaternion) Yaw() float32 {
	return math32.Atan2(v[0]*v[1]+v[3]*v[2], 0.5-v[1]*v[1]-v[2]*v[2])
}

func (a Quaternion) Product(b vecTypes.Quaternion) vecTypes.Vector {
	other := b.(Quaternion)
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

func (v Quaternion) Slerp(v1 vecTypes.Vector, time, spin float32) vecTypes.Vector {
	other := v1.(Quaternion)
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

func (v Quaternion) SlerpLong(v1 vecTypes.Vector, time, spin float32) vecTypes.Vector {
	other := v1.(Quaternion)
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

func (v Quaternion) Multiply(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Quaternion)
	v[0] *= other[0]
	v[1] *= other[1]
	v[2] *= other[2]
	v[3] *= other[3]
	return v
}

func (v Quaternion) Dot(v1 vecTypes.Vector) float32 {
	other := v1.(Quaternion)
	return v[0]*other[0] + v[1]*other[1] + v[2]*other[2] + v[3]*other[3]
}

func (v Quaternion) Cross(vecTypes.Vector) vecTypes.Vector {
	panic("vec.Quaternion.Cross: unsupported operation")
}

func (v Quaternion) Refract2D(vecTypes.Vector, float32, float32) (vecTypes.Vector, bool) {
	panic("vec.Quaternion.Refract2D: unsupported operation")
}

func (v Quaternion) Refract3D(vecTypes.Vector, float32, float32) (vecTypes.Vector, bool) {
	panic("vec.Quaternion.Refract3D: unsupported operation")
}

func (v Quaternion) Reflect(n vecTypes.Vector) vecTypes.Vector {
	nVec := n.(Quaternion)
	d := v.Dot(n) * 2
	v[0] = -v[0] + d*nVec[0]
	v[1] = -v[1] + d*nVec[1]
	v[2] = -v[2] + d*nVec[2]
	v[3] = -v[3] + d*nVec[3]
	return v
}

func (v Quaternion) Interpolate(v1 vecTypes.Vector, t float32) vecTypes.Vector {
	other := v1.(Quaternion)
	v[0] = v[0] + t*(other[0]-v[0])
	v[1] = v[1] + t*(other[1]-v[1])
	v[2] = v[2] + t*(other[2]-v[2])
	v[3] = v[3] + t*(other[3]-v[3])
	return v
}

func (v Quaternion) Sum() float32 {
	return v[0] + v[1] + v[2] + v[3]
}

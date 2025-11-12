// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math"
	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

const vector2DSize = 2

type Vector2D [2]float32

func (v Vector2D) Sum() float32 {
	var sum float32
	for _, val := range v {
		sum += val
	}
	return sum
}

func (v Vector2D) Vector() vecTypes.Vector {
	return Vector(v[:])
}

func (v Vector2D) Slice(start, end int) vecTypes.Vector {
	if end < 0 {
		end = len(v)
	}
	return Vector(v[start:end])
}

func (v Vector2D) XY() (float32, float32) {
	return v[0], v[1]
}

func (v Vector2D) XYZ() (float32, float32, float32) {
	panic("vec.Vector2D.XYZ: unsupported operation")
}

func (v Vector2D) XYZW() (float32, float32, float32, float32) {
	panic("vec.Vector2D.XYZW: unsupported operation")
}

func (v Vector2D) SumSqr() float32 {
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v Vector2D) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v Vector2D) DistanceSqr(v1 vecTypes.Vector) float32 {
	other := readVector(v1, "Vector2D.DistanceSqr", vector2DSize)
	dx := v[0] - other[0]
	dy := v[1] - other[1]
	return dx*dx + dy*dy
}

func (v Vector2D) Distance(v1 vecTypes.Vector) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Vector2D) Clone() vecTypes.Vector {
	clone := new(Vector2D)
	copy(clone[:], v[:])
	return clone
}

func (v Vector2D) CopyFrom(start int, v1 vecTypes.Vector) vecTypes.Vector {
	src := readVector(v1, "Vector2D.CopyFrom", vector2DSize)
	copy(v[start:], src)
	return v
}

func (v Vector2D) CopyTo(start int, v1 vecTypes.Vector) vecTypes.Vector {
	dst := writeVector(v1, "Vector2D.CopyTo", vector2DSize)
	copy(dst, v[start:])
	return v1
}

func (v Vector2D) Clamp(min, max vecTypes.Vector) vecTypes.Vector {
	minVec := readVector(min, "Vector2D.Clamp.min", vector2DSize)
	maxVec := readVector(max, "Vector2D.Clamp.max", vector2DSize)
	for i := range v {
		v[i] = math.Clamp(v[i], minVec[i], maxVec[i])
	}
	return v
}

func (v Vector2D) FillC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] = c
	}
	return v
}

func (v Vector2D) Neg() vecTypes.Vector {
	for i := range v {
		v[i] = -v[i]
	}
	return v
}

func (v Vector2D) Add(v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector2D.Add", vector2DSize)
	for i := range v {
		v[i] += other[i]
	}
	return v
}

func (v Vector2D) AddC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] += c
	}
	return v
}

func (v Vector2D) Sub(v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector2D.Sub", vector2DSize)
	for i := range v {
		v[i] -= other[i]
	}
	return v
}

func (v Vector2D) SubC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] -= c
	}
	return v
}

func (v Vector2D) MulC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] *= c
	}
	return v
}

func (v Vector2D) MulCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector2D.MulCAdd", vector2DSize)
	for i := range v {
		v[i] += other[i] * c
	}
	return v
}

func (v Vector2D) MulCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector2D.MulCSub", vector2DSize)
	for i := range v {
		v[i] -= other[i] * c
	}
	return v
}

func (v Vector2D) DivC(c float32) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector2D.DivC: divide by zero")
	}
	for i := range v {
		v[i] /= c
	}
	return v
}

func (v Vector2D) DivCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector2D.DivCAdd: divide by zero")
	}
	other := readVector(v1, "Vector2D.DivCAdd", vector2DSize)
	for i := range v {
		v[i] += other[i] / c
	}
	return v
}

func (v Vector2D) DivCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector2D.DivCSub: divide by zero")
	}
	other := readVector(v1, "Vector2D.DivCSub", vector2DSize)
	for i := range v {
		v[i] -= other[i] / c
	}
	return v
}

func (v Vector2D) Normal() vecTypes.Vector {
	m := v.Magnitude()
	if m == 0 {
		panic("vec.Vector2D.Normal: zero magnitude")
	}
	return v.DivC(m)
}

func (v Vector2D) NormalFast() vecTypes.Vector {
	s := v.SumSqr()
	if s == 0 {
		panic("vec.Vector2D.NormalFast: zero magnitude")
	}
	return v.MulC(math.FastISqrt(s))
}

func (v Vector2D) Multiply(v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector2D.Multiply", vector2DSize)
	for i := range v {
		v[i] *= other[i]
	}
	return v
}

func (v Vector2D) Dot(v1 vecTypes.Vector) float32 {
	other := readVector(v1, "Vector2D.Dot", vector2DSize)
	return v[0]*other[0] + v[1]*other[1]
}

func (v Vector2D) Cross(vecTypes.Vector) vecTypes.Vector {
	panic("vec.Vector2D.Cross: unsupported operation")
}

func (v Vector2D) Refract2D(n vecTypes.Vector, ni, nt float32) (vecTypes.Vector, bool) {
	nVec := readVector(n, "Vector2D.Refract2D.normal", vector2DSize)
	NdotV := nVec[0]*v[0] + nVec[1]*v[1]
	var nMult float32
	if NdotV > 0 {
		nMult = ni / nt
	} else {
		nMult = nt / ni
	}

	sinT := Vector2D{}
	cosV := Vector2D{}
	cosV[0] = nVec[0] * NdotV
	cosV[1] = nVec[1] * NdotV
	sinT[0] = (cosV[0] - v[0]) * nMult
	sinT[1] = (cosV[1] - v[1]) * nMult
	lenSinT := sinT[0]*sinT[0] + sinT[1]*sinT[1]
	if lenSinT >= 1 {
		return v, false
	}
	NdotT := math32.Sqrt(1 - lenSinT)
	if NdotV < 0 {
		NdotT = -NdotT
	}
	v[0] = sinT[0] - nVec[0]*NdotT
	v[1] = sinT[1] - nVec[1]*NdotT

	return v, true
}

func (v Vector2D) Refract3D(vecTypes.Vector, float32, float32) (vecTypes.Vector, bool) {
	panic("vec.Vector2D.Refract3D: unsupported operation")
}

func (v Vector2D) Reflect(n vecTypes.Vector) vecTypes.Vector {
	nVec := readVector(n, "Vector2D.Reflect", vector2DSize)
	d := v.Dot(n)
	v[0] = -v[0] + 2*d*nVec[0]
	v[1] = -v[1] + 2*d*nVec[1]
	return v
}

func (v Vector2D) Interpolate(v1 vecTypes.Vector, t float32) vecTypes.Vector {
	other := readVector(v1, "Vector2D.Interpolate", vector2DSize)
	v[0] = v[0] + t*(other[0]-v[0])
	v[1] = v[1] + t*(other[1]-v[1])
	return v
}

func (v Vector2D) Axis() vecTypes.Vector {
	panic("vec.Vector2D.Axis: unsupported operation")
}

func (v Vector2D) Theta() float32 {
	panic("vec.Vector2D.Theta: unsupported operation")
}

func (v Vector2D) Conjugate() vecTypes.Vector {
	panic("vec.Vector2D.Conjugate: unsupported operation")
}

func (v Vector2D) Roll() float32 {
	panic("vec.Vector2D.Roll: unsupported operation")
}

func (v Vector2D) Pitch() float32 {
	panic("vec.Vector2D.Pitch: unsupported operation")
}

func (v Vector2D) Yaw() float32 {
	panic("vec.Vector2D.Yaw: unsupported operation")
}

func (v Vector2D) Product(vecTypes.Quaternion) vecTypes.Vector {
	panic("vec.Vector2D.Product: unsupported operation")
}

func (v Vector2D) Slerp(vecTypes.Vector, float32, float32) vecTypes.Vector {
	panic("vec.Vector2D.Slerp: unsupported operation")
}

func (v Vector2D) SlerpLong(vecTypes.Vector, float32, float32) vecTypes.Vector {
	panic("vec.Vector2D.SlerpLong: unsupported operation")
}

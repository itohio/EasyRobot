// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math"
	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

const vector3DSize = 3

type Vector3D [3]float32

func (v *Vector3D) view() Vector {
	return v[:]
}

func (v *Vector3D) Sum() float32 {
	var sum float32
	for _, val := range v {
		sum += val
	}
	return sum
}

func (v *Vector3D) Vector() Vector {
	return v.view()
}

func (v *Vector3D) Slice(start, end int) vecTypes.Vector {
	if end < 0 {
		end = len(v)
	}
	return Vector(v[start:end])
}

func (v *Vector3D) XY() (float32, float32) {
	return v[0], v[1]
}

func (v *Vector3D) XYZ() (float32, float32, float32) {
	return v[0], v[1], v[2]
}

func (v *Vector3D) XYZW() (float32, float32, float32, float32) {
	panic("vec.Vector3D.XYZW: unsupported operation")
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

func (v *Vector3D) DistanceSqr(v1 vecTypes.Vector) float32 {
	other := readVector(v1, "Vector3D.DistanceSqr", vector3DSize)
	dx := v[0] - other[0]
	dy := v[1] - other[1]
	dz := v[2] - other[2]
	return dx*dx + dy*dy + dz*dz
}

func (v *Vector3D) Distance(v1 vecTypes.Vector) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v *Vector3D) Clone() vecTypes.Vector {
	if v == nil {
		return nil
	}
	clone := new(Vector3D)
	copy(clone[:], v[:])
	return clone
}

func (v *Vector3D) CopyFrom(start int, v1 vecTypes.Vector) vecTypes.Vector {
	src := readVector(v1, "Vector3D.CopyFrom", vector3DSize)
	copy(v[start:], src)
	return v.view()
}

func (v *Vector3D) CopyTo(start int, v1 vecTypes.Vector) vecTypes.Vector {
	dst := writeVector(v1, "Vector3D.CopyTo", vector3DSize)
	copy(dst, v[start:])
	return v1
}

func (v *Vector3D) Clamp(min, max vecTypes.Vector) vecTypes.Vector {
	minVec := readVector(min, "Vector3D.Clamp.min", vector3DSize)
	maxVec := readVector(max, "Vector3D.Clamp.max", vector3DSize)
	for i := range v {
		v[i] = math.Clamp(v[i], minVec[i], maxVec[i])
	}
	return v.view()
}

func (v *Vector3D) FillC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] = c
	}
	return v.view()
}

func (v *Vector3D) Neg() vecTypes.Vector {
	for i := range v {
		v[i] = -v[i]
	}
	return v.view()
}

func (v *Vector3D) Add(v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector3D.Add", vector3DSize)
	for i := range v {
		v[i] += other[i]
	}
	return v.view()
}

func (v *Vector3D) AddC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] += c
	}
	return v.view()
}

func (v *Vector3D) Sub(v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector3D.Sub", vector3DSize)
	for i := range v {
		v[i] -= other[i]
	}
	return v.view()
}

func (v *Vector3D) SubC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] -= c
	}
	return v.view()
}

func (v *Vector3D) MulC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] *= c
	}
	return v.view()
}

func (v *Vector3D) MulCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector3D.MulCAdd", vector3DSize)
	for i := range v {
		v[i] += other[i] * c
	}
	return v.view()
}

func (v *Vector3D) MulCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector3D.MulCSub", vector3DSize)
	for i := range v {
		v[i] -= other[i] * c
	}
	return v.view()
}

func (v *Vector3D) DivC(c float32) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector3D.DivC: divide by zero")
	}
	for i := range v {
		v[i] /= c
	}
	return v.view()
}

func (v *Vector3D) DivCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector3D.DivCAdd: divide by zero")
	}
	other := readVector(v1, "Vector3D.DivCAdd", vector3DSize)
	for i := range v {
		v[i] += other[i] / c
	}
	return v.view()
}

func (v *Vector3D) DivCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector3D.DivCSub: divide by zero")
	}
	other := readVector(v1, "Vector3D.DivCSub", vector3DSize)
	for i := range v {
		v[i] -= other[i] / c
	}
	return v.view()
}

func (v *Vector3D) Normal() vecTypes.Vector {
	m := v.Magnitude()
	if m == 0 {
		panic("vec.Vector3D.Normal: zero magnitude")
	}
	return v.DivC(m)
}

func (v *Vector3D) NormalFast() vecTypes.Vector {
	s := v.SumSqr()
	if s == 0 {
		panic("vec.Vector3D.NormalFast: zero magnitude")
	}
	return v.MulC(math.FastISqrt(s))
}

func (v *Vector3D) Multiply(v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector3D.Multiply", vector3DSize)
	for i := range v {
		v[i] *= other[i]
	}
	return v.view()
}

func (v *Vector3D) Dot(v1 vecTypes.Vector) float32 {
	other := readVector(v1, "Vector3D.Dot", vector3DSize)
	return v[0]*other[0] + v[1]*other[1] + v[2]*other[2]
}

func (v *Vector3D) Cross(v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector3D.Cross", vector3DSize)
	x := v[0]
	y := v[1]
	z := v[2]
	v[0] = y*other[2] - z*other[1]
	v[1] = z*other[0] - x*other[2]
	v[2] = x*other[1] - y*other[0]
	return v.view()
}

func (v *Vector3D) Refract2D(vecTypes.Vector, float32, float32) (vecTypes.Vector, bool) {
	panic("vec.Vector3D.Refract2D: unsupported operation")
}

func (v *Vector3D) Refract3D(n vecTypes.Vector, ni, nt float32) (vecTypes.Vector, bool) {
	nVec := readVector(n, "Vector3D.Refract3D.normal", vector3DSize)
	NdotV := nVec[0]*v[0] + nVec[1]*v[1] + nVec[2]*v[2]
	var nMult float32
	if NdotV > 0 {
		nMult = ni / nt
	} else {
		nMult = nt / ni
	}

	sinT := Vector3D{}
	cosV := Vector3D{}
	cosV[0] = nVec[0] * NdotV
	cosV[1] = nVec[1] * NdotV
	cosV[2] = nVec[2] * NdotV
	sinT[0] = (cosV[0] - v[0]) * nMult
	sinT[1] = (cosV[1] - v[1]) * nMult
	sinT[2] = (cosV[2] - v[2]) * nMult
	lenSinT := sinT[0]*sinT[0] + sinT[1]*sinT[1] + sinT[2]*sinT[2]
	if lenSinT >= 1 {
		return v, false
	}
	NdotT := math32.Sqrt(1 - lenSinT)
	if NdotV < 0 {
		NdotT = -NdotT
	}
	v[0] = sinT[0] - nVec[0]*NdotT
	v[1] = sinT[1] - nVec[1]*NdotT
	v[2] = sinT[2] - nVec[2]*NdotT

	return v, true
}

func (v *Vector3D) Reflect(n vecTypes.Vector) vecTypes.Vector {
	nVec := readVector(n, "Vector3D.Reflect", vector3DSize)
	d := v.Dot(n) * 2
	v[0] = -v[0] + d*nVec[0]
	v[1] = -v[1] + d*nVec[1]
	v[2] = -v[2] + d*nVec[2]
	return v.view()
}

func (v *Vector3D) Interpolate(v1 vecTypes.Vector, t float32) vecTypes.Vector {
	other := readVector(v1, "Vector3D.Interpolate", vector3DSize)
	v[0] = v[0] + t*(other[0]-v[0])
	v[1] = v[1] + t*(other[1]-v[1])
	v[2] = v[2] + t*(other[2]-v[2])
	return v.view()
}

func (v *Vector3D) Axis() vecTypes.Vector {
	panic("vec.Vector3D.Axis: unsupported operation")
}

func (v *Vector3D) Theta() float32 {
	panic("vec.Vector3D.Theta: unsupported operation")
}

func (v *Vector3D) Conjugate() vecTypes.Vector {
	panic("vec.Vector3D.Conjugate: unsupported operation")
}

func (v *Vector3D) Roll() float32 {
	panic("vec.Vector3D.Roll: unsupported operation")
}

func (v *Vector3D) Pitch() float32 {
	panic("vec.Vector3D.Pitch: unsupported operation")
}

func (v *Vector3D) Yaw() float32 {
	panic("vec.Vector3D.Yaw: unsupported operation")
}

func (v *Vector3D) Product(vecTypes.Quaternion) vecTypes.Vector {
	panic("vec.Vector3D.Product: unsupported operation")
}

func (v *Vector3D) Slerp(vecTypes.Vector, float32, float32) vecTypes.Vector {
	panic("vec.Vector3D.Slerp: unsupported operation")
}

func (v *Vector3D) SlerpLong(vecTypes.Vector, float32, float32) vecTypes.Vector {
	panic("vec.Vector3D.SlerpLong: unsupported operation")
}

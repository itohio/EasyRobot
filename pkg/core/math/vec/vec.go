// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math"
	"github.com/itohio/EasyRobot/pkg/core/math/primitive/fp32"
	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

var _ vecTypes.Vector = (Vector)(nil)

type Vector []float32

func New(size int) Vector {
	return make(Vector, size)
}

func NewFrom(v ...float32) Vector {
	return v[:]
}

func (v Vector) View() vecTypes.Vector {
	return Vector(v[:])
}

func (v Vector) Sum() float32 {
	if len(v) == 0 {
		return 0
	}
	return fp32.Sum(v, len(v), 1)
}

func (v Vector) Slice(start, end int) vecTypes.Vector {
	if end < 0 {
		end = len(v)
	}
	return v[start:end]
}

func (v Vector) XY() (float32, float32) {
	return v[0], v[1]
}

func (v Vector) XYZ() (float32, float32, float32) {
	return v[0], v[1], v[2]
}

func (v Vector) XYZW() (float32, float32, float32, float32) {
	return v[0], v[1], v[2], v[3]
}

func (v Vector) SumSqr() float32 {
	if len(v) == 0 {
		return 0
	}
	return fp32.SqrSum(v, len(v), 1)
}

func (v Vector) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v Vector) DistanceSqr(v1 vecTypes.Vector) float32 {
	other := readVector(v1, "Vector.DistanceSqr", len(v))
	clone := v.Clone().(Vector)
	clone.Sub(other)
	return clone.SumSqr()
}

func (v Vector) Distance(v1 vecTypes.Vector) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Vector) Clone() vecTypes.Vector {
	if v == nil {
		return nil
	}

	clone := make(Vector, len(v))
	copy(clone, v)
	return clone
}

func (v Vector) CopyFrom(start int, v1 vecTypes.Vector) vecTypes.Vector {
	src := readVector(v1, "Vector.CopyFrom", len(v))
	copy(v[start:], src)
	return v
}

func (v Vector) CopyTo(start int, v1 vecTypes.Vector) vecTypes.Vector {
	dst := writeVector(v1, "Vector.CopyTo", len(v))
	copy(dst, v[start:])
	return v1
}

func (v Vector) Clamp(min, max vecTypes.Vector) vecTypes.Vector {
	minVec := readVector(min, "Vector.Clamp.min", len(v))
	maxVec := readVector(max, "Vector.Clamp.max", len(v))
	for i := range v {
		v[i] = math.Clamp(v[i], minVec[i], maxVec[i])
	}
	return v
}

func (v Vector) FillC(c float32) vecTypes.Vector {
	if len(v) == 0 {
		return v
	}
	for i := range v {
		v[i] = c
	}
	return v
}

func (v Vector) Neg() vecTypes.Vector {
	if len(v) == 0 {
		return v
	}
	fp32.Scal(v, 1, len(v), -1.0)
	return v
}

func (v Vector) Add(v1 vecTypes.Vector) vecTypes.Vector {
	if len(v) == 0 {
		return v
	}
	other := readVector(v1, "Vector.Add", len(v))
	fp32.Axpy(v, other, 1, 1, len(v), 1.0)
	return v
}

func (v Vector) AddC(c float32) vecTypes.Vector {
	if len(v) == 0 {
		return v
	}
	fp32.SumArrInPlace(v, c, len(v))
	return v
}

func (v Vector) Sub(v1 vecTypes.Vector) vecTypes.Vector {
	if len(v) == 0 {
		return v
	}
	other := readVector(v1, "Vector.Sub", len(v))
	fp32.Axpy(v, other, 1, 1, len(v), -1.0)
	return v
}

func (v Vector) SubC(c float32) vecTypes.Vector {
	if len(v) == 0 {
		return v
	}
	fp32.DiffArrInPlace(v, c, len(v))
	return v
}

func (v Vector) MulC(c float32) vecTypes.Vector {
	if len(v) == 0 {
		return v
	}
	fp32.Scal(v, 1, len(v), c)
	return v
}

func (v Vector) MulCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if len(v) == 0 {
		return v
	}
	other := readVector(v1, "Vector.MulCAdd", len(v))
	fp32.Axpy(v, other, 1, 1, len(v), c)
	return v
}

func (v Vector) MulCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if len(v) == 0 {
		return v
	}
	other := readVector(v1, "Vector.MulCSub", len(v))
	fp32.Axpy(v, other, 1, 1, len(v), -c)
	return v
}

func (v Vector) DivC(c float32) vecTypes.Vector {
	if len(v) == 0 || c == 0 {
		return v
	}
	fp32.Scal(v, 1, len(v), 1.0/c)
	return v
}

func (v Vector) DivCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector.DivCAdd", len(v))
	for i := range v {
		v[i] += other[i] / c
	}
	return v
}

func (v Vector) DivCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Vector.DivCSub", len(v))
	for i := range v {
		v[i] -= other[i] / c
	}
	return v
}

func (v Vector) Normal() vecTypes.Vector {
	d := v.Magnitude()
	return v.DivC(d)
}

func (v Vector) NormalFast() vecTypes.Vector {
	d := v.SumSqr()
	return v.MulC(math.FastISqrt(d))
}

func (v Vector) Axis() vecTypes.Vector {
	if len(v) < 3 {
		panic("vec.Vector.Axis: requires length >= 3")
	}
	return v[:3]
}

func (v Vector) Theta() float32 {
	if len(v) < 4 {
		panic("vec.Vector.Theta: requires length >= 4")
	}
	return v[3]
}

func (v Vector) Conjugate() vecTypes.Vector {
	if len(v) < 3 {
		panic("vec.Vector.Conjugate: requires length >= 3")
	}
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v Vector) Roll() float32 {
	return math32.Atan2(v[3]*v[0]+v[1]*v[2], 0.5-v[0]*v[0]-v[1]*v[1])
}
func (v Vector) Pitch() float32 {
	return math32.Asin(-2.0 * (v[0]*v[2] - v[3]*v[1]))
}
func (v Vector) Yaw() float32 {
	return math32.Atan2(v[0]*v[1]+v[3]*v[2], 0.5-v[1]*v[1]-v[2]*v[2])
}

func (a Vector) Product(b vecTypes.Quaternion) vecTypes.Vector {
	other := readVector(b, "Vector.Product", 4)
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

func (v Vector) Slerp(v1 vecTypes.Vector, time, spin float32) vecTypes.Vector {
	other := readVector(v1, "Vector.Slerp", len(v))
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

	for i := range v {
		v[i] = k1*v[i] + k2*other[i]
	}
	return v
}

func (v Vector) SlerpLong(v1 vecTypes.Vector, time, spin float32) vecTypes.Vector {
	other := readVector(v1, "Vector.SlerpLong", len(v))
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

	for i := range v {
		v[i] = k1*v[i] + k2*other[i]
	}
	return v
}

func (v Vector) Multiply(v1 vecTypes.Vector) vecTypes.Vector {
	if len(v) == 0 {
		return v
	}
	other := readVector(v1, "Vector.Multiply", len(v))
	fp32.ElemMul(v, v, other, []int{len(v)}, []int{1}, []int{1}, []int{1})
	return v
}

func (v Vector) Dot(v1 vecTypes.Vector) float32 {
	if len(v) == 0 {
		return 0
	}
	other := readVector(v1, "Vector.Dot", len(v))
	return fp32.Dot(v, other, 1, 1, len(v))
}

func (v Vector) Cross(v1 vecTypes.Vector) vecTypes.Vector {
	if len(v) < 3 {
		panic("vec.Vector.Cross: requires at least 3 dimensions")
	}
	other := readVector(v1, "Vector.Cross", len(v))
	t := []float32{v[0], v[1], v[2]}
	v[0] = t[1]*other[2] - t[2]*other[1]
	v[1] = t[2]*other[0] - t[0]*other[2]
	v[2] = t[0]*other[1] - t[1]*other[0]
	return v
}

func (v Vector) Refract2D(n vecTypes.Vector, ni, nt float32) (vecTypes.Vector, bool) {
	nVec := readVector(n, "Vector.Refract2D.normal", 2)
	cosV := make(Vector, 2)
	sinT := make(Vector, 2)

	NdotV := nVec[0]*v[0] + nVec[1]*v[1]

	var nMult float32
	if NdotV > 0 {
		nMult = ni / nt
	} else {
		nMult = nt / ni
	}

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

func (v Vector) Refract3D(n vecTypes.Vector, ni, nt float32) (vecTypes.Vector, bool) {
	nVec := readVector(n, "Vector.Refract3D.normal", 3)
	cosV := make(Vector, 3)
	sinT := make(Vector, 3)

	NdotV := nVec[0]*v[0] + nVec[1]*v[1] + nVec[2]*v[2]

	var nMult float32
	if NdotV > 0 {
		nMult = ni / nt
	} else {
		nMult = nt / ni
	}
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

func (v Vector) Reflect(n vecTypes.Vector) vecTypes.Vector {
	nVec := readVector(n, "Vector.Reflect", len(v))
	NdotV := v.Dot(n) * 2
	for i := range v {
		v[i] = -v[i] + NdotV*nVec[i]
	}
	return v
}

func (v Vector) Interpolate(v1 vecTypes.Vector, t float32) vecTypes.Vector {
	other := readVector(v1, "Vector.Interpolate", len(v))
	d := other.Clone().(Vector)
	d.Sub(v)
	return v.MulCAdd(t, d)
}

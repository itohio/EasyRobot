// Generated code. DO NOT EDIT

package vec

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math"
	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

var _ vecTypes.Vector = Quaternion{}

const quaternionSize = 4

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
	var sum float32
	for _, val := range v {
		sum += val * val
	}
	return sum
}

func (v Quaternion) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v Quaternion) DistanceSqr(v1 vecTypes.Vector) float32 {
	other := readVector(v1, "Quaternion.DistanceSqr", quaternionSize)
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
	clone := new(Quaternion)
	copy(clone[:], v[:])
	return clone
}

func (v Quaternion) CopyFrom(start int, v1 vecTypes.Vector) vecTypes.Vector {
	src := readVector(v1, "Quaternion.CopyFrom", quaternionSize)
	copy(v[start:], src)
	return v
}

func (v Quaternion) CopyTo(start int, v1 vecTypes.Vector) vecTypes.Vector {
	dst := writeVector(v1, "Quaternion.CopyTo", quaternionSize)
	copy(dst, v[start:])
	return v1
}

func (v Quaternion) Clamp(min, max vecTypes.Vector) vecTypes.Vector {
	minVec := readVector(min, "Quaternion.Clamp.min", quaternionSize)
	maxVec := readVector(max, "Quaternion.Clamp.max", quaternionSize)
	for i := range v {
		v[i] = math.Clamp(v[i], minVec[i], maxVec[i])
	}
	return v
}

func (v Quaternion) FillC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] = c
	}
	return v
}

func (v Quaternion) Neg() vecTypes.Vector {
	for i := range v {
		v[i] = -v[i]
	}
	return v
}

func (v Quaternion) Add(v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Quaternion.Add", quaternionSize)
	for i := range v {
		v[i] += other[i]
	}
	return v
}

func (v Quaternion) AddC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] += c
	}
	return v
}

func (v Quaternion) Sub(v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Quaternion.Sub", quaternionSize)
	for i := range v {
		v[i] -= other[i]
	}
	return v
}

func (v Quaternion) SubC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] -= c
	}
	return v
}

func (v Quaternion) MulC(c float32) vecTypes.Vector {
	for i := range v {
		v[i] *= c
	}
	return v
}

func (v Quaternion) MulCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Quaternion.MulCAdd", quaternionSize)
	for i := range v {
		v[i] += other[i] * c
	}
	return v
}

func (v Quaternion) MulCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := readVector(v1, "Quaternion.MulCSub", quaternionSize)
	for i := range v {
		v[i] -= other[i] * c
	}
	return v
}

func (v Quaternion) DivC(c float32) vecTypes.Vector {
	if c == 0 {
		panic("vec.Quaternion.DivC: divide by zero")
	}
	for i := range v {
		v[i] /= c
	}
	return v
}

func (v Quaternion) DivCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Quaternion.DivCAdd: divide by zero")
	}
	other := readVector(v1, "Quaternion.DivCAdd", quaternionSize)
	for i := range v {
		v[i] += other[i] / c
	}
	return v
}

func (v Quaternion) DivCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Quaternion.DivCSub: divide by zero")
	}
	other := readVector(v1, "Quaternion.DivCSub", quaternionSize)
	for i := range v {
		v[i] -= other[i] / c
	}
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
	other := readVector(b, "Quaternion.Product", quaternionSize)
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
	other := readVector(v1, "Quaternion.Slerp", quaternionSize)
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
	other := readVector(v1, "Quaternion.SlerpLong", quaternionSize)
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
	other := readVector(v1, "Quaternion.Multiply", quaternionSize)
	for i := range v {
		v[i] *= other[i]
	}
	return v
}

func (v Quaternion) Dot(v1 vecTypes.Vector) float32 {
	other := readVector(v1, "Quaternion.Dot", quaternionSize)
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
	nVec := readVector(n, "Quaternion.Reflect", quaternionSize)
	d := v.Dot(n) * 2
	for i := range v {
		v[i] = -v[i] + d*nVec[i]
	}
	return v
}

func (v Quaternion) Interpolate(v1 vecTypes.Vector, t float32) vecTypes.Vector {
	other := readVector(v1, "Quaternion.Interpolate", quaternionSize)
	for i := range v {
		v[i] = v[i] + t*(other[i]-v[i])
	}
	return v
}

func (v Quaternion) Sum() float32 {
	var sum float32
	for _, val := range v {
		sum += val
	}
	return sum
}

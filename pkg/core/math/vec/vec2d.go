package vec

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/pkg/core/math"
	vecTypes "github.com/itohio/EasyRobot/pkg/core/math/vec/types"
)

var _ vecTypes.Vector = Vector2D{}

type Vector2D [2]float32

func (v Vector2D) Sum() float32 {
	return v[0] + v[1]
}

func (v Vector2D) View() vecTypes.Vector {
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
	return v[0], v[1], 0
}

func (v Vector2D) XYZW() (float32, float32, float32, float32) {
	return v[0], v[1], 0, 0
}

func (v Vector2D) SumSqr() float32 {
	return v[0] * v[1]
}

func (v Vector2D) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v Vector2D) DistanceSqr(v1 vecTypes.Vector) float32 {
	other := v1.(Vector2D)
	dx := v[0] - other[0]
	dy := v[1] - other[1]
	return dx*dx + dy*dy
}

func (v Vector2D) Distance(v1 vecTypes.Vector) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Vector2D) Clone() vecTypes.Vector {
	return v
}

func (v Vector2D) CopyFrom(start int, v1 vecTypes.Vector) vecTypes.Vector {
	src := v1.View().(Vector)
	copy(v[:], src[start:])
	return v
}

func (v Vector2D) CopyTo(start int, v1 vecTypes.Vector) vecTypes.Vector {
	dst := v1.View().(Vector)
	copy(dst, v[start:])
	return v1
}

func (v Vector2D) Clamp(min, max vecTypes.Vector) vecTypes.Vector {
	minVec := min.(Vector2D)
	maxVec := max.(Vector2D)
	v[0] = math.Clamp(v[0], minVec[0], maxVec[0])
	v[1] = math.Clamp(v[1], minVec[1], maxVec[1])
	return v
}

func (v Vector2D) FillC(c float32) vecTypes.Vector {
	v[0] = c
	v[1] = c
	return v
}

func (v Vector2D) Neg() vecTypes.Vector {
	v[0] = -v[0]
	v[1] = -v[1]
	return v
}

func (v Vector2D) Add(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector2D)
	v[0] += other[0]
	v[1] += other[1]
	return v
}

func (v Vector2D) AddC(c float32) vecTypes.Vector {
	v[0] += c
	v[1] += c
	return v
}

func (v Vector2D) Sub(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector2D)
	v[0] -= other[0]
	v[1] -= other[1]
	return v
}

func (v Vector2D) SubC(c float32) vecTypes.Vector {
	v[0] -= c
	v[1] -= c
	return v
}

func (v Vector2D) MulC(c float32) vecTypes.Vector {
	v[0] *= c
	v[1] *= c
	return v
}

func (v Vector2D) MulCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector2D)
	v[0] += other[0] * c
	v[1] += other[1] * c
	return v
}

func (v Vector2D) MulCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector2D)
	v[0] -= other[0] * c
	v[1] -= other[1] * c
	return v
}

func (v Vector2D) DivC(c float32) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector2D.DivC: divide by zero")
	}
	v[0] /= c
	v[1] /= c
	return v
}

func (v Vector2D) DivCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector2D.DivCAdd: divide by zero")
	}
	other := v1.(Vector2D)
	v[0] += other[0] / c
	v[1] += other[1] / c
	return v
}

func (v Vector2D) DivCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector2D.DivCSub: divide by zero")
	}
	other := v1.(Vector2D)
	v[0] -= other[0] / c
	v[1] -= other[1] / c
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
	other := v1.(Vector2D)
	v[0] *= other[0]
	v[1] *= other[1]
	return v
}

func (v Vector2D) Dot(v1 vecTypes.Vector) float32 {
	other := v1.(Vector2D)
	return v[0]*other[0] + v[1]*other[1]
}

// Cross returns a Vector3D that is the cross product of the 3D-equivalent vectors from 2D vectors.
// That is, it lifts both 2D vectors to Z=0 in 3D and returns the cross product as a Vector3D.
func (v Vector2D) Cross(v1 vecTypes.Vector) vecTypes.Vector {
	other, ok := v1.(Vector2D)
	if !ok {
		panic("vec.Vector2D.Cross: input is not a Vector2D")
	}
	// The cross product in 3D of (x1, y1, 0) and (x2, y2, 0) yields (0, 0, x1*y2 - y1*x2)
	z := v[0]*other[1] - v[1]*other[0]
	return Vector3D{0, 0, z}
}

func (v Vector2D) Refract2D(n vecTypes.Vector, ni, nt float32) (vecTypes.Vector, bool) {
	nVec := n.(Vector2D)
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
	nVec := n.(Vector2D)
	d := v.Dot(n)
	v[0] = -v[0] + 2*d*nVec[0]
	v[1] = -v[1] + 2*d*nVec[1]
	return v
}

func (v Vector2D) Interpolate(v1 vecTypes.Vector, t float32) vecTypes.Vector {
	other := v1.(Vector2D)
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

// Slerp computes the spherical linear interpolation between v and v1 at fraction t.
// tol is unused here for compatibility, but not typically needed for 2D SLERP.
// Algorithm: standard 2D slerp between normalized vectors.
func (v Vector2D) Slerp(v1 vecTypes.Vector, t float32, tol float32) vecTypes.Vector {
	other := v1.(Vector2D)

	// Normalize v and other
	var vNorm, oNorm Vector2D
	copy(vNorm[:], v[:])
	copy(oNorm[:], other[:])
	vNorm = vNorm.Normal().(Vector2D)
	oNorm = oNorm.Normal().(Vector2D)

	// Compute dot product and clamp to [-1, 1]
	dot := vNorm.Dot(oNorm)
	if dot > 1.0 {
		dot = 1.0
	}
	if dot < -1.0 {
		dot = -1.0
	}

	// Linear interpolation if angle is too small
	const epsilon = 1e-6
	if math32.Abs(dot) > 1.0-epsilon {
		return vNorm.Interpolate(oNorm, t)
	}

	theta := math32.Acos(dot)
	sinTheta := math32.Sin(theta)

	a := math32.Sin((1-t)*theta) / sinTheta
	b := math32.Sin(t*theta) / sinTheta

	return Vector2D{
		vNorm[0]*a + oNorm[0]*b,
		vNorm[1]*a + oNorm[1]*b,
	}
}

// SlerpLong computes the "long" path spherical interpolation between v and v1 at fraction t.
// Equivalent to negating one vector if the dot is positive, then slerping.
func (v Vector2D) SlerpLong(v1 vecTypes.Vector, t float32, tol float32) vecTypes.Vector {
	other := v1.(Vector2D)

	// Normalize v and other
	var vNorm, oNorm Vector2D
	copy(vNorm[:], v[:])
	copy(oNorm[:], other[:])
	vNorm = vNorm.Normal().(Vector2D)
	oNorm = oNorm.Normal().(Vector2D)

	// Compute dot product and clamp to [-1, 1]
	dot := vNorm.Dot(oNorm)
	if dot > 1.0 {
		dot = 1.0
	}
	if dot < -1.0 {
		dot = -1.0
	}

	// Take the long way around the sphere by negating one vector if dot > 0
	if dot > 0 {
		oNorm[0] = -oNorm[0]
		oNorm[1] = -oNorm[1]
		dot = vNorm.Dot(oNorm)
		if dot < -1.0 {
			dot = -1.0
		}
	}

	// Linear interpolation if angle is too small
	const epsilon = 1e-6
	if math32.Abs(dot) > 1.0-epsilon {
		return vNorm.Interpolate(oNorm, t)
	}

	theta := math32.Acos(dot)
	sinTheta := math32.Sin(theta)

	a := math32.Sin((1-t)*theta) / sinTheta
	b := math32.Sin(t*theta) / sinTheta

	return Vector2D{
		vNorm[0]*a + oNorm[0]*b,
		vNorm[1]*a + oNorm[1]*b,
	}
}

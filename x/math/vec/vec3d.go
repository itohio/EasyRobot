package vec

import (
	"github.com/chewxy/math32"
	"github.com/itohio/EasyRobot/x/math"
	vecTypes "github.com/itohio/EasyRobot/x/math/vec/types"
)

var _ vecTypes.Vector = Vector3D{}

type Vector3D [3]float32

func (v Vector3D) Len() int {
	return 3
}

func (v Vector3D) Sum() float32 {
	return v[0] + v[1] + v[2]
}

func (v Vector3D) Release() {
}

func (v Vector3D) View() vecTypes.Vector {
	return Vector(v[:])
}

func (v Vector3D) Slice(start, end int) vecTypes.Vector {
	if end < 0 {
		end = len(v)
	}
	return Vector(v[start:end])
}

func (v Vector3D) XY() (float32, float32) {
	return v[0], v[1]
}

func (v Vector3D) XYZ() (float32, float32, float32) {
	return v[0], v[1], v[2]
}

func (v Vector3D) XYZW() (float32, float32, float32, float32) {
	return v[0], v[1], v[2], 0
}

func (v Vector3D) SumSqr() float32 {
	return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]
}

func (v Vector3D) Magnitude() float32 {
	return math32.Sqrt(v.SumSqr())
}

func (v Vector3D) DistanceSqr(v1 vecTypes.Vector) float32 {
	other := v1.(Vector3D)
	dx := v[0] - other[0]
	dy := v[1] - other[1]
	dz := v[2] - other[2]
	return dx*dx + dy*dy + dz*dz
}

func (v Vector3D) Distance(v1 vecTypes.Vector) float32 {
	return math32.Sqrt(v.DistanceSqr(v1))
}

func (v Vector3D) Clone() vecTypes.Vector {
	return v
}

func (v Vector3D) CopyFrom(start int, v1 vecTypes.Vector) vecTypes.Vector {
	src := v1.View().(Vector)
	copy(v[:], src[start:])
	return v
}

func (v Vector3D) CopyTo(start int, v1 vecTypes.Vector) vecTypes.Vector {
	dst := v1.View().(Vector)
	copy(dst, v[start:])
	return v1
}

func (v Vector3D) Clamp(min, max vecTypes.Vector) vecTypes.Vector {
	minVec := min.(Vector3D)
	maxVec := max.(Vector3D)
	v[0] = math.Clamp(v[0], minVec[0], maxVec[0])
	v[1] = math.Clamp(v[1], minVec[1], maxVec[1])
	v[2] = math.Clamp(v[2], minVec[2], maxVec[2])
	return v
}

func (v Vector3D) FillC(c float32) vecTypes.Vector {
	v[0], v[1], v[2] = c, c, c
	return v
}

func (v Vector3D) Neg() vecTypes.Vector {
	v[0] = -v[0]
	v[1] = -v[1]
	v[2] = -v[2]
	return v
}

func (v Vector3D) Add(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector3D)
	v[0] += other[0]
	v[1] += other[1]
	v[2] += other[2]
	return v
}

func (v Vector3D) AddC(c float32) vecTypes.Vector {
	v[0] += c
	v[1] += c
	v[2] += c
	return v
}

func (v Vector3D) Sub(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector3D)
	v[0] -= other[0]
	v[1] -= other[1]
	v[2] -= other[2]
	return v
}

func (v Vector3D) SubC(c float32) vecTypes.Vector {
	v[0] -= c
	v[1] -= c
	v[2] -= c
	return v
}

func (v Vector3D) MulC(c float32) vecTypes.Vector {
	v[0] *= c
	v[1] *= c
	v[2] *= c
	return v
}

func (v Vector3D) MulCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector3D)
	v[0] += other[0] * c
	v[1] += other[1] * c
	v[2] += other[2] * c
	return v
}

func (v Vector3D) MulCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector3D)
	v[0] -= other[0] * c
	v[1] -= other[1] * c
	v[2] -= other[2] * c
	return v
}

func (v Vector3D) DivC(c float32) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector3D.DivC: divide by zero")
	}
	v[0] /= c
	v[1] /= c
	v[2] /= c
	return v
}

func (v Vector3D) DivCAdd(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector3D.DivCAdd: divide by zero")
	}
	other := v1.(Vector3D)
	v[0] += other[0] / c
	v[1] += other[1] / c
	v[2] += other[2] / c
	return v
}

func (v Vector3D) DivCSub(c float32, v1 vecTypes.Vector) vecTypes.Vector {
	if c == 0 {
		panic("vec.Vector3D.DivCSub: divide by zero")
	}
	other := v1.(Vector3D)
	v[0] -= other[0] / c
	v[1] -= other[1] / c
	v[2] -= other[2] / c
	return v
}

func (v Vector3D) Normal() vecTypes.Vector {
	m := v.Magnitude()
	if m == 0 {
		panic("vec.Vector3D.Normal: zero magnitude")
	}
	return v.DivC(m)
}

func (v Vector3D) NormalFast() vecTypes.Vector {
	s := v.SumSqr()
	if s == 0 {
		panic("vec.Vector3D.NormalFast: zero magnitude")
	}
	return v.MulC(math.FastISqrt(s))
}

func (v Vector3D) Multiply(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector3D)
	v[0] *= other[0]
	v[1] *= other[1]
	v[2] *= other[2]
	return v
}

func (v Vector3D) Dot(v1 vecTypes.Vector) float32 {
	other := v1.(Vector3D)
	return v[0]*other[0] + v[1]*other[1] + v[2]*other[2]
}

func (v Vector3D) Cross(v1 vecTypes.Vector) vecTypes.Vector {
	other := v1.(Vector3D)
	x := v[0]
	y := v[1]
	z := v[2]
	v[0] = y*other[2] - z*other[1]
	v[1] = z*other[0] - x*other[2]
	v[2] = x*other[1] - y*other[0]
	return v
}

func (v Vector3D) Refract2D(vecTypes.Vector, float32, float32) (vecTypes.Vector, bool) {
	panic("vec.Vector3D.Refract2D: unsupported operation")
}

func (v Vector3D) Refract3D(n vecTypes.Vector, ni, nt float32) (vecTypes.Vector, bool) {
	nVec := n.(Vector3D)
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

func (v Vector3D) Reflect(n vecTypes.Vector) vecTypes.Vector {
	nVec := n.(Vector3D)
	d := v.Dot(n) * 2
	v[0] = -v[0] + d*nVec[0]
	v[1] = -v[1] + d*nVec[1]
	v[2] = -v[2] + d*nVec[2]
	return v
}

func (v Vector3D) Interpolate(v1 vecTypes.Vector, t float32) vecTypes.Vector {
	other := v1.(Vector3D)
	v[0] = v[0] + t*(other[0]-v[0])
	v[1] = v[1] + t*(other[1]-v[1])
	v[2] = v[2] + t*(other[2]-v[2])
	return v
}

func (v Vector3D) Axis() vecTypes.Vector {
	panic("vec.Vector3D.Axis: unsupported operation")
}

func (v Vector3D) Theta() float32 {
	panic("vec.Vector3D.Theta: unsupported operation")
}

func (v Vector3D) Conjugate() vecTypes.Vector {
	panic("vec.Vector3D.Conjugate: unsupported operation")
}

func (v Vector3D) Roll() float32 {
	panic("vec.Vector3D.Roll: unsupported operation")
}

func (v Vector3D) Pitch() float32 {
	panic("vec.Vector3D.Pitch: unsupported operation")
}

func (v Vector3D) Yaw() float32 {
	panic("vec.Vector3D.Yaw: unsupported operation")
}

func (v Vector3D) Product(vecTypes.Quaternion) vecTypes.Vector {
	panic("vec.Vector3D.Product: unsupported operation")
}

// Slerp computes the spherical linear interpolation between v and v1 at fraction t.
// tol is unused here for compatibility, but not typically needed for 3D SLERP.
// Algorithm: standard 3D slerp between normalized vectors.
func (v Vector3D) Slerp(v1 vecTypes.Vector, t float32, tol float32) vecTypes.Vector {
	other := v1.(Vector3D)

	// Normalize both vectors
	var vNorm, oNorm Vector3D
	copy(vNorm[:], v[:])
	copy(oNorm[:], other[:])

	vNorm = vNorm.Normal().(Vector3D)
	oNorm = oNorm.Normal().(Vector3D)

	// Compute dot product and clamp to [-1, 1]
	dot := vNorm.Dot(oNorm)
	if dot > 1.0 {
		dot = 1.0
	}
	if dot < -1.0 {
		dot = -1.0
	}

	// If the vectors are too close, linear interpolate
	const epsilon = 1e-6
	if math32.Abs(dot) > 1.0-epsilon {
		return vNorm.Interpolate(oNorm, t)
	}

	theta := math32.Acos(dot)
	sinTheta := math32.Sin(theta)

	// slerp(v0, v1, t) = (sin((1-t)*theta)/sin(theta))*v0 + (sin(t*theta)/sin(theta))*v1
	a := math32.Sin((1-t)*theta) / sinTheta
	b := math32.Sin(t*theta) / sinTheta
	return Vector3D{
		vNorm[0]*a + oNorm[0]*b,
		vNorm[1]*a + oNorm[1]*b,
		vNorm[2]*a + oNorm[2]*b,
	}
}

// SlerpLong computes the "long" path spherical linear interpolation between v and v1 at fraction t.
// This is equivalent to negating one vector if the dot is positive, then slerping.
func (v Vector3D) SlerpLong(v1 vecTypes.Vector, t float32, tol float32) vecTypes.Vector {
	other := v1.(Vector3D)

	// Normalize both vectors
	var vNorm, oNorm Vector3D
	copy(vNorm[:], v[:])
	copy(oNorm[:], other[:])

	vNorm = vNorm.Normal().(Vector3D)
	oNorm = oNorm.Normal().(Vector3D)

	dot := vNorm.Dot(oNorm)
	if dot > 1.0 {
		dot = 1.0
	}
	if dot < -1.0 {
		dot = -1.0
	}

	const epsilon = 1e-6

	// Long arc: flip one direction to ensure angle > pi/2
	if dot > 0 {
		for i := 0; i < 3; i++ {
			oNorm[i] = -oNorm[i]
		}
		dot = vNorm.Dot(oNorm)
		if dot < -1.0 {
			dot = -1.0
		}
	}

	if math32.Abs(dot) > 1.0-epsilon {
		return vNorm.Interpolate(oNorm, t)
	}

	theta := math32.Acos(dot)
	sinTheta := math32.Sin(theta)
	a := math32.Sin((1-t)*theta) / sinTheta
	b := math32.Sin(t*theta) / sinTheta

	return Vector3D{
		vNorm[0]*a + oNorm[0]*b,
		vNorm[1]*a + oNorm[1]*b,
		vNorm[2]*a + oNorm[2]*b,
	}
}

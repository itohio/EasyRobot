package types

// Accessors encapsulate read access helpers for vector components.
type Accessors interface {
	Sum() float32
	Slice(start, end int) Vector
	XY() (float32, float32)
	XYZ() (float32, float32, float32)
	XYZW() (float32, float32, float32, float32)
	SumSqr() float32
	Magnitude() float32
	DistanceSqr(v1 Vector) float32
	Distance(v1 Vector) float32
	Clone() Vector
	CopyFrom(start int, v1 Vector) Vector
	CopyTo(start int, v1 Vector) Vector
}

// Modifiers describe element-wise vector mutations.
type Modifiers interface {
	Clamp(min, max Vector) Vector
	FillC(c float32) Vector
	Neg() Vector
	Add(v1 Vector) Vector
	AddC(c float32) Vector
	Sub(v1 Vector) Vector
	SubC(c float32) Vector
	MulC(c float32) Vector
	MulCAdd(c float32, v1 Vector) Vector
	MulCSub(c float32, v1 Vector) Vector
	DivC(c float32) Vector
	DivCAdd(c float32, v1 Vector) Vector
	DivCSub(c float32, v1 Vector) Vector
	Multiply(v1 Vector) Vector
}

// Quaternion extends Vector semantics for quaternion-specific behavior.
type Quaternion interface {
	Vector
}

// Orientation includes quaternion/axis helpers commonly used in robotics.
type Orientation interface {
	Axis() Vector
	Theta() float32
	Conjugate() Vector
	Roll() float32
	Pitch() float32
	Yaw() float32
	Product(b Quaternion) Vector
	Slerp(v1 Vector, time, spin float32) Vector
	SlerpLong(v1 Vector, time, spin float32) Vector
}

// Geometry encompasses vector operations involving normals and reflections.
type Geometry interface {
	Normal() Vector
	NormalFast() Vector
	Dot(v1 Vector) float32
	Cross(v1 Vector) Vector
	Refract2D(n Vector, ni, nt float32) (Vector, bool)
	Refract3D(n Vector, ni, nt float32) (Vector, bool)
	Reflect(n Vector) Vector
	Interpolate(v1 Vector, t float32) Vector
}

// Vector composes the common vector operations surfaced by the vec package.
type Vector interface {
	Accessors
	Modifiers
	Orientation
	Geometry
}

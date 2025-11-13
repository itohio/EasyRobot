// Package types hosts the common vector interfaces used across math modules.
//
// Receiver semantics summary:
//   - Implementations use value receivers. For array-backed vectors (Vector2D/3D/4D/Quaternion)
//     this means methods operate on a copy; callers must capture the returned value to observe
//     the mutation.
//   - Slice-backed `vec.Vector` implementations share the underlying slice, so value receivers
//     still mutate the original data in place.
//   - `View()` exists to provide a `vec.Vector` view irrespective of the backing store.
package types

// Accessors expose read-only helpers and safe copying utilities.
type Accessors interface {
	// Sum returns the sum of all components.
	Sum() float32
	// Slice returns the sub-range `[start:end]` as a vector view/copy.
	Slice(start, end int) Vector
	// XY/XYZ/XYZW expose the first 2/3/4 components respectively.
	XY() (float32, float32)
	XYZ() (float32, float32, float32)
	XYZW() (float32, float32, float32, float32)
	// SumSqr returns the sum of squared components.
	SumSqr() float32
	// Magnitude returns the Euclidean norm.
	Magnitude() float32
	// DistanceSqr returns squared Euclidean distance to v1.
	DistanceSqr(v1 Vector) float32
	// Distance returns Euclidean distance to v1.
	Distance(v1 Vector) float32
	// Clone returns a deep copy (slice backed) or value copy (array backed).
	Clone() Vector
	// View returns a `vec.Vector` view/copy of the receiver.
	View() Vector
	// CopyFrom copies from v1 into this vector starting at index start.
	CopyFrom(start int, v1 Vector) Vector
	// CopyTo copies this vector into v1 starting at index start.
	CopyTo(start int, v1 Vector) Vector
}

// Modifiers capture element-wise arithmetic operations.
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

// Quaternion extends Vector semantics for quaternion-specific behaviour.
type Quaternion interface {
	Vector
}

// Orientation groups quaternion / Euler helpers used for attitude calculations.
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

// Geometry encompasses vector operations involving normals and refraction.
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

// Vector provides the complete set of behaviors required by the vec package.
//
// Implementations:
//   - vec.Vector: Generic, heap-allocated, dynamic shape vector for use cases where size is not known at compile time.
//   - Fixed vector types (e.g., vec.Vector3D, mat.Quaternion): Stack-allocated, optimized for their specific, static sizes.
//
// Implementation Notes and Best Practices:
// - Both vec.Vector and fixed-sized vectors like Vector3D implement this interface. However, their semantics differ:
//   - Fixed vectors (e.g., Vector3D, Quaternion) expect other fixed vectors or matrices of the exact same size as arguments.
//   - The generic vec.Vector may accept other heap-allocated Vector instances, supporting variable sizes.
//
// - For maximum efficiency, always prefer fixed-sized vectors (e.g., Vector3D for 3D math) for statically sized operations. These are faster (stack-allocated, inlined) than generic heap-based vectors.
// - Do NOT take the address (&) of any value implementing Vector or fixed vector/matrix types (like Vector3D, Matrix3x3) unless explicitly required for a destination parameter. Never pass pointers except where documented and necessary for "destination" arguments.
// - If you require a concrete struct (for example, to access struct fields), type assert the value, e.g. `v.(vec.Vector3D)`.
// - All interfaces consuming or returning Vector must document exactly what concrete types they expect (fixed, generic, which dimension, etc).
// - Example: For 3D operations, prefer using vec.Vector3D on the stack rather than the generic heap-allocated Vector.
type Vector interface {
	Accessors
	Modifiers
	Orientation
	Geometry
}

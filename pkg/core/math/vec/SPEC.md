# math/vec - Vector Operations Specification

## Overview

The `vec` package provides vector mathematics for 2D, 3D, 4D, and arbitrary-dimensional vectors, optimized for robotics applications with emphasis on embedded systems compatibility and performance.

## Design Principles

1. **In-Place Operations**: Most operations modify the vector in-place to minimize allocations
2. **Method Chaining**: Operations return the vector to enable method chaining
3. **Zero Allocations**: Critical paths avoid memory allocations
4. **Float32 Precision**: Uses `float32` for embedded-friendly precision
5. **Type Safety**: Fixed-size types (Vec2D, Vec3D, Vec4D) for compile-time size guarantees
6. **Highly optimized**: Fixed-size vectors use direct math computations without any loops
7. **Interface Compliance**: All vector variants implement the `types.Vector` interface; operations that are not semantically valid for a specific variant must panic with a descriptive message.

## Types

### Vector
Generic variable-length vector type (slice of float32).

```go
type Vector []float32
```

**Characteristics:**
- Variable length (runtime-determined)
- Slice-based (allows sub-slicing)
- Backed by `[]float32`
- Used for arbitrary-dimensional operations

### Vector2D
Fixed-size 2D vector.

```go
type Vector2D [2]float32
```

**Characteristics:**
- Fixed size at compile time
- Array-based (value semantics)
- Optimized for 2D operations
- Access via `[0]` (X), `[1]` (Y)

### Vector3D
Fixed-size 3D vector.

```go
type Vector3D [3]float32
```

**Characteristics:**
- Fixed size at compile time
- Array-based (value semantics)
- Optimized for 3D operations
- Access via `[0]` (X), `[1]` (Y), `[2]` (Z)

### Vector4D
Fixed-size 4D vector (can represent homogeneous coordinates or quaternions).

```go
type Vector4D [4]float32
```

**Characteristics:**
- Fixed size at compile time
- Array-based (value semantics)
- Supports quaternion operations
- Access via `[0]` (X), `[1]` (Y), `[2]` (Z), `[3]` (W)

### Quaternion
Quaternion representation (4 components: i, j, k, w).

```go
type Quaternion [4]float32
```

**Characteristics:**
- Same storage as Vector4D
- Specialized methods for quaternion operations
- Supports rotation operations

## Operations

### Construction and Access

| Method | Type | Description |
|--------|------|-------------|
| `New(size int) Vector` | Vector | Creates new zero-initialized vector |
| `NewFrom(v ...float32) Vector` | Vector | Creates vector from variadic arguments |
| `Clone() Vector/*Vector` | All | Creates deep copy of vector |
| `Slice(start, end int) Vector` | All | Returns slice view of vector |
| `XY() (float32, float32)` | Vector, Vector2D | Returns X, Y components |
| `XYZ() (float32, float32, float32)` | Vector, Vector3D, Vector4D, Quaternion | Returns X, Y, Z components |
| `XYZW() (float32, float32, float32, float32)` | Vector, Vector4D, Quaternion | Returns X, Y, Z, W components |
| `CopyFrom(start int, v1 Vector)` | All | Copies from Vector into this vector |
| `CopyTo(start int, v1 Vector)` | All | Copies this vector to Vector |

### Arithmetic Operations

All arithmetic operations are **in-place** and return the vector for chaining.

| Method | Type | Description | BLAS Equivalent |
|--------|------|-------------|-----------------|
| `Add(v1) Vector/*Vector` | All | `v = v + v1` | Axpy (alpha=1.0) |
| `Sub(v1) Vector/*Vector` | All | `v = v - v1` | Custom (DiffArr) |
| `MulC(c float32) Vector/*Vector` | All | `v = v * c` | Scal |
| `DivC(c float32) Vector/*Vector` | All | `v = v / c` | Scal (alpha=1/c) |
| `AddC(c float32) Vector/*Vector` | All | `v = v + c` | Custom (SumArrInPlace) |
| `SubC(c float32) Vector/*Vector` | All | `v = v - c` | Custom (DiffArrInPlace) |
| `Neg() Vector/*Vector` | All | `v = -v` | Scal (alpha=-1.0) |
| `Multiply(v1) Vector/*Vector` | All | `v = v .* v1` (element-wise) | HadamardProduct |

### Fused Multiply-Add Operations

| Method | Type | Description | BLAS Equivalent |
|--------|------|-------------|-----------------|
| `MulCAdd(c float32, v1) Vector/*Vector` | All | `v = v + v1 * c` | Axpy |
| `MulCSub(c float32, v1) Vector/*Vector` | All | `v = v - v1 * c` | Axpy (alpha=-c) |
| `DivCAdd(c float32, v1) Vector/*Vector` | All | `v = v + v1 / c` | Custom |
| `DivCSub(c float32, v1) Vector/*Vector` | All | `v = v - v1 / c` | Custom |

### Geometric Operations

| Method | Type | Description | BLAS Equivalent |
|--------|------|-------------|-----------------|
| `Dot(v1) float32` | All | Dot product `v · v1` | Dot |
| `Cross(v1) Vector/*Vector` | Vector, Vector3D | Cross product `v × v1` | Custom |
| `Magnitude() float32` | All | Euclidean norm `||v||` | Nrm2 |
| `SumSqr() float32` | All | Sum of squares `Σv²` | SqrSum (via primitive) |
| `Sum() float32` | All | Sum of elements `Σv` | Sum (via primitive) |
| `Normal() Vector/*Vector` | All | Normalize: `v / ||v||` | Nrm2 + Scal |
| `NormalFast() Vector/*Vector` | All | Fast normalize using fast inverse sqrt | Custom |
| `Distance(v1) float32` | All | Euclidean distance | Custom |
| `DistanceSqr(v1) float32` | All | Squared distance | Custom |

### Vector-Specific Operations

| Method | Type | Description |
|--------|------|-------------|
| `FillC(c float32) Vector/*Vector` | All | Fill all elements with constant `c` |
| `Clamp(min, max) Vector/*Vector` | All | Clamp each element to `[min[i], max[i]]` |
| `Reflect(n) Vector/*Vector` | Vector, Vector2D, Vector3D | Reflection: `v - 2*(v·n)*n` |
| `Refract2D(n, ni, nt) (Vector, bool)` | Vector | 2D refraction |
| `Refract3D(n, ni, nt) (Vector, bool)` | Vector | 3D refraction |
| `Interpolate(v1, t float32) Vector/*Vector` | All | Linear interpolation: `v + t*(v1-v)` |

### Quaternion-Specific Operations

| Method | Type | Description |
|--------|------|-------------|
| `Product(q) Quaternion` | Quaternion, Vector4D | Quaternion multiplication |
| `Conjugate() Quaternion` | Quaternion, Vector4D | Quaternion conjugate |
| `Axis() Vector` | Quaternion, Vector4D | Extract rotation axis (first 3 components) |
| `Theta() float32` | Quaternion, Vector4D | Extract rotation angle (4th component) |
| `Slerp(q1, time, spin float32) Quaternion` | Quaternion, Vector4D | Spherical linear interpolation |
| `SlerpLong(q1, time, spin float32) Quaternion` | Quaternion, Vector4D | Spherical linear interpolation (long path) |
| `Roll() float32` | Quaternion, Vector4D | Extract roll angle |
| `Pitch() float32` | Quaternion, Vector4D | Extract pitch angle |
| `Yaw() float32` | Quaternion, Vector4D | Extract yaw angle |

## Current Implementation Details

### Dependency on Primitives

The `Vector` type currently uses `primitive` package functions:
- `primitive.Sum()` - for Sum()
- `primitive.SqrSum()` - for SumSqr()
- `primitive.DiffArr()` - for Sub()
- `primitive.MulArrInPlace()` - for Neg() and MulC()
- `primitive.SumArrAdd()` - for Add()
- `primitive.SumArrInPlace()` - for AddC()
- `primitive.DiffArrInPlace()` - for SubC()
- `primitive.DivArrInPlace()` - for DivC()
- `primitive.MulArrAdd()` - for MulCAdd()
- `primitive.HadamardProduct()` - for Multiply()
- `primitive.DotProduct()` - for Dot() (deprecated, calls `primitive.Dot()`)

### Fixed-Size Vector Implementations

`Vector2D`, `Vector3D`, `Vector4D`, and `Quaternion` use **inline loops** instead of primitive functions:
- Small fixed sizes (2, 3, 4 elements)
- Inline loops are more efficient for small sizes
- Avoids function call overhead
- Compiler can optimize better

## Performance Characteristics

### Vector (Variable Length)
- Uses primitive functions for operations
- Function call overhead for small vectors (< 10 elements)
- Better performance for larger vectors
- Allows stride-based operations via primitives

### Vector2D/3D/4D/Quaternion (Fixed Size)
- Inline loop implementations
- Zero function call overhead
- Compiler optimizations (loop unrolling, SIMD)
- Optimal for small fixed-size vectors

## Memory Layout

### Vector
- Backed by `[]float32` slice
- Contiguous memory layout
- Can share underlying array (slicing)

### Vector2D/3D/4D
- Backed by arrays `[2]float32`, `[3]float32`, `[4]float32`
- Stack-allocated (when not referenced)
- Fixed-size guarantees

## Edge Cases and Error Handling

### Zero Division
- `DivC()` checks for `c == 0` and returns early
- `Normal()` may divide by zero if magnitude is zero (currently no check)

### Empty Vectors
- Most operations check `len(v) == 0` and return early
- Some operations may panic if bounds are exceeded

### Dimension Mismatch
- `Add()`, `Sub()`, `Dot()`, etc. may panic if lengths don't match
- Fixed-size vectors have compile-time guarantees

## Future Improvements

1. **Migration to BLAS Level 1**: Replace deprecated primitive functions with BLAS Level 1 operations
2. **Normalization Safety**: Add checks for zero-norm vectors in `Normal()`
3. **Bounds Checking**: Add optional bounds checking in debug mode
4. **SIMD Optimization**: Optimize fixed-size vectors with SIMD instructions
5. **Error Returns**: Consider returning errors instead of panicking for invalid operations

## Dependencies

- `github.com/chewxy/math32` - Math functions (Sqrt, Atan2, etc.)
- `github.com/itohio/EasyRobot/pkg/core/math` - Fast math utilities (FastISqrt)
- `github.com/itohio/EasyRobot/pkg/core/math/primitive` - BLAS/LAPACK primitives

## Testing

- Unit tests for all operations
- Benchmark tests for performance critical paths
- Edge case tests (empty vectors, zero division, etc.)
- Fixed-size vs. variable-size performance comparisons
- Behavioural parity tests verifying slice-backed `Vector` operations mutate receivers in-place while array-backed `Vector2D`/`Vector3D`/`Vector4D` return modified copies
- Regression tests ensuring cloning, slicing, and mixed-type arithmetic preserve original operands when value semantics are expected


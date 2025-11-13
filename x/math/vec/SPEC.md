# math/vec – Vector Operations Specification

## Overview

The `vec` package provides the low-level vector math building blocks that the EasyRobot stack relies on.  It offers two families of vector types:

- `Vector` (slice backed): dynamic-length helpers built on top of `[]float32`, optimised for in-place operations and zero-copy interop.
- Fixed-size vectors (`Vector2D`, `Vector3D`, `Vector4D`, `Quaternion`): array-backed value types that expose a `vec/types.Vector` implementation without requiring heap allocations.

Understanding the difference between these two families is critical because it determines whether an operation mutates the original value or whether the caller must capture the returned value.

## Semantics at a Glance

| Type | Backing | Receiver style | `View()` result | Mutation behaviour |
|------|---------|----------------|-----------------|--------------------|
| `Vector` | `[]float32` slice | value (slice semantics) | shares underlying slice | Methods mutate the underlying data in-place. No reassignment is required. |
| `Vector2D` / `Vector3D` / `Vector4D` | `[N]float32` array | value (array semantics) | returns a `Vector` backed by a copy of the array | Methods operate on a copy of the receiver. The returned value **must be captured** to observe the change. |
| `Quaternion` | `[4]float32` array | value (array semantics) | returns a `Vector` backed by a copy of the array | Same as other fixed-size vectors; additionally exposes quaternion-specific helpers. |

Key consequences:

1. `View()` replaces the former `Vector()` accessor. For array-backed types the view is a copy; mutating it does not affect the original value unless the caller reassigns.
2. `Clone()` simply returns the value (for fixed-size types) or copies the slice (for `Vector`).
3. All fixed-size methods must be used in a functional style:
```go
   v := vec.Vector3D{1, 2, 3}
   v = v.AddC(10).(vec.Vector3D) // capture result, original literal is unchanged otherwise
   ```
4. Slice-based `Vector` continues to support in-place mutations:
```go
   v := vec.NewFrom(1, 2, 3)
   v.AddC(10)        // modifies v in-place
   fmt.Println(v[0]) // => 11
   ```

## Interfaces (`vec/types`)

`types.Vector` is the common interface implemented by both the slice and fixed-size variants.  Notable requirements:

- `View() Vector` – returns a `vec.Vector` view/copy of the receiver.
- Accessor methods (`Sum`, `Magnitude`, `Distance`, `Clone`, etc.) use value receivers.
- Mutator-like methods (`Add`, `MulC`, `Clamp`, etc.) also use value receivers and return the updated value. Consumers should always capture the returned `Vector`.
- Specialised interfaces (`Quaternion`, `Orientation`, `Geometry`) build on the base `Vector` interface to surface quaternion and geometric helpers.

The interface intentionally documents (in comments) that slice types mutate in-place, while array types require reassignment.

## Type Reference

### `Vector`
- Backed by a `[]float32` slice.
- Supports arbitrary length with constructors `New(size int)` and `NewFrom(values ...float32)`.
- Ideal when zero allocations and in-place operations are required.

### `Vector2D`, `Vector3D`, `Vector4D`
- Compile-time sized arrays (`[2]`, `[3]`, `[4]`).
- Provide the same API surface as `Vector`, but every mutator returns an updated copy that must be stored by the caller.
- `View()` produces a `vec.Vector` that contains a copy of the array contents; modifying the returned slice does **not** mutate the original literal.
- Geometry helpers panic for operations that are undefined for the dimensionality (e.g., `Vector2D.Cross`).

### `Quaternion`
- Alias for `[4]float32` with quaternion-specific helpers (axis/angle extraction, Euler angles, quaternion product, SLERP).
- Inherits the same value semantics as other fixed-size vectors.

## Operation Categories

The following groups are available on every vector variant.  Unless otherwise noted, they return a `vec/types.Vector` that represents the updated value.

- **Accessors** – `Sum`, `Magnitude`, `Distance`, `Clone`, `View`, `Slice`, `XY/XYZ/XYZW`.
- **Arithmetic** – `Add`, `Sub`, `MulC`, `DivC`, `AddC`, `SubC`, `Neg`, `MulCAdd`, `MulCSub`, `DivCAdd`, `DivCSub`, `Multiply` (Hadamard product).
- **Geometry** – `Dot`, `Cross` (dimension permitting), `Normal`, `NormalFast`, `Reflect`, `Refract*`, `Interpolate`.
- **Quaternion-specific** – `Axis`, `Theta`, `Conjugate`, `Roll/Pitch/Yaw`, `Product`, `Slerp`, `SlerpLong` (implemented for `Vector4D` and `Quaternion`).

Operations that are not defined for a particular dimensionality panic with a descriptive message; this behaviour is relied upon by the current tests.

## Error Handling & Edge Cases

- Division by zero panics in `DivC*` helpers.
- Normalisation helpers panic on zero-magnitude inputs.
- Cross-dimensional conversions (e.g., treating `Vector` as `Vector3D`) rely on the caller to supply compatible types; mismatches panic during type assertions.

## Performance Notes

- Slice-based `Vector` delegates to BLAS-like helpers in `pkg/core/math/primitive/fp32` to avoid manual loops and to minimise allocations.
- Fixed-size vectors are simple value types; operations are implemented with straightforward arithmetic so the compiler can inline and eliminate heap allocations when possible.  Returning by value keeps APIs ergonomic while still avoiding heap churn inside tight loops.

## Testing Expectations

- Unit tests assert that slice-based `Vector` mutates in place, whereas fixed-size vectors only change when the caller reassigns the return value.
- Behavioural tests cover clone/view behaviour to ensure copies are independent.
- Quaternion tests validate rotational helpers alongside arithmetic consistency.

Keeping this distinction between slice-backed and array-backed vectors is essential when adding new operations: every new mutator must continue to return an updated value so that both families of vectors satisfy `types.Vector` without accidental heap allocations or hidden side effects.

## Interface Reference (vec/types)

`types.Vector` is the root interface. It embeds four specialised sub-interfaces, making every implementation responsible for the full arithmetic/geometry/quaternion surface area.  Methods always return a `types.Vector`; callers must cast back to the concrete type when necessary.

### Accessors

| Method | Description | Notes |
|--------|-------------|-------|
| `Sum() float32` | Sum of all components. | Works on any dimensionality. |
| `Slice(start, end int) Vector` | Logical slice, analogous to Go's slicing rules. | For fixed-size values this returns a copy; for `Vector` it shares the slice. |
| `XY()`, `XYZ()`, `XYZW()` | Convenience component extractors. | Unsupported combinations panic on fixed-size types. |
| `SumSqr() float32` | Sum of squared components (‖v‖²). | |
| `Magnitude() float32` | Euclidean norm (√‖v‖²). | |
| `DistanceSqr(v1 Vector) float32` | Squared Euclidean distance to `v1`. | Implementations expect dimension match and panic if violated. |
| `Distance(v1 Vector) float32` | Euclidean distance. | |
| `Clone() Vector` | Returns a duplicate with independent backing. | Fixed-size vectors return the value (copy); slice vectors allocate. |
| `View() Vector` | View/copy of the receiver as `vec.Vector`. | Slice vectors share storage; fixed-size vectors copy. |
| `CopyFrom(start int, v1 Vector) Vector` | Copy elements starting at `start`. | Behaviour mirrors `View` semantics. |
| `CopyTo(start int, v1 Vector) Vector` | Copy into the destination vector. | |

### Modifiers

These mutate the conceptual vector and return the updated value. Remember to capture the result for fixed-size implementations.

| Method | Description |
|--------|-------------|
| `Clamp(min, max Vector)` | Component-wise clamp into `[min, max]`. |
| `FillC(c float32)` | Fill all components with a constant. |
| `Neg()` | Unary negation. |
| `Add(v1 Vector)` / `Sub(v1 Vector)` | Component-wise addition/subtraction. |
| `AddC(c float32)` / `SubC(c float32)` | Add/subtract a scalar constant. |
| `MulC(c float32)` / `DivC(c float32)` | Scale by a constant (panic on `c == 0` for division). |
| `MulCAdd(c, v1)` / `MulCSub(c, v1)` | Fused multiply-add/sub: `v += v1*c` or `v -= v1*c`. |
| `DivCAdd(c, v1)` / `DivCSub(c, v1)` | Fused divide-add/sub: `v += v1/c`, `v -= v1/c`. |
| `Multiply(v1 Vector)` | Hadamard (element-wise) product. |

### Geometry

| Method | Description |
|--------|-------------|
| `Normal()` | Normalise to unit length (panic on zero magnitude). |
| `NormalFast()` | Fast inverse-sqrt normalisation (panic on zero magnitude). |
| `Dot(v1 Vector) float32` | Dot product. |
| `Cross(v1 Vector)` | Cross product (dimension-specific support). |
| `Refract2D(n, ni, nt)` / `Refract3D(n, ni, nt)` | Snell refraction helpers (panic if unsupported). |
| `Reflect(n Vector)` | Reflect around normal `n`. |
| `Interpolate(v1, t)` | Linear interpolation. |

### Orientation / Quaternion

| Method | Description |
|--------|-------------|
| `Axis()` | Rotation axis (first three components). |
| `Theta()` | Rotation angle / scalar part. |
| `Conjugate()` | Quaternion conjugate (negates vector part). |
| `Roll()`, `Pitch()`, `Yaw()` | Euler angle extraction. |
| `Product(b Quaternion)` | Hamilton product with another quaternion. |
| `Slerp(v1, time, spin)` / `SlerpLong(v1, time, spin)` | Spherical linear interpolation (short/long arc). |

These tables mirror today’s implementation and should be kept in sync whenever new helpers are added to the interfaces.


# Vector Interface Conformance Plan

## Goal

Ensure every vector variant in `pkg/core/math/vec` (`Vector`, `Vector2D`,
`Vector3D`, `Vector4D`, and `Quaternion`) presents a consistent API by
implementing the `types.Vector` interface.

## Scope

- Normalize method signatures so fixed-size vectors expose the same shape as the
  slice-based `Vector` type.
- Provide panic-backed stubs for operations that are semantically invalid for a
  given variant (for example quaternion-only orientation helpers on `Vector2D`).
- Preserve existing in-place semantics and performance characteristics.

## Approach

1. Audit the interface in `pkg/core/math/vec/types` and list every required
   method.
2. Update each vector variant so its method signatures match the interface,
   returning `vec.Vector` where required and accepting `vec.Vector`
   dependencies.
3. Route variant-specific logic through thin adapters to avoid code duplication
   while keeping functions under 30 lines.
4. Introduce explicit `panic` calls with descriptive messages for unsupported
   operations to surface early failures instead of silent no-ops.
5. Ensure helper conversions between fixed-size arrays and the generic
   `Vector` view stay allocation free (slicing over the underlying array).

## Risks & Mitigations

- **Signature Breakage**: Existing call sites may rely on concrete types.
  Mitigate by adjusting callers within the package and documenting the change.
- **Unexpected Panics**: New panics could surface in downstream consumers.
  Provide clear messages to aid migration and keep behaviour consistent across
  variants.

## Validation

- Run existing unit tests in `pkg/core/math/vec`.
- Add targeted compile-time assertions (e.g., `_ = types.Vector(new(Vector2D))`)
  to guarantee interface conformance.

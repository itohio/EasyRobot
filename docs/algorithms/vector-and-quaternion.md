---
type: algorithm
status: partial
tags: [easyrobot, vec, quaternion, float32]
workstream: W9
module: "[[mat-vec]]"
summary: "33 of 61 vec tests empty; `Vector.Product(Quaternion)` panics; `Theta()` returns w, not an angle"
source: [x/math/vec/vec.go, x/math/vec/vec2d.go, x/math/vec/vec3d.go, x/math/vec/vec4d.go, x/math/vec/quaternion.go, x/math/vec/smart_view.go, x/math/vec/types/types.go, x/math/math.go]
tests: [x/math/vec/vec_test.go, x/math/vec/vec2d_test.go, x/math/vec/vec3d_test.go, x/math/vec/vec4d_test.go, x/math/vec/quaternion_test.go, x/math/math_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): x/math/vec 61/61 pass, 33 vacuous (vec_test.go tables with '//TODO: Add test cases': Sub, SubC, MulC*, DivC*, Normal, NormalFast, Axis, Theta, Conjugate, Roll, Pitch, Yaw, Product, Slerp, SlerpLong, Multiply, Dot, Cross, Refract2D/3D, Reflect, Interpolate, Distance(Sqr), CopyFrom/To, Clamp, FillC, Neg); x/math 4/4 pass; probe InfraNotes/vecprobe"
---
# Vectors and quaternions (x/math/vec)

**What it does:** float32 vector algebra: dynamic `Vector []float32` (in-place, mutates receiver's backing), fixed `Vector2D/3D/4D [N]float32` and `Quaternion [4]float32` (value receivers → return modified copies), all behind `vecTypes.Vector`; ops: add/sub/scale/dot/cross/norm/fast-norm/distance/clamp/reflect/refract/lerp, quaternion product, conjugate, slerp (short/long with extra spins), roll/pitch/yaw. `SmartVector` ref-counted view (`smart_view.go:10-93`).

**Where:** `x/math/vec/vec.go:14-446`; `x/math/vec/quaternion.go:13` `Quaternion`; `:243-254` Hamilton `Product` (**[x, y, z, w], w last**); `:226-231` `Conjugate`; `:202-216` `Normal`/`NormalFast` (panic on zero); `:233-241` Roll/Pitch/Yaw (ZYX, aerospace); `:256-290` `Slerp(v1, t, spin)` (flips sign for shortest path), `:292-320` `SlerpLong`; `:218-224` `Axis()` = xyz, `Theta()` = **w component, not an angle**. `x/math/math.go:66-92` `FastISqrt` (Quake magic + 1 Newton step).

**Inputs → outputs:** float32; quaternions not normalised automatically (Slerp assumes unit inputs).

## Completeness review

- Complete: yes (API); Quaternion `Cross/Refract*` intentionally panic (`quaternion.go:336-346`).
- Tested: 28 value-checking tests (`TestVector_Add/AddC/Clone/Sum/SumSqr/Magnitude/XY…`, `TestVector2D_*`, `TestVector3D_CrossDoesNotMutatePointerReceiver`, `…NormalProducesUnitVector`, `TestQuaternion_ProductIdentity`, `TestQuaternion_NormalProducesUnitQuaternion`); **33 vacuous** incl. every Slerp/Product/Roll/Pitch/Yaw/Cross/Dot table.
- Correct: **mostly yes, with defects** (probe):
  - `Slerp(id, z90, 0.5)` = `[0 0 0.3827 0.9239]` ✔; shortest path with `−z90` ✔; `Vector` Slerp same ✔.
  - z90 Roll/Pitch/Yaw = `0, 0, 1.5708` ✔; `z90·z90` = `[0 0 1 0]` ✔; `Vector3D`/`Vector` Cross ✔.
  - `Vector.Product(Quaternion)` **panics**: `b.View().(Quaternion)` but `Quaternion.View()` returns `Vector` (`vec.go:261`, `quaternion.go:21-23`).
  - `NormalFast` error ≈0.15 % (`[3 4]` → `[0.599 0.799]`) — one Newton iteration; fine for control, not for renormalising quaternions repeatedly.
  - `Vector{0,0}.Normal()` returns `[0 0]` (no panic), whereas `Quaternion.Normal` panics on zero — inconsistent.
- Convention clash: dense `mat.Matrix.Orientation` interprets `Theta()` (w) as an angle and produces a wrong matrix; `Matrix3x3.Orientation` is correct — see [[mat-fixed-size-and-dense]].

**Embedded suitability:** float32, no allocations for fixed-size types; but interface-returning methods box `[N]float32` values (heap escape [INFERENCE]); `Vector` methods mutate in place (aliasing surprises); type assertions per op panic on mixed types.

## TODOs

- [ ] Fix `Vector.Product` to accept `Quaternion`/`*Quaternion`/`Vector` — x/math/vec/vec.go:261 #p1 #ER-021
- [ ] Rename or document `Theta()` (returns w) and fix its misuse in `mat.Matrix.Orientation` — x/math/vec/quaternion.go:222, x/math/mat/mat.go:263 #p2
- [ ] Fill the 33 empty table tests (Slerp, Product, Euler, Cross, Dot, Refract) — x/math/vec/vec_test.go:489-1431 #p2
- [ ] Consistent zero-norm policy across Vector/Quaternion — x/math/vec/vec.go:216, quaternion.go:204 #p3

## Links

- Module: [[mat-vec]]
- Related: [[mat-fixed-size-and-dense]], [[homogeneous-transforms]], [[kinematics]]

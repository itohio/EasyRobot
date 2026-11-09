---
type: algorithm
status: partial
tags: [easyrobot, mat, homogeneous, kinematics, se3]
workstream: W9
module: "[[mat-vec]]"
summary: "`HomogenousInverse`, `FromQuaternion`, `FromEuler` return zero rotation (value-receiver results…"
source: [x/math/mat/homogeneous.go, x/math/mat/mat4x4.go, x/math/mat/mat3x3.go]
tests: [x/math/mat/homogeneous_test.go, x/math/mat/mat4x4_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): TestMatrix4x4_Homogenous (2), TestMatrix4x4_HomogenousInverse (2 — assert only Rows/Cols of the product), TestMatrix4x4_Get/SetRotation/Translation, TestMatrix4x4_Col3D pass; probe InfraNotes/matprobe: HomogenousInverse, HomogenousFromQuaternion, HomogenousFromEuler return zero rotation"
---
# Homogeneous transforms (4×4 SE(3))

**What it does:** Build/decompose/invert 4×4 homogeneous transforms `[R t; 0 1]` on `Matrix4x4`: from R+t, from quaternion+t, from Euler (ZYX: R = Rz·Ry·Rx)+t, closed-form inverse `[Rᵀ −Rᵀt; 0 1]`, get/set rotation and translation, column extraction.

**Where:** `x/math/mat/homogeneous.go:19-28` `Homogenous(rot *Matrix3x3, t)`, `:34-38` `HomogenousFromQuaternion`, `:44-60` `HomogenousFromEuler`, `:68-94` `HomogenousInverse(dst)`, `:98-106` `SetRotation`, `:112-117` `SetTranslation`, `:123-128` `GetTranslation`, `:134-141` `GetRotation`, `:145-153` `Col3D`. Pointer receivers on `*Matrix4x4`; but the helpers call **value-receiver** `Matrix3x3` methods (`Orientation`, `RotationZ/Y/X`, `Mul`, `Transpose`, `MulVec`) and discard their return values.

**Inputs → outputs:** `Matrix3x3` rotation / `vec.Quaternion{x,y,z,w}` / Euler `Vector3D{roll(x), pitch(y), yaw(z)}` radians + `Vector3D` translation → `*Matrix4x4`.

## Completeness review

- Complete: yes (API surface).
- Tested: `Homogenous` with identity and translation (values); `HomogenousInverse` tests compute `M·M⁻¹` but assert **only dimensions** (`homogeneous_test.go:75-79, 97-101`) — value-vacuous. No test for `HomogenousFromQuaternion/FromEuler`.
- Correct: **no** (probe, R = RotZ(90°), t = (1,2,3)):
  - `HomogenousInverse` → rotation rows all 0, translation 0 (want row0 `[0 1 0 −2]`). Cause: `rotT.Transpose(&rot)` (`:79`) discards the returned value (value receiver), so `rotT` stays zero; `GetTranslation(trans)` (`:74`) is passed a value.
  - `HomogenousFromQuaternion(q90z, t)` → `[[0 0 0 1][0 0 0 2][0 0 0 3][0 0 0 1]]`: `rot3x3.Orientation(&rot)` (`:36`) result discarded.
  - `HomogenousFromEuler((0,0,π/2), t)` → same zero rotation: `rz.RotationZ`, `ry.RotationY`, `rx.RotationX`, `temp1.Mul`, `temp2.Mul` (`:47-56`) all discard results.
  - `Homogenous(&rot, t)` with a precomputed rotation ✔ (used by probe/inverse tests).
  - Full inverse alternative `Matrix4x4.Inverse` is also wrong for such matrices ([[matrix-inverse-and-lu]]); dense `Matrix.Inverse` is correct.

**Embedded suitability:** fixed-size stack values, no allocations, float32 — suitable for ESP32 once fixed.

## TODOs

- [ ] Assign results of value-receiver calls (`rotT = rot.Transpose(...)`, `rot3x3 = Matrix3x3{}.Orientation(rot)…`) in HomogenousInverse/FromQuaternion/FromEuler — x/math/mat/homogeneous.go:36,47-56,74-82 #p1 #ER-018
- [ ] Value-checking tests: M·M⁻¹ = I, FromEuler/FromQuaternion vs known matrices — x/math/mat/homogeneous_test.go:75-101 #p1 #ER-018
- [ ] Document Euler convention (ZYX intrinsic = roll-pitch-yaw) in API — x/math/mat/homogeneous.go:44 #p3

## Links

- Module: [[mat-vec]]
- Related: [[mat-fixed-size-and-dense]], [[matrix-inverse-and-lu]], [[vector-and-quaternion]], [[kinematics]]

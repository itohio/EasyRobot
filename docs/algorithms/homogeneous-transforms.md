---
type: algorithm
status: tested
tags: [easyrobot, mat, homogeneous, kinematics, se3]
workstream: W9
module: "[[mat-vec]]"
summary: "4×4 SE(3) build/decompose/invert; FromQuaternion/FromEuler/HomogenousInverse fixed and value-tested (ER-018)."
source: [x/math/mat/homogeneous.go, x/math/mat/mat4x4.go, x/math/mat/mat3x3.go]
tests: [x/math/mat/homogeneous_test.go, x/math/mat/mat4x4_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -race -count=1 ./x/math/mat/... → ok (2026-11-09): TestMatrix4x4_HomogenousKnownMatrices and TestMatrix4x4_HomogenousInverseKnown failed before the ER-018 fix (zero rotation, probe InfraNotes/matprobe) and pass; review probe: Euler order Rz·Ry·Rx and xyzw quaternion agree with Matrix3x3.Rotation*/Orientation [[ER-018-leg-helpers-gait-scheduler|ER-018]]"
source_revision: 1bf6234e676daa87d3e9eb191bb1dc2420f05ff9
---
# Homogeneous transforms (4×4 SE(3))

**What it does:** build/decompose/invert 4×4 homogeneous transforms `[R t; 0 1]` on `Matrix4x4`: from R+t, from quaternion+t, from Euler (ZYX: R = Rz·Ry·Rx)+t, closed-form inverse `[Rᵀ −Rᵀt; 0 1]`, get/set rotation and translation, column extraction.

**Where:** `x/math/mat/homogeneous.go`: `Homogenous(rot *Matrix3x3, t)`, `HomogenousFromQuaternion`, `HomogenousFromEuler`, `HomogenousInverse(dst)`, `SetRotation`, `SetTranslation`, `GetTranslation`, `GetRotation`, `Col3D`. Pointer receivers on `*Matrix4x4`. The helpers call value-receiver `Matrix3x3` methods (`Orientation`, `RotationZ/Y/X`, `Mul`, `Transpose`, `MulVec`); since ER-018 their return values are kept (before, they were discarded and every rotation came out zero).

**Inputs → outputs:** `Matrix3x3` rotation / `vec.Quaternion{x,y,z,w}` / Euler `Vector3D{roll(x), pitch(y), yaw(z)}` radians + `Vector3D` translation → `*Matrix4x4`.

## Completeness review

- Complete: yes (API surface).
- Tested: `TestMatrix4x4_HomogenousKnownMatrices` (FromQuaternion and FromEuler vs known matrices), `TestMatrix4x4_HomogenousInverseKnown` (inverse of R = RotZ(90°), t = (1,2,3) vs `[0 1 0 −2; …]` and M·M⁻¹ = I), `TestHomogenous_MatchesComposition` (general angles, in-place inverse), `TestHomogenous_NoAllocations` (0 per call; was 5 / 1 / 3 for FromEuler / FromQuaternion / HomogenousInverse through interface boxing, review K4), plus the earlier construction/get/set tests. The old `TestMatrix4x4_HomogenousInverse` still asserts only dimensions; the new tests cover values.
- Correct: yes. The only callers of the changed helpers are tests. The rotation → quaternion direction (`Matrix.Quaternion` returns the conjugate) is still wrong — tracked in [[dh-forward-kinematics]] (ER-021), not used here.

**Embedded suitability:** computed in place on fixed-size values (no interface calls), 0 allocations, float32. `HomogenousInverse(dst)` allows `dst == m`.

## TODOs

- [x] Assign results of value-receiver calls in HomogenousInverse/FromQuaternion/FromEuler — x/math/mat/homogeneous.go #p1 #ER-018
- [x] Value-checking tests: M·M⁻¹ = I, FromEuler/FromQuaternion vs known matrices — x/math/mat/homogeneous_test.go #p1 #ER-018
- [ ] Document Euler convention (ZYX intrinsic = roll-pitch-yaw) in API — x/math/mat/homogeneous.go:44 #p3

## Links

- Module: [[mat-vec]]
- Related: [[mat-fixed-size-and-dense]], [[matrix-inverse-and-lu]], [[vector-and-quaternion]], [[kinematics]]

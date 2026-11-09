---
type: module
group: "Tensor and NN"
status: partial
tags: [easyrobot, math, linear-algebra, kinematics]
workstream: [W3, W9]
packages: [x/math/mat, x/math/mat/types, x/math/vec, x/math/vec/types, x/math/shape, x/math]
test_result: pass
---
# mat-vec

**Purpose:** float32 small-matrix and vector math for robotics: fixed-size `Matrix2x2…4x4`, `Matrix3x4/4x3`, dense `Matrix [][]float32`, decompositions (LU/inverse, QR, SVD, Cholesky, NNLS/LDP, pseudo-inverse, damped least squares), homogeneous transforms, Jacobian columns; `Vector`, `Vector2D/3D/4D`, `Quaternion`; shape/stride helpers used by tensors. Consumed by [[kinematics]], [[filters]], [[motion-and-gait]].

**Packages (non-test LOC):**

| Package | Files | LOC | Notes |
|---|---|---|---|
| `x/math/mat` | 17 | 3665 | `mat.go` dense, `mat2x2…mat4x4.go`, `inverse.go`, `qr.go`, `svd.go`, `cholesky.go`, `nnls.go`, `pseudo_inverse.go`, `jacobian.go`, `homogeneous.go`, `smart_view.go`, **`benchmark_inverse.go` (non-test file importing `testing`)** |
| `x/math/mat/types` | 1 | 118 | matrix interfaces |
| `x/math/vec` | 6 | 2060 | `vec.go`, `vec2d/3d/4d.go`, `quaternion.go`, `smart_view.go` |
| `x/math/vec/types` | 1 | 112 | vector interfaces |
| `x/math/shape` | 3 | 348 | `Shape`, strides, `AdvanceOffsets*`, `MAX_DIMS=16` |
| `x/math` | 1 | 117 | `math.go`: `SQR`, `Pytag`, `Quad`, `Number`; `//go:generate go run ../../../cmd/codegen -i gen/vec.tpl …` points to a `cmd/codegen` and `gen/` templates that do **not exist** in the repo |

Declared docs: `mat/SPEC.md`, `vec/SPEC.md`, `x/math/SPEC.md`.

## Public API

- `mat.New(rows, cols, backing...)`, `Matrix.Inverse(dst)`, `.PseudoInverse(dst)`, `.DampedLeastSquares(...)`, `.LU`, `.Det`, `.Quaternion()`, `.Orientation()`, `QR`, `SVD`, `Cholesky`, `NNLS(A, B, dst, range)`, `LDP`; `CalculateJacobianColumn(jointPos, axis, eePos, revolute)`; fixed types with value-receiver methods.
- `vec.Vector` (`[]float32`), `Vector3D` etc. (arrays), `Quaternion [4]float32`; `SmartVector`/`VectorView` with refcount.
- `shape.Shape`, `ComputeStrides`, `IsContiguous`, `AdvanceOffsets(…)` (used by every strided kernel in [[tensor]], [[primitive-kernels]], [[iter-workers]]).

## Algorithms

<!-- vault:algorithms h=472106149636 -->

Counts: 5 tested · 0 implemented · 5 partial · 0 stub · 0 unverified (10 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[homogeneous-transforms]] | tested | pass | 4×4 SE(3) build/decompose/invert; FromQuaternion/FromEuler/HomogenousInverse fixed and value-tested (ER-018). |
| [[mat-cholesky]] | tested | pass | correct; re-factors and allocates on every solve |
| [[mat-nnls]] | tested | pass | exact solutions by probe; destroys caller's A |
| [[mat-numeric-jacobian]] | tested | pass | analytic geometric Jacobian column (not finite differences) |
| [[mat-svd]] | tested | pass | reconstruction value-tested; rows < cols unsupported |
| [[mat-fixed-size-and-dense]] | partial | pass | 59 of 145 mat tests are empty tables; `Matrix.Orientation` wrong (`mat.go:261-291`)… |
| [[mat-pseudo-inverse]] | partial | pass | fails for wide matrices (rows < cols, `la.go:760` Gesvd rejection) → DH IK with > 3 joints falls… |
| [[mat-qr]] | partial | pass | Q orthonormal but Q·R ≠ A (error 3–12, `qr.go:57,109`); tests check lengths only |
| [[matrix-inverse-and-lu]] | partial | pass | `Matrix4x4.Inverse` pivoted since ER-018; dense `LU` drops the permutation and dense `Det` divides by zero pivots (ER-021) |
| [[vector-and-quaternion]] | partial | pass | 33 of 61 vec tests empty; `Vector.Product(Quaternion)` panics; `Theta()` returns w, not an angle |

<!-- /vault:algorithms -->

## Build targets and dependencies

- float32 throughout (`chewxy/math32`) — good for ESP32 FPU; fixed-size types are arrays with value receivers → stack-allocated, TinyGo-friendly.
- Dense `Matrix` is `[][]float32` (row slices); `Flat()` uses `unsafe.Slice` when rows are contiguous (`mat.go:57-82`); dense decompositions allocate scratch per call (`inverse.go:60-62`, Cholesky).
- `benchmark_inverse.go` puts `testing` into production deps (`go list -deps ./x/math/mat` lists `testing`).
- `x/math/math.go:7-8` `go:generate` is dead (missing generator and templates); never run `go generate` in this repo (baseline: `drivers/manipulator` would flash a board).

## Tests and evidence

- `go test -count=1 -json ./x/math/mat/... ./x/math/vec/... ./x/math/shape/... ./x/math` (2026-11-09): mat 145 pass, vec 61 pass, shape 18 pass, x/math 4 pass; `mat/types`, `vec/types` no tests. Same as baseline.
- Many passes are vacuous: empty `//TODO: Add test cases` tables in `mat3x4_test.go`, `mat4x3_test.go` (20 each), `mat_full_test.go` (11), `vec_test.go` (20) — 59/145 mat and 33/61 vec tests check nothing (InfraNotes count).
- Value probes: `~/src/tmp/easyrobot/tensor/InfraNotes/{matprobe,matprobe2,vecprobe}`; kinematics cross-check in `docs/evidence/ER-R2-ik-review.md` (E1, E2).

## API notes (ease of use, composability)

- Good: fixed-size types are cheap values; dense and fixed types share method names; `dst` parameters avoid allocation.
- **Value-receiver trap:** several fixed-size methods modify or return a copy and the caller's result is thrown away (`homogeneous.go`), producing silent zero rotations — a pointer-receiver/return-value convention must be uniform and enforced by tests.
- Conventions undocumented at the API: quaternion order (w-first?) and handedness, Euler order, row- vs column-vector convention; `Quaternion()`'s sign bug shows the cost.
- Errors: decompositions return sentinel errors (`ErrNotSquare`, `ErrSingular`), good; but `ErrUnsupportedOperation` for wide pseudo-inverse is surprising for IK users (should transpose internally).

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Pivoted LU for `Matrix4x4.Inverse`; permutation in `Matrix.LU`; pivoted `Det` — x/math/mat/mat4x4.go:364, inverse.go:139, mat.go:564-596 #p1 #ER-021
- [ ] Fill the 92 empty table tests (mat + vec) or delete them — mat3x4_test.go:20, mat4x3_test.go:20, mat_full_test.go:64, vec_test.go:277 #p2
- [ ] Move `benchmark_inverse.go` into a `_test.go` file — x/math/mat/benchmark_inverse.go:1 #p3

## Inferred objectives (proposals)

- **Proposal:** before W3 gait work, make mat/vec a verified base: property tests (R·Rᵀ = I, T·T⁻¹ = I, Q·R = A, quaternion ↔ matrix round-trip) on random poses, and document frame/quaternion conventions in one place.
- **Proposal:** generate fixed-size types from one template again (restoring `cmd/codegen` or using generics) so fixes land in all sizes at once.

---
type: algorithm
status: partial
tags: [easyrobot, mat, linear-algebra, float32]
workstream: W9
module: "[[mat-vec]]"
summary: "59 of 145 mat tests are empty tables; `Matrix.Orientation` wrong (`mat.go:261-291`)…"
source: [x/math/mat/mat.go, x/math/mat/mat2x2.go, x/math/mat/mat3x3.go, x/math/mat/mat3x4.go, x/math/mat/mat4x3.go, x/math/mat/mat4x4.go, x/math/mat/smart_view.go, x/math/mat/types/types.go, x/math/mat/types_alias.go, x/math/mat/benchmark_inverse.go, x/math/math.go]
tests: [x/math/mat/mat_full_test.go, x/math/mat/mat2x2_test.go, x/math/mat/mat3x3_full_test.go, x/math/mat/mat3x4_test.go, x/math/mat/mat4x3_test.go, x/math/mat/mat4x4_test.go, x/math/mat/matrix_semantics_test.go, x/math/mat/mat_ik_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): x/math/mat 145/145 pass, of which 59 are vacuous (table has '//TODO: Add test cases', 0 subtests run: 24 in mat3x4_test.go, 24 in mat4x3_test.go, 11 in mat_full_test.go); probe InfraNotes/matprobe"
---
# Matrix types: dense `Matrix` and fixed-size Mat2x2…Mat4x4

**What it does:** float32 matrices for robotics/kinematics: dense `Matrix [][]float32` (row slices over one flat backing) and value-typed fixed arrays `Matrix2x2 [2][2]float32`, `Matrix3x3`, `Matrix3x4`, `Matrix4x3`, `Matrix4x4`, all implementing `matTypes.Matrix` (`x/math/mat/types/types.go:17-101`: Core, Rotations, Arithmetic, Multiplication, Factorization). Ref-counted wrappers `SmartMatrix`/`MatrixView` (`smart_view.go:10-93`, atomic counter, `Release` is a no-op for plain `Matrix`).

**Where:** `x/math/mat/mat.go:14` `Matrix`; `:29-40` `New` (allocates row-header slice + backing unless `len(backing)==rows*cols`); `:45-65` `IsContiguous` (pointer arithmetic), `:68-95` `Flat` (`unsafe.Slice` zero-copy when contiguous, else copy); `:100-106` `View` (allocates new row header); rotations `:222-259` (column-vector convention, RotZ = [c −s; s c]); `:261-291` `Orientation(q)`; `:496-516` `Mul` → `fp32.Gemm_NN`; `:529-553` `MulVec/MulVecT` → `fp32.Gemv_*`; `:557-578` `Det`; `:583-622` `LU`; `:626-648` `Quaternion`. Fixed-size files: `mat2x2.go:12`, `mat3x3.go:12` (`Orientation` `:85-110` Hamilton x,y,z,w — correct), `mat3x4.go:12`, `mat4x3.go:12`, `mat4x4.go:12`. Value receivers: setters return a new value (`TestMatrix3x3_SettersReturnNewValues`).

**Inputs → outputs:** float32 only; quaternions are `vec.Quaternion{x,y,z,w}` (w last). Interfaces return `matTypes.Matrix` (boxing fixed-size values into interfaces → heap alloc per call [INFERENCE: escape]).

## Completeness review

- Complete: **no** — `Matrix2x2.Quaternion` and `Matrix3x3.Quaternion` panic "unsupported operation" (`mat2x2.go:296`, `mat3x3.go:334`); `Matrix4x4.Quaternion` (`mat4x4.go:395`) delegates to the 3×3 one [INFERENCE: panics too].
- Tested: 145 pass; **59 vacuous** (zero subtests): all 24 Matrix3x4 and 24 Matrix4x3 tests (`mat3x4_test.go`, `mat4x3_test.go`, incl. `_Mul`, `_MulVec`, `_Orientation`, `_Quaternion`, `_Transpose`), and 11 in `mat_full_test.go`: `TestMatrix_Rotation2D/X/Y/Z`, `_Orientation`, `_SetRow`, `_SetCol`, `_SetDiagonal`, `_Det`, `_LU`, `_Quaternion`. 86 tests check values (Mat2x2/3x3/4x4 suites, `TestMatrix_Mul/Add/Transpose/…`).
- Correct: **no** (probe `tmp/easyrobot/tensor/InfraNotes/matprobe`, q = 90° about Z):
  - `Matrix3x3.Orientation(q)` → `[[0 −1 0][1 0 0][0 0 1]]` ✔; `Matrix4x3.Orientation` ✔ (delegates).
  - dense `Matrix.Orientation(q)` → `[[0.880 0.459 0][0.459 0.880 0][0 0 1]]` ✘ — treats `q[3]` as an angle (`Theta()` returns w) and uses symmetric off-diagonals (`mat.go:281-289`), not a rotation matrix.
  - `Matrix.Quaternion(RotZ(90°))` → `*vec.Quaternion{0 0 0.707 −0.707}` ✘ (−90°, sign of vector part flipped, `mat.go:632-643`); returns a pointer, and `Matrix.Orientation` type-asserts a value (`mat.go:262`) → round-trip **panics** (interface conversion).
  - `Matrix.Det([[0 1][1 0]])` → **NaN** (no pivoting, `mat.go:564-571`); want −1.

**Other defects**
- `x/math/mat/mat.go:505-514` — `Mul` uses `ldC = N`; a destination wider than N (or non-contiguous: `Flat` copies) writes the wrong layout/loses results [INFERENCE; non-contiguous square case passed in probe].
- `x/math/mat/benchmark_inverse.go:3-15` — non-test file imports `testing` and exports `Benchmark*` → `testing` linked into production deps (lead-reported via `go list -deps`).
- `x/math/mat/mat.go:42-43` `Release` no-op while `SmartMatrix` implies pooling.
- `x/math/math.go:7-8` `//go:generate go run ../../../cmd/codegen -i gen/vec.tpl …` — path resolves outside the repo and `x/math/gen/` does not exist; generator is gone. `x/math/math.go` `Quad`: for a==0 returns `b/c` (root of bx+c is −c/b); for b==0 returns `(t,t)` instead of `(±t)`. Only `TestClamp`, `TestFastISqrt`, `TestPytag`, `TestSQR` exist (pass).

**Embedded suitability:** float32 throughout (good for ESP32-S3 single-precision FPU); fixed-size types are stack values but the interface-returning API boxes them; dense `Matrix` allocates row headers in `New`/`View`, `unsafe` in `Flat`/`IsContiguous`; Det/Inverse helpers allocate work buffers; type assertions on interfaces everywhere (`a.View().(Matrix)`), panics on mismatch.

## TODOs

- [ ] Fix dense `Matrix.Orientation` to the Hamilton formula used by Matrix3x3 (accept value or pointer) — x/math/mat/mat.go:261-291 #p1 #ER-021
- [ ] Partial pivoting in `Matrix.Det` — x/math/mat/mat.go:564 #p1 #ER-021
- [ ] Fill the 59 empty table tests (Mat3x4/Mat4x3/dense rotations, Det, LU, Quaternion) — x/math/mat/mat3x4_test.go, mat4x3_test.go, mat_full_test.go:1309 #p2
- [ ] Move benchmark_inverse.go to `_test.go` — x/math/mat/benchmark_inverse.go:3 #p3
- [ ] Remove dead `//go:generate` lines or restore generator — x/math/math.go:7-8 #p3
- [ ] Fix `Quad` linear/±root cases and test it — x/math/math.go:37 #p2

## Links

- Module: [[mat-vec]]
- Related: [[matrix-inverse-and-lu]], [[homogeneous-transforms]], [[vector-and-quaternion]], [[kinematics]], [[primitive-kernels]]

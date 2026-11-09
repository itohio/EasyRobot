---
type: algorithm
status: partial
tags: [easyrobot, mat, inverse, lu, linear-algebra]
workstream: W9
module: "[[mat-vec]]"
summary: "`Matrix4x4.Inverse` silently wrong for a 90° rotation pose (no pivoting, `mat4x4.go:364`…"
source: [x/math/mat/inverse.go, x/math/mat/mat.go, x/math/mat/mat4x4.go, x/math/mat/mat3x3.go, x/math/mat/mat2x2.go]
tests: [x/math/mat/inverse_test.go, x/math/mat/mat4x4_test.go, x/math/mat/mat2x2_test.go, x/math/mat/mat3x3_full_test.go, x/math/mat/mat_full_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): TestMatrix_Inverse (5 subtests, M·M⁻¹≈I), TestMatrix2x2_Inverse (3), TestMatrix3x3_Inverse (2), TestMatrix4x4_Inverse (identity only), TestMatrix4x4_LUProducesFactors pass; TestMatrix_LU, TestMatrix_Det vacuous; probe InfraNotes/matprobe"
---
# Matrix inverse, determinant and LU

**What it does:** Square-matrix inversion and LU factorisation for dense and fixed-size float32 matrices.

**Where:**
- Dense `Matrix.Inverse` `x/math/mat/inverse.go:27-73`: copies into `work`, `fp32.Getrf_IP` (LU with partial pivoting, `ipiv`), `fp32.Getri` → dst. Destination may be `Matrix` or `*MatrixNxM` (panic message wrongly says "Matrix2x2" `:52`).
- `Matrix2x2.Inverse` `:76-97` and `Matrix3x3.Inverse` `:100-126`: closed-form adjugate/det, singular if |det| < `SingularityTolerance = 1e-6` (absolute, `:14`).
- `Matrix4x4.Inverse` `:129-168`: pivoted `Det` (`mat4x4.go:328-362`) for singularity, then **unpivoted Doolittle `LU`** (`mat4x4.go:364-393`) + forward/back substitution.
- Dense `Matrix.LU(L,U)` `x/math/mat/mat.go:583-622`: `fp32.Getrf` with pivoting but the **permutation `ipiv` is discarded**; fallback unpivoted Doolittle only on error.
- Dense `Matrix.Det` `mat.go:557-578`: Gaussian elimination without pivoting.

**Inputs → outputs:** n×n float32 → n×n inverse, `ErrNotSquare`/`ErrSingular`; LU writes L (unit lower), U.

## Completeness review

- Complete: yes for square inputs.
- Tested: dense inverse checked via `M·M⁻¹ ≈ I` (identity, known 2×2, rotation, singular, non-square); 2×2/3×3 likewise; `TestMatrix4x4_Inverse` only the identity; `TestMatrix4x4_LUProducesFactors` checks L·U = A on a diagonally dominant matrix (no pivot needed). `TestMatrix_LU`, `TestMatrix_Det` are empty tables (pass vacuously).
- Correct: **no** (probe `tmp/easyrobot/tensor/InfraNotes/matprobe`):
  - `Matrix4x4.Inverse` of homogeneous `[RotZ(90°) | t=(1,2,3)]` (zero at [0][0]) → err nil, row0 `[0 1 0 −2.727]`; correct `[0 1 0 −2]`. Silent wrong answer.
  - Dense `Matrix.Inverse` of the same → row0 `[−4e-8 1 0 −2]` ✔ (pivoted).
  - `Matrix.LU([[0 1][1 0]])` → L = I, U = I (L·U ≠ A; permutation lost).
  - `Matrix.Det([[0 1][1 0]])` → NaN.
  - `Matrix4x4.Inverse` of a rank-deficient matrix → `ErrSingular` ✔.

**Defects**
- `x/math/mat/mat4x4.go:364-393` + `x/math/mat/inverse.go:139` — no pivoting → wrong inverse for any 4×4 with a zero (or tiny) leading pivot, e.g. 90° rotations — common in kinematics.
- `x/math/mat/mat.go:595-596` — pivots discarded; LU returned for PA, not A.
- `x/math/mat/mat.go:564-571` — Det divides by zero pivot.
- `x/math/mat/inverse.go:78,102,131` — absolute det tolerance 1e-6 rejects well-conditioned small-scale matrices (e.g. 0.01·I₃, det 1e-6) and accepts ill-conditioned large ones.
- `x/math/mat/inverse.go:60-62` — allocates `work` and `ipiv` per call.

**Embedded suitability:** float32; fixed-size paths allocation-free; dense path allocates 2 buffers per call; interface type switches/panics on destination type.

## TODOs

- [ ] Use pivoted LU (or `HomogenousInverse` / closed form) in `Matrix4x4.Inverse`; add non-trivial 4×4 tests — x/math/mat/inverse.go:139, x/math/mat/mat4x4.go:364 #p1 #ER-018
- [ ] Return/apply the permutation in `Matrix.LU` (PA = LU API) — x/math/mat/mat.go:583-622 #p1 #ER-021
- [ ] Pivoting in dense `Det`; fill `TestMatrix_Det`/`TestMatrix_LU` — x/math/mat/mat.go:557, x/math/mat/mat_full_test.go:1309,1336 #p1 #ER-021
- [ ] Relative singularity test (scale by norm) — x/math/mat/inverse.go:14 #p2
- [ ] Accept caller scratch to avoid allocs — x/math/mat/inverse.go:60 #p3

## Links

- Module: [[mat-vec]]
- Related: [[mat-fixed-size-and-dense]], [[homogeneous-transforms]], [[mat-pseudo-inverse]], [[kinematics]]

---
type: algorithm
status: partial
tags: [easyrobot, filter, kalman]
workstream: W7
module: "[[filters]]"
summary: "Covariance aliasing kills filter after 1 update"
source: [x/math/filter/kalman/kalman.go]
tests: [x/math/filter/kalman/kalman_test.go]
test_result: pass
complete: yes
correct: no
evidence: "kalman tests 6/6 pass; probe P collapses to 0 after first update, estimate freezes"
---
# Linear Kalman filter

**What it does:** Discrete linear KF (predict `x=Fx(+Bu)`, `P=FPFᵀ+Q`; update with gain `K=PHᵀS⁻¹`, `P=(I−KH)P`) over `mat.Matrix`/`vec.Vector` (float32).

**Where:** `x/math/filter/kalman/kalman.go:10-45` (`Kalman`), `New` :54, `NewWithControl` :109, `Predict` :200, `PredictWithControl` :228, `UpdateMeasurement`/`update` :268-344, `Update` (Filter interface) :348-369.

**Inputs → outputs:** dims n (state), m (meas), k (control); F n×n, H m×n, B n×k, Q n×n, R m×m. `UpdateMeasurement(z m)` → state via `Output()`; `P` exported field. Implements `filter.Filter[vec.Vector, vec.Vector]`; `timestep` argument ignored (F must already include dt).

## Completeness review

- Complete: yes for textbook scope (predict/update/control). No Joseph-form update, no square-root/UD form, no gating, no handling of missing measurements other than "all-zero vector = no measurement" (:353-364).
- Tested: 6 tests (`kalman_test.go`) — single predict/update steps with loose bounds (e.g. position in [0.8,1.2], :58-72), reset, interface shape. None checks covariance or more than one update. Run: go test -count=1 -json ./x/math/filter/... (2026-11-09) → ok, 6 pass.
- Correct: **no**. `kalman.go:338` `copy(k.P, k.tempM)` copies *row slice headers* (`mat.Matrix` is `[][]float32`), so after the first update `P` aliases `tempM`; the next `zeroMatrix(k.tempM)` (:208, :307) zeroes `P`. Probe (scalar random-walk, q=0.01, r=1, 8 measurements): step 0 matches reference (x=0.5025, P=0.5025); from step 1 `P=0.0000` and `x` stays 0.5025 while reference converges to 0.90 (P=0.13). The filter is effectively dead after one update.
- Other defects: `:287` allocates the innovation vector every update; `:308` writes the n×m product `P·Hᵀ` into the n×n `tempM` (works only because `Gemm` packs with stride m and the following `K.Mul` reads with the same stride — fragile); `:312-315` silently substitutes identity when `S` is singular; `:353-364` treats a legitimate all-zero measurement as missing.

## TODOs

- [ ] Fix covariance aliasing: copy element-wise (`for i := range P { copy(P[i], tempM[i]) }`) — kalman.go:338 #p1 #ER-019
- [ ] Add multi-step test vs scalar reference (P and x after N updates) — kalman_test.go #p1 #ER-019
- [ ] Use a dedicated n×m temp for `P·Hᵀ` — kalman.go:308 #p2
- [ ] Preallocate innovation vector — kalman.go:287 #p3
- [ ] Explicit "no measurement" API instead of zero-vector sentinel; report singular S instead of identity fallback — kalman.go:312,353 #p2
- [ ] Optional Joseph-form update for numerical robustness in float32 #p3

## Links

- Module: [[filters]]
- Shares the aliasing defect with [[extended-kalman-filter]]; consumed by nothing else in-tree (EKF is used by [[ekf-raycast-localization]]).

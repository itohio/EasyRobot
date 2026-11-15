---
type: algorithm
status: partial
tags: [easyrobot, filter, kalman, ekf]
workstream: [W7, W8]
module: "[[filters]]"
summary: "Same aliasing; Jacobian at predicted state"
source: [x/math/filter/ekalman/ekalman.go]
tests: [x/math/filter/ekalman/ekalman_test.go]
test_result: pass
complete: yes
correct: no
evidence: "ekalman tests 6/6 pass; probe kfprobe: P collapses to 0 after first update"
---
# Extended Kalman filter

**What it does:** EKF with user-supplied nonlinear `f(x,u,dt)`, `h(x)` and optional analytic Jacobians; falls back to central-difference numerical Jacobians (`JacobianEpsilon`, default 1e-5).

**Where:** `x/math/filter/ekalman/ekalman.go`: types :11-24, `EKF` :28-73, `New` :89, `NewWithControl` :153, numerical Jacobians :259-281 and :299-321, `Predict` :334-358, `PredictWithControl` :362-390, `UpdateMeasurement` :395-465, `Update` :469-490.

**Inputs → outputs:** n, m, k dims; Q n×n, R m×m; callbacks return fresh `vec.Vector`/`mat.Matrix`. Output state vector; `P` exported. Implements `filter.Filter[vec.Vector, vec.Vector]`.

## Completeness review

- Complete: yes for the standard EKF loop; no iterated EKF, no UKF, no outlier gating.
- Tested: 6 tests — pendulum and numerical-Jacobian cases assert only that the state stays within ±1 or ±10 (`ekalman_test.go:85,95,137,146`); `TestEKF_WithControl` checks velocity after one control prediction to 1e-3 (:207). Run: go test -count=1 -json ./x/math/filter/... (2026-11-09) → ok, 6 pass.
- Correct: **no**.
  - Same covariance aliasing as [[linear-kalman-filter]]: `ekalman.go:459` `copy(e.P, e.tempM)` copies row headers → `P` becomes 0 at the next predict. Probe (`kfprobe`, identity f/h, numerical Jacobians): P=0.5025 after step 0, then 0.0000; x creeps 0.509→0.536 vs reference 0.90.
  - State Jacobian evaluated at the *predicted* state: `:336-340` (`copy(e.x, xPred)` then `computeStateJacobian(e.x, …)`), same at :368-372. Standard EKF linearises at the prior estimate x̂ₖ₋₁. Matters for strongly nonlinear f.
  - Numerical Jacobians allocate two vectors per state dimension per call (:265-270, :305-310) plus whatever `f`/`h` allocate; innovation allocated per update (:407).
  - `Update` treats an all-zero measurement as missing (:474-485).

## TODOs

- [ ] Fix covariance aliasing — ekalman.go:459 #p1 #ER-019
- [ ] Linearise f at the prior state (compute Jacobian before overwriting x) — ekalman.go:336-340,368-372 #p1 #ER-019
- [ ] Multi-step regression test vs float64 reference EKF (e.g. range-bearing tracking) — ekalman_test.go #p1 #ER-019
- [ ] Preallocate xPlus/xMinus and innovation buffers; allow `f`/`h` to write into dst — ekalman.go:265,305,407 #p2
- [ ] Replace zero-vector "no measurement" sentinel — ekalman.go:474 #p2

## Links

- Module: [[filters]]
- Used by [[ekf-raycast-localization]] (`x/math/filter/slam/slam.go:229`).

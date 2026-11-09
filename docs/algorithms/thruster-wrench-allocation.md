---
type: algorithm
status: tested
tags: [easyrobot, kinematics, thrusters, allocation]
workstream: W3
module: "[[kinematics]]"
summary: "Forward: sums thrust vectors, `r × F` moments and reaction torques into a body wrench and…"
source: [x/math/control/kinematics/thrusters/model.go, x/math/control/kinematics/thrusters/forward.go, x/math/control/kinematics/thrusters/inverse.go, x/math/control/kinematics/thrusters/helpers.go]
tests: [x/math/control/kinematics/thrusters/thrusters_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test thrusters → ok, 3 tests (2026-11-09): quadcopter hover forward/inverse values, infeasible-under-limits error"
---
# Thruster wrench allocation

**What it does:** Forward: sums thrust vectors, `r × F` moments and reaction torques into a body wrench and accelerations. Backward: least-squares allocation of a desired wrench to `[thrust, torque]` per thruster.

**Where:** `thrusters/model.go:71-112` (`Forward`), `:114-163` (`Backward`); `forward.go:8-41` (`applyForward`); `inverse.go:11-34` (`allocate`), `:51-66` (allocation matrix 6×2N), `:68-89` (`solveAllocation`), `:91-108` (damped fallback, λ=1e-3).

**Inputs → outputs:** `[thrust_i, torque_i]` 2N×1 ↔ wrench `[Fx Fy Fz Tx Ty Tz]`. Limits per thruster (`Range`).

## Completeness review

- Complete: yes for unconstrained allocation.
- Tested: `TestQuadcopterForwardHover` (wrench values), `TestQuadcopterInverseHover` (equal thrusts, zero torques, forward re-check), `TestInverseInfeasibleDueToLimits`. Pass (2026-11-09).
- Correct: yes within scope. Note: for N ≥ 4 thrusters the matrix is 6×2N (rows < cols), which `Gesvd` rejects (`primitive/fp32/la.go:760`), so the damped fallback always runs [INFERENCE from code; the hover tests pass through that path]. Limits are **not** honoured by optimisation: any saturation returns `ErrInfeasible` (`inverse.go:30`); `kinematics/SPEC.md:106` claims a limit-honouring damped allocation.

## TODOs

- [ ] Constrained allocation (e.g. `fp32.Gnnls` or box-constrained LS) instead of failing on saturation; fix SPEC claim — thrusters/inverse.go:30; kinematics/SPEC.md:106 #p2
- [ ] Skip the always-failing SVD path when rows < cols — thrusters/inverse.go:75 #p3

## Links

- Module: [[kinematics]] · [[rigid-body-velocity-wrench-model]] · depends on [[mat-pseudo-inverse]] (fails for rows<cols, so the damped path runs) and the pivoted dense `Matrix.Inverse` of JJᵀ (correct per [[matrix-inverse-and-lu]])

---
type: algorithm
status: tested
tags: [easyrobot, kinematics, dh, ik]
workstream: W3
module: "[[kinematics]]"
summary: "Damped least-squares position IK for any number of joints; limit-aware, restarts, typed errors, best-effort q, allocation-free (ER-017)."
source: [x/math/control/kinematics/joints/dh/denavithartenberg.go, x/math/control/kinematics/joints/dh/config.go, x/math/control/kinematics/types/types.go]
tests: [x/math/control/kinematics/joints/dh/denavithartenberg_test.go, x/math/control/kinematics/joints/dh/ik_robust_test.go, x/math/control/kinematics/joints/dh/config_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race -count=1 ./x/math/control/kinematics/... → ok (2026-11-09). Before: n>3 ErrUnsupportedOperation; probe R nominal seed 401/500; 156 allocs/call; review round: 455/2000 false ErrJointLimit, gantry in mm 0/300, NaN target wrote the previous answer. After: 300/300 redundant chains, probe R 500/500 (zero seed 500/500), 2000/2000 full-limit set, 0 allocs [[ER-017-dh-planar-ik-robustness|ER-017]]"
source_revision: 1bf6234e676daa87d3e9eb191bb1dc2420f05ff9
---
# DH damped least-squares IK

**What it does:** position-only iterative IK for a DH chain of any length: damped least-squares steps until `|e| < eps`, with joint limits respected, restarts out of local minima, and a typed reason on failure. Controls always receive the best in-limit pose found (except for invalid input).

**Method (since ER-017):**
- Step: `Δq = S·J'ᵀ (J' J'ᵀ + λ² I)⁻¹ e` with a 3×3 Cholesky solve, so n > 3 works. Prismatic columns are scaled by the chain length (`J' = J·S`) so damping is unit-independent.
- Adaptive λ, the Cartesian error clamped per step, a per-joint step cap, and angle wrapping into [Min, Max]. Joints sitting on a limit and pushed outward drop out of the step.
- Restarts (up to 4, sharing the iteration budget): an attempt that creeps along a limit (< 10 % distance gain per 8 iterations while a joint is blocked) restarts by flipping the elbow (negate the next parallel joint and re-aim the blocked one), re-aiming the blocked joint at the target, or pointing it away; a singular stall nudges the joints.
- Failure classification: a blocked failure gets up to 30 extra iterations with limits ignored. If they converge inside the limits, that solution is returned; if they converge (or get ≥ 4× closer) only past a limit → `ErrJointLimit`; if they end at a stationary point with a nonzero residual (|Jᵀe| ≤ 1e-2·‖J‖·|e|, e.g. the inner workspace hole) → `ErrUnreachable`; otherwise `ErrNoConvergence`. Beyond total link length or a stall with no blocked joint → `ErrUnreachable`. Every tuning constant is documented in `dh/SPEC.md`.
- All buffers are allocated in `New`; `Backward` does not allocate (when the caller passes `mattype.Matrix` values). The best-effort pose starts from this call's seed, so a float32-overflowing residual returns the seed, never a previous call's answer. One solver instance per goroutine/leg (preallocated workspace, not concurrency-safe).

**Where:** `dh/denavithartenberg.go` (`New`, `Backward`, `dlsStep`, restarts), `dh/config.go` (`Config.Validate`, `CalculateTransform`), `kinematics/types` (`ErrInvalidConfig`, `ErrUnreachable`, `ErrJointLimit`, `ErrNoConvergence`, `ErrNonFinite`).

**Inputs → outputs:** `dh.New(eps, maxIterations, cfg...) (*DenavitHartenberg, error)` (rejects invalid configs, e.g. zero-width limits); `state` n×1 seed, `destination` 7×1 target (rows 0–2 used), `controls` n×1 result. `maxIterations` counts every attempted step including rejected ones and restarts; worst case `maxIterations + 30` iterations per call. Non-finite target or seed → `ErrNonFinite`, controls untouched. Joint kinds Index 0–3 (θ, α, r, d) all work in the Jacobian.

## Completeness review

- Complete: no — orientation targets are still ignored (position only).
- Tested: `TestBackward_RedundantChains` (4-link planar, 6-DOF: 300/300 each), `TestBackward_AlphaAndLengthJoints` (200/200), `TestBackward_HexlegRealisticWorkspace` (probe R nominal and warm 500/500), `TestBackward_FullLimitsNoFalseJointLimit` (2000/2000, 0 `ErrJointLimit`), `TestBackward_TypedFailures` (F1/E2 → `ErrJointLimit`, E1 → `ErrUnreachable`, best-effort q in limits and closer than the seed), `TestBackward_SingularSeed`, `TestBackward_PrismaticUnitInvariance` (gantry/SCARA m vs mm equal), `TestBackward_NonFiniteInput`, `TestBackward_NoAllocations`, review fixes `TestBackward_OverflowResidualNoStaleAnswer`, `TestBackward_InnerHoleUnreachable` ((0,0,0), (0.039,0,0)), `TestBackward_BlockedClassification` (blocked local minimum → `ErrNoConvergence`; in-limit lifted branch → nil), `TestBackward_BestEffortIsMinimumResidual` (≤ a 49³ grid minimum), `TestConfig_Validate`, `TestNew_RejectsInvalidConfig`, finite-difference Jacobian check for all four joint kinds. Mutants D15/D16 and "best = last attempt" are killed. The lead's probe (30/60/90 mm, ±π limits, seed 0.1) went from 29/500 to 500/500.
- Correct: yes on these sets. Limitations: the `ErrJointLimit` check only searches near the best pose (2–3/2000 random-seed targets and 1/9000 nominal-seed targets still get a false `ErrJointLimit`; 8/500 on a 4-joint prismatic chain); classification of rare borderline targets can differ between normal and `-race` builds (floating-point codegen); inner-hole best effort is a local, not global, minimum.

## TODOs

- [x] Make IK work for n>3 (damped least squares) — denavithartenberg.go `dlsStep` #p1 #ER-017
- [x] Return best-effort `q` and a typed reason (unreachable / joint limit) instead of leaving `controls` untouched — denavithartenberg.go `Backward` #p2
- [x] Tests: 3-DOF leg cold/warm seeds, >3-DOF chain, joint-limit target — ik_robust_test.go #p2
- [x] Preallocate J/J⁺/Δq in the struct; no per-iteration allocation — denavithartenberg.go `New` #p3
- [ ] Optional orientation IK (6×n J, quaternion error) — denavithartenberg.go #p3
- [ ] Global check before reporting `ErrJointLimit` (rare false reports from random seeds) — denavithartenberg.go #p3

## Links

- Module: [[kinematics]] · [[dh-geometric-jacobian]] · [[dh-forward-kinematics]] · [[planar-3dof-leg-kinematics]] · [[leg-chain-config-and-servo-map]] · [[W3-legged-ik]] · [[mat-pseudo-inverse]] · [[mat-numeric-jacobian]]

---
type: algorithm
status: partial
tags: [easyrobot, kinematics, dh, ik]
workstream: W3
module: "[[kinematics]]"
summary: "partial (fails for >3 joints; fragile from a cold seed)"
source: [x/math/control/kinematics/joints/dh/denavithartenberg.go, x/math/mat/pseudo_inverse.go, x/math/primitive/fp32/la.go]
tests: [x/math/control/kinematics/joints/dh/denavithartenberg_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test dh → ok (2026-11-09); probe C/D/R/Z in [[ER-R2-ik-review]]: n>3 always ErrUnsupportedOperation; hexapod leg cold-seed convergence 37–81%, warm-seed 100% in ≤7 iterations"
---
# DH Jacobian pseudo-inverse IK

**What it does:** Position-only iterative IK: Newton steps `Δq = J⁺·(p_target − p)` until `|e| < eps`.

**Where:** `dh/denavithartenberg.go:99-119` (`Backward`), `:145-178` (`inverseInternal`), `:182-229` (`ikSolverJacobianPos`); `J.PseudoInverse` → `mat/pseudo_inverse.go:18` → `fp32.Gepseu` (`primitive/fp32/la.go:671`) → `Gesvd` (`la.go:755`).

**Inputs → outputs:** `state` n×1 seed, `destination` 7×1 target (only rows 0–2 used), `controls` n×1 result (written **only on success**). `eps` is a distance tolerance (compared as `|e|² < eps²`, `:164`); `maxIterations` steps. Errors: `ErrNoConvergence`, `kintypes.ErrUnsupportedOperation`.

## Completeness review

- Complete: no. Orientation target ignored (`loadEffector` reads it, nothing uses it). **Chains with more than 3 joints never solve**: J is 3×n and `Gesvd` requires rows ≥ cols (`la.go:760`), so `PseudoInverse` fails → `ErrUnsupportedOperation` (`:216-217`). Probe G1 (6-DOF) and G2 (4-link planar) fail; section C: 6-DOF 0/1500.
- Tested: `TestDenavitHartenberg_Backward` (2-link, targets (2,0,0) and (0,2,0) from zero seed, an unreachable and an inconsistent target expecting an error), `TestDenavitHartenberg_RoundTrip` (4 poses, seeded at the true pose). Pass (2026-11-09). Tests never use a 3-DOF leg, a >3-DOF chain, joint limits or a cold seed far from the answer.
- Correct: no (scope gaps + robustness). Probe C (500 reachable targets each; target = FK(q_true)):
  - hexapod leg 39/45/100 mm: zero seed 355/500, nominal seed (0, 0.3, −1.6) 184/500, warm seed (q_true ± 0.2 rad) **500/500, p90 3 iterations**.
  - realistic single-knee foot workspace (probe R): zero 404/500, nominal 401/500, warm 499/500; a reference damped least-squares solver (λ = 5 mm, step ≤ 0.25 rad) gets 441/449/500 on the same targets.
  - Causes: undamped full step with no step limit; the SVD cutoff is an absolute 1e-10 (`la.go:712`), meaningless in float32, so near-singular poses give huge steps; joints are clamped inside FK (`:126`) but J ignores the clamp, so iterates pin against a limit; stretched zero seed is singular (probe D1: stays at q=0). Failures return no best-effort `q`.
  - Allocates `J`, `Jinv`, `deltaParams` every iteration (`:184-185`, `:221`) plus pool buffers in `Gepseu`.

## TODOs

- [ ] Make IK work for n>3: use `mat.DampedLeastSquares` (pseudo_inverse.go:48) or SVD of Jᵀ — denavithartenberg.go:216 #p1 #ER-017
- [ ] Return best-effort `q` and a typed reason (unreachable / joint limit) instead of leaving `controls` untouched — denavithartenberg.go:110 #p2
- [ ] Tests: 3-DOF leg cold/warm seeds, >3-DOF chain, joint-limit target (failing-test-first, see [[ER-R2-ik-review]] ER-017) — denavithartenberg_test.go #p2
- [ ] Preallocate J/J⁺/Δq in the struct; no per-iteration allocation — denavithartenberg.go:184 #p3
- [ ] Optional orientation IK (6×n J, quaternion error) — denavithartenberg.go:243 #p3

## Links

- Module: [[kinematics]] · [[dh-geometric-jacobian]] · [[dh-forward-kinematics]] · [[planar-3dof-leg-kinematics]] · [[ER-R2-ik-review]] · [[W3-legged-ik]] · wide-matrix pinv failure / DLS alternative: [[mat-pseudo-inverse]] · FD Jacobian helper: [[mat-numeric-jacobian]]

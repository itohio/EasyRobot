---
type: algorithm
status: implemented
tags: [easyrobot, kinematics, dh, jacobian]
workstream: W3
module: "[[kinematics]]"
summary: "none direct"
source: [x/math/control/kinematics/joints/dh/denavithartenberg.go]
tests: []
test_result: none
complete: yes
correct: yes
evidence: "probe section B (tmp/easyrobot/ik, 2026-11-09): code J vs float64 analytic J max abs 4.4e-7, vs central FD (h=1e-2) max abs 5.5e-5 over 1000 random poses × 7 chains; no repo test targets J directly"
---
# DH geometric position Jacobian

**What it does:** Builds the 3×n position Jacobian of a DH chain used by the IK step.

**Where:** `dh/denavithartenberg.go:182-214` inside `ikSolverJacobianPos` (not exported; built fresh each IK iteration).

**Inputs → outputs:** cached `H0i` from the last FK → `J` (3×n, `mat.New`). Column i: revolute (`Index 0`) `z_i × (p_n − o_i)`; prismatic (`Index 3`) `z_i`, where `z_i`, `o_i` are column 2 and 3 of `H0i[i]` (the frame *before* joint i, i.e. the axis the joint acts about in standard DH). `Index 1/2` (α / r joints) → `ErrUnsupportedOperation` (`:212`). No orientation rows.

## Completeness review

- Complete: yes for revolute and prismatic joints; orientation rows (`ω_i = z_i` / 0) absent, so orientation IK is impossible.
- Tested: no repo test inspects J. Only exercised indirectly by `TestDenavitHartenberg_Backward`/`_RoundTrip` (2-link, reachable targets) → pass (2026-11-09).
- Correct: **yes**. Derivation in [[ER-R2-ik-review]] §2. Probe B, 1000 random poses per chain (hexapod legs 39/45/100 mm and 52/66/133 mm, planar 2- and 3-link, the repo test chains, revolute+prismatic, 6-DOF): |J_code − J_analytic(float64)| ≤ 4.4e-7; |J_code − FD| ≤ 5.5e-5 at h=1e-2 (≤3.1e-4 at h=1e-3, float32 cancellation). FD-vs-analytic error equals code-vs-FD error, so the residual is FD noise. The IK step the code applies equals float64 least squares `(JᵀJ)⁻¹Jᵀe` to ≤3.4e-4 relative on full-rank chains. **The owner's Jacobian is right; the problems are in the solver around it** ([[dh-jacobian-pseudoinverse-ik]]).

## TODOs

- [ ] Export `Jacobian(q, dst)` (allocation-free, 6×n with ω rows) and test it against FD — denavithartenberg.go:182 #p2
- [ ] Support α/r joint indices in J or reject them in `New` — denavithartenberg.go:212 #p3

## Links

- Module: [[kinematics]] · [[dh-forward-kinematics]] · [[dh-jacobian-pseudoinverse-ik]] · [[ER-R2-ik-review]]

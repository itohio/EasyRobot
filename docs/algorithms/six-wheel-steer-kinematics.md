---
type: algorithm
status: implemented
tags: [easyrobot, kinematics, wheels, rover]
workstream: W6
module: "[[kinematics]]"
summary: "Rocker-bogie style: front and rear pairs steer, middle pair fixed; same centre-referenced geometry…"
source: [x/math/control/kinematics/wheels/steer6/drive.go, x/math/control/kinematics/wheels/internal/rigid/math.go]
tests: []
test_result: none
complete: yes
correct: yes
evidence: "no tests; probe L (2026-11-09): v=1 ω=2 round trip exact with middle axle at x=0 and at x=0.05"
---
# Six-wheel rover steering kinematics (steer6)

**What it does:** Rocker-bogie style: front and rear pairs steer, middle pair fixed; same centre-referenced geometry as [[dual-axle-steer4-kinematics]].

**Where:** `wheels/steer6/drive.go:99-136` (`Forward`), `:138-186` (`Backward`), `:188-207` (`steeringFor`, `steerAngle`).

**Inputs → outputs:** `[ω_fl ω_fr ω_ml ω_mr ω_rl ω_rr δ_fl δ_fr δ_rl δ_rr]` ↔ `[v ω δ_fl δ_fr δ_rl δ_rr]`.

## Completeness review

- Complete: yes.
- Tested: no test file (2026-11-09).
- Correct: yes when the middle axle is at x=0 (probe L). With `MiddleOffset ≠ 0` the fixed middle wheels scrub (lateral velocity `ω·x_m`), yet the least-squares round trip still reports exact v, ω — no warning.

## TODOs

- [ ] Tests; warn or error when MiddleOffset ≠ 0 with fixed middle wheels — steer6/drive.go:39 #p3

## Links

- Module: [[kinematics]] · [[wheel-twist-least-squares]]

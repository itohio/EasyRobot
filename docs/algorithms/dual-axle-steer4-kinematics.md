---
type: algorithm
status: implemented
tags: [easyrobot, kinematics, wheels]
workstream: W6
module: "[[kinematics]]"
summary: "Front and rear steering, centre-referenced ICR: each wheel angle `atan(x/(R − y))`, `R = v/ω`."
source: [x/math/control/kinematics/wheels/steer4dual/drive.go, x/math/control/kinematics/wheels/internal/rigid/math.go]
tests: []
test_result: none
complete: yes
correct: yes
evidence: "no tests; probe L (2026-11-09): v=1 ω=2 → deltas (0.359 0.245 −0.359 −0.245) → back v=1.0000 ω=2.0000"
---
# Dual-axle 4-wheel steering kinematics (steer4dual)

**What it does:** Front and rear steering, centre-referenced ICR: each wheel angle `atan(x/(R − y))`, `R = v/ω`.

**Where:** `wheels/steer4dual/drive.go:94-126` (`Forward`), `:128-162` (`Backward`), `:164-183` (`steeringFor`, `steerAngle`).

**Inputs → outputs:** `[ω_fl ω_fr ω_rl ω_rr δ_fl δ_fr δ_rl δ_rr]` ↔ `[v ω δ_fl δ_fr δ_rl δ_rr]`.

## Completeness review

- Complete: yes (no steering-limit clamp; DESIGN lists it as optional).
- Tested: no test file (2026-11-09).
- Correct: yes by derivation (heading of a wheel at (x,y) for twist (v,ω) is `atan2(ωx, v−ωy) = atan(x/(R−y))`) and probe L round trip. Below `|v|,|ω| < 1e-6` steering returns 0 (pure spin not supported); `steerAngle` forces `den ≥ 1e-6`.

## TODOs

- [ ] Table tests (straight, turn, counter-phase, round trip) — wheels/steer4dual #p2
- [ ] Pure rotation (v=0) support and steering limits — steer4dual/drive.go:165 #p3

## Links

- Module: [[kinematics]] · [[wheel-twist-least-squares]] · [[ackermann-steer4-kinematics]] · [[six-wheel-steer-kinematics]]

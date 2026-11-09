---
type: algorithm
status: partial
tags: [easyrobot, kinematics, wheels, ackermann]
workstream: W6
module: "[[kinematics]]"
summary: "Front-steered 4-wheel chassis: wheel rates + two steering angles ↔ `[v, ω, δ_fl, δ_fr]`."
source: [x/math/control/kinematics/wheels/steer4/drive.go, x/math/control/kinematics/wheels/internal/rigid/math.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no tests; probe L (2026-11-09): v=1 round trip exact but steering angles disagree with rigid-body wheel velocity by 0.079 rad (ω=0.5) and 0.359 rad (ω=2)"
---
# Ackermann front-steer 4-wheel kinematics (steer4)

**What it does:** Front-steered 4-wheel chassis: wheel rates + two steering angles ↔ `[v, ω, δ_fl, δ_fr]`.

**Where:** `wheels/steer4/drive.go:99-127` (`Forward`, via [[wheel-twist-least-squares]]), `:129-163` (`Backward`), `:165-179` (`steeringFor`).

**Inputs → outputs:** params `[ω_fl ω_fr ω_rl ω_rr δ_fl δ_fr]` ↔ effector `[v ω δ_fl δ_fr]`. Steering: `R = |v/ω|`, inner/outer `atan(L/(R ∓ T/2))` (`:173-174`).

## Completeness review

- Complete: no. `FrontDrive`/`RearDrive` stored and never used (`:28-29`, `:54-55`): all four wheels always get rates. Panics on bad config instead of returning errors (`:43`, `:46`).
- Tested: no test file (2026-11-09).
- Correct: **no** (geometry mismatch). Steering angles assume the turn centre on the rear-axle line (`atan(L/…)`), but wheel positions are centre-referenced (`±L/2`, `:69-72`) and wheel rates come from a centre-referenced twist. Probe L: L=0.3, T=0.2, v=1: ω=0.5 → δ=(0.157, 0.142), max heading mismatch 0.079 rad; ω=2 → δ=(0.644, 0.464), mismatch 0.359 rad; rear wheels at heading 0 have lateral slip `ω·L/2`. `Forward(Backward(x))` still returns v, ω exactly because both sides use the same least-squares model. [[dual-axle-steer4-kinematics]] uses consistent centre-referenced geometry.

## TODOs

- [ ] Make steering and wheel-rate geometry share one reference point (rear axle or centre) — steer4/drive.go:69-72,173-174 #p1 #ER-038
- [ ] Implement or remove FrontDrive/RearDrive — steer4/drive.go:28 #p3
- [ ] Tests: wheel heading equals rigid-body velocity direction at each wheel — wheels/steer4 #p2

## Links

- Module: [[kinematics]] · [[wheel-twist-least-squares]] · [[dual-axle-steer4-kinematics]]

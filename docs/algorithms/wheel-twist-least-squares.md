---
type: algorithm
status: implemented
tags: [easyrobot, kinematics, wheels]
workstream: W6
module: "[[kinematics]]"
summary: "Fits `(v, ω)` to N steered wheels by solving the 2×2 normal equations of `wheel_i·r = cos δ_i·v +…"
source: [x/math/control/kinematics/wheels/internal/rigid/math.go]
tests: []
test_result: none
complete: yes
correct: yes
evidence: "no direct tests; exercised by probe L round trips of steer4/steer4dual/steer6 (exact, 2026-11-09)"
---
# Wheel-speed → chassis twist least squares

**What it does:** Fits `(v, ω)` to N steered wheels by solving the 2×2 normal equations of `wheel_i·r = cos δ_i·v + (x_i sin δ_i − y_i cos δ_i)·ω`.

**Where:** `wheels/internal/rigid/math.go:10-35` (`SolveTwist`), `:37-41` (`AssignWheelRates`), `:43-47` (`WheelLinearVelocity`).

**Inputs → outputs:** radius, wheel rad/s, headings (rad), wheel positions (m, x forward, y left) → `v` (m/s), `ω` (rad/s). Returns (0,0) when the normal matrix is singular (`|det| < 1e-6`, `:28`).

## Completeness review

- Complete: yes (no lateral-velocity term: assumes no side slip).
- Tested: indirectly only (probe L).
- Correct: yes — derivation matches `WheelLinearVelocity` (projection of rigid-body velocity onto the wheel heading); round trips exact. The singular fallback silently returns zero motion (e.g. all wheels at 90°).

## TODOs

- [ ] Unit test with known geometry; return an error instead of (0,0) when singular — rigid/math.go:28 #p3

## Links

- Module: [[kinematics]] · [[ackermann-steer4-kinematics]] · [[dual-axle-steer4-kinematics]] · [[six-wheel-steer-kinematics]]

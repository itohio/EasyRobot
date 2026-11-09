---
type: algorithm
status: tested
tags: [easyrobot, kinematics, wheels]
workstream: W6
module: "[[kinematics]]"
summary: "Four mecanum wheel rates ↔ chassis twist `[vx, vy, ω]` with fixed 3×4 / 4×3 matrices."
source: [x/math/control/kinematics/wheels/mecanum/drive.go]
tests: [x/math/control/kinematics/wheels/mecanum/drive_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test wheels/mecanum → ok, 3 tests/6 subtests (2026-11-09); matrices checked by hand: F·I = identity"
---
# Mecanum drive kinematics

**What it does:** Four mecanum wheel rates ↔ chassis twist `[vx, vy, ω]` with fixed 3×4 / 4×3 matrices.

**Where:** `wheels/mecanum/drive.go:63-85` (`Forward`), `:90-121` (`Backward`), `transform3x4` `:123`, `transform4x3` `:133`.

**Inputs → outputs:** `[ω_fl ω_fr ω_rl ω_rr]` rad/s → `[vx vy ω]`; forward `r/4·[1 1 1 1; 1 −1 −1 1; −c c −c c]`, `c = 2/(baseX+baseY)`; inverse `1/r·[1 1 −k; 1 −1 k; 1 −1 −k; 1 1 k]`, `k=(baseX+baseY)/2` (base = full wheel-centre distances, so `k = lx+ly` of the textbook form).

## Completeness review

- Complete: yes.
- Tested: `TestDriveForward`, `TestDriveBackward`, `TestDriveRoundTrip` (value checks). Pass (2026-11-09).
- Correct: yes — hand check: `F·I = I₃` (ω row: c·k = 1); tests pass. Allocation-free fixed-size matrices.

## TODOs

- [ ] Remove unused helpers `vec3FromArray`/`vec4FromArray` — mecanum/drive.go:148-154 #p3

## Links

- Module: [[kinematics]]

---
type: algorithm
status: implemented
tags: [easyrobot, kinematics, wheels]
workstream: W6
module: "[[kinematics]]"
summary: "Maps left/right wheel speeds ↔ chassis `[v, ω]`."
source: [x/math/control/kinematics/wheels/differential/drive.go]
tests: []
test_result: none
complete: yes
correct: yes
evidence: "no tests; probe L (2026-11-09): Backward(v=1,ω=0.5)→Forward round trip exact; wheel outputs are linear speeds, radius unused"
---
# Differential drive kinematics

**What it does:** Maps left/right wheel speeds ↔ chassis `[v, ω]`.

**Where:** `wheels/differential/drive.go:67-92` (`Forward`), `:94-125` (`Backward`).

**Inputs → outputs:** `[wl, wr]` → `[v, ω]`: `v = (wl+wr)/2`, `ω = (wr−wl)/B`; inverse `wl,r = v ∓ ω·B/2`.

## Completeness review

- Complete: yes.
- Tested: no test file (`go test` reports `[no test files]`, 2026-11-09).
- Correct: yes, self-consistent (probe L: v=1, ω=0.5 → wheels (0.95, 1.05) → back (1.000, 0.500)). But doc comments say "wheel rates" while values are **linear speeds**; `wheelRadius` is stored and never used (`:11`, `:27`). For rad/s you must divide by r yourself (probe: rad/s would be 19/21).

## TODOs

- [ ] Use wheel radius (rad/s in/out) or rename/document as linear speeds — differential/drive.go:80,116 #p2
- [ ] Add table tests (straight, spin, round trip) — wheels/differential #p2

## Links

- Module: [[kinematics]] · [[mecanum-drive-kinematics]]

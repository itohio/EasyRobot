---
type: algorithm
status: partial
tags: [easyrobot, motion, gait, legged, trajectory]
workstream: W3
module: "[[motion-and-gait]]"
summary: "Per-leg foot curve per sub-phase; swing apex = SwingHeight, monotonic touchdown (ER-018). Not allocation-free; frame unspecified."
source: [x/math/control/motion/gait/support/path_planner.go, x/math/control/motion/gait/types/contracts.go]
tests: [x/math/control/motion/gait/support/path_planner_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race -count=1 ./x/math/control/motion/gait/support → ok (2026-11-09); before the fix TestSwingApexEqualsSwingHeight: apex 0.1633 m vs 0.0800, TestTouchdownMonotonicDescent: z rises 0.0800→0.0840, TestLiftStartsAtStanceHeight: overshoot 0.0504 > 0.0500, support-p3-before.txt) [[ER-018-leg-helpers-gait-scheduler|ER-018]]"
source_revision: 1bf6234e676daa87d3e9eb191bb1dc2420f05ff9
---
# Gait support endpoint planner (foot trajectory per phase)

**What it does:** per leg, given an externally supplied sub-phase (`Lift`, `Swing`, `Touchdown`, `Support`) and progress ∈ [0,1], interpolates the foot from the pose at phase entry to a phase target: lift = +LiftHeight, swing = foothold + LiftHeight with an apex SwingHeight above the higher of lift-off and foothold, touchdown = foothold + TouchdownClearance (+ terrain height), support = foothold. Finite-difference velocity; contact flag from phase.

**Curves (since ER-018):** lift and touchdown use `smoothShape`, a vertical minimum-jerk move s(τ) = 10τ³ − 15τ⁴ + 6τ⁵. Swing uses `swingShape`: minimum-jerk in x/y; in z a rise to the apex and a fall, each half minimum-jerk, apex reached exactly at τ = 0.5. Every sub-phase boundary has zero velocity (C¹). `SupportProfile.Normalize` clamps LiftHeight ≤ SwingHeight and TouchdownClearance ≤ LiftHeight, so touchdown always descends ([[2026-11-09-gaits-w3]] §3).

**Not used by the walker (review G7):** nothing in the repo imports `gait/support`; `gait/walk` has its own allocation-free swing curve. The two define `SwingHeight` differently: here it is the apex above the higher of lift-off and foothold height (after a vertical `LiftHeight` segment); in `walk` it is the apex of a cosine bump above the straight lift-off→touchdown line. Recorded in `gait/DESIGN.md`; sharing one profile is a TODO.

**Where:** `gait/support/path_planner.go` (`Update`, `syncSubPhase`, terrain modes, targets, `smoothShape`, `swingShape`); height semantics documented in `gait/types/contracts.go` (`SupportProfile`).

**Inputs → outputs:** `SupportUpdateRequest{LegID, DesiredPose, Phase{Mode, Progress}, Timestamp, Delta, Observations}` → `FootState{Pose, Velocity, Contact, Phase, PhaseProgress}`. Phase mode + progress can come from [[gait-phase-scheduler]] (`LegState.PhaseState`). The frame of `DesiredPose` is still not specified.

## Completeness review

- Complete: no — no stride/foothold computation (that lives in [[gait-foothold-planner]], which uses its own allocation-free swing curve), no `Preview`, no reach/velocity limits, no mid-swing re-targeting (DESIGN claims these). `interpolateQuat` may allocate, so the planner is not allocation-free.
- Tested: `path_planner_test.go`, 7 profiles (including stance at z = −0.08, LiftHeight > SwingHeight, LiftHeight 0, TouchdownClearance > LiftHeight): apex = SwingHeight ± 1 mm, lift starts at stance height without overshoot, monotonic touchdown ending at foothold + clearance, no jump at any phase boundary.
- Correct: yes for the curve shapes above. Contact during touchdown is still assumed at progress ≥ 0.5 (`determineContact`).

## TODOs

- [x] Fix swing profile: single lift–swing–touchdown curve with apex = SwingHeight and monotonic descent; tests from probe K (ER-018) — path_planner.go `smoothShape`/`swingShape` #p1 #ER-018
- [ ] Specify the frame of DesiredPose/FootState (body vs world) and make support phase consistent — gait/types/contracts.go:50 #p2
- [ ] Unit tests for phase wrap and variable dt — gait/support #p2
- [ ] Remove per-call allocation (`interpolateQuat`) so the planner can run per tick on an MCU — path_planner.go #p3
- [ ] Share one swing profile (and one `SwingHeight` definition) between `gait/support` and `gait/walk` — path_planner.go, walk/walk.go #p3
- [ ] Contact from observations only when available; do not assume contact at 50 % touchdown — path_planner.go `determineContact` #p3

## Links

- Module: [[motion-and-gait]] · [[gait-phase-scheduler]] · [[gait-foothold-planner]] · [[gait-rigidbody-planner]] · [[interpolation]] · [[W3-legged-ik]]

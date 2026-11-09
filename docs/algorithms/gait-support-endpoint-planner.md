---
type: algorithm
status: partial
tags: [easyrobot, motion, gait, legged, trajectory]
workstream: W3
module: "[[motion-and-gait]]"
summary: "partial (clearance stacking, touchdown rises)"
source: [x/math/control/motion/gait/support/path_planner.go, x/math/control/motion/gait/types/contracts.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no tests; probe K (2026-11-09): lift peaks 0.131 m for LiftHeight 0.05, swing peaks 0.160 m for SwingHeight 0.08, touchdown rises 0.080→0.163 before descending"
---
# Gait support endpoint planner (foot trajectory per phase)

**What it does:** Per leg, given an externally supplied sub-phase (`Lift`, `Swing`, `Touchdown`, `Support`) and progress ∈ [0,1], interpolates the foot from the pose at phase entry to a phase target: lift = +LiftHeight, swing = desired + SwingHeight, touchdown = desired (+ terrain height), support = desired. Lift/swing/touchdown use a quadratic "arc" through an apex; support is linear. Finite-difference velocity; contact flag from phase.

**Where:** `gait/support/path_planner.go:96-129` (`Update`), `:150-181` (`syncSubPhase`), `:183-208` (terrain modes), `:227-251` (targets), `:366-384` (`arcShape`, `quadraticThroughMidpoint`), `:386-394` (default shapes). Contracts in `gait/types/contracts.go`.

**Inputs → outputs:** `SupportUpdateRequest{LegID, DesiredPose, Phase{Mode, Progress}, Timestamp, Delta, Observations}` → `FootState{Pose, Velocity, Contact, Phase, PhaseProgress}`. Frame of `DesiredPose` is not specified (world in `gait/DESIGN.md`; the support phase moves the foot, which only makes sense in the body frame).

## Completeness review

- Complete: no — no phase generation (caller must supply mode + progress), no stride/foothold computation, no `Preview`, no reach/velocity limits, no mid-swing re-targeting (DESIGN claims all of these). Terrain hook and per-phase shape override (`WithPhaseShape`) exist.
- Tested: no test file (IMPLEMENTATION_PLAN.md lists table tests that do not exist).
- Correct: **no** for default clearances. `arcShape` adds its amplitude (= max(Lift, Swing) = 0.08) on top of the phase target height (`:370`), so heights stack: probe K (ground 0, stride 0.1 m): lift peak 0.131, swing peak 0.160, and touchdown first **rises** from 0.080 to 0.163 before descending to 0 — a foot that lifts before landing. Contact during touchdown is assumed at progress ≥ 0.5 (`:266`).

## TODOs

- [ ] Fix swing profile: single lift–swing–touchdown curve with apex = SwingHeight and monotonic descent; tests from probe K (ER-018) — path_planner.go:366-377,386-394 #p1 #ER-018
- [ ] Specify the frame of DesiredPose/FootState (body vs world) and make support phase consistent — gait/types/contracts.go:50 #p2
- [ ] Unit tests for each sub-phase, phase wrap, variable dt — gait/support #p2
- [ ] Contact from observations only when available; do not assume contact at 50 % touchdown — path_planner.go:266 #p3

## Links

- Module: [[motion-and-gait]] · [[gait-phase-scheduler]] · [[gait-rigidbody-planner]] · [[interpolation]] (Bezier for swing curves) · [[W3-legged-ik]]

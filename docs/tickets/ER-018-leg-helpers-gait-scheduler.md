---
id: ER-018
title: "Leg/joint helpers, gait scheduler, strides and gait→IK→servo glue"
type: ticket
status: done
kind: implementation
owner: easyrobot
workstream: W3
milestone: M-a
maps: [T-IK-01, T-IK-07, T-IK-08, T-IK-09, T-IK-10]
tags: [easyrobot, ticket, w3, gait, kinematics]
created: 2026-11-09
updated: 2026-11-09
source_revision: 1bf6234e676daa87d3e9eb191bb1dc2420f05ff9
---
# ER-018 — Leg/joint helpers, gait scheduler, strides and gait→IK→servo glue

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

Everything M-a's host simulation needs on top of the IK, widened per order changes O1 and O2 ([[roadmap#2.1 The agreed order, and where it should change|roadmap §2.1]]). Accepted ER-R2 recommendation 2: a minimal phase scheduler first (tripod/wave/ripple, walk/trot). Scope:

1. **Joint/leg helper package** `x/math/control/kinematics/chain` per IK review §5 (evidence: `~/src/docs/evidence/ER-R2-ik-review.md#5. Joint-configuration helpers (proposed API sketch)`, outside the repo): `Joint`, `ServoMap`, `Chain.Validate/DH/Planar3/Model/ToServo/FromServo/CheckZeroPose`, `LegGeometry`, `Hexapod`, `Quadruped`, `Body.FootInLeg`, `Body.SolveLegs`; radians/metres internally, degrees only in `ServoMap` (accepted decision 3). Today the only servo mapping is `drivers/manipulator/main.go:333`.
2. **Gait phase scheduler** filling the stub `x/math/control/motion/gait/types/contracts.go:155-196` (`GaitTemplate.DutyCycles`, `PhaseMap` declared only; [[gait-phase-scheduler]]), presets tripod/wave/ripple and walk/trot ([[2026-11-09-gaits-w3]] §1: phase scheduler, not CPG).
3. **Swing profile**: clearances stack and touchdown rises before landing (E16) — `gait/support/path_planner.go:366-377`, `:386-394`, `:233-243` ([[gait-support-endpoint-planner]]).
4. **Strides/footholds with body→leg transforms** (O1): foothold = neutral + v·T_stance/2 + ω×r; `gait/support` currently takes `DesiredPose` from the caller (IK review §6 (evidence: `~/src/docs/evidence/ER-R2-ik-review.md#6. Hexapod/quadruped gap analysis`, outside the repo)).
5. **Gait → leg IK → servo glue** (O1): scheduler → footholds → swing → analytic IK → `ServoMap` → servo array, allocation-free per tick.
6. **VAJ1D fixes** (E14, E15): `x/math/control/motion/vaj1d.go:59` (`x1 < 1` should be `< 0`), velocity snap `:92-96`; drop the workaround at `motion/rigidbody/motion.go:427` ([[vaj1d-jerk-limited-profile]]).
7. **Two ER-021 transform fixes pulled forward** (O2): homogeneous helpers return zero rotation because value-receiver results are discarded — `x/math/mat/homogeneous.go:36`, `:47-56`, `:74-82`; `Matrix4x4.Inverse` unpivoted — `x/math/mat/mat4x4.go:364`, `x/math/mat/inverse.go:139` ([[homogeneous-transforms]], [[matrix-inverse-and-lu]]).

## Dependencies and scope

Dependencies: [[ER-017-dh-planar-ik-robustness|ER-017]] (analytic leg IK with typed errors; may run in parallel, the helper package consumes it). Feeds [[ER-037-body-pose-static-stability|ER-037]] (stability, M-a stage 3) and [[ER-039-manipulator-integration|ER-039]]. Device stage 2 needs [[ER-025-device-fixes|ER-025]] and the TinyGo stage of [[ER-026-build-green|ER-026]].

Not in scope: No ESP32 run, no CPG, no gait transitions, no support polygon (ER-037), no terrain adaptation.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [x] **T-IK-01** — `TestVAJ1D_ReachesTarget`, `TestVAJ1D_ReversalStopsWithinStoppingDistance`, review round `TestVAJ1D_StepProfile` (12 cases: exact landing, ≤ 1.5× time-optimal, FD |a| ≤ 1.05·maxA, jerk ≤ 1.05·maxJ). Fix: signed state, jerk bisection with a time-optimal brake check, float32 carry; workaround removed and `TestForwardStraightPathReachesCruiseSpeed` added (rigidbody now reaches MaxSpeed). [[vaj1d-jerk-limited-profile]], [[rigidbody-path-follower]].
- [x] **T-IK-07** — `TestHexlegDHMatchesPlanar3` (< 1e-6 m, 2000 poses), `TestServoRoundTrip`, `TestValidate` (19 cases), `TestFootInLegInvertsForwardTransform`, `TestSolveLegsReproducesJointAngles` (0 allocs). New package `kinematics/chain` ([[leg-chain-config-and-servo-map]]).
- [x] **T-IK-08** — `TestSwingApexEqualsSwingHeight` (0.1633 → 0.0800 ± 1 mm), `TestTouchdownMonotonicDescent`, `TestLiftStartsAtStanceHeight`, `TestSwingCycleContinuity`. Fix: minimum-jerk lift/touchdown, apex-exact swing, `Normalize` clamps ([[gait-support-endpoint-planner]]).
- [x] **T-IK-09** — `TestTripod`, `TestWave`, `TestRipple`, `TestTrot`, `TestWalk`, `TestPhaseWraps`, `TestAdvanceDoesNotAllocate` and 5 more on a 4096-point phase grid. New package `gait/scheduler` ([[gait-phase-scheduler]]).
- [x] **T-IK-10** — `TestStraightWalk`, `TestTurnInPlace`, `TestCrabWalk`: stance drift ≤ 9.5e-8 m per run; `TestClipScalesTwist`. New package `gait/walk` ([[gait-foothold-planner]]).
- [x] O2 transforms — `TestMatrix4x4_HomogenousKnownMatrices`, `TestMatrix4x4_HomogenousInverseKnown`, `TestMatrix4x4_InverseNeedsPivoting` failed first, pass ([[homogeneous-transforms]], [[matrix-inverse-and-lu]]).
- [x] Glue — `TestHexapodTripodSim`: 15 s (tripod walk + turn in place), every foot reachable (min margin 9.1 mm), every joint inside its limits, 0 IK errors, `AllocsPerRun` = 0 per tick; `TestQuadrupedWalkSim`.

## Effort uncertainty and revision trigger

Product's guess: M-a stage 1 **M–L** with ER-017; this ticket ≈ **M–L** after widening (O1/O2). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

Report: ER-017-018 (evidence: `~/src/docs/evidence/ER-017-018.md`, outside the repo). Open: joint speed is reported, not limited (hexapod sim peaks 8.0 rad/s against a 5.24 rad/s servo) — TODO in [[gait-foothold-planner]].

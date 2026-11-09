---
id: ER-018
title: "Leg/joint helpers, gait scheduler, strides and gait→IK→servo glue"
type: ticket
status: ready
kind: implementation
owner: easyrobot
workstream: W3
milestone: M-a
maps: [T-IK-01, T-IK-07, T-IK-08, T-IK-09, T-IK-10]
tags: [easyrobot, ticket, w3, gait, kinematics]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-018 — Leg/joint helpers, gait scheduler, strides and gait→IK→servo glue

[[tickets/index|Ticket index]] · [[roadmap]] · [[ER-R2]] · [[todo]]

## Goal and evidence

Everything M-a's host simulation needs on top of the IK, widened per order changes O1 and O2 ([[roadmap#2.1 The agreed order, and where it should change|roadmap §2.1]]). Accepted ER-R2 recommendation 2: a minimal phase scheduler first (tripod/wave/ripple, walk/trot). Scope:

1. **Joint/leg helper package** `x/math/control/kinematics/chain` per [[ER-R2-ik-review#5. Joint-configuration helpers (proposed API sketch)|IK review §5]]: `Joint`, `ServoMap`, `Chain.Validate/DH/Planar3/Model/ToServo/FromServo/CheckZeroPose`, `LegGeometry`, `Hexapod`, `Quadruped`, `Body.FootInLeg`, `Body.SolveLegs`; radians/metres internally, degrees only in `ServoMap` (accepted decision 3). Today the only servo mapping is `drivers/manipulator/main.go:333`.
2. **Gait phase scheduler** filling the stub `x/math/control/motion/gait/types/contracts.go:155-196` (`GaitTemplate.DutyCycles`, `PhaseMap` declared only; [[gait-phase-scheduler]]), presets tripod/wave/ripple and walk/trot ([[2026-11-09-gaits-w3]] §1: phase scheduler, not CPG).
3. **Swing profile**: clearances stack and touchdown rises before landing (E16) — `gait/support/path_planner.go:366-377`, `:386-394`, `:233-243` ([[gait-support-endpoint-planner]]).
4. **Strides/footholds with body→leg transforms** (O1): foothold = neutral + v·T_stance/2 + ω×r; `gait/support` currently takes `DesiredPose` from the caller ([[ER-R2-ik-review#6. Hexapod/quadruped gap analysis|IK review §6]]).
5. **Gait → leg IK → servo glue** (O1): scheduler → footholds → swing → analytic IK → `ServoMap` → servo array, allocation-free per tick.
6. **VAJ1D fixes** (E14, E15): `x/math/control/motion/vaj1d.go:59` (`x1 < 1` should be `< 0`), velocity snap `:92-96`; drop the workaround at `motion/rigidbody/motion.go:427` ([[vaj1d-jerk-limited-profile]]).
7. **Two ER-021 transform fixes pulled forward** (O2): homogeneous helpers return zero rotation because value-receiver results are discarded — `x/math/mat/homogeneous.go:36`, `:47-56`, `:74-82`; `Matrix4x4.Inverse` unpivoted — `x/math/mat/mat4x4.go:364`, `x/math/mat/inverse.go:139` ([[homogeneous-transforms]], [[matrix-inverse-and-lu]]).

## Dependencies and scope

Dependencies: [[ER-017-dh-planar-ik-robustness|ER-017]] (analytic leg IK with typed errors; may run in parallel, the helper package consumes it). Feeds [[ER-037-body-pose-static-stability|ER-037]] (stability, M-a stage 3) and [[ER-039-manipulator-integration|ER-039]]. Device stage 2 needs [[ER-025-device-fixes|ER-025]] and the TinyGo stage of [[ER-026-build-green|ER-026]].

Not in scope: No ESP32 run, no CPG, no gait transitions, no support polygon (ER-037), no terrain adaptation.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] **T-IK-01** — failing test first: 0→0.5 ends within 1e-3; 0→100 ends within 1e-3; FD |a| ≤ 1.05·maxA; reversal never exceeds the reversal position + stopping distance. Fix: `vaj1d.go:59` `x1<0`; remove velocity snap `:92-96`; signed state; drop `motion.go:427` workaround.
- [ ] **T-IK-07** — failing test first: hexleg `Chain.DH()` FK == `Planar3()` FK; `ToServo`/`FromServo` round trip; `Validate` catches bad limits and servo range. Fix: §5 API.
- [ ] **T-IK-08** — failing test first: apex == SwingHeight ± 1 mm; touchdown z monotonic non-increasing; lift starts at the stance height. Fix: rewrite `arcShape`/targets (Bezier from [[interpolation]]).
- [ ] **T-IK-09** — failing test first: tripod: always exactly 3 stance legs, offsets {0, ½}; wave: ≥5 stance legs; trot: diagonal pairs. Fix: implement `GaitScheduler` with presets.
- [ ] **T-IK-10** — failing test first: straight walk: stance feet fixed in the world (< 1 mm drift per cycle); turn in place. Fix: foothold generator + `Body.FootInLeg`.
- [ ] O2 transforms: failing value tests M·M⁻¹ = I for a non-trivial 4×4 and `FromEuler`/`FromQuaternion` vs known matrices (`x/math/mat/homogeneous_test.go:75-101`), then the fixes.
- [ ] Glue: a host simulation of a hexapod tripod over a full cycle — every foot reachable, every joint inside its limits, `testing.AllocsPerRun` = 0 for one tick (M-a stage 1 checkpoint, [[roadmap#M-a — Hexapod / quadruped walking (W3)|roadmap M-a]]).

## Effort uncertainty and revision trigger

Product's guess: M-a stage 1 **M–L** with ER-017; this ticket ≈ **M–L** after widening (O1/O2). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

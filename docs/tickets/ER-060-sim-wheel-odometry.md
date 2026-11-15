---
id: ER-060
title: "Sim S1: wheel odometry — encoder ticks → 2D pose from the robot description"
type: ticket
status: proposed
kind: implementation
owner: easyrobot
workstream: [W3, W6]
milestone: M-d
tags: [easyrobot, ticket, simulation, sim-s1]
created: 2026-11-12
updated: 2026-11-12
---
# ER-060 — Sim S1: wheel odometry — encoder ticks → 2D pose from the robot description

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`)

**Brick:** dead-reckoning from wheel encoders. It is a robot brick, used by the sim base and the hardware base alike.

## Goal and evidence

Design review (accepted): odometry was inside the sim adapter, but it is exactly what the real robot needs too. It moves here, next to the kinematics, so it reads only encoder ticks and the robot's geometry.

Decisions (2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule: primitives like the others (gait, matrix ops, filters), interface-based where performance allows, built on graphs, sets and other primitives; simple, solid, single responsibility; bricks first, then the house.

## Interface sketch

Package `x/robot/wheeled/odometry` (or next to `wheels/differential`; the implementer picks):

```go
type Odometry struct { /* geometry, last ticks, pose */ }
func New(w model.Wheeled /* x/robot/model */, countsPerRev int) *Odometry
func (o *Odometry) Update(ticks []int32, out *wheeled.Pose2D) error // integrates the tick deltas since the last call
```

## Dependencies and scope

Dependencies: ER-052, ER-055 and ER-057 (the over-the-fake acceptance case), ER-059 (`Pose2D`); ER-038 (wheel rates in rad/s).

- Uses the differential kinematics (`differential.Forward`) for the tick-delta → twist step; no second copy of the kinematics.
- Handles counter wrap; typed error for a wrong wheel count.
- No allocation per call.

Not in scope: fusion with the IMU (a filter brick, later); ground truth (the sim's, used only to measure drift).

Source/test edits are approved once this ticket is scheduled. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] Synthetic ticks for a straight 2 m: pose x within 1 mm of 2 m at 360 counts/rev and the description's radius.
- [ ] Synthetic ticks for a turn in place of 2π: heading within 0.5° of 2π.
- [ ] Over the fake (ER-055) through `simbase` (ER-057): twist (0.2 m/s, 0) for 10 s → odometry within 1 cm of ground truth; twist (0, 0.5 rad/s) for 10 s → heading within 1° of 5 rad.
- [ ] A counter wrap mid-run doesn't jump the pose; 0 allocations per `Update`.

## Open questions

None.

## Effort uncertainty and revision trigger

Size guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Log

- 2026-11-12: review fixes: depends on ER-055 and ER-057 for its over-the-fake case and on ER-059 for `Pose2D`; the description comes from `x/robot/model`; created date corrected to the 2026-11-12 fold.

## Implementation evidence

None. Proposed ticket; becomes `ready` after review.

---
id: ER-057
title: "Sim S1: `simbase` — the sim implementation of `wheeled.Base` and `Clock` over `Sim`"
type: ticket
status: proposed
kind: implementation
owner: easyrobot
workstream: [W3, W6]
milestone: M-d
tags: [easyrobot, ticket, simulation, sim-s1]
created: 2026-11-10
updated: 2026-11-12
---
# ER-057 — Sim S1: `simbase` — the sim implementation of `wheeled.Base` and `Clock` over `Sim`

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; verify accepted) · Brief: `~/src/docs/tasks/SIM-tickets.md`

**Brick:** the sim side of the robot seam: `SetTwist` becomes wheel velocity targets in a `Command`, `Tick` advances one lockstep step, and `Encoders` and `IMU` read the `State` through the sensor models. It holds no control law and no odometry.

## Goal and evidence

ux's UX-27 S1 (major; `docs/ux/UX-27-sim.md`, Leader accepted): this ticket used to be a sim-typed adapter with odometry inside; it is now the sim implementation of ER-059's `Base` and `Clock`, so the same controller loop runs on the robot later. Odometry moved to its own robot brick, ER-060 (UX-27 S7). The differential kinematics already map a chassis twist to wheel rates (`differential.Backward`, observed).

Owner's answers (roster `decisions.md`, 2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule (verbatim): "i want primitives like other primitives(gait, matrix ops, filters, etc.) - interface based where possible for performance reasons, using graphs, sets and other primitives. kiss and solid. single responsibility principle is very important to me. building bricks and then the hous is important to me unless we know good bricks and good doors and good windows."

## Interface sketch

Package `x/sim/simbase` (product's proposal):

```go
func New(s sim.Sim, w model.Wheeled, imu sensor.IMUModel, enc []sensor.EncoderModel) (*Base, *Clock)
var _ wheeled.Base  = (*Base)(nil)
var _ wheeled.Clock = (*Clock)(nil)
```

## Dependencies and scope

Dependencies: ER-049, ER-052, ER-054, ER-059; ER-038 (differential wheel rates in rad/s).

- `SetTwist` stores the twist; `Tick` turns it into wheel velocity targets in rad/s (from the description's radius, consistent with ER-038's fix), sends one `Command`, and reads the `State`.
- `Encoders` applies ER-054's encoder models to the joint positions; `IMU` applies ER-054's IMU model to the exact IMU values.
- `Now` is the sim time from the last `State`.
- **Optional interfaces** (ER-059; ux's UX-27 second rating R1, R3): `Ranger` and `Contact` read `State.sensors` by name (a `rangefinder` value in metres; a `touch` value above zero as contact), and `Truth` returns the ground-truth body pose projected to 2D. Lookup by name uses an index built at `Load`, so a call allocates nothing.
- No allocation per call after warm-up.

Not in scope: odometry (ER-060); path tracking and any control law (ER-058 uses the existing ones); obstacle avoidance (M-d's decision trees, later).

Source/test edits follow the owner's autonomy instruction once Leader schedules this ticket. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] Passes ER-059's `basetest.Conformance` over the fake backend (ER-055).
- [ ] Wheel targets for 10 sampled twists equal `differential.Backward` divided by the radius (< 1e-6).
- [ ] After `SetTwist(0.2, 0)` and 500 `Tick`s at 50 Hz on the fake, `Now()` = 10 s and the encoder ticks match the fake's wheel angles within 1 tick.
- [ ] With a model that has a `rangefinder` named `tof_front` (ER-053), `Range("tof_front")` returns the sidecar's value; `Range("nope")` returns `ErrNoSensor`; `Pose` matches `State.body_pose` projected to 2D.
- [ ] 0 allocations per `SetTwist`, `Tick`, `Encoders`, `IMU`, `Range`, `Contact` and `Pose`.

## Open questions

None.

## Effort uncertainty and revision trigger

Product's guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Implementation evidence

None. Proposed ticket; becomes `ready` after ux's UX-27 second rating and verify.

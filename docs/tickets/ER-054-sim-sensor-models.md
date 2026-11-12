---
id: ER-054
title: "Sim S1: sensor models — IMU noise and wheel-encoder quantisation from ground truth"
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
# ER-054 — Sim S1: sensor models — IMU noise and wheel-encoder quantisation from ground truth

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; verify accepted) · Brief: `~/src/docs/tasks/SIM-tickets.md`

**Brick:** turning the simulator's exact values into what real sensors would report. It has no transport and no physics.

## Goal and evidence

Note §1: IMU and encoders are must-haves. The `State` from any backend carries exact values (ER-048); the noise and quantisation are applied in Go so every backend (MuJoCo, the fake, later Cogent Lab) gets the same seeded sensor behaviour, and the controller sees what the robot would see.

Owner's answers (roster `decisions.md`, 2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule (verbatim): "i want primitives like other primitives(gait, matrix ops, filters, etc.) - interface based where possible for performance reasons, using graphs, sets and other primitives. kiss and solid. single responsibility principle is very important to me. building bricks and then the hous is important to me unless we know good bricks and good doors and good windows."

## Interface sketch

Package `x/sim/sensor`:

```go
type IMUModel interface { Apply(truth *devicespb.IMUReading, out *devicespb.IMUReading) } // bias, white noise, seeded (types.devices.IMUReading, ER-048)
type EncoderModel interface { Ticks(jointPos float32) int32 }          // counts per revolution
func NewIMU(cfg IMUConfig, seed uint64) IMUModel
func NewEncoder(countsPerRev int) EncoderModel
```

## Dependencies and scope

Dependencies: ER-048.

- IMU: per-axis bias and Gaussian white noise from config (accel m/s², gyro rad/s); seeded, so the same seed gives the same stream.
- Encoder: floor(angle / 2π × counts) with wrap handling over many turns.
- No allocation per call.
- **Add a sensor (recipe, UX-27 S2):** with ER-048's generic `State.sensors` and ER-051 forwarding every named MJCF sensor, a new sensor touches three places:
  1. the robot description and exporter (ER-052, ER-053): its placement and its MJCF element with a unique name;
  2. a Go model here, if it needs noise or quantisation (it reads `State.sensors` by name);
  3. to use it in a controller, the matching optional interface beside `wheeled.Base` (ER-059: `Ranger`, `Contact`), implemented by `simbase` (ER-057) from `State.sensors`; a new kind of sensor adds one new small interface, never a method on `Base` (ux's UX-27 second rating R1).
  The proto, the sidecar, the client and `Base` itself need no change. If the fake (ER-055) should report it, the fake gets a line too.
- **Later, for O-007 vision navigation (not a dependency now):** a camera model (QVGA/QQVGA frames with the real cameras' intrinsics), ground-truth pose per frame, and IMU streams for VO scale. The sim can serve bench-style datasets once those land; the owner's current plan is bench and handheld data first, no robot.

Not in scope: camera, lidar; no noise model for contact sensors (they pass through `State.sensors`).

Source/test edits follow the owner's autonomy instruction once Leader schedules this ticket. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] With zero noise and bias, IMU output equals the input within 1e-7.
- [ ] With noise σ, the sample standard deviation over 10,000 samples is within 5 % of σ; two runs with the same seed are bit-identical, different seeds differ.
- [ ] Encoder at 360 counts/rev: error vs the exact angle ≤ 1 count over 100 revolutions forward and back.
- [ ] 0 allocations per `Apply` and `Ticks`.

## Open questions

- Default noise figures: from the owner's IMU (MPU6050 datasheet values, product's proposal) or measured on the bench.

## Effort uncertainty and revision trigger

Product's guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Log

- 2026-11-12: verify's fixes (`docs/verification/SIM-tickets.md`, accepted with 11 fixes; Leader msg m-40a81d6669b0): the IMU model takes `types.devices.IMUReading`; the recipe says three places.

## Implementation evidence

None. Proposed ticket; becomes `ready` after ux's UX-27 second rating and verify.

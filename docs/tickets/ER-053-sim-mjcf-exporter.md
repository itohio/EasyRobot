---
id: ER-053
title: "Sim S1: MJCF exporter — wheeled description → MJCF bytes with named actuators and sensors"
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
# ER-053 — Sim S1: MJCF exporter — wheeled description → MJCF bytes with named actuators and sensors

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; verify accepted) · Brief: `~/src/docs/tasks/SIM-tickets.md`

**Brick:** one function that turns a ER-052 description into MJCF. It doesn't touch MuJoCo itself.

## Goal and evidence

Decision 4: generate MJCF from Go structs. Note §4 S1a lists what the exporter writes (for the hexapod; the wheeled equivalent here): bodies, joints with axes and limits, actuators with force ranges from the motor model, contact friction, IMU site with accelerometer and gyro, named so the sidecar finds them.

Owner's answers (roster `decisions.md`, 2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule (verbatim): "i want primitives like other primitives(gait, matrix ops, filters, etc.) - interface based where possible for performance reasons, using graphs, sets and other primitives. kiss and solid. single responsibility principle is very important to me. building bricks and then the hous is important to me unless we know good bricks and good doors and good windows."

## Interface sketch

Package `x/sim/mjcf`:

```go
func Wheeled(w model.Wheeled, opt Options) ([]byte, error) // model = x/robot/model (ER-052)
type Options struct { PhysicsDT float32; Floor bool }
// Names: actuators "wheel_<i>", joints "wheel_<i>_joint", sensors "imu_acc", "imu_gyro", "wheel_<i>_pos"
```

## Dependencies and scope

Dependencies: ER-052.

- Chassis as a box with its mass; wheels as cylinders on hinge joints about their axle; casters as frictionless-ish spheres; a floor plane with the description's friction.
- Wheel motors as **velocity actuators** with `forcerange = ±StallTorque` and `ctrlrange = ±NoLoadSpeed` (the motor model's limits). The velocity gain `kv` is `StallTorque / NoLoadSpeed` (N·m per rad/s), so the actuator follows the motor's straight-line torque–speed curve instead of MuJoCo's default `kv = 1` [INFERENCE, verify's note].
- An IMU site with `accelerometer` and `gyro`; a `jointpos` sensor per wheel for the encoders (ER-054 quantises them).
- Every sensor gets a unique `name`, since the sidecar forwards each named sensor into `State.sensors` (ER-048, UX-27 S2).
- Output is deterministic: the same description gives byte-identical MJCF.

Not in scope: loading or validating in MuJoCo (ER-051 does that); legged export (later, from `chain.Body`); URDF (later if needed).

Source/test edits follow the owner's autonomy instruction once Leader schedules this ticket. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] A golden test: `DifferentialBase(...)` exports to MJCF equal to a checked-in file; a second run is byte-identical.
- [ ] The XML parses with `encoding/xml`; every name the sidecar needs (actuators, joints, sensors) is present, each once.
- [ ] Wheel joint positions and axes in the MJCF match the description within 1e-6 m and 1e-6 rad.
- [ ] With ER-051 available, MuJoCo compiles the exported model without warnings (one integration test). Without it the test skips with `skipped: MuJoCo sidecar not installed (ER-051; needs the owner's download go)` (UX-27 S8).

## Open questions

- Velocity actuators vs torque with a Go-side motor model (product: velocity actuators first; the motor curve later if ER-058 shows it matters).

## Effort uncertainty and revision trigger

Product's guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Log

- 2026-11-12: verify's fixes (`docs/verification/SIM-tickets.md`, accepted with 11 fixes; Leader msg m-40a81d6669b0): velocity actuators get `kv = StallTorque / NoLoadSpeed`; the description package is `x/robot/model`.

## Implementation evidence

None. Proposed ticket; becomes `ready` after ux's UX-27 second rating and verify.

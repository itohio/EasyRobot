---
id: ER-048
title: "Sim S1: `types/sim` protobuf contract — Command, State, ModelLoad"
type: ticket
status: proposed
kind: design + implementation
owner: easyrobot
workstream: [W3, W6]
milestone: M-d
tags: [easyrobot, ticket, simulation, sim-s1]
created: 2026-11-10
updated: 2026-11-12
---
# ER-048 — Sim S1: `types/sim` protobuf contract — Command, State, ModelLoad

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; verify accepted) · Brief: `~/src/docs/tasks/SIM-tickets.md`

**Brick:** the wire contract between any Go controller and any simulator backend. It holds data shapes only, no behaviour.

## Goal and evidence

The owner's rule is that protobuf schemas are the wire contract (note §3). EasyRobot keeps its protos in `proto/types/<pkg>/` and generates them into `types/<pkg>/` (observed: `proto/types/core/frame.proto` → `types/core/frame.pb.go`). The sim needs its own package.

Owner's answers (roster `decisions.md`, 2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule (verbatim): "i want primitives like other primitives(gait, matrix ops, filters, etc.) - interface based where possible for performance reasons, using graphs, sets and other primitives. kiss and solid. single responsibility principle is very important to me. building bricks and then the hous is important to me unless we know good bricks and good doors and good windows."

## Interface sketch

`proto/types/sim/sim.proto`, generated to `types/sim`:

```proto
message ModelLoad { bytes mjcf = 1; uint32 physics_dt_us = 2; uint32 substeps = 3; uint64 seed = 4; }
message Reset { uint64 seed = 1; }
message Error { string code = 1; string message = 2; }   // e.g. STEP_GAP, SHAPE, MODEL
message Envelope { oneof m { ModelLoad load = 1; Command command = 2; Reset reset = 3; State state = 4; Error error = 5; } } // one frame on the wire (ER-050)
message Command {
  uint64 step = 1;                        // lockstep counter; the backend refuses a gap or a repeat
  repeated float position_target = 2;     // per actuator, radians (servos)
  repeated float velocity_target = 3;     // per actuator, rad/s (wheel motors)
  repeated float torque = 4;              // per actuator, N·m (when the actuator is torque-driven)
  RenderRequest render = 5;               // optional; camera later
}
message State {
  uint64 step = 1; uint64 sim_time_us = 2;
  repeated float joint_position = 3; repeated float joint_velocity = 4;
  Pose body_pose = 5; Twist body_twist = 6;   // ground truth, world frame
  Imu imu = 7;                                // accel (m/s²), gyro (rad/s), no noise; noise is ER-054's
  repeated Contact contacts = 8;
  repeated float actuator_force = 9; repeated bool actuator_saturated = 10;
  types.core.Frame frame = 11;                // optional; camera later
  repeated SensorReading sensors = 12;        // every named MJCF sensor (UX-27 S2)
}
message SensorReading { string name = 1; repeated float values = 2; } // the MJCF sensor's name and its sensordata slice
```

## Dependencies and scope

Dependencies: none.

- Field names and units in comments; every repeated field is indexed by the actuator or joint order the model defines (ER-052).
- Reuse `types/core` `Frame` and existing pose/twist messages where they exist (`proto/types/control`); new ones only where none fits.
- **One place to add a sensor** (ux's UX-27 S2, major, Leader accepted): `State.sensors` carries every named MJCF sensor generically, so a new sensor (a ToF, a bumper, a line sensor) needs no proto change and no sidecar change. The typed `imu` field stays for speed.
- A golden binary fixture of one `Command` and one `State` is checked in, so a later schema change shows up as a diff.

Not in scope: transport (ER-050); behaviour; lidar; camera rendering (later slice).

Source/test edits follow the owner's autonomy instruction once Leader schedules this ticket. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] `make proto` (or the repo's generator) builds `types/sim` with no warnings; generated code is committed. **Offline** (ux's UX-27 second rating R2): no `protoc` is on the host and the Makefile's `install-tools` fetches `@latest`, so the build uses `buf` v1.75.0 and `protoc-gen-go` v1.36.12 (the version in `go.mod`) built from the local module cache with `GOPROXY=off` (both observed in `~/go/pkg/mod`), as abugaga's ABU-084 did; no download. If either is missing at build time, it joins ER-051's single approval list instead of being fetched.
- [ ] A round-trip test marshals and unmarshals each `Envelope` case, including a `Command` and a `State` for a 2-wheel robot with IMU and two contacts; byte-equal on re-marshal.
- [ ] That `State` is ≤ 512 B on the wire (product's budget; a frame and generic sensors excluded).
- [ ] A `State` with three `SensorReading`s (1, 3 and 8 values) round-trips byte-equal.
- [ ] The golden fixtures decode with the generated code.

## Open questions

- Whether `Pose`/`Twist` come from `proto/types/control` or `types/math` (the implementer picks the existing one; no duplicate).

## Effort uncertainty and revision trigger

Product's guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Implementation evidence

None. Proposed ticket; becomes `ready` after ux's UX-27 second rating and verify.

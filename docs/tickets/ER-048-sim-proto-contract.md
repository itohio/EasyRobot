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

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; reviewed)

**Brick:** the wire contract between any Go controller and any simulator backend. It holds data shapes only, no behaviour.

## Goal and evidence

The owner's rule is that protobuf schemas are the wire contract (note §3). EasyRobot keeps its protos in `proto/types/<pkg>/` and generates them into `types/<pkg>/` (observed: `proto/types/core/frame.proto` → `types/core/frame.pb.go`). The sim needs its own package.

Decisions (2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule: primitives like the others (gait, matrix ops, filters), interface-based where performance allows, built on graphs, sets and other primitives; simple, solid, single responsibility; bricks first, then the house.

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
  types.devices.IMUReading imu = 7;           // acceleration (m/s²), angular_velocity (rad/s); exact, noise is ER-054's
  repeated Contact contacts = 8;
  repeated float actuator_force = 9; repeated bool actuator_saturated = 10;
  types.core.Frame frame = 11;                // optional; camera later
  repeated SimSensor sensors = 12;            // every named MJCF sensor
}
message SimSensor { string name = 1; repeated float values = 2; } // an MJCF sensor's name and its sensordata slice
// Defined here, reusing existing messages (proto/types/control has no Pose or Twist):
message Pose    { types.math.Vector3D position = 1; types.math.Quaternion orientation = 2; } // world frame
message Twist   { types.math.Vector3D linear = 1;   types.math.Vector3D angular = 2; }       // m/s, rad/s
message Contact { string geom = 1; float normal_force = 2; types.math.Vector3D position = 3; } // N, world frame
message RenderRequest { string camera = 1; uint32 width = 2; uint32 height = 3; }             // camera later
```

## Dependencies and scope

Dependencies: none.

- Field names and units in comments; every repeated field is indexed by the actuator or joint order the model defines (ER-052).
- Reuse what exists: `types.core.Frame`, `types.math.Vector3D` and `Quaternion`, and `types.devices.IMUReading` for the IMU (its magnetometer, temperature and timestamp fields stay unset). `types.devices.SensorReading` is a typed oneof for device drivers, so the sim's generic name-and-values reading is a separate `SimSensor`, not a second `SensorReading`.
- **One place to add a sensor**: `State.sensors` carries every named MJCF sensor generically, so a new sensor (a ToF, a bumper, a line sensor) needs no proto change and no sidecar change. The typed `imu` field stays for speed.
- A golden binary fixture of one `Command` and one `State` is checked in, so a later schema change shows up as a diff.

Not in scope: transport (ER-050); behaviour; lidar; camera rendering (later slice).

Source/test edits are approved once this ticket is scheduled. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] The generated Go code for `types/sim` builds with no warnings and is committed. **Offline recipe**: no `protoc` on the host; build `buf` v1.75.0 and `protoc-gen-go` v1.36.12 (the version in `go.mod`) from **copies** of their directories in `~/go/pkg/mod` (`go install …@version` with `GOPROXY=off` fails); `~/.cache/buf` must hold `wellknowntypes`; run `cd proto && buf generate --path types/sim` (the buf module root is `proto/`) so the 27 existing `.pb.go` headers aren't rewritten (v1.36.10 → v1.36.12). If any piece is missing, it joins ER-051's single approval list instead of being fetched.
- [ ] A round-trip test marshals and unmarshals each `Envelope` case, including a `Command` and a `State` for a 2-wheel robot with IMU and two contacts; byte-equal on re-marshal.
- [ ] That `State` is ≤ 512 B on the wire (a frame and generic sensors excluded).
- [ ] A `State` with three `SimSensor`s (1, 3 and 8 values) round-trips byte-equal.
- [ ] `buf build` compiles the sketch's messages (measured 220 B for the 2-wheel `State`, 307 B with three sensors).
- [ ] The golden fixtures decode with the generated code.

## Open questions

None.

## Effort uncertainty and revision trigger

Size guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Log

- 2026-11-12: review fixes: the IMU reuses `types.devices.IMUReading`; `Pose`, `Twist`, `Contact`, `RenderRequest` defined from `types.math` (no Pose/Twist in `proto/types/control`); the generic reading is `SimSensor`, apart from `types.devices.SensorReading`; the offline generation recipe (module-directory copies, `wellknowntypes`, `--path proto/types/sim`).

## Implementation evidence

None. Proposed ticket; becomes `ready` after review.
- 2026-11-12: SIM-tickets review addendum: `cd proto && buf generate --path types/sim` (buf's module root is `proto/`); current `State` sizes (220/307 B).

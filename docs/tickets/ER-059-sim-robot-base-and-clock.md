---
id: ER-059
title: "Sim S1: robot-facing `wheeled.Base` interface and `Clock` — the seam between sim and real robot"
type: ticket
status: proposed
kind: design + implementation
owner: easyrobot
workstream: [W3, W6]
milestone: M-d
tags: [easyrobot, ticket, simulation, sim-s1]
created: 2026-11-12
updated: 2026-11-12
---
# ER-059 — Sim S1: robot-facing `wheeled.Base` interface and `Clock` — the seam between sim and real robot

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; verify accepted) · Brief: `~/src/docs/tasks/SIM-tickets.md`

**Brick:** the interface a wheeled controller talks to, whether the robot is simulated or real, plus the clock that drives its loop. It has no implementation of its own beyond contract tests.

## Goal and evidence

ux's UX-27 S1 (major; `docs/ux/UX-27-sim.md`, Leader accepted): `Sim` (ER-049) is a simulator interface, which a real robot can't implement, and the controller loop was built on `simpb` types, with no clock seam (sim time in lockstep vs wall time on the robot). The note's goal is the same controller code on the sim and the robot (§3).

Owner's answers (roster `decisions.md`, 2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule (verbatim): "i want primitives like other primitives(gait, matrix ops, filters, etc.) - interface based where possible for performance reasons, using graphs, sets and other primitives. kiss and solid. single responsibility principle is very important to me. building bricks and then the hous is important to me unless we know good bricks and good doors and good windows."

## Interface sketch

Package `x/robot/wheeled` (product's proposal for the path; next to the kinematics is also fine):

```go
type Imu struct { Accel, Gyro [3]float32 }          // m/s², rad/s; the robot's own type, not simpb (ER-060 and the hardware base share it)
type Pose2D struct { X, Y, Theta float64 }          // metres, radians; float64 so odometry doesn't drift (ER-060)
type Base interface {
    SetTwist(v, omega float32) error  // chassis twist; the base maps it to its actuators
    Encoders(out []int32) error       // ticks per wheel, in the description's wheel order
    IMU(out *Imu) error               // accel m/s², gyro rad/s, as the robot's sensor reports them
}
type Clock interface {
    Now() time.Duration               // since the loop started
    Tick() error                      // wait for the next control period (sim: advances one lockstep step; robot: sleeps to the wall-clock period)
}
// Optional, segregated interfaces beside Base (UX-27 R1, R3): a base implements only the ones it has;
// the controller finds them by type assertion: if r, ok := base.(wheeled.Ranger); ok { … }
type Ranger interface  { Range(name string, out *float32) error }   // metres; a ToF or ultrasonic sensor by its name
type Contact interface { Contact(name string) (bool, error) }       // a bumper or touch switch by its name
type Truth interface   { Pose(out *Pose2D) error }                  // ground truth; the sim base only, never hardware
// basetest.Conformance(t, newBase func() (Base, Clock)) — the shared contract tests,
// plus a check for each optional interface the base under test claims
```

## Dependencies and scope

Dependencies: none.

- Two implementations: the sim one is ER-057 (`simbase`, over `Sim`); the hardware one (motor and encoder drivers in `x/devices/motor` and `x/devices/encoder`, a wall clock) is a later ticket and not part of S1.
- The controller loop in ER-058 uses only `Base` and `Clock`; no `simpb` type crosses this interface.
- **New sensors reach the controller without changing `Base`** (ux's UX-27 second rating R1, major, Leader accepted): each sensor family is a small optional interface beside `Base` (`Ranger`, `Contact` now; a new family adds one new small interface, never a method on `Base`). `simbase` (ER-057) implements them from `State.sensors` by name; the hardware base implements the ones its drivers provide. An unknown name returns `ErrNoSensor` naming it.
- **Ground truth** (ux's UX-27 second rating R3): `Truth` is implemented only by `simbase`; ER-058's loop type-asserts it for the log, and the column stays empty on hardware.
- Errors typed (`ErrShape` for a wrong wheel count, `ErrStopped`, `ErrNoSensor`).
- No allocation per call; the caller owns the buffers.

Not in scope: any implementation; the hardware base (later).

Source/test edits follow the owner's autonomy instruction once Leader schedules this ticket. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] `x/robot/wheeled` compiles with `Base`, `Clock` and typed errors; `go vet` clean.
- [ ] `basetest.Conformance` passes against a minimal in-test stub (records the twist, returns scripted ticks, a manual clock); a broken stub that ignores the twist fails it.
- [ ] `basetest` checks every optional interface a base claims: a stub claiming `Ranger` returns its scripted range by name and `ErrNoSensor` for an unknown name; a stub without `Contact` fails the type assertion and the suite skips that check with the reason.
- [ ] No allocation per `Range`, `Contact` and `Pose` call.

## Open questions

- Whether the hardware base ticket lands in M-d stage 2 (the real robot) or earlier for bench tests: Leader's call.

## Effort uncertainty and revision trigger

Product's guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Log

- 2026-11-12: verify's fixes (`docs/verification/SIM-tickets.md`, accepted with 11 fixes; Leader msg m-40a81d6669b0): `Imu` and `Pose2D` defined here as robot types, shared by ER-060; the import test moved to ER-058's loop package; created date corrected to the 2026-11-12 fold.

## Implementation evidence

None. Proposed ticket; becomes `ready` after ux's UX-27 second rating and verify.

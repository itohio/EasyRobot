---
id: ER-052
title: "Sim S1: robot description structs — a wheeled base with masses and motor torque and speed"
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
# ER-052 — Sim S1: robot description structs — a wheeled base with masses and motor torque and speed

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`)

**Brick:** Go structs that describe one robot's physical parameters. They are the single source of truth for geometry, mass and actuator limits. They don't export or simulate anything.

## Goal and evidence

Decision 4: robot descriptions stay Go structs; add link mass and servo torque and speed fields; generate MJCF. First slice wheeled. Today the differential drive knows only wheel radius and base (`x/math/control/kinematics/wheels/differential/drive.go:25`, observed); `chain.Body` covers legs, not wheels (`chain/legged.go:57`).

Decisions (2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule: primitives like the others (gait, matrix ops, filters), interface-based where performance allows, built on graphs, sets and other primitives; simple, solid, single responsibility; bricks first, then the house.

## Interface sketch

Package `x/robot/model` (the robot's source of truth lives with the robot, not under `x/sim`; hexapod later extends `chain.Body` with the same fields):

```go
type Motor struct { StallTorque, NoLoadSpeed float32 } // N·m, rad/s
type Wheel struct { Radius, Width, Mass float32; Pos vec.Vector3D; Motor Motor; Driven bool }
type Caster struct { Radius, Mass float32; Pos vec.Vector3D }
type Wheeled struct {
    Chassis struct { Size vec.Vector3D; Mass float32 }
    Wheels  []Wheel   // order = actuator order in Command/State
    Casters []Caster
    IMU     struct { Pos vec.Vector3D }
    Friction float32  // tyre/ground
}
func DifferentialBase(radius, track, chassisMass float32, m Motor) Wheeled // the M-d robot's shape
func (w Wheeled) Validate() error
```

## Dependencies and scope

Dependencies: none.

- `Validate` refuses non-positive masses or radii, zero motor limits and an empty driven set, each with a typed error naming the field. Wheels overlapping the chassis are allowed: MuJoCo filters contacts between a parent body and its children [INFERENCE], so the exporter needs no gap.
- `DifferentialBase` matches the differential kinematics: its `track` is the `base` that `differential.New` takes.
- Defaults for motor stall torque and no-load speed are placeholders marked `TODO(owner: part numbers)` until the owner supplies them (decision 7, open).

Not in scope: the exporter (ER-053); legged bodies (later, extending `chain.Body`); mecanum and steer variants (later; same structs, more wheels).

Source/test edits are approved once this ticket is scheduled. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] `DifferentialBase(0.033, 0.16, 0.6, Motor{0.1, 20})` validates; each invalid field case returns its typed error (table test).
- [ ] The kinematic `differential.New(r, track)` built from the description gives the same chassis twist as the description's wheel geometry for 10 sampled wheel-rate pairs (< 1e-6).

## Open questions

- **Decision 7 (open):** the owner's motor and servo part numbers (stall torque, no-load speed). Until then the numbers are placeholders and ER-058's results are labelled with them.
- The M-d robot's real dimensions (radius, track, masses): the owner's, or measured when the frame exists.

## Effort uncertainty and revision trigger

Size guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Log

- 2026-11-12: review fixes: the package moves to `x/robot/model` (layering: robot bricks don't depend on `x/sim`); the "wheels inside the chassis" refusal is dropped.

## Implementation evidence

None. Proposed ticket; becomes `ready` after review.

---
id: ER-055
title: "Sim S1: kinematic fake backend — a pure-Go differential integrator implementing `Sim` for CI"
type: ticket
status: proposed
kind: implementation
owner: easyrobot
workstream: [W3, W6]
milestone: M-d
tags: [easyrobot, ticket, simulation, sim-s1]
created: 2026-11-10
updated: 2026-11-10
---
# ER-055 — Sim S1: kinematic fake backend — a pure-Go differential integrator implementing `Sim` for CI

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; verify accepted) · Brief: `~/src/docs/tasks/SIM-tickets.md`

**Brick:** a physics-free `Sim` implementation so controllers and the conformance suite run in CI without MuJoCo.

## Goal and evidence

MuJoCo needs a download (decision 2) and won't be in every test run. A kinematic fake is the CI stand-in: it integrates the differential kinematics that already exist (`wheels/differential`), with no forces or slip. It is not a physics backend and claims nothing about dynamics.

Owner's answers (roster `decisions.md`, 2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule (verbatim): "i want primitives like other primitives(gait, matrix ops, filters, etc.) - interface based where possible for performance reasons, using graphs, sets and other primitives. kiss and solid. single responsibility principle is very important to me. building bricks and then the hous is important to me unless we know good bricks and good doors and good windows."

## Interface sketch

`x/sim/fake`:

```go
func NewDifferential(w model.Wheeled) sim.Sim // integrates wheel velocity targets → pose, exact IMU (gyro = ω, accel from v̇)
```

## Dependencies and scope

Dependencies: ER-049, ER-052; ER-038's differential fix (wheel rates in rad/s from the radius) if it lands first, otherwise this ticket converts linear speeds explicitly.

- Uses `differential.Forward` for the chassis twist (no second copy of the kinematics).
- Integrates pose per substep (`physics_dt`), honouring the motor's no-load speed as a clamp and reporting `actuator_saturated` when clamped.
- Deterministic and allocation-free per step.

Not in scope: forces, slip, contacts (MuJoCo's job).

Source/test edits follow the owner's autonomy instruction once Leader schedules this ticket. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] Passes ER-049's conformance suite.
- [ ] Wheels at equal rate r·ω for 10 s: straight-line distance = v·t within 1e-4 m; opposite rates: heading change = ω·t within 1e-5 rad.
- [ ] A target above the no-load speed is clamped and flagged `actuator_saturated`.
- [ ] 0 allocations per `Step`.

## Open questions

None.

## Effort uncertainty and revision trigger

Product's guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Implementation evidence

None. Proposed ticket; becomes `ready` after ux's UX-27 and verify.

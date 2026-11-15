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
updated: 2026-11-12
---
# ER-055 — Sim S1: kinematic fake backend — a pure-Go differential integrator implementing `Sim` for CI

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; reviewed)

**Brick:** a physics-free `Sim` implementation so controllers and the conformance suite run in CI without MuJoCo.

## Goal and evidence

MuJoCo needs a download (decision 2) and won't be in every test run. A kinematic fake is the CI stand-in: it integrates the differential kinematics that already exist (`wheels/differential`), with no forces or slip. It is not a physics backend and claims nothing about dynamics.

Decisions (2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule: primitives like the others (gait, matrix ops, filters), interface-based where performance allows, built on graphs, sets and other primitives; simple, solid, single responsibility; bricks first, then the house.

## Interface sketch

`x/sim/fake`:

```go
func NewDifferential(w model.Wheeled) sim.Sim // model = x/robot/model (ER-052); integrates wheel velocity targets → pose, exact IMU (gyro = ω, accel from v̇)
```

## Dependencies and scope

Dependencies: ER-049, ER-052; ER-038's differential fix (wheel rates in rad/s from the radius) if it lands first, otherwise this ticket converts linear speeds explicitly.

- Uses `differential.Forward` for the chassis twist (no second copy of the kinematics).
- Integrates pose per substep (`physics_dt`) in **float64** internal state (float32 accumulation over 5,000 substeps drifts by about 1.2e-4 m and 1.9e-4 rad at the rates below, measured), converting to float32 only when filling `State`; honouring the motor's no-load speed as a clamp and reporting `actuator_saturated` when clamped.
- Deterministic and allocation-free per step.

Not in scope: forces, slip, contacts (MuJoCo's job).

Source/test edits are approved once this ticket is scheduled. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] Passes ER-049's conformance suite.
- [ ] At v = 0.2 m/s (equal wheel rates) for 10 s with 2 ms substeps: straight-line distance = v·t within 1e-6 m. At ω = 2.5 rad/s (opposite rates) for 10 s: heading change = ω·t, wrapped, within 1e-6 rad. A float32 accumulation of the same run is checked to fail these bounds, so the float64 state is shown to matter.
- [ ] A target above the no-load speed is clamped and flagged `actuator_saturated`.
- [ ] 0 allocations per `Step`.

## Open questions

None.

## Effort uncertainty and revision trigger

Size guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Log

- 2026-11-12: review fixes: float64 internal state; the rates are stated (0.2 m/s, 2.5 rad/s, 10 s, 2 ms substeps) and the bounds are 1e-6 m and 1e-6 rad, with a float32 run shown to miss them.

## Implementation evidence

None. Proposed ticket; becomes `ready` after review.
- 2026-11-12: review addendum: doubled comment in the sketch merged.

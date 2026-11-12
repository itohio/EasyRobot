---
id: ER-049
title: "Sim S1: Go `Sim` interface and a backend conformance suite"
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
# ER-049 — Sim S1: Go `Sim` interface and a backend conformance suite

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; verify accepted) · Brief: `~/src/docs/tasks/SIM-tickets.md`

**Brick:** the one Go interface every simulator backend implements, plus the shared tests that prove a backend honours it.

## Goal and evidence

Note §3: "one Go interface, implemented by every backend". The controller code stays the same whatever the backend: MuJoCo sidecar now, `purego` later, a kinematic fake for CI (ER-055), maybe Cogent Lab (decision 6, open).

Owner's answers (roster `decisions.md`, 2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule (verbatim): "i want primitives like other primitives(gait, matrix ops, filters, etc.) - interface based where possible for performance reasons, using graphs, sets and other primitives. kiss and solid. single responsibility principle is very important to me. building bricks and then the hous is important to me unless we know good bricks and good doors and good windows."

## Interface sketch

Package `x/sim` (product's proposal for the path):

```go
type Sim interface {
    Load(m *simpb.ModelLoad) error
    Step(cmd *simpb.Command, out *simpb.State) error // lockstep: apply, advance substeps, fill out; no allocation per step
    Reset(seed uint64) error
    Close() error
}
// camera later: type Renderer interface { Render(cam string, w, h int, buf []byte) (core.Frame, error) }
```

`x/sim/simtest.Conformance(t, newBackend func() sim.Sim)`: the shared suite.

## Dependencies and scope

Dependencies: ER-048.

- `Step` reuses the caller's `out` (no allocation per step; the suite checks it).
- Errors are typed: `ErrNotLoaded`, `ErrStepGap` (a command whose `step` isn't the next one), `ErrShape` (wrong actuator count), `ErrClosed`.
- The conformance suite checks: load → step increments `step` and `sim_time_us` by `substeps × physics_dt`; a gap or repeat is refused; a wrong-length command is refused; `Reset(seed)` returns to step 0; `Close` is idempotent.
- Rendering is a separate, optional interface, so a backend without a camera isn't forced to stub it (interface segregation).

Not in scope: any backend implementation; transport.

Source/test edits follow the owner's autonomy instruction once Leader schedules this ticket. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] `x/sim` compiles with the interface and typed errors; `go vet` clean.
- [ ] The conformance suite passes against a minimal in-test stub backend; a deliberately broken stub (skips the step-gap check) fails it. Every later backend (ER-050 client, ER-055 fake, ER-051 sidecar) runs the same suite.
- [ ] The suite includes an allocation check (`testing.AllocsPerRun` on `Step`): 0 for in-process backends (ER-055) and for the client (ER-050), whose `State` decoding is a hand-written `protowire` decoder into reused structs, because stock `proto.Unmarshal` allocates (verify measured 21 per `State`, 33 with three sensors; verify's SIM-tickets fix 1).

## Open questions

- Whether `Step` should take k control steps per call (batching) now or only if ER-056's speed check fails (product: only then).

## Effort uncertainty and revision trigger

Product's guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Log

- 2026-11-12: verify's fixes (`docs/verification/SIM-tickets.md`, accepted with 11 fixes; Leader msg m-40a81d6669b0): the allocation check's bound: 0 for in-process backends and for the client with ER-050's hand decoder.

## Implementation evidence

None. Proposed ticket; becomes `ready` after ux's UX-27 and verify.

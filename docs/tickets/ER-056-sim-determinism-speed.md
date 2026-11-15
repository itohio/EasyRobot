---
id: ER-056
title: "Sim S1: determinism and speed checks with the note's kill criteria"
type: ticket
status: proposed
kind: measurement
owner: easyrobot
workstream: [W3, W6]
milestone: M-d
tags: [easyrobot, ticket, simulation, sim-s1]
created: 2026-11-10
updated: 2026-11-10
---
# ER-056 — Sim S1: determinism and speed checks with the note's kill criteria

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; reviewed)

**Brick:** the measurement harness that decides whether a backend is fit for CI and experiments. It doesn't change any backend.

## Goal and evidence

Note §4 S1e and S1f: the same seed and inputs must give bit-identical `State` streams; headless must reach ≥ 10× real time on the Mac; the S1b round trip must stay ≤ 2 ms. The kill criteria come from the note.

Decisions (2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule: primitives like the others (gait, matrix ops, filters), interface-based where performance allows, built on graphs, sets and other primitives; simple, solid, single responsibility; bricks first, then the house.

## Interface sketch

`x/sim/simtest.Bench` and a `cmd/simcheck`:

```text
simcheck --backend unix:///tmp/sim.sock --model wheeled --seconds 60 --runs 2
  → determinism: sha256 of the State stream per run; equal or the first differing step
  → speed: sim seconds / wall seconds; round-trip median and p99
```

## Dependencies and scope

Dependencies: ER-050, ER-051, ER-053, ER-055.

- Runs the same scripted command sequence (a fixed pattern of wheel targets) twice per backend.
- Prints one line per check with pass/fail against the threshold, and the kill action when it fails.

Not in scope: tuning anything to pass.

Source/test edits follow once this ticket is scheduled. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] Fake backend: two 60 s runs give equal stream hashes; speed ≥ 100× real time (the floor for the fake).
- [ ] MuJoCo sidecar: two 60 s runs give equal hashes, **or** the first differing step and field are reported, and the tolerance is documented before any CI use (kill: not bit-identical → find the nondeterminism or document the tolerance).
- [ ] MuJoCo sidecar headless: ≥ 10× real time at 50 Hz control and 2 ms physics (kill: < 10× → batch steps per round trip, ER-049's open question).
- [ ] Round trip median ≤ 2 ms (kill: → batching or `purego`, a project decision).

## Open questions

None.

## Effort uncertainty and revision trigger

Size guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Implementation evidence

None. Proposed ticket; becomes `ready` after review.

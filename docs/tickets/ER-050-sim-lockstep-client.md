---
id: ER-050
title: "Sim S1: lockstep client — `Sim` over length-prefixed protobuf on a Unix socket or TCP"
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
# ER-050 — Sim S1: lockstep client — `Sim` over length-prefixed protobuf on a Unix socket or TCP

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; reviewed)

**Brick:** the Go side of the transport. It implements `Sim` by sending one request and reading one reply per step. It knows no physics.

## Goal and evidence

Note §3 transport: length-prefixed protobuf over TCP or a Unix socket first; dndm once its route defects are fixed (`docs/architecture/dndm-and-protobuf.md`).

Decisions (2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule: primitives like the others (gait, matrix ops, filters), interface-based where performance allows, built on graphs, sets and other primitives; simple, solid, single responsibility; bricks first, then the house.

## Interface sketch

`x/sim/client`:

```go
func Dial(addr string, opts ...Option) (*Client, error) // "unix:///tmp/sim.sock" or "tcp://127.0.0.1:7600"
var _ sim.Sim = (*Client)(nil)
```

Frame: 4-byte big-endian length + one `Envelope` (ER-048).

## Dependencies and scope

Dependencies: ER-048, ER-049.

- One outstanding request at a time (lockstep); a reply's `step` must match the command's.
- Timeouts per call (default 1 s), errors typed and wrapped with the address.
- Buffers reused across steps; no allocation per step after warm-up.
- A test server in Go (wrapping ER-049's exported `simtest.NewStub`, configured with 2 wheels, 2 contacts and 3 sensors) lets the client pass the conformance suite without MuJoCo.

Not in scope: the sidecar (ER-051); dndm transport (later).

Source/test edits follow the standing autonomy instruction once this ticket is scheduled. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] The client, against the in-process test server over a Unix socket, passes ER-049's conformance suite.
- [ ] Round trip for a 2-wheel `Command`/`State` over a Unix socket: median ≤ 100 µs, p99 ≤ 500 µs on the Mac (`simtest.NewStub` behind the test server; transport cost only).
- [ ] After 1,000 warm-up steps, 0 allocations per `Step` (`testing.AllocsPerRun`) with a 2-wheel, 2-contact, 3-sensor `State`. Encoding uses `MarshalAppend` into a reused buffer (0 allocations, measured); **decoding is a hand-written `protowire` decoder into reused structs**, matching sensor names against the names from the previous `State` without allocating strings (`ModelLoad` carries only the MJCF).
- [ ] A server that dies mid-run gives a typed error naming the address within the timeout, not a hang.

## Open questions

- Unix socket vs TCP as the default (product: Unix socket on the host; TCP for a sim on another machine).

## Effort uncertainty and revision trigger

Size guess: **S** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Log

- 2026-11-12: review fixes: 0 allocations per `Step` through a hand-written `protowire` decoder (stock `proto.Unmarshal` allocates 21); the conformance run uses ER-049's in-test stub server, not ER-055, so no new dependency.

## Implementation evidence

None. Proposed ticket; becomes `ready` after review.
- 2026-11-12: SIM-tickets review addendum: the test server wraps `simtest.NewStub` (2 wheels, 2 contacts, 3 sensors); sensor names match the previous `State`.

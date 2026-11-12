---
id: ER-051
title: "Sim S1: MuJoCo sidecar — load MJCF, step in lockstep, fill State, headless with an optional viewer"
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
# ER-051 — Sim S1: MuJoCo sidecar — load MJCF, step in lockstep, fill State, headless with an optional viewer

[[tickets/index|Ticket index]] · [[roadmap]] · Design: [[2026-11-10-simulation]] (ideation `abc52ed`, corrections `c61b697`; verify accepted) · Brief: `~/src/docs/tasks/SIM-tickets.md`

**Brick:** the one process that owns MuJoCo. It serves ER-050's protocol and does nothing else.

## Goal and evidence

Decision 1: MuJoCo in a sidecar, driven from Go. Decision 2: the MuJoCo 4.x download is approved **when S1 starts**, either the release binary or the Python package in a uv venv under `_scratch`. No Go binding exists (note §2).

Owner's answers (roster `decisions.md`, 2026-11-10): SIM decisions 1–4 approved. MuJoCo 4.x (Apache-2.0) runs as a sidecar driven from Go over protobuf, in lockstep (a `purego` binding later). MuJoCo is downloaded at the start of S1, not before. The contract is a `types/sim` proto plus a Go `Sim` interface. Robot descriptions are Go structs with mass and actuator torque and speed, plus an MJCF exporter. **First slice: wheeled.** Design rule (verbatim): "i want primitives like other primitives(gait, matrix ops, filters, etc.) - interface based where possible for performance reasons, using graphs, sets and other primitives. kiss and solid. single responsibility principle is very important to me. building bricks and then the hous is important to me unless we know good bricks and good doors and good windows."

## Interface sketch

`tools/sim/mujoco_sidecar` (product's proposal: Python with the `mujoco` package; the C API via `purego` is the later step):

```text
mujoco_sidecar --listen unix:///tmp/sim.sock [--viewer]
  ModelLoad → mj_loadXML from bytes, set timestep = physics_dt, store substeps and seed
  Command   → write ctrl[], mj_step × substeps, read qpos/qvel/sensordata/contacts/actuator_force → State
  Reset     → mj_resetData, reseed
```

## Dependencies and scope

Dependencies: ER-048, ER-050 (protocol), ER-053 (an MJCF to load). The download needs the owner's go at S1 start.

- **Download gate, one approval** (decision 2: at the start of S1; ux's UX-27 S3, major, Leader accepted): the first action of this ticket is to ask Leader for the owner's go on the **complete** list below, once. Every item is recorded with version, source and sha256 in the evidence; nothing outside the list is installed.

  | Item | Why | Source (product's proposal; exact versions pinned at the ask) |
  |---|---|---|
  | `uv` (if not already on the host) | creates the venv under `~/src/tmp/` | astral.sh release or Homebrew |
  | Python 3.12 (uv-managed) | the sidecar's runtime | uv's managed Python |
  | `mujoco` 4.x | the physics engine and its Python binding | PyPI `mujoco` |
  | `numpy` | `mujoco`'s dependency | PyPI |
  | `protobuf` (Python runtime) | decoding `Envelope` frames | PyPI `protobuf` |
  | Python code for `types/sim` | the messages in Python | generated with `protoc --python_out` from the repo's `protoc` (no extra plugin), or a small hand-written decoder if `protoc` isn't installed (no download then) |
  | viewer dependencies (`glfw`, pulled in by `mujoco`'s viewer) | `--viewer` only | PyPI, as `mujoco` requires |

  - **macOS viewer:** MuJoCo's interactive viewer on macOS needs its `mjpython` launcher (#verify against the installed version); `--viewer` starts the sidecar under `mjpython`, and headless runs use plain `python`.
- The sidecar maps the typed `State` fields from named MJCF sensors and sites (the names ER-053 emits), not from array positions guessed at, and copies **every named MJCF sensor** into `State.sensors` (name and its `sensordata` slice), so a new sensor needs no sidecar change (UX-27 S2).
- **One command** (UX-27 S6): a small Go launcher, `x/sim/sidecar.Start(ctx, opts) (addr string, stop func() error, err error)`, starts the sidecar on a temp Unix socket (under `mjpython` with the viewer) and stops it; `cmd/sim-wheeled --backend mujoco` and `cmd/simcheck` use it.
- Headless by default; `--viewer` opens MuJoCo's passive viewer for debugging.
- Exit codes and one-line errors for a bad model (MuJoCo's compile message passed through).

Not in scope: physics tuning; camera rendering (later); a `purego` binding (later).

Source/test edits follow the owner's autonomy instruction once Leader schedules this ticket. No download, hardware or flashing is authorised by this ticket unless it says so.

## Acceptance evidence required later

- [ ] With the owner's go recorded for the whole list, everything installed under `~/src/tmp/` with version, source and sha256 per item in the evidence; nothing else installed.
- [ ] An MJCF with an extra named sensor (a `rangefinder`) shows up in `State.sensors` with its name and value, with no sidecar code change.
- [ ] `sidecar.Start` returns a working address; `stop` leaves no process behind (checked with `pgrep`).
- [ ] ER-050's client, against the sidecar with ER-053's wheeled model, passes ER-049's conformance suite.
- [ ] A bad MJCF gives `Error` with MuJoCo's message; the sidecar stays up for the next `ModelLoad`.
- [ ] Round trip per control step (10 substeps of 2 ms) ≤ 2 ms median on the Mac (note's S1b kill criterion; if it fails: batch steps or move to `purego`, recorded as a decision for Leader).
- [ ] `--viewer` shows the wheeled robot moving under a scripted command (owner's or ux's visual check).

## Open questions

- Python sidecar vs a C sidecar (product: Python, smallest; the `purego` step removes the sidecar later anyway).

## Effort uncertainty and revision trigger

Product's guess: **M** (inferred). Revisit when a kill criterion fires or an open question changes the scope.

## Implementation evidence

None. Proposed ticket; becomes `ready` after ux's UX-27 second rating and verify.

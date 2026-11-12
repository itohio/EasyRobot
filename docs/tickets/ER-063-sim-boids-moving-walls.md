---
id: ER-063
title: "Boids with moving walls (port from tinygui) — x/math/swarm/boids"
type: ticket
status: ready
kind: implementation
owner: easyrobot
workstream:
milestone: none
tags: [easyrobot, ticket, simulation, tufty]
created: 2026-11-12
updated: 2026-11-12
---
# ER-063 — Boids with moving walls (port from tinygui) — x/math/swarm/boids

[[tickets/index|Ticket index]] · [[simulation-algorithms]] · Brief: `~/src/docs/tasks/ER-SIMALGO.md` (owner 2026-11-12: "liquid engine belongs in easyrobot …"; paths and RNG per Leader's decisions there)

**Brick:** Reynolds flocking in fixed point, inside a box that turns and shakes in the boids' world (the owner's "window into the boids' world").

## Goal and evidence

The flock handed over by tinygui, **pinned to tinytufty2040 `cfbd1b0`** (`app/boids.go`, `app/boidsbox.go`, `app/boids_test.go` at that commit; handover `~/src/docs/evidence/ER-SIMALGO-handover.md`): Q8 positions and velocities, separation/alignment/cohesion by a scan with an early reject, speed limited with the octagonal norm, signed distances divided with `/16` (`>>4` biased the flock), walls of a box whose angle and centre move (a `sinQ14` table). Its tests: stay inside the box within speed under turns, spins and shakes; untouched at all 256 box angles when no wall reaches; pushed by a moving wall; octant follows heading. Badge baseline: 64 boids 2.1–2.4 ms per step. tinygui's working tree changes these files after `cfbd1b0`; if a later tinytufty2040 commit should be the source instead, Leader re-pins it here before the port starts.

## Dependencies and scope

Package `x/math/swarm/boids`: `Flock` (N fixed at `New`), `Params`, `Box` as a concrete type (no interface in the per-boid loop, Leader's decision), `Step`, positions and headings for drawing. O(N²) scan for N ≤ 128; a spatial grid only if measured necessary. Not in scope: trails and colours (demo).

Port the fixed form, not `cfbd1b0` verbatim (review R-ER-SIMALGO-EF on `2c3114b`): before porting, read review R-TG-1 (`~/src/docs/review/R-TG-1.md`) for findings in the ported code (per-frame soft float, per-event allocations, T10) and carry the fixes into the port. The demo-only parts it flags, such as the Dash sparkline, stay out of the package.

Every piece: check written first (red), tests, host benchmark, `tinygo build -target=tufty2040` with no allocation in the package, verify, then commit. No hardware or flashing is authorised by this ticket; tinygui runs it on the badge.

## Acceptance evidence required later

- [ ] tinygui's four boids tests pass on the port, red first.
- [ ] Speed stays within [VMin, VMax] for 10,000 steps; no boid leaves the box when it turns 22.5° per press.
- [ ] 0 allocations per `Step`; host benchmark at N = 128; tufty2040 build.

## Effort uncertainty and revision trigger

easyrobot's guess: **M**. Revisit when the badge's frame budget or the owner's scope changes.

## Implementation evidence

None.

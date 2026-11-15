---
id: ER-063
title: "Boids with moving walls (port from a badge demo) — x/math/swarm/boids"
type: ticket
status: done
kind: implementation
owner: easyrobot
workstream:
milestone: none
tags: [easyrobot, ticket, simulation, tufty]
created: 2026-11-12
updated: 2026-11-12
---
# ER-063 — Boids with moving walls (port from a badge demo) — x/math/swarm/boids

[[tickets/index|Ticket index]] · [[simulation-algorithms]] · The liquid engine belongs in EasyRobot.

**Brick:** Reynolds flocking in fixed point, inside a box that turns and shakes in the boids' world (the owner's "window into the boids' world").

## Goal and evidence

The flock from the badge app, **pinned to one demo revision** (its boids, boids-box and boids test files): Q8 positions and velocities, separation/alignment/cohesion by a scan with an early reject, speed limited with the octagonal norm, signed distances divided with `/16` (`>>4` biased the flock), walls of a box whose angle and centre move (a `sinQ14` table). Its tests: stay inside the box within speed under turns, spins and shakes; untouched at all 256 box angles when no wall reaches; pushed by a moving wall; octant follows heading. Badge baseline: 64 boids 2.1–2.4 ms per step. The badge app changes these files after that revision; if a later demo revision should be the source instead, it is re-pinned here before the port starts.

## Dependencies and scope

Package `x/math/swarm/boids`: `Flock` (N fixed at `New`), `Params`, `Box` as a concrete type (no interface in the per-boid loop), `Step`, positions and headings for drawing. O(N²) scan for N ≤ 128; a spatial grid only if measured necessary. Not in scope: trails and colours (demo).

Port the fixed form, not that revision verbatim: before porting, read the earlier review of that code for findings in the ported code (per-frame soft float, per-event allocations, T10) and carry the fixes into the port. The demo-only parts it flags, such as the Dash sparkline, stay out of the package.

Every piece: check written first (red), tests, host benchmark, `tinygo build -target=tufty2040` with no allocation in the package, check, then commit. No hardware or flashing is authorised by this ticket; it is run on the badge separately.

## Acceptance evidence required later

- [x] The original four boids tests pass on the port, red first (against a missing package, 2026-11-12).
- [x] Speed stays within [VMin, VMax] for 10,000 steps; no boid leaves the box when it turns 22.5° per press (`TestTurnPerPressFor10000Steps`).
- [x] 0 allocations per `Step` (raw mallocs over 1,000 steps); host benchmark at N = 128 (70–88 µs per step); tufty2040 build (allocations only in `New`).

## Effort uncertainty and revision trigger

Estimate: **M**. Revisit when the badge's frame budget or the scope changes.

## Implementation evidence

`x/math/swarm/boids` (`boids.go`, `boids_test.go`), note [[boids-flocking]], `x/math/SPEC.md` §14. Ported from the pinned demo revision. That review has no finding in the ported files; its numerics note holds, and the sine table is now a literal. The one change to the flock's behaviour: a velocity that steering cancels to zero keeps its old heading instead of turning east, a bias the new mirror-symmetry test found. 14 tests, 25 mutants all fail; the flock core was bit-identical to the original in 60 of 60 runs. RP2040 step time not measured (no board run).

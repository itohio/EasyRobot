---
id: ER-066
title: "Ray casting: grid DDA traversal and a 2.5D column helper — x/math/grid"
type: ticket
status: proposed
kind: implementation
owner: easyrobot
workstream:
milestone: none
tags: [easyrobot, ticket, simulation, tufty]
created: 2026-11-12
updated: 2026-11-12
---
# ER-066 — Ray casting: grid DDA traversal and a 2.5D column helper — x/math/grid

[[tickets/index|Ticket index]] · [[simulation-algorithms]] · Brief: `~/src/docs/tasks/ER-SIMALGO.md` (owner 2026-11-12: "liquid engine belongs in easyrobot …"; paths and RNG per Leader's decisions there)

**Brick:** exact grid traversal (Amanatides–Woo DDA) returning the first occupied cell, the distance and the side hit, plus a per-column wall height for a 2.5D view.

## Goal and evidence

Owner (ER-SIMALGO): ray casting. `x/math/grid/raycast.go` already casts lidar rays against an occupancy grid by stepping; DDA visits each crossed cell exactly once.

## Dependencies and scope

In `x/math/grid` beside `RayCast`. The renderer (textures, the screen) stays in the demo.

Every piece: check written first (red), tests, host benchmark, `tinygo build -target=tufty2040` with no allocation in the package, verify, then commit. No hardware or flashing is authorised by this ticket; tinygui runs it on the badge.

## Acceptance evidence required later

- [ ] DDA visits exactly the cells a brute-force supersampled walk crosses, for random rays (incl. axis-parallel and corner-crossing rays).
- [ ] Column heights: fish-eye corrected (perpendicular distance), tested against a hand-built room.
- [ ] 0 allocations per ray; benchmark; tufty2040 build.

## Effort uncertainty and revision trigger

easyrobot's guess: **S**. Revisit when the badge's frame budget or the owner's scope changes.

## Implementation evidence

None.

---
id: ER-065
title: "Fluid dynamics: Stam stable fluids, 2D first — x/math/fluid"
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
# ER-065 — Fluid dynamics: Stam stable fluids, 2D first — x/math/fluid

[[tickets/index|Ticket index]] · [[simulation-algorithms]] · Brief: `~/src/docs/tasks/ER-SIMALGO.md` (owner 2026-11-12: "liquid engine belongs in easyrobot …"; paths and RNG per Leader's decisions there)

**Brick:** a grid fluid solver (velocity + density): semi-Lagrangian advection, diffusion, a pressure projection; 2D first, then 3D.

## Goal and evidence

Owner (ER-SIMALGO): "full fluid dynamics later". Stam's stable fluids are unconditionally stable at any step, which suits a fixed frame rate.

## Dependencies and scope

Package `x/math/fluid`: fixed-size grids, Jacobi or Gauss–Seidel pressure solve with a set iteration count, boundaries as solid cells. float32 vs fixed point decided by the RP2040 measurement. Not in scope: free surfaces (the automaton covers liquids).

Every piece: check written first (red), tests, host benchmark, `tinygo build -target=tufty2040` with no allocation in the package, verify, then commit. No hardware or flashing is authorised by this ticket; tinygui runs it on the badge.

## Acceptance evidence required later

- [ ] Divergence after the projection below a set tolerance; a still fluid stays still; a vortex keeps its circulation within a bound over N steps.
- [ ] Mass of the advected density conserved within a bound.
- [ ] Step time at 32×32 on the RP2040 measured; 0 allocations per step; tufty2040 build.

## Effort uncertainty and revision trigger

easyrobot's guess: **L**. Revisit when the badge's frame budget or the owner's scope changes.

## Implementation evidence

None.

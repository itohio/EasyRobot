---
id: ER-061
title: "2D liquid automaton with continuous gravity (port from tinytufty2040) — x/math/automata/liquid"
type: ticket
status: blocked
kind: implementation
owner: easyrobot
workstream:
milestone: none
tags: [easyrobot, ticket, simulation, tufty]
created: 2026-11-12
updated: 2026-11-12
---
# ER-061 — 2D liquid automaton with continuous gravity (port from tinytufty2040) — x/math/automata/liquid

[[tickets/index|Ticket index]] · [[simulation-algorithms]] · Brief: `~/src/docs/tasks/ER-SIMALGO.md` (owner 2026-11-12: "liquid engine belongs in easyrobot …"; paths and RNG per Leader's decisions there)

**Brick:** a falling-liquid cellular automaton on a w×h byte grid driven by a **continuous gravity vector** (from the IMU), so a small tilt moves the liquid proportionally. It is used for the badge's water and its fire's fuel.

## Goal and evidence

**Source (Leader m-e0a5dd1ae1f5):** the continuous-gravity engine in `../tinytufty2040/app/liquid.go` with `water_test.go` (and the fuel test in `fire_test.go`), **pinned to the tinytufty2040 commit that lands it (TG-2 step 8) — pending that commit**; port nothing until the revision is named here. In tinygui's working tree as of 2026-11-12, `setTilt(gx, gy)` makes cells fall along the two nearest lattice directions in proportion to the angle, and `water_test.go` adds `TestWaterSurfaceFollowsASlightTilt` to the three earlier tests. The rest is the handed-over engine (tinytufty2040 `cfbd1b0`; handover `~/src/docs/evidence/ER-SIMALGO-handover.md`; tinygui history `0c4a4fc`, `e023c19`, `e5cf189`, `62d7d1f`): cells keep a sideways direction; a step falls up to 3 cells, tries the diagonals, then slides along the surface (up to 32 cells of travel, legs of up to 16); the scan order alternates on both axes; an optional two-cell pour. Badge baseline for `cfbd1b0`'s engine: 64×64 step ≈ 5.2 ms (RP2040 200 MHz, TinyGo 0.43).

## Dependencies and scope

**Owner (2026-11-12, after a badge test):** the water stayed frozen when the badge was tilted slightly, because `cfbd1b0`'s engine snaps gravity to 8 directions. The port takes `SetGravity(gx, gy)` as a vector, in any unit (only its direction matters), and follows the pinned engine's split between the two nearest lattice directions, so sideways flow scales with the tilt. Gravity exactly along an axis or a diagonal behaves as the snapped engine.

Package `x/math/automata/liquid`: `Grid` over a caller buffer or one allocation at `New`; `Set/At/Count`; `SetGravity(gx, gy)`; `Step` with a private xorshift32 (deterministic seed). Not in scope: rendering, fire (demo), obstacles (add when a demo needs them).

Every piece: check written first (red), tests, host benchmark, `tinygo build -target=tufty2040` with no allocation in the package, verify, then commit. No hardware or flashing is authorised by this ticket; tinygui runs it on the badge.

## Acceptance evidence required later

- [ ] The pinned commit's water tests (four, incl. `TestWaterSurfaceFollowsASlightTilt`) and the fuel-never-consumed test pass on the port, red against an empty package first.
- [ ] `Step` conserves the cell count (the fuel invariant) over 10,000 steps at gravity angles every 5° around the circle, with pour off.
- [ ] Proportional tilt: from a level pool, the net sideways transport per step at tilts of 2°, 5°, 10° and 20° from vertical increases monotonically and is ≈ linear for small angles; tilt 0 gives no net drift. The same test is red against `cfbd1b0`'s snapped engine (2° and 5° give zero).
- [ ] No cell moves twice in a step (moved bit), tested.
- [ ] Same seed and steps → identical grids; 0 allocations per `Step`; host benchmark on a 64×64 grid (like the badge baseline) and tufty2040 build.

## Effort uncertainty and revision trigger

easyrobot's guess: **S**. Revisit when the badge's frame budget or the owner's scope changes.

## Implementation evidence

None.

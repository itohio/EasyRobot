---
id: ER-064
title: "3D liquid automaton driven by the IMU's continuous gravity vector"
type: ticket
status: proposed
kind: implementation
owner: easyrobot
workstream:
milestone: none
tags: [easyrobot, ticket, simulation, tufty]
created: 2026-11-12
updated: 2026-11-14
---
# ER-064 — 3D liquid automaton driven by the IMU's continuous gravity vector

[[tickets/index|Ticket index]] · [[simulation-algorithms]] · Scope (2026-11-12): the liquid engine belongs in EasyRobot

**Brick:** the 2D liquid's rules (ER-061) on a w×h×d grid, driven by the IMU's gravity vector as a continuous direction, not snapped to 26 directions. Small tilts move the liquid proportionally.

## Goal and evidence

Request: the liquid engine is 2D now, 3D next. The badge's MPU6050 gives a gravity vector ([[mpu6050-imu-driver]], tilt via [[mahony-ahrs]]). Owner, 2026-11-12: a slight tilt left the snapped 2D water frozen, so the engine must follow continuous gravity in 3D too.

## Dependencies and scope

Same package as ER-061 or `liquid3`, decided when ER-061 lands. Memory budget on the RP2040 (264 KB RAM): e.g. 32×32×16 = 16 KB at one byte per cell. Rendering a 3D grid on the 2D screen is the demo's.

Every piece: check written first (red), tests, host benchmark, `tinygo build -target=tufty2040` with no allocation in the package, check, then commit. No hardware or flashing is authorised by this ticket; it is run on the badge separately.

## Acceptance evidence required later

- [ ] Count conserved for gravity directions sampled over the sphere; a basin levels perpendicular to an oblique gravity within a bounded number of steps.
- [ ] Proportional tilt: net sideways transport grows monotonically with tilts of 2°, 5°, 10° and 20° about either horizontal axis. Tilt 0 gives no drift.
- [ ] Step time on the RP2040 within the badge's frame budget (measured on the badge); 0 allocations.

## Plan: 3-D with several liquids

Owner, 2026-11-13: after the 2-D optimisation, a 3-D simulation with several liquids, "different intra-particle interaction parameters so I can specify how much each particle interacts with other kinds of particles". Plan only; nothing is built yet.

**Outcome.** A few kinds of liquid (say water, oil, honey) in a 3-D box on the badge. For every pair of kinds, a kind with itself included, the owner sets how strongly they push apart, how strongly they cling, how close they settle and how much they drag each other along. Tilting pours them through each other: oil floats on water as a separate layer, two miscible kinds swirl into one, a sticky kind clings to itself and the walls.

**Model.** The PIC/FLIP engine of [[flip-liquid-2d]] (ER-061), fixed point and allocation-free, with a species byte per particle and two tables:

- per species: density (relative, Q8) and optionally its own FLIP share (a viscous kind is more PIC);
- per pair, symmetric (checked by `New`): `Spacing` (the distance a pair settles at; unlike kinds further apart than like kinds keep layers apart), `Push` (the share of an overlap one separation pass removes; 1/2 today), `Pull` (cohesion between `Spacing` and `Range`; negative repels; like-with-like above unlike gives surface tension and demixing; with a wall kind, wetting), `Range` (under one cell), `Drag` (velocity mixing of overlapping particles; miscible pairs drag, immiscible ones slip).

They enter the step in the separation pass (`Spacing`, `Push`, `Pull` and `Drag` per pair, one table read per pair check), the transfer to the grid (weights scaled by density, so the grid carries momentum), and a buoyancy term (gravity scaled by the particle's density against its cell's mean). The buoyancy term approximates a variable-density solve (1/ρ on faces), which is the fallback if layering fails. One species with the default pair entry must reproduce the committed 2-D engine bit for bit.

**3-D.** A w×h×d grid with three face sets; a third position and velocity component; stencils of 8 corners instead of 4; a solve over 6 neighbours instead of 4; the accelerometer's full gravity vector, continuous (the acceptance above).

**Memory**, by the 2-D layout: 16 B per particle (3 positions, 3 velocities, bucket index, last move, species), 6 B per face, 5 B per cell. These are raw bytes; `New` adds the header and TinyGo's 16-byte block rounding, about 1.5–2 % more. The 2-D `Size` equals `New` exactly on 32-bit TinyGo, so "`Size` equals `New`" carries over as a test.

| Box (cells, wall ring included) | Particles | Faces | Bytes (raw) | Badge (about 27 KB free) |
|---|---|---|---|---|
| 12×10×8 | 600 | 3,176 | ≈ 33.5 KB | too big; fits only with 16-bit face storage between steps (about 6 KB less) |
| 10×8×6 | 300 | 1,628 | ≈ 17 KB | fits (8×6×4 cells of liquid space) |

**Cost**, an estimate: stencils 4 → 8 corners, solve neighbours 4 → 6, and the pair neighbourhood 9 → 27 cells, so pair checks per particle may roughly triple at equal packing. The 2-D badge step at 24×18/400 measured 25.3 ms before the optimisation and 18.23 ms after it (separation change, `053cae1`; per-phase timing). The 3-D size on the badge is to be set by timing the 3-D engine there, not by extrapolation.

**Tiers**, one engine and API; the tier sets box, particle count and sweeps:

| Tier | Limit | 2-D | 3-D |
|---|---|---|---|
| Badge (RP2040: Cortex-M0+ without FPU or divide instruction; the SIO divider is unused by TinyGo) | memory, then arithmetic | 32×24/700 (21.4 KB) or 24×18/400 (12.3 KB) | 10×8×6/300 (≈ 17 KB); 12×10×8 with 16-bit faces |
| ESP32-S3 (2 × LX7 at 240 MHz, hardware divide, FPU; 416 KB DRAM in TinyGo's `esp32s3.ld`) | SRAM | 64×48, thousands of particles | 24×16×12/3000 ≈ 160 KB; the build tag selecting `sdiv` narrows to cores without a divide |
| Desktop | none practical | any | 64³/500,000 ≈ 14.1 MB; for tuning the pair matrix, GIFs, and trying ghost-fluid pressure |

**Route.** S1 adds species and the pair matrix to `x/math/fluid/flip` (one species equal to the committed engine bit for bit). S2 is the 3-D engine in its own package (`x/math/fluid/flip3`), sharing the method, not code paths. The 2-D tests port one by one: stillness over gravity directions sampled on the sphere, slosh, reversal (a 2-D fix is frozen but not committed), the surface acceptance over directions, `Size` equal to `New`. The −90° terrace (4.5° at gravity 0.02) carries over; ghost-fluid pressure would fix it and is deferred.

**Stages**, each test-first, gated and verified:

1. S1, several liquids in 2-D (new ticket): one species reproduces the committed engine's golden hashes; two immiscible kinds of different density layer heavy below within N steps at any gravity direction and stay layered when still (the stillness sweep, 72 directions × 5 gravity strengths, per species); a miscible pair mixes; cohesion holds a drop together; an asymmetric matrix is rejected; 0 allocations; the badge step time with two species.
2. S2, 3-D with one liquid (this ticket's acceptance plus the 2-D checks lifted to 3-D; memory and badge time at 10×8×6).
3. S3, 3-D with several liquids (S1's tests in 3-D).
4. S4, the badge: drawing and a hand check.

**Decisions for the owner:** at most 4 kinds (recommended: 4×4 matrix, 2-bit tag) or 8; whether densities matter (oil floats on water) or only interaction strengths; 2-D first (recommended) or straight to 3-D; 10×8×6 now or wait for 16-bit face storage for 12×10×8.

**Risks:** cohesion against stillness (a pull makes particles creep at rest; every species pair must pass the stillness sweep, and the pull may need the rest bar); the buoyancy approximation holds for modest density ratios (oil to water) but not air to water; the pair table adds one load per pair check (13,786 per step at 32×24/700 in 2-D), small next to the check.

Verified and rechecked.

## Effort uncertainty and revision trigger

Estimate: **M**. Revisit when the badge's frame budget or the scope changes.

## Implementation evidence

None.

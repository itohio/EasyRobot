---
id: ER-061
title: "2D liquid engine: PIC/FLIP particles on a grid, fixed point — x/math/fluid/flip"
type: ticket
status: in-progress
kind: implementation
owner: easyrobot
workstream:
milestone: none
tags: [easyrobot, ticket, simulation, tufty]
created: 2026-11-12
updated: 2026-11-13
---
# ER-061 — 2D liquid engine: PIC/FLIP particles on a grid, fixed point — x/math/fluid/flip

[[tickets/index|Ticket index]] · [[simulation-algorithms]] · [[flip-liquid-2d]] · Related: [[ER-065-sim-fluid-dynamics|ER-065]] (stable fluids, no free surface)

**Brick:** a real 2D liquid for the Tufty badge and any small screen: it keeps momentum, waves travel and reflect, it splashes, its volume is conserved, and at rest its surface lies flat across any gravity vector the IMU gives. Integer only (the RP2040 has no FPU), no allocation per step. The badge app renders it.

## Goal and evidence

**2026-11-13:** the previous liquid simulation did not behave like a liquid at all and had to be fixed. The cellular automaton (the badge app's `app/liquid.go` and this ticket's earlier plan to port it) is dropped: a byte grid where cells fall and slide has no velocity, so it cannot slosh, carry a wave or splash. This ticket now builds a particle liquid instead. Acceptance from the order: momentum (sloshes after a tilt and settles), waves travel and reflect, volume conserved, surface perpendicular to an arbitrary gravity vector at rest, no jitter when still, splashes on a fast tilt; fixed point; zero allocations per step; step time measured on the tufty2040 at a size that gives a usable frame rate; host GIFs of a tilt sequence.

## Method: three options compared

Costs are per frame at the same particle count (~700), counted from the algorithms, not measured; only the chosen one was built.

| | Liquid behaviour | At rest | RP2040 cost per frame |
|---|---|---|---|
| **SPH, weakly compressible** (Monaghan; Becker & Teschner 2007) | lively, splashes; compressible, so it bounces and the level breathes unless very stiff | jitters unless the time step is tiny; clumps at the free surface | 2 neighbour passes with kernel powers and a square root per pair, and the stiffness needs 4–10 substeps a frame: the most expensive |
| **Position-based fluids** (Macklin & Müller 2013) | incompressible, stable at large steps, splashes; needs artificial viscosity and surface terms | settles if enough iterations; under-iterated it compresses at the bottom | 3–4 iterations × 3 neighbour passes (density, λ, Δp), each with kernel gradients: about 3× FLIP |
| **PIC/FLIP** (Zhu & Bridson 2005; Müller's "FLIP water") — **chosen** | particles carry velocity, a coarse grid makes it divergence-free; FLIP keeps little numerical viscosity, so sloshing, waves and splashes look right | gravity is balanced by pressure on the grid; needs care at rest (below) | 2 grid transfers (4-point stencils), Gauss–Seidel sweeps over the liquid cells only (~400 at 32×24), 2 short separation passes; no square root or divide per pair |

**Why PIC/FLIP:** it is the cheapest of the three that is truly incompressible, its cost is mostly the grid (cells, not particle pairs), and the grid maps naturally onto the badge's pixelated look. Its known weakness, noise at rest, is handled by five measures, each pinned by a test except where [[flip-liquid-2d]] says otherwise:

1. the pressure solve starts from last step's pressure (warm start);
2. FLIP fades into PIC below a calm speed;
3. particles slower than 0.8 of a step's gravity don't move;
4. a liquid cell stays liquid while a particle is within 3/16 cell of it, so surface cells don't flicker;
5. a still particle keeps only 7/8 per step of a push back against its last move, so a lone surface particle can't rock across a cell edge for ever; the last moves are forgotten whenever gravity turns by about 0.45°, so tilting back isn't damped.

Volume is held by pushing overlapping particles apart and by a density drift term, which pushes out of packed cells and pulls loose ones together deep in the liquid.

## Dependencies and scope

**Package:** `x/math/fluid/flip`; the path is recorded here. `x/math/fluid` itself stays for [[ER-065-sim-fluid-dynamics|ER-065]]'s stable-fluids grid solver (smoke and gas, no free surface); a free-surface particle liquid is a different method, so it gets its own subpackage. `x/math/automata/liquid` is not created, and the earlier pin to the badge app's automaton is dropped with it.

**API:** `New(w, h, n, Params)` (the only allocation: `Size(w, h, n)` bytes, the `Liquid` and its arrays, exact on 32-bit TinyGo), `Reset`, `Step`, `Raster(dst, w, h)` for drawing at any pixel size; `GX, GY` gravity (Q16) and `X, Y` (1/256 cell), `VX, VY` (1/16384 cell per step) particles are plain fields. One instance can serve several pages (Water, Fire's fuel). Not in scope: obstacles inside the box, two-way coupling with rigid bodies, rendering (the badge app's).

**Hardware:** no hardware or flashing in this ticket; a separate badge timer program is flashed to measure it.

## Acceptance evidence required later

- [x] Momentum: `TestSloshesAndSettlesAfterTilt` — after tilts of 20°–45° and back at gravity 0.01–0.02 the mean crosses its rest position 5–8 times with decaying peaks, then no particle moves and the surface is level (worst 3.0°).
- [x] Waves travel and reflect: `TestDamBreakWaveTravelsAndReflects` — a dam's front reaches the far wall no faster than Ritter's 2√(gh₀) (step 45 against 25), climbs it, and the reflection raises the near wall again (step 185).
- [x] Volume conserved: the particle count is fixed. `TestVolumeConservedThroughShaking`: after five fast tilts the level comes back within 0.5 cell (0.05 measured).
- [x] Surface perpendicular at rest: `TestRestSurfacePerpendicularToGravity` at 0°, 25°, −40°, −60°, 90°, 135°, 180°: fitted surface within 3° of perpendicular (worst 2.4°), depth spread ≤ 1.5 cells (worst 1.00).
- [x] No jitter when still, at any gravity direction (review: 6 of 36 directions flickered for ever): `TestStillLiquidDoesNotJitter` at every 5° and gravity 0.01, 0.0125, 0.015, 0.0175, 0.02 (360 cases): drift < 0.05 cell and ≤ 10 pixel changes in 1000 frames; 0 in every case. Red first: the r2 engine fails 68 of them.
- [x] A slight tilt moves the liquid: `TestSlightTiltMovesTheLiquid` — turning gravity from 1° or 2° on one side to the other moves the mean ≥ 0.3 cell that way (0.30–1.03; a level surface would move 0.37 for the 1° turn that gives 0.30).
- [x] Splashes: `TestFastTiltSplashes` — a 100° swing in 4 steps throws 10 droplets clear; the same swing over 400 steps throws 4; none at rest.
- [x] Physics sanity (review recommendation): `TestSloshPeriodMatchesWaveTheory` — slosh period 174 and 250 steps against linear wave theory's 164 and 233, and √2 longer at half gravity (1.44).
- [x] Fixed point, no float in the package; 0 allocations per `Step` and `Raster` (`TestStepDoesNotAllocate`); TinyGo 0.43 tufty2040 build with allocations only in `New`.
- [x] Host GIFs of a tilt sequence, a fast-tilt splash, a dam break and a still pool (evidence below).
- [x] Small enough for the badge (budget: ≤ ~25 KB at 32×24/700 and ≤ ~15 KB at 24×18/400; about 27 KB free). 16-bit particle storage, the pre-solve copy holding the transfer weights, a shared density/bucket array, 16-bit warm-start pressure, and face validity, walls and open sides worked out instead of stored. `New` on 32-bit TinyGo (`TotalAlloc` around it, `wasip1`, `-gc=leaking`, the review's method; `Size` counts the `Liquid` and its arrays as TinyGo rounds them, and equals it):
  - 32×24 cells, 700 particles: 21,408 B, all of the checks above; the r3 engine took 52.6 KB by the same method (the r2 engine's 52.9 KB is its badge heap, below);
  - 24×18, 400: 12,288 B;
  - 28×21, 524: 16,400 B, still at all 108 directions × gravities tried, sloshes with ≥ 4 crossings, settles within 3.7°, period within 15 %;
  - 24×18, 373: 12,032 B, still, sloshes 3 times, overdamped at strong gravity;
  - 20×15 (8,336 B) and 16×12 (5,392 B): still, but they pour rather than slosh.
  - `TestSizeIsWhatNewAllocates` pins `Size`; `TestDeepPoolStaysStill` pins the 16-bit pressure (it saturates past 25 cells of depth at 0.02, without wrapping round).
- [ ] Step time on the tufty2040 at 32×24 cells and 700 particles, measured with the badge timer. If too slow for a usable frame rate, the size to use (fewer particles or cells) is chosen from that measurement. Measured on the previous engine: 38.3–39.3 ms per `Step`, about 25 steps/s, room for about one step per demo frame; the size is still to be chosen. This engine: 25.3 ms at 24×18/400 and 44.8 ms at 32×24/700; after task 2e (exact divide-free division under TinyGo) 20.8 and 36.6 ms (Task 2e: fromGrid −44 %, toGrid and separate −18 %, the rest unchanged).

## Effort uncertainty and revision trigger

Estimate: **M**. Revisit when the badge measurement is in (frame budget) or the GIFs are judged.

## Implementation evidence

- `x/math/fluid/flip/flip.go`, `flip_test.go`: `go test -race -short` and a full `go test` of `./x/math/fluid/flip/` pass (17 tests, 2026-11-13; 18 with `TestSdivIsGoDivision` since task 2e). The first tests failed to build without the package; they were written alongside the engine. The stillness sweep was red first against the r2 engine. 42 of 43 mutants fail a test; the survivor (rest density from the 0.3 radius rather than the 76/256 positions hold, 1.5 % apart) is measured and explained there.
- Memory and per-size behaviour: `flip-sizes-r4.txt` (bytes, host µs per step, still sweep, slosh, slight tilt and period at five sizes).
- Host: `BenchmarkStep` 105.5–107.3 µs per step at 700 particles on 32×24 cells at a quiet moment, 0 allocs (Apple M5, shared machine).
- TinyGo 0.43 `-target=tufty2040` (the badge timer, now timing five sizes and printing their bytes): 20,644 B flash; `-print-allocs=flip` reports only `New`.
- Tufty 2040 (2026-11-13, the r2 engine): `Step` 38.3–39.3 ms, `Raster(64×48)` 0.31 ms, 0 heap allocations per 30 steps, `New` 52.9 KB.
- Performance task 2e (division): golden hashes on both division paths equal `6671770`'s; badge per phase before and after (Task 2e: fromGrid −44 %, toGrid and separate −18 %, the rest unchanged).
- Performance task 2b (the solve: one cell byte per update, rows with no liquid skipped): golden hashes on both division paths equal `6671770`'s; badge timed per phase before and after.
- Performance task 2a (the separation: position in registers, neighbour cells as slices): golden hashes on both division paths equal `6671770`'s; badge timed per phase before and after.
- Performance task 2c (fromGrid: blend on plain values, cell reads inlined): golden hashes on both division paths equal `6671770`'s; badge timed per phase before and after.
- Performance task 2d (toGrid: interior-only normalisation, face arrays in locals): golden hashes on both division paths equal `6671770`'s, old and new `toGrid` identical in 12,960 states over every box 3×3–20×20 (a throwaway equivalence test); badge timed per phase after.
- GIFs and their stats were made with this engine (tilt, splash, dam, still, still at 60°; 60° flickered before the jitter fix).

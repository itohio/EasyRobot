---
type: algorithm
status: tested
tags: [easyrobot, simulation, fluid, game, tinygo]
workstream:
module: "[[simulation-algorithms]]"
summary: "2D PIC/FLIP liquid in a walled box: particles on a staggered grid, warm-started pressure solve; sloshes, waves, splashes, flat still surface across any gravity; integer only, 16-bit storage (21 KB at 32×24 cells, 700 particles), no alloc after New"
source: [x/math/fluid/flip/flip.go]
tests: [x/math/fluid/flip/flip_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test ./x/math/fluid/flip pass (17 tests, 2026-11-13; -race with -short): still pool at every 5° × 5 gravities (360 cases) 0 pixel changes in 1000 frames (the r2 engine fails 68), sloshes decay and settle level within 3.0°, 1° and 2° tilts both ways move the liquid, slosh period within 6–8 % of linear wave theory and √2 longer at half gravity, dam front ≤ Ritter speed and reflects, level back within 0.05 cell after shaking, still surface within 2.4° of perpendicular at 7 angles, fast tilt 10 droplets vs slow 4, a 50-deep pool still; Size = what New allocates on 32-bit TinyGo, the Liquid included (21,408 B at 32×24/700, exact at 7 sizes; the r2 engine 52.9 KB on the badge); 42 of 43 mutants fail (ER-061-liquid/flip-mutants-r4.txt); 0 allocs per Step; host 105–107 µs per step; tinygo 0.43 tufty2040 build, allocations only in New; Tufty 2040 (tinygui, r2 engine) 38.3–39.3 ms per Step, this engine not yet timed there"
---
# 2D liquid: PIC/FLIP particles on a grid (flip)

**What it does:** simulates a liquid in a walled 2D box under a gravity vector that can point anywhere (an accelerometer's in-plane reading). Particles carry the liquid's velocity; each step a coarse staggered grid makes that velocity divergence-free, so the liquid is incompressible, keeps momentum (sloshes, carries waves, splashes) and at rest lies flat across gravity. Integer only, for the owner's Tufty 2040 badge (RP2040, no FPU; ER-061), usable for any small-screen demo.

**Where:** `x/math/fluid/flip/flip.go`:

- types and setup: `Params`, `DefaultParams` (:36-60), `Liquid` (:80-101), `Size` (:107), `New` (:114-141), `Reset` (:147-167);
- `Step` (:175-192) and its stages: `gravity` (:197), `advect` (:240-283), `walls` (:305), `bucket`, `separate`, `pair`, `half` (:329-423), `toGrid`, `mark` (:452-563), `solve`, `push` (:576-646), `fromGrid`, `blend` (:655-703);
- drawing: `Raster` (:706).

**Inputs → outputs:** `New(w, h, n, Params)` (3 ≤ w, h ≤ 255 cells, the outer ring is wall; 1 ≤ n ≤ 65535) makes n particles packed at rest on the floor (`Reset`) and allocates `Size(w, h, n)` bytes (the `Liquid` and its arrays), nothing after. Positions `X, Y` are `uint16` in 1/256 cell (`Cell`), velocities `VX, VY` `int16` in 1/16384 cell per step (`VOne`), gravity `GX, GY` and `Params` in 1/65536 (`One`). Set gravity, call `Step` (one or more per frame), then `Raster(dst, pw, ph)` counts particles per pixel over the whole box for drawing at any size (2 px per cell gives the badge's pixelated look).

**Method (per step):**

1. **To the grid:** cells are wall (the ring), liquid (a particle's centre is in it, or it was liquid and a particle is within `Stick`, 3/16 cell, of it), or air. Particle velocities are spread onto the faces with bilinear weights (u at (i, j+½), v at (i+½, j)) and normalised by the weight sums; wall faces are zeroed; a copy is kept for FLIP.
2. **Gravity** is added to every face with liquid on a side.
3. **Pressure (Gauss–Seidel, over-relaxed 1.9):**
   - each sweep makes each liquid cell's divergence zero through its open sides;
   - the solve **starts from last step's pressure**, so a liquid at rest is already balanced and few sweeps hold it;
   - a **drift** term asks for outflow from cells packed denser than `Reset` packs them, and for inflow to loose cells deep inside (all four neighbours liquid). This holds the volume after shaking.
4. **Back to particles:** each particle takes the grid velocity (PIC) blended with its own plus the grid's change (FLIP), 90 % FLIP, over the faces next to liquid. Below `Calm` the blend fades to PIC, so a liquid coming to rest damps instead of circulating.
5. **Move:** velocities are limited to `MaxV` (under a cell per step). A particle slower than `RestQ8`/256 of gravity's pull per step does not move. The grid leaves surface particles a fraction of that, so a still liquid stays exactly still; a tilt leaves gravity unbalanced and moves it within a few steps. A particle that does not move keeps only `HoldQ8`/256 (7/8) per step of a velocity pointing back against its last move. Without that, a lone surface particle on a cell edge rocks for ever: it creeps across the edge, the cells change hands, the push reverses, its speed builds up past the bar while it stands, and it creeps back (at 6 of 36 gravity directions, up to 184 pixel changes per 1000 steps). Each time gravity turns by more than 1/128 of its strength (about 0.45°) the last moves are forgotten, so tilting back the other way is not damped.
6. **Separate:** particles closer than two radii d₀ are each pushed half the overlap along their line: (d₀ − d)·Δ/(2d), rounded to the nearest position unit (truncated, a push falls a unit short of a whole-unit move and a particle pressed into its neighbour wobbles by that unit for ever). Only overlapping pairs cost a square root (8 steps, d² < 2¹⁵) and a divide; a liquid packed at rest has none. The pairs are found from cell buckets (each cell with itself and its four later neighbours).
7. **Walls:** particles are kept a radius inside the ring and lose their speed into it.

**Numbers:** the arithmetic is int32 in Q16 (velocities, gravity, pressure) with Q8 weights; every product is checked to stay in int32 (the speed limit bounds the transfers). Storage is 16-bit where it can be: positions Q8 (`uint16`, so boxes up to 255 cells), velocities Q14 (`int16`, up to ±2 cells per step), a particle moves by its velocity rounded to 1/256 cell. Divides normalise the transfers (one per face and two per particle) and size the separation pushes.

**Memory:** `Size(w, h, n)` = 11 bytes per particle (positions, velocities, a bucket index, the last-move bits) + 6 per grid face (a 32-bit velocity, and a 16-bit copy that first holds the transfer's weight sums) + 5 per cell (type and last step's liquid bit in one byte; 16-bit density, shared with the separation buckets; 16-bit pressure): the arrays, plus the `Liquid` itself (244 bytes on 32-bit), each of the 14 allocations rounded up to TinyGo's 16-byte heap blocks. On 32-bit TinyGo that is exactly what `New` takes (`ER-061-liquid/flip-new-wasip1-size.txt`). Face validity, wall faces and open sides are worked out from the indices rather than stored. The pressure is kept in 16 bits for the warm start: it holds g·depth up to half a cell per step (0.02 cell per step² under 25 cells of liquid) and saturates beyond (`TestDeepPoolStaysStill`, 50 cells deep).

| Box (cells), particles | `Size` | `New` on 32-bit TinyGo | previous engine | host per step | liquid behaviour (fpv-gate's checks, per size) |
|---|---:|---:|---:|---:|---|
| 32×24, 700 | 21,408 B | 21,408 B | 52.6 KB (the r3 engine; fpv-gate, same method: m-30911cf13ed8, `docs/evidence/ER-SIMALGO.md` L233) | 106–107 µs | all tests; still 0 of 360 directions × gravities flicker; sloshes ≥ 5 crossings |
| 28×21, 524 | 16,400 B | 16,400 B | — | 76 µs | still 0 of 108; sloshes ≥ 4 crossings, settles within 3.7°; period within 15 % |
| 24×18, 400 | 12,288 B | 12,288 B | — | — | (373 particles tested, below) |
| 24×18, 373 | 12,032 B | 12,032 B | — | 53 µs | still 0 of 108; sloshes 3 crossings, then overdamped at strong gravity |
| 20×15, 248 | 8,336 B | 8,336 B | — | 36 µs | still; overdamped (2 crossings): pours more than sloshes |
| 16×12, 148 | 5,392 B | 5,392 B | — | 21 µs | still; no real slosh, surface up to 5° off |

`New` on 32-bit TinyGo: `TotalAlloc` around `New` in a `wasip1` build with `-gc=leaking` (fpv-gate's method; `tmp/easyrobot/simalgo/flipmem`, `ER-061-liquid/flip-new-wasip1-r4.txt`, `flip-new-wasip1-size.txt`).

Host times are on a shared Apple M5 (32×24 at a quiet moment, the others under load, when 32×24 took 146 µs); the badge timer `fliptufty` prints `Size`, the heap it grew and µs per step for each size. The previous engine's 32×24 step took 38.3–39.3 ms on the badge; on the host this one is as fast at a quiet moment (105.5–107.3 µs against 100–105 µs) and 15–20 % slower under load.

**Limits:**

- Rectangular box only, no obstacles inside.
- Tuned for gravity of 0.01–0.02 cell per step²; take more steps per frame rather than stronger gravity (at 0.035 a still surface flickers again).
- Smooth the accelerometer: gravity noise moves a still liquid by itself, and noise above 0.45° keeps the last moves forgotten (a lone surface particle can then rock again).
- Fewer than 12 sweeps let the warm-started solve run away (measured on the previous engine).
- Packing varies a little: the level after violent shaking is within about 0.1–0.2 cell of the start, not exact.
- The surface is level to within one row of particles (0.57 cell): the last part-filled row comes to rest lying to one side as a terrace, which a fitted line reads as a tilt. How much depends on how full that row is, not on the direction: at gravity −90° and 0.02, 680–720 particles rest at +0.4°, +4.1°, −0.5°, −4.4°, +4.5° (700), −0.6°, −1.0°, −0.2°, 0.0°. Over every 15° plus five off-grid directions at three gravities (87 cases) 8 are outside 3° (worst 4.5°: fpv-gate's −90° at 0.02). The pressure solve sees whole cells, so a step under one cell drives no flow, and the rest bar holds it; levelling it needs a sub-cell free surface in the solve (ghost-fluid pressure), not a parameter (`Calm`, `FlipQ8`, `RestQ8`, `HoldQ8`, `Stick` tried: none better). The mean x can then be up to about 0.6 cell from where it started.
- Small sloshes die fast: released from 8° at gravity 0.02 the pool crosses its rest twice (from 20° four times, 30° seven). A slosh moving the mean by under about half a cell is a wave under one cell, which the grid does not carry: pure FLIP with no rest bar still loses it in two crossings (and then never settles). A less damped default (`Calm` 1/32) adds a crossing or two at gravity 0.01 but leaves the still pool moving in many of the 360 directions × gravities, so the default stays.
- A wedge of liquid in a corner can creep for a few thousand steps before it is fully level.

## Method choice

ER-061 compares weakly compressible SPH, position-based fluids and PIC/FLIP on liquid behaviour and RP2040 cost. PIC/FLIP is the cheapest that is truly incompressible: its cost is mostly the grid, not particle pairs. Its weakness, noise at rest, needed five measures, each found by measurement here and pinned by a test (see the mutants):

- **warm start:** without it, gravity is never quite balanced. With 1/65536-cell positions the damped push-back hid that (the warm start's mutant survived); with 1/256-cell positions a jammed layer against the ceiling kept a speed just above the rest bar and jostled for ever (115° at 0.0175: 472 changes per 1000 steps);
- **calm PIC:** pure FLIP keeps a slanted surface circulating;
- **rest bar:** surface particles otherwise creep;
- **sticky cells:** a surface particle on a cell edge otherwise flips the cell each step and flickers;
- **damped push-back:** a lone surface particle otherwise rocks across a cell edge for ever at some gravity directions (fpv-gate found 6 of 36). Stickier cells alone (1/4 cell) still left one; damping every push of a still particle stopped the rocking but ignored tilts under 3°; a wake-up bar above the rest bar kept the whole pool sloshing. Damping only the push back, and forgetting the last moves when gravity turns, stops the rocking and still follows a 0.5° tilt.

The first build without drift lost 10–20 % of its level after shaking; drift that only pushed out made it boil. Pulling loose interior cells together fixed both.

## Completeness review

- Complete: no. Everything in ER-061 except choosing the badge's size from the badge timing (below).
- Tested (`flip_test.go`; 32×24 box, 700 particles, gravity 0.01–0.02):
  - **Acceptance:**
    - `TestSloshesAndSettlesAfterTilt`: 4 tilts at 3 gravities, 5–8 rest crossings with decaying peaks, then still and level (fitted surface ≤ 3.5°, worst 3.0°; spread ≤ 1.5 cells).
    - `TestDamBreakWaveTravelsAndReflects`: front at the far wall at step 45, not before Ritter's 25; reflection back at step 185.
    - `TestVolumeConservedThroughShaking`: level within 0.5 cell, 0.05 measured.
    - `TestRestSurfacePerpendicularToGravity`: 7 angles, fitted surface ≤ 3° off, worst 2.4°; depth spread ≤ 1.5 cells.
    - `TestStillLiquidDoesNotJitter`: every 5° at gravity 0.01, 0.0125, 0.015, 0.0175 and 0.02 (360 cases), 6000 steps after the turn: no particle drifts 0.05 cell and at most 10 pixel changes in 1000 frames; 0 in every case. The r2 engine fails 68 of them (`ER-061-liquid/flip-jitter-fail-before.txt`). `-short` runs every 30° at 0.02.
    - `TestSlightTiltMovesTheLiquid`: turned from 1° or 2° on one side to the other, at 2 gravities: mean x moves ≥ 0.3 cell that way (0.30–1.03; the 0.30 is a 1° turn at g 0.02, where a level surface would move 0.37).
    - `TestFastTiltSplashes`: 10 droplets on a fast tilt, 4 on a slow one, 0 at rest.
  - **Physics:** `TestSloshPeriodMatchesWaveTheory`: slosh period 174 and 250 steps against 164 and 233 from linear wave theory (≤ 15 %), and halving gravity lengthens it 1.44 times (√2 within 5 %).
  - **Mechanics:**
    - `TestOverlappingParticlesEndTwoRadiiApart` (exactly 2r, on the same line);
    - `TestParticleStopsAtWall`, `TestSpeedLimit`;
    - `TestResetForgetsThePast` (Reset steps exactly like New: after the liquid was against the ceiling, with a particle a hair under a formerly liquid cell, and under the same gravity as before);
    - `TestRasterMapsCellsToPixels` (unequal x and y scales), `TestNewRejectsParams`;
    - `TestSizeIsWhatNewAllocates`, `TestDeepPoolStaysStill` (16×64 cells, 50 deep: the 16-bit pressure saturates, the pool stays still);
    - `TestStepDoesNotAllocate` (Step and Raster).
  - The first tests failed to build without the package (`ER-061-liquid/flip-fail-before.txt`); they were written alongside the engine, not before it. The 360-case stillness sweep was red first against the r2 engine (68 failures).
- Correct: yes on these properties; 42 of 43 mutants fail a test (`ER-061-liquid/flip-mutants-r4.txt`), among them each part of the compact storage: no warm start, pressure wrapping instead of saturating, separation pushes truncated, a move truncated instead of rounded, velocities stored truncated, every stencil face counted, `Size` forgetting the pressure. The survivor: the rest density from the 0.3 radius instead of the 76/256 positions hold (1.5 % apart; kept, so the density aims at what `Reset` packs).
- Speed: host `BenchmarkStep` 105.5–107.3 µs per step at a quiet moment (Apple M5, shared machine, 0 allocs; `ER-061-liquid/flip-bench-host-r4.txt`; under load both engines reach 120–150 µs). Tufty 2040 (tinygui, 2026-11-13, the r2 engine, 32×24 cells, 700 particles): `Step` 38.3–39.3 ms, `Raster(64×48)` 0.31 ms, 0 heap allocations per step (`docs/evidence/er-timing/fliptufty-serial.txt`). That is about 25 steps/s with nothing else running; the demo frame leaves room for about one step. This engine is not yet timed on the badge; `fliptufty` now prints the bytes and µs per step for five sizes.
- TinyGo 0.43 `-target=tufty2040` (`fliptufty`, five sizes): 20,644 B flash; `-print-allocs=flip` reports only `New` (`flip-tinygo-tufty2040-r4.txt`).
- GIFs (made with this engine): `~/src/docs/evidence/ER-061-liquid/flip-{tilt,splash,dam,still,still-60}.gif`; `still-60` is a direction that flickered before the jitter fix.

## TODOs

- [ ] Pick the badge's size from the badge timing (38–39 ms per step at 32×24 cells and 700 particles on the previous engine; `fliptufty` times five sizes): a step every frame or every second frame, fewer particles or a smaller box — x/math/fluid/flip/flip.go:175 #p1 #ER-061
- [ ] Before cutting quality: profile on the badge (on the host separation is 40 % of a step, the solve a quarter) — x/math/fluid/flip/flip.go:356 #p2

## Links

- Module: [[simulation-algorithms]] · Related: [[boids-flocking]], [[rigid-body-2d-contact]] (same badge) · Ticket: [[ER-061-sim-liquid-2d]]

---
type: module
group: "Math and maps"
status: tested
tags: [easyrobot, simulation, physics, game]
workstream:
packages: [x/math/physics/rigid2d, x/math/swarm/boids, x/math/fluid/flip]
test_result: pass
---
# Simulation algorithms (games and demos)

**Purpose:** generic simulation bricks for demos and games, first for the owner's Tufty badge (RP2040, TinyGo): small fixed-step models with no allocation per step. The robot simulator (MuJoCo lockstep, `x/sim`, ER-048…060) is a different thing and lives elsewhere.

**Packages:** `x/math/physics/rigid2d` (rigid2d.go 266, rigid2d_test.go 569), `x/math/swarm/boids` (boids.go 380, boids_test.go 351) and `x/math/fluid/flip` (flip.go 722, flip_test.go 681); no build tags. Planned here, by Leader's path decision (ER-SIMALGO): `x/math/automata/life`, `x/math/fluid` (ER-065, stable fluids); ray casting goes to `x/math/grid` ([[grid-maps]]). The liquid is `x/math/fluid/flip` (ER-061, easyrobot's call after the owner dropped the cellular automaton, 2026-11-13).

## Public API

- `rigid2d.World{Params, Balls, Segments, Boxes}.Step(dt)`. `Ball{X, Y, VX, VY, W, R, Mass, InertiaK}`. Kinematic `Segment{A, B, V, Material}` and `Box{Min, Max, V, Material}`. `Material{Restitution, Friction}`. `Params{GX, GY, Magnus, LinearDamping, AngularDamping}`.
- Composition for Pong: paddles are `Box`es whose `VY` the game sets from the buttons (or from the IMU tilt); walls are `Segment`s; the IMU's gravity can feed `GX, GY`.
- `boids.New(n, w, h, Params) (*Flock, error)`; per frame `Flock.Box.Advance()` then `Flock.Step()`. `Box{Target, Spin}` and `Box.Shake(dir, dist)` move the window; `Flock.Cell(i)` gives a boid's cell and heading octant for drawing. Fixed point (1/256 cell), integer only.
- `flip.New(w, h, n, Params) (*Liquid, error)` allocates the `Liquid` and `flip.Size(w, h, n)` bytes of arrays: 21,408 B in all on 32-bit TinyGo for 32×24 cells and 700 particles, 12,288 B for 24×18 and 400; one instance can serve several pages (Water, Fire's fuel). Set `Liquid.GX, GY` (the IMU's in-plane gravity, Q16 cells per step², smoothed so its noise stays under about 0.45°), call `Step()` one or more times a frame, draw with `Raster(dst, pw, ph)` (particles per pixel over the box). `Reset()` refills the pool. Integer only; particles are stored in 16 bits (`X, Y` in 1/256 cell, `VX, VY` in 1/16384 cell per step).

## Algorithms

<!-- vault:algorithms h=adcb8c199c8d -->

Counts: 3 tested · 0 implemented · 0 partial · 0 stub · 0 unverified (3 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[boids-flocking]] | tested | pass | fixed-point Reynolds flocking inside a box that turns and shakes in the boids' world; integer only, no alloc after New |
| [[flip-liquid-2d]] | tested | pass | 2D PIC/FLIP liquid in a walled box: particles on a staggered grid, warm-started pressure solve; sloshes, waves, splashes, flat still surface across any gravity; integer only, 16-bit storage (21 KB at 32×24 cells, 700 particles), no alloc after New |
| [[rigid-body-2d-contact]] | tested | pass | spinning balls vs moving walls/boxes: restitution, Coulomb friction, Magnus, damping; no alloc |

<!-- /vault:algorithms -->

## Build targets and dependencies

Pure Go, no allocation per step. `rigid2d` is float32 (`chewxy/math32`); TinyGo 0.43 `-target=tufty2040`: 17,996 B flash for a Pong world (`pongtufty`, walled table), no allocation in `rigid2d` (2026-11-12). `boids` is integer only; 16,092 B flash for 128 boids (`boidstufty`), allocations only in `New` (2026-11-12). `flip` is integer only; about 20 KB flash in `fliptufty` (which times five box sizes), allocations only in `New` (2026-11-13).

## Tests and evidence

`go test -race -count=1 ./x/math/physics/rigid2d/ ./x/math/swarm/boids/` and `go test -race -short` plus a full `go test` (its 360-case stillness sweep) of `./x/math/fluid/flip/` pass, 2026-11-13; details in [[rigid-body-2d-contact]], [[boids-flocking]] and [[flip-liquid-2d]].

## TODOs

Module-level items only; algorithm defects live in the algorithm notes.

None at module level.

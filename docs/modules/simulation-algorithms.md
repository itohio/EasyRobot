---
type: module
group: "Math and maps"
status: tested
tags: [easyrobot, simulation, physics, game]
workstream:
packages: [x/math/physics/rigid2d, x/math/swarm/boids]
test_result: pass
---
# Simulation algorithms (games and demos)

**Purpose:** generic simulation bricks for demos and games, first for the owner's Tufty badge (RP2040, TinyGo): small fixed-step models with no allocation per step. The robot simulator (MuJoCo lockstep, `x/sim`, ER-048…060) is a different thing and lives elsewhere.

**Packages:** `x/math/physics/rigid2d` (rigid2d.go 266, rigid2d_test.go 569) and `x/math/swarm/boids` (boids.go 380, boids_test.go 351); no build tags. Planned here, by Leader's path decision (ER-SIMALGO): `x/math/automata/liquid`, `x/math/automata/life`, `x/math/fluid`; ray casting goes to `x/math/grid` ([[grid-maps]]).

## Public API

- `rigid2d.World{Params, Balls, Segments, Boxes}.Step(dt)`. `Ball{X, Y, VX, VY, W, R, Mass, InertiaK}`. Kinematic `Segment{A, B, V, Material}` and `Box{Min, Max, V, Material}`. `Material{Restitution, Friction}`. `Params{GX, GY, Magnus, LinearDamping, AngularDamping}`.
- Composition for Pong: paddles are `Box`es whose `VY` the game sets from the buttons (or from the IMU tilt); walls are `Segment`s; the IMU's gravity can feed `GX, GY`.
- `boids.New(n, w, h, Params) (*Flock, error)`; per frame `Flock.Box.Advance()` then `Flock.Step()`. `Box{Target, Spin}` and `Box.Shake(dir, dist)` move the window; `Flock.Cell(i)` gives a boid's cell and heading octant for drawing. Fixed point (1/256 cell), integer only.

## Algorithms

<!-- vault:algorithms h=d1bf26c97470 -->

Counts: 2 tested · 0 implemented · 0 partial · 0 stub · 0 unverified (2 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[boids-flocking]] | tested | pass | fixed-point Reynolds flocking inside a box that turns and shakes in the boids' world; integer only, no alloc after New |
| [[rigid-body-2d-contact]] | tested | pass | spinning balls vs moving walls/boxes: restitution, Coulomb friction, Magnus, damping; no alloc |

<!-- /vault:algorithms -->

## Build targets and dependencies

Pure Go, no allocation per step. `rigid2d` is float32 (`chewxy/math32`); TinyGo 0.43 `-target=tufty2040`: 17,996 B flash for a Pong world (`pongtufty`, walled table), no allocation in `rigid2d` (2026-11-12). `boids` is integer only; 16,092 B flash for 128 boids (`boidstufty`), allocations only in `New` (2026-11-12).

## Tests and evidence

`go test -race -count=1 ./x/math/physics/rigid2d/ ./x/math/swarm/boids/` pass, 2026-11-12; details in [[rigid-body-2d-contact]] and [[boids-flocking]].

## TODOs

Module-level items only; algorithm defects live in the algorithm notes.

None at module level.

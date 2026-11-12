---
type: module
group: "Math and maps"
status: tested
tags: [easyrobot, simulation, physics, game]
workstream:
packages: [x/math/physics/rigid2d]
test_result: pass
---
# Simulation algorithms (games and demos)

**Purpose:** generic simulation bricks for demos and games, first for the owner's Tufty badge (RP2040, TinyGo): small fixed-step models with no allocation per step. The robot simulator (MuJoCo lockstep, `x/sim`, ER-048…060) is a different thing and lives elsewhere.

**Packages:** `x/math/physics/rigid2d` (rigid2d.go 232, rigid2d_test.go 410; no build tags). Planned here, by Leader's path decision (ER-SIMALGO): `x/math/automata/liquid`, `x/math/automata/life`, `x/math/swarm/boids`, `x/math/fluid`; ray casting goes to `x/math/grid` ([[grid-maps]]).

## Public API

- `rigid2d.World{Params, Balls, Segments, Boxes}.Step(dt)`. `Ball{X, Y, VX, VY, W, R, Mass, InertiaK}`. Kinematic `Segment{A, B, V, Material}` and `Box{Min, Max, V, Material}`. `Material{Restitution, Friction}`. `Params{GX, GY, Magnus, LinearDamping, AngularDamping}`.
- Composition for Pong: paddles are `Box`es whose `VY` the game sets from the buttons (or from the IMU tilt); walls are `Segment`s; the IMU's gravity can feed `GX, GY`.

## Algorithms

<!-- vault:algorithms h=bd40af9513fb -->

Counts: 1 tested · 0 implemented · 0 partial · 0 stub · 0 unverified (1 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[rigid-body-2d-contact]] | tested | pass | spinning balls vs moving walls/boxes: restitution, Coulomb friction, Magnus, damping; no alloc |

<!-- /vault:algorithms -->

## Build targets and dependencies

Pure Go, float32 (`chewxy/math32`), no allocation per step. TinyGo 0.43 `-target=tufty2040`: 17,996 B flash for a Pong world (`pongtufty`, walled table), no allocation in `rigid2d` (2026-11-12).

## Tests and evidence

`go test -race -count=1 ./x/math/physics/rigid2d/` pass, 2026-11-12; details in [[rigid-body-2d-contact]].

## TODOs

Module-level items only; algorithm defects live in the algorithm notes.

None at module level.

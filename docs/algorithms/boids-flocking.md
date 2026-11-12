---
type: algorithm
status: tested
tags: [easyrobot, swarm, simulation, game, tinygo]
workstream:
module: "[[simulation-algorithms]]"
summary: "fixed-point Reynolds flocking inside a box that turns and shakes in the boids' world; integer only, no alloc after New"
source: [x/math/swarm/boids/boids.go]
tests: [x/math/swarm/boids/boids_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -race ./x/math/swarm/boids pass (14 tests, 2026-11-12, after fpv-gate ER-SIMALGO-boids063): tinygui's four boids tests ported, red first against a missing package (ER-SIMALGO-logs/boids-fail-before.txt); 10,000 steps with 22.5° turns stay in the box within speed; mirror symmetry and order independence exact; 25 mutants all fail (boids-mutants.txt); 0 mallocs in 1,000 steps; host 70–88 µs per step at N = 128 (busy shared host, boids-bench-host.txt); tinygo 0.43 tufty2040 build, allocations only in New (boids-tinygo-tufty2040.txt); RP2040 step time not measured (no board run)"
---
# Boids flocking in a moving box (boids)

**What it does:** steps a flock of boids (Reynolds: separation, alignment, cohesion) in fixed point, inside a box that turns and shakes in the boids' world. The box is a window onto free space: moving its walls sweeps them through the flock, and boids away from the walls keep their world heading. Generic: any demo or game with a flock on a small screen; ported for the owner's Tufty 2040 badge (ER-063).

**Where:** `x/math/swarm/boids/boids.go` — `Params`, `DefaultParams` (:35-48), `Box` (:53-63), `Flock`, `New` (:67-98), `Seed` (:102-114), `limit` (:134-149), `Step` (:154-201), `walls` (:209-237), `Scatter` (:251-255), `Cell` (:259-263), `Octant` (:267-289), `Box.Advance`, `Shake` (:293-322), `ToBox`, `ToWorld`, `VecToBox`, `VecToWorld` (:337-360), `sinQ14` (:363-380).

**Inputs → outputs:** `New(n, w, h, Params)` (w, h ≤ 256 cells) makes n boids in a w×h-cell box centred in the world, seeded with `Seed(1)`. `Flock.X, Y, VX, VY` are world coordinates in 1/256 cell (velocities per step); the first `N` are live. `Box.Target` (256 per turn; `TurnStep` = 22.5°, one press), `Box.Spin`, `Box.Shake(dir, dist)`, then `Box.Advance()` and `Flock.Step()` once per frame. `Cell(i)` gives a boid's cell in the box and its heading octant on screen, for drawing; `ToBox`/`VecToBox` convert anything else.

**Method (per step):**
- **Steering:** each boid scans the flock (early reject on |dx|, |dy| > R, then distance² in (1/16 cell)² ≤ R²) and takes the neighbours as they were at the start of the step:
  - alignment `(mean v − v)/8`, cohesion `(mean offset)/64`, separation `−Σ offset / 8` over neighbours closer than `Sep`;
  - every term divides, never shifts: a shift rounds negative values towards −∞, and tinygui found it pushed the flock into the top-left corner. The mirror test pins it.
- **Speed:** kept between `VMin` and `VMax` measured with the octagonal norm max + 3/8·min (no square root). A velocity that steering cancels to zero keeps the old one. tinygui's code turned it east, which the mirror test exposed as a bias; this is the one change to the flock's behaviour from the source (the API changes too: a seeded private xorshift instead of a package-global one, `Shake` takes its direction).
- **Walls (box frame):** only boids inside the `Margin` or past a wall are converted to box coordinates: they get `Push` inwards, and one past a wall is put back and bounces. All other boids keep their world position and velocity bit-exactly, so turning the box doesn't turn them.
- **Box:** `Angle` steps one unit per `Advance` towards `Target` (Q14 sine table, 256 entries); a shake throws the centre out and back over `ShakeLen` = 20 steps.

**Limits:** O(N²) scan, fine for the N ≤ 128 a badge runs (a spatial grid only if measured necessary); positions are int32, so a box beyond 256 cells is refused; no obstacles inside the box.

**Port (ER-063):** from tinytufty2040 `cfbd1b0` (`app/boids.go`, `app/boidsbox.go`, `app/boids_test.go`), which carries tinygui's `/16` fix. Review R-TG-1 has no finding in these files: its T10 items are the Dash sparkline and Life's seed. Its numerics note (Q8 and a Q14 sine table, no float per frame) holds here, and the sine table is now a literal, so no float at init either. The trails, colours and buttons stay in the demo.

## Completeness review

- Complete: yes for ER-063: flock, moving box, tinygui's tests, speed and containment over 10,000 steps, no allocation per step, host benchmark at N = 128, tufty2040 build.
- Tested (`boids_test.go`, the badge's 64×48 box and default parameters):
  - **tinygui's four tests, ported:** `TestStayInTheBoxWithinSpeed` (128 boids, spin, turns, scatter, shakes, 3,000 steps), `TestKeepWorldHeadingWhenTheBoxTurns` (all 256 angles), `TestPushedByAMovingWall`, `TestOctantFollowsHeading` (plus cases either side of 22.5°, including (256, 106) on the threshold).
  - **Acceptance:** `TestTurnPerPressFor10000Steps`: 10,000 steps, a 22.5° press every 40 steps in alternating bursts: every boid inside the box and within [VMin, VMax] (±3 for rounding) at every step.
  - **Arithmetic:** `TestMirrorSymmetry`: a flock and its x and y mirror images stay exact mirrors for 2,000 steps (catches any shift on a negative value). `TestOrderIndependent`: the flock stored backwards gives the same flock, backwards, for 500 steps.
  - **Rules:** `TestRules`: separation widens a 1-cell gap; alignment gives exactly (122, −15) and (0, −112) for a side-by-side pair heading east and north; cohesion steers a pair together, and with a diagonal neighbour gives exactly ±12 in both x and y (fpv-gate F1).
  - **Box and walls:** `TestSpinRate` (the target rises one per two steps, the angle follows); `TestFarThrowIsClampedIntoTheBox` (a boid two box widths out is put at the far wall, not reflected out).
  - **API:** `TestSeed`, `TestNewRefusesBadParams`, `TestCell`.
  - **Allocation:** `TestStepDoesNotAllocate`: raw `MemStats.Mallocs` delta 0 over 1,000 steps of 128 boids, spinning and shaking.
  - Run: `go test -race -count=1 ./x/math/swarm/boids/` pass, 2026-11-12.
- Correct: yes on these properties; 25 mutants (every shift-for-division, each rule removed, cohesion per axis, speed limits, wall push, bounce and clamp, the early return, rotation sign, spin rate, shake direction, the stall fallback, in-place steering, a boid counting itself, the octant threshold at 106 and 128, the cohesion rate, the seed) each fail a test (`ER-SIMALGO-logs/boids-mutants.txt`).
- Speed: host `BenchmarkStep128` 70–88 µs per step (Apple M5, busy shared machine, 0 allocs). The badge baseline from tinygui is 2.1–2.4 ms per step for 64 boids on the RP2040; the port's RP2040 time is not measured. `tmp/easyrobot/simalgo/boidstufty` (128 boids) prints µs per step when flashed.
- TinyGo 0.43 `-target=tufty2040`: 16,092 B flash; `-print-allocs=boids` reports only `New`'s buffer and `Flock`.

## TODOs

- [ ] Step time on the Tufty 2040 (RP2040) at 64 and 128 boids against tinygui's 2.1–2.4 ms baseline — x/math/swarm/boids/boids.go:154 #p2
- [ ] Spatial grid for the neighbour scan only if the badge needs more than 128 boids — x/math/swarm/boids/boids.go:160 #p3

## Links

- Module: [[simulation-algorithms]] · Related: [[rigid-body-2d-contact]] (same badge) · Ticket: [[ER-063-sim-boids-moving-walls]]

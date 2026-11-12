---
type: algorithm
status: tested
tags: [easyrobot, physics, simulation, game, tinygo]
workstream:
module: "[[simulation-algorithms]]"
summary: "spinning balls vs moving walls/boxes: restitution, Coulomb friction, Magnus, damping; no alloc"
source: [x/math/physics/rigid2d/rigid2d.go]
tests: [x/math/physics/rigid2d/rigid2d_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -race ./x/math/physics/rigid2d pass (25 tests, 2026-11-12, after fpv-gate ER-SIMALGO-E and review R-ER-SIMALGO-EF); 24 mutants all fail (ER-SIMALGO-logs/rigid2d-mutants-r2.txt); squeeze and exact-hit fix (review E2, E5): red first (ER-SIMALGO-logs/e2-fail-before.txt), 6 mutants of the fix and 4 earlier segment mutants fail (e2-mutants.txt); review E3/E4 test gaps: surface normal velocity, first-ball-only and both InertiaK guards now fail a test (e3e4-mutants.txt); host step with contacts: 0.58 % contact steps 197–323 ns, every step in contact 78–207 ns (busy shared host), 0 allocs; allocation guard (review E1): 0 mallocs in 1,000 steps with an impulse on a segment and a box every step and a changing dt, and one injected allocation per impulse, cache refresh, box or segment contact fails it (ER-SIMALGO-logs/e1-alloc-mutants.txt); tinygo 0.43 tufty2040 build, no allocation in the package; RP2040 step time not measured (no board run)"
---
# 2D ball contact with spin (rigid2d)

**What it does:** steps balls (circles with angular velocity) among kinematic line segments and axis-aligned boxes that move at a set velocity (walls, paddles), for the owner's spinning Pong. Generic: any 2D game or demo with balls bouncing off moving surfaces.

**Where:** `x/math/physics/rigid2d/rigid2d.go` — `Material`, `Ball`, `Segment`, `Box`, `Params`, `World` (:32-99), `Step` (:102-145), segment contact (:147-197), box contact (:200-224), `contact` impulse (:229-259).

**Inputs → outputs:** `World{Params{GX, GY, Magnus, LinearDamping, AngularDamping}, Balls, Segments, Boxes}`; `Step(dt)` moves the kinematic shapes by their velocity, then each ball, and resolves contacts. `Ball.Contacts` counts the contacts of the last step (e.g. for a sound on a hit); a ball resting on a surface touches it every step (Contacts 1, a micro-bounce of |VY| ≈ 0.02 m/s at 240 Hz).

**Method (per ball, per step):**
- **Magnus:** the velocity is turned by 2·atan(θ/2), θ = Magnus·ω·dt (acceleration Magnus·ω·perp(v), ∝ ω × v), in the Cayley form.
  - That rotation keeps |v| in exact arithmetic, so lift does no work; in float32 |v| drifts about 0.5 % per 100k steps (fpv-gate's probe).
  - Then gravity, then damping by e^(−c·dt); the exponentials are computed once per dt and coefficients.
- **Contact:**
  - The closest point gives the normal. Segments are two-sided; for a segment's interior the normal is the line's own and points to the side the ball was on at the start of the step (against the segment where it was then), so a ball pushed onto or across the line in this step is pushed back, not through. A ball centred exactly on a segment's end is pushed back the way it came. A box with the centre inside pushes out through the nearest face. The overlap is pushed out.
  - Boxes are resolved before segments, so walls win: a ball squeezed between a paddle and a wall stays on its side of the wall and may be left overlapping the paddle (review E2: with segments first, the paddle's push-out carried the ball through the wall).
  - If the contact point (v + ω × r, r = −R·n) approaches the surface (moving at its velocity), the normal impulse is (1 + e)·(−vn)·m.
  - The tangential impulse that would stop the slip is −vt·m/(1 + 1/k), with I = k·m·R², clamped to μ times the normal impulse (Coulomb). In the slip regime that takes exactly μ·(1+e)·|vn| off the tangential speed. The spin changes by −jt/(k·R).
  - `InertiaK ≤ 0`: contacts never change the spin, but a `W` set by the caller is kept and still drives Magnus and the slip.
- **Consequences (tested):** a slippery wall barely changes the spin; a grippy wall leaves the ball rolling; a grippy paddle moving along its face, along y or x, imparts spin in proportion to its speed; a strongly spun ball's path curls back.

**Limits:** no ball–ball contacts; discrete contacts (keep speed·dt under about the radius, or substep); restitution and friction belong to the surface, not the ball; a ball squeezed between a paddle and a wall overlaps the paddle.

## Completeness review

- Complete: yes for the brief (ER-SIMALGO §Pong): impulse solver with restitution and Coulomb friction per surface, spin, Magnus, linear and angular damping, fixed step, zero allocation, tufty2040 build, step benchmark.
- Tested (`rigid2d_test.go`, a 20 mm, 2.7 g thin-shell ball at 240 Hz):
  - **Spin from walls and paddles:**
    - `TestStillWallSpinDependsOnFriction`: 45° onto a still wall; μ 0.02 gives a rim speed < 0.1 m/s, μ 0.8 makes it roll off within 1 %.
    - `TestMovingPaddleSpinProportionalToSpeed`: paddle speeds 0.5, 1 and 2 m/s give spin 1:2:4 within 1 %, clockwise for a paddle moving up on the left; a still paddle gives none.
    - `TestBoxMovingAlongX`: a box moving along x drags the ball's bottom, giving counter-clockwise spin and a contact point at the box's speed.
    - `TestConveyorFloor`: the contact point leaves with the belt's speed.
  - **Impulse rules:**
    - `TestRestitution`: e = 0, 0.5, 1.
    - `TestSlidingFrictionIsCoulombsLimit`: a sliding hit loses exactly μ(1+e)|vn|.
    - `TestCoulombAndEnergyAtAStillWall`: 2,000 random hits; the tangential impulse stays ≤ μ·normal, sliding never reverses, a still wall never adds energy.
    - `TestSeparatingContactKeepsVelocity`.
    - `TestPaddleStrikesARestingBall`: a paddle at 2 m/s into a resting ball gives vx = (1+e)·2 for e = 0 and 0.8 (review E3).
    - `TestEveryBallGetsContacts`: two balls hitting different walls both bounce (review E4).
    - `TestInfiniteInertiaHit`: `InertiaK` 0 and −1 keep the spin as set, stay finite, and take the point-mass tangential impulse (vy → ω·R) (review E4).
  - **Magnus and damping:**
    - `TestMagnusCurvesAndCanComeBack`: weak spin keeps going, curving to +y; strong spin comes back ≥ 0.3 m towards the paddle; clockwise spin curves to −y.
    - `TestMagnusTurnAndGravityAxes`: the turn per step is 2·atan(θ/2); gravity (3, −4) acts on both axes.
    - `TestDampingDecay`: energy falls every step, spin and speed follow e^(−c·t) within 0.1 %.
    - `TestDampingChangesTakeEffect`: both coefficients, the step size, and the angular coefficient alone.
  - **Geometry:**
    - `TestRestsOnTheFloor`.
    - `TestBoxContactPushesOut`: corner and centre-inside cases.
    - `TestSegmentEnd`.
    - `TestSegmentFromBelow`: the top wall hit from below.
    - `TestSqueezedBallDoesNotTunnel`: a paddle rising at 0.3 and 3 m/s into the top wall, and on through it, never puts the ball's centre less than R below the wall (red before: through at step 48).
    - `TestSegmentExactHits`: a centre landing exactly on the line from below, exactly on end A along the segment, exactly on end B diagonally, and a ball starting on the line, each pushed out the way it came or is going.
    - `TestFastWallSweepsTheBall`: a wall passing the ball's centre in one step sweeps it along.
    - `TestKinematicShapesMove`.
  - **Determinism and allocation:** `TestDeterministic` (5,000 steps, bit-identical). `TestStepDoesNotAllocate`: two balls resting under gravity, one on a segment and one on a box, so both contact paths apply an impulse every step (checked), and a dt that changes every step, so the damping cache refreshes every step. It asserts a raw `MemStats.Mallocs` delta of 0 over 1,000 steps; `AllocsPerRun`'s integer division hid fewer than one allocation per step (review E1).
  - Run: `go test -race -count=1 ./x/math/physics/rigid2d/`, pass, 2026-11-12.
- Correct: yes on these properties: 24 mutants of the solver, the Magnus rotation, gravity, damping, the damping cache and geometry each fail a test (`ER-SIMALGO-logs/rigid2d-mutants-r2.txt`).
- Speed (host, Apple M5, a busy shared machine, so wide spreads; `ER-SIMALGO-logs/rigid2d-bench-host.txt`):
  - `BenchmarkStepPong` (a walled table, 4 segments, 2 moving paddles, replaying its first 10 s; 0.58 % of steps resolve a contact): 197–323 ns per step.
  - `BenchmarkStepInContact` (a contact every step): 78–207 ns.
  - The RP2040 has no FPU, so float32 is soft-float there; the step time on the badge is not measured yet. `tmp/easyrobot/simalgo/pongtufty` (the same table) prints µs per 1,000 steps when flashed.

## TODOs

- [ ] Step time on the Tufty 2040 (RP2040, soft float); switch to fixed point only if it does not fit the frame — x/math/physics/rigid2d/rigid2d.go:102 #p2
- [ ] Ball–ball contacts if a demo needs several balls — x/math/physics/rigid2d/rigid2d.go:229 #p3

## Links

- Module: [[simulation-algorithms]] · Related: [[pedestrian-dead-reckoning]] (same badge)

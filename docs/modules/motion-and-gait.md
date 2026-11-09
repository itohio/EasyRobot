---
type: module
group: "Control and planning"
status: partial
tags: [easyrobot, motion, gait, trajectory, legged]
workstream: [W3, W6]
packages: [x/math/control/motion, x/math/control/motion/rigidbody, x/math/control/motion/gait/types, x/math/control/motion/gait/support, x/math/control/motion/gait/rigidbody]
test_result: pass
---
# Motion and gait (x/math/control/motion)

**Purpose:**
- Time-parameterised motion: a jerk-limited 1-D profile for servos and joints.
- A unicycle path follower and tracking controller.
- The start of a legged gait stack: contracts, a foot-trajectory planner and a body planner stub.

**Packages** (non-test LOC / test LOC; no build tags):

| Package | LOC | Tests | Notes |
|---|---|---|---|
| `motion` (`vaj1d.go`) | 196 | — | package name `vaj` |
| `motion/rigidbody` (9 files) | 1559 | 438 | |
| `motion/gait/types` (`contracts.go`) | 196 | — | |
| `motion/gait/support` (`path_planner.go`) | 420 | — | |
| `motion/gait/rigidbody` (`planner.go`) | 91 | — | |

Docs read as claims: `motion/IMPLEMENTATION_PLAN.md` (generic planner, not implemented), `motion/rigidbody/SPEC.md`, `motion/gait/DESIGN.md`, `motion/gait/IMPLEMENTATION_PLAN.md`.

## Public API

- **`vaj.New1D(maxV, maxA, maxJ) VAJ1D`.** Set `Input`/`Target`, call `Update(dt)`, read `Output`/`Velocity`. Value type, allocation-free. Used by `drivers/manipulator` (one per servo) and by `motion/rigidbody`.
- **`rigidbody.NewMotion(Constraints, Parameters) (*Motion, error)`.** Implements `kinematics/types.Bidirectional`. `SetPath([]vec.Vector3D)` / `SetWaypointMatrix(m)` take row layouts 3, 6, 7 or 14. `Forward` previews the next state on the path; `Backward` produces `[linear, angular, effort…]`.
- **Gait contracts (`gait/types`):**
  - `PhaseMode` (Lift/Swing/Touchdown/Support), `PhaseState`, `EndpointPose`, `FootState`
  - `SupportEndpointPlanner`, `RigidBodyPlanner`, `TerrainProvider`
  - `GaitTemplate`, `GaitInstance`, `GaitScheduler`, `TransitionPolicy`
- **Implementations:**
  - `support.NewSupportPathPlanner(opts...)` with `WithTerrainProvider`, `WithSupportProfile`, `WithPhaseShape`.
  - `gait/rigidbody.NewPlanner(WithInitialPose)` is a stub.
- **Composition today:** the caller must generate phase modes and progress, desired foot poses, body motion and IK calls. Only the foot-curve step exists ([[W3-legged-ik]]).

## Algorithms

<!-- vault:algorithms h=7e4604be26de -->

Counts: 4 tested · 0 implemented · 2 partial · 1 stub · 0 unverified (7 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[gait-phase-scheduler]] | tested | pass | Per-leg stance/swing phase from a periodic pattern (offset, duty); tripod, ripple, wave, walk, trot presets; allocation-free. |
| [[polyline-path-sampler]] | tested | pass | Arc-length parameterised piecewise-linear path with optional per-waypoint quaternion (slerp) and… |
| [[rigidbody-path-follower]] | tested | pass | Unicycle path follower; jerk-limited progress to the path end at MaxSpeed (ER-018); lateral speed cap only reported |
| [[vaj1d-jerk-limited-profile]] | tested | pass | Signed, near-time-optimal jerk-limited 1-D profile; lands exactly on target within v/a/j limits (ER-018). |
| [[gait-support-endpoint-planner]] | partial | pass | Per-leg foot curve per sub-phase; swing apex = SwingHeight, monotonic touchdown (ER-018). Not allocation-free; frame unspecified. |
| [[path-tracking-pid-controller]] | partial | pass | partial (lateral sign, heading wrap) |
| [[gait-rigidbody-planner]] | stub | none |  |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Deps: `x/math/vec`, `x/math/mat`, `x/math/control/pid`, `x/math/control/kinematics/types`, `chewxy/math32`, stdlib `context`, `time`, `sync`, `fmt`, `math` (float64 used only in parameter validation).
- TinyGo/embedded:
  - VAJ1D: allocation-free value type, MCU-friendly once fixed.
  - Gait planner: map-based leg state (`map[string]*legState`), `time.Time`, `context` and a mutex per call. It works but allocates and is heavier than needed for 6 legs at 50–100 Hz.
  - `motion/rigidbody`: allocates `mat.New` in constraint resizing and slices in path factories. Fine on the host.
- No CGO, no goroutines.

## Tests and evidence

- `go test -count=1 -json ./x/math/control/...` (2026-11-09):
  - `motion/rigidbody` ok, 10 tests (including the closed-loop `TestTriangularChassisFollowsSinusoid`).
  - `motion`, `gait/types`, `gait/support`, `gait/rigidbody`: no test files.
- Probe `tmp/easyrobot/ik` sections J (VAJ1D), K (gait support), M (Backward signs); numbers in [[ER-R2-ik-review]] §3.6.

## API notes (ease of use, composability)

- **VAJ1D** has the right shape for servo streams: value type with public `Input/Target/Output`. Limits can only be set at construction; there is no `SetLimits` to apply the `config.Motion` constraints the manipulator receives.
- **The gait contracts are sensible and composable:**
  - Per-leg planner, terrain provider and phase-shape strategy are injected through options.
  - The request structs are explicit, carrying `Delta` and `Timestamp`.
- **Missing glue:**
  - The frame of `DesiredPose` is unspecified (body or world). The support phase only makes sense in the body frame.
  - Phase progress must come from a scheduler that does not exist.
  - Leg identity is a string; nothing binds legs to a body or to IK.
- **`motion/rigidbody.Motion`:**
  - Uses the kinematics `Bidirectional` interface for a planner, so `Forward` means "plan" and `Backward` means "control". It composes in the toy test but reads oddly.
  - The 6-row state matrix duplicates the `State` struct.
  - `Forward` returns a point on the path, not an integrated state.
- **Quaternion layouts:** `motion/rigidbody` uses `[w x y z]`, DH uses `[x y z w]`.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Stride/foothold generator and body→leg transforms (ER-018) — gait (new) #p2
- [ ] Support polygon / static stability margin (ER-037) — gait (new) #p2
- [ ] Specify the body/world frame for FootState — gait/types/contracts.go:56 #p2
- [ ] Tests for vaj1d, gait/support (the plan lists them; none exist) — motion/gait/IMPLEMENTATION_PLAN.md #p2
- [ ] `SetLimits` on VAJ1D so firmware can honour `ManipulatorConfig.Motion` — vaj1d.go:22; drivers/manipulator/main.go:351 #p2

## Inferred objectives (proposals)

- **Proposal:** build a minimal, allocation-free legged loop for ESP32:
  - phase scheduler → foothold generator → swing/stance curves → body→hip transform → analytic leg IK → servo map.
  - Run it at a fixed 50 Hz with one VAJ1D per joint, or with direct interpolation.
- **Proposal:** use the existing contracts (`PhaseMode`, `FootState`, `TerrainProvider`) as the integration seams. Defer the general `GaitTemplate` role-binding machinery until two robot morphologies exist.
- **Proposal:** offer gait selection (tripod vs wave vs ripple) as a discrete action to the W6 decision-tree behaviours.

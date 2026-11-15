---
type: module
group: "Control and planning"
status: partial
tags: [easyrobot, kinematics, control]
workstream: [W3, W6]
packages: [x/math/control/kinematics/types, x/math/control/kinematics/chain, x/math/control/kinematics/joints/dh, x/math/control/kinematics/joints/planar, x/math/control/kinematics/rigidbody, x/math/control/kinematics/thrusters, x/math/control/kinematics/wheels/differential, x/math/control/kinematics/wheels/mecanum, x/math/control/kinematics/wheels/steer4, x/math/control/kinematics/wheels/steer4dual, x/math/control/kinematics/wheels/steer6, x/math/control/kinematics/wheels/internal/rigid]
test_result: pass
---
# Kinematics (x/math/control/kinematics)

**Purpose:** Forward/inverse kinematics behind one matrix-in/matrix-out interface. Covers serial chains (DH, planar legs/arms), wheeled bases (differential, mecanum, Ackermann, dual-axle, 6-wheel rover), thruster allocation and a rigid-body velocity→wrench model. This is the foundation for W3 (legged IK) and for wheeled W6 robots.

**Packages** (non-test LOC / test LOC; no build tags; pure Go, float32 via `chewxy/math32`):

| Package | Files | LOC | Tests |
|---|---|---|---|
| `types` | `types.go` | 85 | — |
| `joints/dh` | `config.go`, `denavithartenberg.go` | 350 | 333 |
| `joints/planar` | `config.go`, `planar2dof.go`, `planar3dof.go` | 319 | 322 |
| `rigidbody` | `model.go`, `tensor_helpers.go` | 243 | 109 |
| `thrusters` | 6 files | 543 | 113 |
| `wheels/differential` | | 137 | — |
| `wheels/mecanum` | | 166 | 124 |
| `wheels/steer4` | | 195 | — |
| `wheels/steer4dual` | | 195 | — |
| `wheels/steer6` | | 219 | — |
| `wheels/internal/rigid` | | 47 | — |

Specs and design docs (`SPEC.md`, `joints/dh/SPEC.md`, `joints/planar/SPEC.md`, `thrusters/DESIGN.md`, `rigidbody/DESIGN.md`, `wheels/DESIGN.md`) were read as claims. Their paths still say `pkg/core/...`.

## Public API

- **Shared interface:** `types.Bidirectional` = `Model` (`Dimensions()`, `Capabilities()`, `ConstraintSet()`) + `Forward(state, destination, controls mattype.Matrix) error` + `Backward(state, destination, controls) error` (`types/types.go:53-75`). Every model is used the same way: you allocate column matrices, a call fills them, and errors are `ErrInvalidDimensions` / `ErrUnsupportedOperation`.
- **Constructors:**
  - `dh.New(eps, maxIter, cfg ...dh.Config) (*DenavitHartenberg, error)` (validates configs since ER-017)
  - `planar.New2DOF([2]Config) (*p2d, error)`, `planar.New3DOF([3]Config) (*p3d, error)`
  - `chain.Chain.Model(...)`, `chain.Body.NewLegSolver(...)` — one joint/leg declaration with servo mapping ([[leg-chain-config-and-servo-map]], ER-018)
  - `differential.New(r, base)`
  - `mecanum.New(r, bx, by)`
  - `steer4.New(Config)`, `steer4dual.New(Config)`, `steer6.New(Config)` (these panic on bad config)
  - `thrusters.NewModel(Body, []Thruster) (*Model, error)`
  - `rigidbody.NewModel(mass, inertia, Config) (*Model, error)`, plus the inertia helpers
- **Composition:** in `motion/rigidbody` tests, `motion/rigidbody.Motion` drives `kinematics/rigidbody.Model` in a closed loop. Since ER-018, `kinematics/chain` turns a leg declaration into a planar/DH model plus servo degrees, and `motion/gait/walk` composes the gait scheduler, footholds, leg IK and servo mapping per tick ([[gait-foothold-planner]]). `drivers/manipulator` still has placeholder adapters that return "not implemented" (`kinematics.planar.go:44-57`, `kinematics.dh.go:44-57`).
- **Typed errors** (`kinematics/types`): `ErrInvalidConfig`, `ErrUnreachable`, `ErrJointLimit`, `ErrNoConvergence`, `ErrNonFinite`, besides `ErrInvalidDimensions` / `ErrUnsupportedOperation`.
- **Proto counterpart:** `types/control/kinematics.JointsConfig{DhParams, PlanarJoints, Constraints}` (`proto/types/control/kinematics/joints.proto`). It mirrors `dh.Config`/`planar.Config` field for field, with no units, offsets, sign or servo channel.

## Algorithms

<!-- vault:algorithms h=656aea0c6b8c -->

Counts: 7 tested · 5 implemented · 2 partial · 0 stub · 0 unverified (14 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[dh-jacobian-pseudoinverse-ik]] | tested | pass | Damped least-squares position IK for any number of joints; limit-aware, restarts, typed errors, best-effort q, allocation-free (ER-017). |
| [[leg-chain-config-and-servo-map]] | tested | pass | One joint/leg declaration → validated DH or analytic leg model, servo degrees, body→hip transforms, allocation-free multi-leg IK. |
| [[mecanum-drive-kinematics]] | tested | pass | Four mecanum wheel rates ↔ chassis twist `[vx, vy, ω]` with fixed 3×4 / 4×3 matrices. |
| [[planar-2dof-arm-kinematics]] | tested | pass | Yaw joint plus one pitch link: FK and closed-form IK. |
| [[planar-3dof-leg-kinematics]] | tested | pass | Closed-form coxa–femur–tibia FK/IK; exact over the whole workspace incl. foot behind the yaw axis; typed errors (ER-017). |
| [[rigid-body-velocity-wrench-model]] | tested | pass | Quasi-static proportional map from body velocity (or velocity error) to a clamped force/torque: `F… |
| [[thruster-wrench-allocation]] | tested | pass | Forward: sums thrust vectors, `r × F` moments and reaction torques into a body wrench and… |
| [[dh-geometric-jacobian]] | implemented | none | none direct |
| [[differential-drive-kinematics]] | implemented | none | Maps left/right wheel speeds ↔ chassis `[v, ω]`. |
| [[dual-axle-steer4-kinematics]] | implemented | none | Front and rear steering, centre-referenced ICR: each wheel angle `atan(x/(R − y))`, `R = v/ω`. |
| [[six-wheel-steer-kinematics]] | implemented | none | Rocker-bogie style: front and rear pairs steer, middle pair fixed; same centre-referenced geometry… |
| [[wheel-twist-least-squares]] | implemented | none | Fits `(v, ω)` to N steered wheels by solving the 2×2 normal equations of `wheel_i·r = cos δ_i·v +… |
| [[ackermann-steer4-kinematics]] | partial | none | Front-steered 4-wheel chassis: wheel rates + two steering angles ↔ `[v, ω, δ_fl, δ_fr]`. |
| [[dh-forward-kinematics]] | partial | pass | partial (orientation quaternion inverted) |

<!-- /vault:algorithms -->


## Build targets and dependencies

- Deps: `x/math/mat`, `x/math/vec`, `x/math/primitive/fp32` (SVD via `mat.PseudoInverse`), `chewxy/math32`. No CGO, no reflection, no goroutines.
- TinyGo/embedded: float32 throughout, which suits an ESP32 FPU (single precision only). Hot paths allocate, though:
  - `mat.New` for J/J⁺ on every DH IK iteration (`denavithartenberg.go:184-185`, `:221`).
  - `Matrix4x4.View()` heap copies (`:139`).
  - `mattype.Matrix` interface calls with type assertions in every `Forward`/`Backward`.
  - Allocation-free: the planar and mecanum solvers (fixed arrays). Callers still have to pass `mat.Matrix` columns.
- `drivers/manipulator` (TinyGo, `sam && xiao`, plus `planar` or `dh` tag) is not built on host. Firmware is TinyGo only and was not run.

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/math/control/...` (2026-11-09):
  - dh ok (7 tests / 15 subtests)
  - planar ok (9 / 18)
  - rigidbody ok (3)
  - thrusters ok (3)
  - mecanum ok (3 / 6)
  - differential, steer4, steer4dual, steer6, internal/rigid, types: no test files
- `go vet ./x/math/control/...`: ok.
- Same as the 2026-11-08 baseline.
- IK probe (sections A, B, C, D, H, I, L, R, X, Y, Z).

## API notes (ease of use, composability)

- **One interface for every model is good for planners and bad for users.**
  - Every call needs three `mat.Matrix` columns with magic layouts (7×1 pose with a quaternion, 6×1 RPY, 6×1 or 8×1 or 10×1 wheel vectors).
  - Layouts live only in doc comments. Nothing types them, so a mistake shows up as a wrong number, not an error.
  - Typed fast paths (`ForwardLeg(q [3]float32) vec.Vector3D`) would serve embedded callers. The matrix interface can stay as an adapter.
- **Configuration is the owner's pain point, and the code confirms it:**
  - `dh.Config` zero-value limits clamp every joint to 0.
  - The meaning of `Index` is a bare int.
  - Units are not stated anywhere.
  - There are no zero offsets, sign or servo mapping.
  - The host CLI already sends degrees/mm into a radians API (`cmd/manipulator/main.go:346-348`).
  - The proposed `chain` helper package (declare, validate, convert to DH/planar, servo mapping, zero-pose check) was sketched in the IK review.
- **Silent clamping everywhere** (FK writes clamped values back, IK clamps, planar returns success for unreachable targets) hides configuration errors. Prefer typed errors plus a best-effort result.
- **`Backward` leaves `controls` untouched on failure.** Callers get no partial answer.
- **State is hidden in the struct** (`params`, `pos`, `H0i`), so a model is not safe for concurrent use and the same model cannot serve 6 legs in parallel goroutines. One model per leg works.
- **Conventions differ:**
  - Quaternion `[x y z w]` (DH) vs `[w x y z]` (proto, `motion/rigidbody`).
  - Planar `pitch = a1+a2` has the opposite sign to the right-hand rule.
  - Wheel "rates" are linear speeds in `differential`.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [x] Joint/leg configuration helper package with validation and servo mapping (ER-018) — x/math/control/kinematics/chain #p2
- [ ] steer4 geometry consistency — wheels/steer4/drive.go:173 #p2
- [ ] Tests for differential, steer4, steer4dual, steer6, internal/rigid — wheels/* #p2
- [ ] Typed allocation-free fast paths for embedded callers — joints/* #p3
- [ ] Update SPEC paths (`pkg/core/...` → `x/...`) and the thrusters limit claim — kinematics/SPEC.md:106 #p3
- [x] Remove the committed ELF binary `cmd/manipulator/manipulator` (9.3 MB) — ER-cleanup #p3

## Inferred objectives (proposals)

- **Proposal:** make `planar.New3DOF` (with errors and offsets) the leg solver for hexapod/quadruped, and keep DH + DLS for arms with more joints.
- **Proposal:** add one leg/body configuration layer that produces DH, planar and servo maps from a single declaration, and send it over `JointsConfig` with explicit units.
- **Proposal:** wheeled W6 robots: fix steer4, add tests, then use `differential` with real wheel rad/s for the ESP32 camera robot.

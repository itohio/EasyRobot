---
type: evidence
status: done
tags: [easyrobot, kinematics, ik, gait, review]
workstream: W3
date: 2026-11-09
head: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-R2 — IK review: DH/planar kinematics, Jacobian, solvers, gait gap

Owner's question (O-005): *"inverse kinematics model (not sure fully if it works, need some helpers to configure joints properly. not sure if i solve jacobian properly). The end goal — a hexapod robot or a quadruped robot inverse kinematics with strides, gaits and timing."*

**Short answer.**
1. **FK is correct**: positions match hand computation and an independent float64 model to 1.5e-8 m.
2. **The Jacobian is correct**: matches the analytic Jacobian to 4.4e-7 and central differences to 5.5e-5.
3. **The IK solver around it is fragile**:
   - 3-DOF hexapod leg from a warm start: 100 % convergence in ≤7 iterations.
   - Same leg from a cold start: 37–81 %.
   - Any chain with **more than 3 joints never solves**.
   - The orientation quaternion is inverted.
4. **The analytic leg IK** (`planar.New3DOF`) is exact for a foot in front of the hip yaw axis. It returns wrong poses **without an error** when the foot is behind that axis, out of reach, or past a joint limit.
5. **Nothing yet produces gaits or timing.** Phase scheduling, strides, body→leg transforms and stability are not written. The foot-swing planner exists but its default clearances are wrong. The servo profile generator (VAJ1D) has a sign bug.
6. **The firmware IK integration is a placeholder.** Its host config mixes degrees/mm with radians.

Related notes: [[kinematics]], [[motion-and-gait]], [[pid-control]], [[W3-legged-ik]]. Upstream math defects (mat-vec slice, independently probed): [[vector-and-quaternion]], [[homogeneous-transforms]], [[mat-pseudo-inverse]], [[matrix-inverse-and-lu]], [[mat-qr]]. Of these, the DH code depends only on `Matrix.Quaternion` (E2) and `PseudoInverse` for wide matrices (E1). It does not call `HomogenousFrom*`, `Matrix4x4.Inverse`, `Orientation` or QR, so those defects do not change the FK/Jacobian results here. Thrusters and kinematics/rigidbody depend on `Inverse` (inertia, JJᵀ).

## 1. Scope, commands, evidence

| What | Command (offline) | Outcome |
|---|---|---|
| Slice tests | `cd EasyRobot && GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/math/control/...` | 7 packages ok, 11 without tests. Top-level tests passed: dh 7 (15 subtests), planar 9 (18), kinematics/rigidbody 3, thrusters 3, mecanum 3 (6), motion/rigidbody 10, pid 8. No failures. |
| Vet | `… go vet ./x/math/control/...` | ok |
| Host CLI build | `… go build -o /dev/null ./cmd/manipulator` | ok |
| Firmware | `drivers/manipulator` (TinyGo, `sam && xiao`) | not built: needs TinyGo; NEVER `go generate` there (main.go:12 flashes a board) |
| Probe | `cd ~/src/tmp/easyrobot/ik && GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run . A H B C D I X Y Z R J K L M` | exit 0; full output `tmp/easyrobot/ik/probe-output-2026-11-09.txt` (134 lines) |

**How the probe works.**
- It imports the real packages through `replace` directives.
- `dhcopy/` is a verbatim copy of `joints/dh` at HEAD plus `zz_probe_export.go`. That file only adds accessors (`ProbeFK` returns `H0i`; `ProbeStep` runs one `ikSolverJacobianPos`) and changes no logic.
- `dhref.go` is an independently written float64 standard-DH FK and analytic Jacobian.
- `dls.go` is a reference damped least-squares IK, used only to show that the targets can be solved.
- Random draws use `math/rand/v2` PCG with fixed seeds (1, 2, 3, 5, 7).

## 2. Model as implemented

- **DH convention: standard (distal) DH.** `A_i = Rz(θ_i)·Tz(d_i)·Tx(r_i)·Rx(α_i)`, written out element by element at `joints/dh/config.go:71-86`. `H0i[0] = I` and `H0i[i+1] = H0i[i]·A_i` (`denavithartenberg.go:121-131`). `Config{Min, Max, Theta, Alpha, R, D, Index}`.
  - `Index` selects which parameter the joint drives: 0 = θ (revolute), 1 = α, 2 = r, 3 = d (prismatic).
  - The Jacobian and IK support only 0 and 3 (`:197-213`).
- **Units.**
  - Angles are radians (`math32.Cos`).
  - Lengths can be any consistent unit.
  - `eps` is a distance tolerance in that unit (`|e|² < eps²`, `:164`).
  - Limits are in the joint's own unit.
- **Limits** are clamped silently inside FK, and the clamped value is written back (`:126`, `config.go:49`).
  - A zero-value `Config` has `Min = Max = 0`, which pins every joint at 0 (probe F2).
- **Output.** `[x y z qx qy qz qw]`. The proto (`manipulator.proto`, `state.proto`) and `motion/rigidbody` use `[w x y z]`.
- **Planar.** `planar.New2DOF` / `New3DOF` take `Config{Min, Max, Length}`.
  - Joint 0 is yaw about +z.
  - Joints 1 and 2 are pitch in the vertical plane through the yaw angle. The tibia angle is relative to the femur; positive is up.
  - Output is `[x y z roll pitch yaw]`, with `pitch = a1 + a2`, which has the opposite sign to the right-hand rule about y.
  - `New3DOF` is the same mechanism as the DH chain α₁ = +π/2, a = (l0, l1, l2), all d = 0. Probe A: the two FKs agree to 3.0e-8 m over 10 000 random poses.

### 2.1 Jacobian derivation (what the code computes)

Write the foot position as

`p(q) = H0i[i] · A_i(q_i) · T_{i+1→n} · [0 0 0 1]ᵀ`.

Let `R_i`, `o_i` be the rotation and origin of `H0i[i]`, and `z_i = R_i·ẑ` (column 2 of `H0i[i]`). The frame before joint i holds the axis that joint i acts on, which is the standard-DH rule.

**Revolute joint (`Index 0`, θ_i = Theta_i + q_i).** Only `Rz(θ_i)` depends on `q_i`, and `∂Rz/∂θ = S(ẑ)·Rz`, where `S(·)` is the skew (cross-product) matrix. With `p_local = R_iᵀ(p − o_i)`:

`∂p/∂q_i = R_i·S(ẑ)·R_iᵀ·(p − o_i) = S(R_i ẑ)·(p − o_i) = z_i × (p − o_i)`

**Prismatic joint (`Index 3`, d_i = D_i + q_i).** `∂Tz/∂d = ẑ`, so `∂p/∂q_i = R_i·ẑ = z_i`.

**Orientation rows (angular velocity, not implemented in the code).** Revolute `ω_i = z_i`; prismatic `ω_i = 0`. A quaternion-error IK would use `q̇ = ½·E(q)·ω`.

**`Index 1` and `Index 2` joints (not supported by the code).**
- α joint: `∂p/∂q = x_{i+1} × (p − o_{i+1})`, where `x_{i+1}` is column 0 of `H0i[i+1]`.
- r joint: `∂p/∂q = x_{i+1}`.

**The code.** `ikSolverJacobianPos` (`denavithartenberg.go:182-214`) sets the revolute column to `R × (dn − di)` and the prismatic column to `R`. Here `R` and `di` are columns 2 and 3 of `H0i[i]`, and `dn` is the translation of `H0i[n]`. This matches the derivation exactly.

**The update** is `Δq = J⁺e` with `e = p* − p`. `J⁺` comes from `mat.Matrix.PseudoInverse` → `fp32.Gepseu`, an SVD whose singular-value cutoff is an absolute `1e-10` (`primitive/fp32/la.go:712`).

## 3. Results

### 3.1 FK vs hand computation (probe A)

Hexapod leg from `cmd/manipulator` (coxa 39, femur 45, tibia 100 mm). Code `dh.Forward` and `planar.New3DOF` against hand values:

| Pose (coxa, femur, tibia) | Hand (m) | DH error | planar3 error |
|---|---|---|---|
| (0, 0, 0) | (0.184, 0, 0) | 0 | 0 |
| (90°, 0, 0) | (0, 0.184, 0) | 8.0e-9 | 8.0e-9 |
| (0, 90°, 0) | (0.039, 0, 0.145) | 7.5e-9 | 7.5e-9 |
| (0, 0, −90°) | (0.084, 0, −0.100) | 7.5e-9 | 7.5e-9 |
| (45°, 30°, −90°) | (0.09049, 0.09049, −0.06410) | 1.5e-8 | 7.5e-9 |

### 3.2 Orientation (probe H) — defect

| Input | Code quaternion `[qx qy qz qw]` | Correct |
|---|---|---|
| revolute θ = +90° about z | `[0 0 0.7071 −0.7071]` (= Rz(−90°)) | `[0 0 0.7071 0.7071]` |
| θ = 30° | `[0 0 −0.2588 0.9659]` | `[0 0 0.2588 0.9659]` |
| α = +90° (Rx) | `[0.7071 0 0 −0.7071]` | `[0.7071 0 0 0.7071]` |

The cause is in `x/math/mat/mat.go:626-648` (`Matrix.Quaternion`). It uses the row-vector (transposed-matrix) form of the Shepperd/Day conversion, so it returns the conjugate. DH output reaches it through `denavithartenberg.go:139`.

### 3.3 Jacobian vs central differences (probe B)

Setup:
- 1000 random poses per chain, kept 2h inside the limits.
- `J_code` is the code's formula evaluated on the code's own float32 `H0i`.
- FD is the central difference of the public `dh.Forward` (float32).
- `An64` is the float64 analytic Jacobian.

The table gives the maximum absolute error per column at h = 1e-2, then the maximum relative error (column norm).

| Chain | \|J_code − FD\| per column | \|J_code − An64\| (max) | rel. error per column |
|---|---|---|---|
| hexleg 39/45/100 mm | 3.3e-6, 3.1e-6, 2.3e-6 | 3.4e-8 | 1.0e-2 *, 2.6e-5, 2.5e-5 |
| hexleg 52/66/133 mm | 4.7e-6, 4.3e-6, 3.2e-6 | 4.2e-8 | 7.3e-4 *, 2.7e-5, 2.9e-5 |
| planar 2-link (1,1) | 3.8e-5, 2.5e-5 | 1.8e-7 | 1.5e-4, 2.6e-5 |
| planar 3-link (1,1,1) | 5.5e-5, 4.4e-5, 2.6e-5 | 4.3e-7 | 7.2e-5, 3.1e-4, 2.9e-5 |
| repo test chain α = π/2, d = 0.5 | 3.7e-5, 2.2e-5 | 2.1e-7 | 1.3e-2 *, 2.6e-5 |
| revolute + prismatic | 3.8e-5, 1.0e-5 | 1.7e-7 | 2.4e-5, 1.0e-5 |
| 6-DOF PUMA-like | 1.7e-5, 2.3e-5, 1.9e-5, 5.9e-6, 6.2e-6, 6.2e-8 | 2.2e-7 | ≤2.7e-3 |

\* The high relative errors are on the first-joint column when its norm is close to 0 (the foot is near the yaw axis). In every cell, `|FD − An64|` equals `|J_code − FD|`, so the residual is finite-difference error, not a code error.

- At h = 1e-3 the FD error **grows** (up to 3.1e-4, and up to 0.94 relative on near-zero columns). That is float32 cancellation, so h = 1e-2 is the right step for this float32 FK.
- **Step check:** the update the code applies (`ProbeStep`) equals float64 `(JᵀJ)⁻¹Jᵀe` to ≤3.4e-4 relative on full-rank chains (hexleg 1.0e-4 and 1.6e-5).
- Through DH, the planar 3-link has a z-row that is identically zero, so J has rank 2 and the comparison is undefined. That chain's large "2.4e10" in the output is the singular reference, not the code.
- The 6-DOF chain errors out; see 3.4.

**Verdict: the Jacobian is right.**

### 3.4 DH IK convergence (probes C, D–G, Z, R)

Targets are `FK(q_true)`, with `q_true` uniform within the limits, so every target is reachable. 500 targets per row, eps = 1e-4, maxIter = 100. "Iterations" is the smallest maxIter that converges; the sequence is deterministic. Residual on success is ≤1.0e-4 everywhere.

| Chain | zero seed | nominal seed | warm seed (q_true ± 0.2 rad) |
|---|---|---|---|
| hexleg 39/45/100 | 355/500 (p50/p90/max 6/9/36 iterations) | 184/500 (5/7/13) | **500/500 (2/3/7)** |
| hexleg 52/66/133 | 374/500 | 204/500 | 500/500 (2/3/7) |
| planar 2-link | 89/500 | 172/500 | 496/500 |
| planar 3-link | 127/500 (13/39/90) | 305/500 | 500/500 |
| repo test chain α = π/2 | 490/500 | 349/500 | 499/500 |
| revolute + prismatic | 203/500 | 265/500 | 491/500 |
| 6-DOF | **0/500, all `ErrUnsupportedOperation`** | 0/500 | 0/500 |

**Realistic hexapod foot workspace (probe R).** Joint ranges coxa ±60°, femur −60…72°, tibia −149…−20° (knee down only); limits as for the hexleg.

| Seed | Code DH IK | Reference DLS (λ = 5 mm, step ≤ 0.25 rad, clamp) | `planar.New3DOF` analytic |
|---|---|---|---|
| zero | 404/500 (6/10/68) | 441/500 | 422/500 |
| nominal (0, 0.3, −1.6) | 401/500 (4/6/15) | 449/500 | 422/500 |
| warm ± 0.2 rad | 499/500 (2/3/5) | 500/500 | 430/500 |
| previous random target's answer | 191/500 | 274/500 | 422/500 |

- All 78 analytic failures have the foot behind the coxa yaw axis (see 3.5).
- **Diagnosis (probe Z, hexleg, uniform targets).**
  - Widening the limits to ±4π moves the zero-seed result from 355 to 198 and the nominal-seed result from 184 to 294. So the limit clamp and the choice of knee branch both matter.
  - The reference DLS reaches 401 (zero seed) and 250 (nominal seed); the rest of its failures end pinned at a joint limit (98 and 249). Uniform targets include knee-up poses that the knee-down seed cannot reach inside the femur limits. This is a property of local methods, but the code gives no best-effort answer or reason.

**Edge cases (probe D–G, hexleg unless noted).**

| Case | Result |
|---|---|
| D1: target 0.999·reach, zero seed (stretched, singular) | `ErrNoConvergence` |
| D2: same target, nominal seed | converged, q = (0, 0.081, −0.118) |
| D3: target on the coxa axis (0, 0, −0.08) | converged |
| E1: target 1.5·reach | `ErrNoConvergence`, `controls` left untouched |
| E2: foot behind the body, needs coxa 180° (limit ±90°) | `ErrNoConvergence` (correct, but no reason given) |
| F1: target needs tibia −2.0 with tibia limit [−π/2, 0] | `ErrNoConvergence` |
| F2: `Config` without Min/Max | FK of (0.5, 0.5, −1) is the stretched pose (0.184, 0, 0) |
| G1: 6-DOF chain | `ErrUnsupportedOperation` |
| G2: 4-link planar chain | `ErrUnsupportedOperation` |
| G3: planar 2-link, target (1, 1, 0) | converged, (0, π/2) |

### 3.5 Analytic leg IK (`planar.New3DOF`, probes I, X, Y)

20 000 random poses with knee ≤ 0. Results are split by the signed horizontal reach of the foot along the coxa direction (probe Y):

| Limits | Foot in front of the yaw axis | Foot behind the yaw axis |
|---|---|---|
| ±π | 1 620/12 767 > 0.1 mm (cause: `a1 > π` clamped, no wrap) | 3 432/7 233 > 0.1 mm |
| leg (±90°, ±90°, [−162°, 0]) | **17/15 382** > 0.1 mm | 4 592/4 618 > 0.1 mm |

- Worst residual is 12 cm. **All of these return `err = nil`.**
- Minimal reproductions (±π limits): `q_true = (0.839, −0.910, −3.069)` (foot 0.3 mm behind the axis) gives `q_ik = (−2.303, π, −π)` with a residual of 0.101 m.

Other probe I results (leg limits):

| Input | Result |
|---|---|
| unreachable target (0.5, 0, 0) | stretched pose (0.184, 0, 0), `err = nil` |
| target behind the body (−0.1, 0, −0.05) | coxa clamped to 90°, FK (0, 0.100, −0.050), `err = nil` |
| directly below the hip (0, 0, −0.1) | exact |
| the `cmd/manipulator` config (limits in degrees, lengths in mm) | 100° coxa and −20° tibia are not clamped: ±90 is read as ±90 rad |

### 3.6 Motion and gait probes (J, K, L, M)

**VAJ1D** (limits v/a/j = 100/100/100 unless noted):

| Move | Result |
|---|---|
| 0→100, dt = 0.01 | final 99.319, max \|a\| 4403 (limit 100) |
| 0→100, dt = 0.1 (manipulator tick) | overshoot 3.754, final 99.70 |
| 0→0.5 | never moves |
| 0→−0.5 | final −0.5003 |
| 0→1.5 rad with (3, 10, 50) | final 0.501 |
| reversal mid-move | direction snaps with no deceleration |

**Gait support planner**, default profile (lift 0.05, swing 0.08), ground at 0, stride 0.10:

| Phase | z range | Notes |
|---|---|---|
| lift | 0 → **0.131** | ends at 0.05 |
| swing | 0.050 → **0.160** | ends at 0.08 |
| touchdown | 0.080 → **0.163** → 0 | rises before landing; contact flagged for 6 of 11 samples |
| support | x 0.1 → 0, z 0 | |

**Wheels** (probe L):
- differential, steer4dual and steer6 round-trip exactly.
- steer4 headings are off from the rigid-body velocity at each wheel by 0.079 rad at ω = 0.5 and 0.359 rad at ω = 2.

**motion/rigidbody Backward** (probe M):

| Situation | Needed | Got |
|---|---|---|
| path on the robot's left | ω > 0 | ω = −0.0628 |
| path on the right | ω < 0 | ω = +0.0628 |
| yaw 3.10 → −3.10 | ω > 0 | ω < 0 |

## 4. Error list (all reproducible with the probe)

| # | Pri | Defect | Where | Repro |
|---|---|---|---|---|
| E1 | p1 | DH IK fails for every chain with >3 joints: J is 3×n and `Gesvd` requires rows ≥ cols | `dh/denavithartenberg.go:216-217`; `primitive/fp32/la.go:760` | G1, G2, C (6-DOF 0/1500) |
| E2 | p1 | DH orientation quaternion is the inverse rotation; layout `[x y z w]` vs `[w x y z]` elsewhere | `x/math/mat/mat.go:626-648`, used at `denavithartenberg.go:139` | H |
| E3 | p1 | DH IK: undamped, no step limit, absolute SVD cutoff 1e-10 in float32; singular start stalls; cold-seed convergence 37–81 % on legs | `denavithartenberg.go:182-229`; `la.go:712` | C, D1, R, Z |
| E4 | p2 | DH IK: silent clamp inside FK, J unaware of limits; on failure no best-effort q and no reason | `denavithartenberg.go:126`, `:110-112` | F1, E1/E2 |
| E5 | p2 | Zero-value `Config` (Min = Max = 0) pins joints at 0; no validation in `New` | `dh/config.go:37-46`; `denavithartenberg.go:35` | F2 |
| E6 | p2 | Orientation target accepted but ignored (doc: "consumes a desired pose") | `denavithartenberg.go:96-99`, `:243-252` | reading |
| E7 | p3 | Per-iteration heap allocation (J, J⁺, Δq, `View()` copy, SVD pool) | `denavithartenberg.go:139`, `:184-185`, `:221` | reading |
| E8 | p1 | planar3 IK wrong with `err = nil` when the foot is behind the yaw axis (unsigned radius, coxa flips by π, then clamped) | `planar/planar3dof.go:122`, `:142` | X, Y, R |
| E9 | p2 | planar3 IK: `a1` not wrapped before `Limit`; unreachable targets and limit violations clamped silently | `planar3dof.go:131`, `:139`, `:142-144` | I, X |
| E10 | p3 | planar2 IK: off-surface targets give `err = nil` | `planar/planar2dof.go:118-121` | reading |
| E11 | p1 | Host CLI sends planar limits in degrees and lengths in mm; firmware IK is a placeholder and returns errors | `cmd/manipulator/main.go:346-348`; `drivers/manipulator/kinematics.planar.go:44-47`, `kinematics.dh.go:44-57` | I, reading |
| E12 | p1 | Firmware maps IK radians to servo degrees with no zero offset or sign (servo range 0…MaxAngle) | `drivers/manipulator/main.go:333` | reading |
| E13 | p2 | Firmware ignores `config.Motion` (speed/accel/jerk): always `New1D(100,100,100)` | `drivers/manipulator/main.go:351-356` | reading |
| E14 | p1 | VAJ1D `if x1 < 1` should be `< 0`: positive moves stop up to 1 unit short and sub-unit moves never start | `x/math/control/motion/vaj1d.go:59` | J |
| E15 | p1 | VAJ1D violates its acceleration limit (velocity snap) and overshoots at coarse dt | `vaj1d.go:92-96` | J |
| E16 | p2 | Gait swing arc stacks clearances; touchdown rises before landing | `gait/support/path_planner.go:366-377`, `:386-394`, `:233-243` | K |
| E17 | p1 | Path-tracking lateral PID has the wrong sign; heading error not wrapped | `motion/rigidbody/motion.go:364-371`, `:504-510` | M |
| E18 | p2 | steer4: Ackermann angles use a rear-axle centre of rotation, wheel rates use the chassis centre; drive flags unused | `wheels/steer4/drive.go:69-72`, `:173-174`, `:28-29` | L |
| E19 | p3 | differential: "wheel rates" are linear speeds; radius unused | `wheels/differential/drive.go:11`, `:27`, `:80`, `:116` | L |
| E20 | p2 | gait/rigidbody `Advance` and `EstimateFromEndpoints` return `ErrNotImplemented` | `gait/rigidbody/planner.go:60-77` | reading |
| E21 | p3 | thrusters SPEC claims limits are honoured; saturation actually returns `ErrInfeasible` | `kinematics/SPEC.md:106`; `thrusters/inverse.go:30` | test `TestInverseInfeasibleDueToLimits` |
| E22 | p3 | `cmd/manipulator/manipulator`: a committed 9.3 MB ELF x86-64 binary | `cmd/manipulator/manipulator` | `file` |
| E23 | p1 | Small SI inertia tensors are rejected as singular: absolute det tolerance 1e-6. A 1 kg, 0.2 m body has det 3.0e-7, so `rigidbody.NewModel` and `thrusters.NewModel` both error; 1.5 kg/0.3 m and 2 kg/0.5 m pass | `x/math/mat/inverse.go:14`, `:78`; `kinematics/rigidbody/model.go:47`; `thrusters/helpers.go:12-15` | `tmp/easyrobot/ik/inertia` (`go run .`, 2026-11-09); reported by TensorNN, reproduced here |

## 5. Joint-configuration helpers (proposed API sketch)

This is a proposal, not implemented. It is grounded in `dh.Config`, `planar.Config`, `servo.Motor` (`x/devices/servo/types.go`), `kintypes.Bidirectional` and `types/control/kinematics.JointsConfig`. Package: `x/math/control/kinematics/chain`.

```go
package chain

type JointKind uint8
const (Revolute JointKind = iota; Prismatic)

// Joint is declared in model units: radians and metres. Degrees appear only in ServoMap.
type Joint struct {
    Name                  string
    Kind                  JointKind
    A, Alpha, D, Offset   float32 // standard DH: r, α, d, θ offset (or d offset for prismatic)
    Min, Max              float32 // model limits; must satisfy Min < Max (zero value rejected)
    MaxVel, MaxAcc        float32 // feed VAJ1D / trajectory timing
    Servo                 ServoMap
}

// ServoMap: servoDeg = ZeroDeg + Sign*Ratio*deg(q). It then goes through servo.Motor (µs).
type ServoMap struct {
    Channel  int     // PCA9685 / PWM channel
    ZeroDeg  float32 // servo angle when q == 0 (calibration)
    Sign     int8    // +1 / -1 (mirror-mounted servos on left/right legs)
    Ratio    float32 // horn/gear ratio, default 1
}

type Chain struct { Name string; Joints []Joint }

func Deg(d float32) float32 // degrees → radians helper for declarations
func (c Chain) Validate() error // Min<Max, finite lengths, |Sign|==1, servo range covers [Min,Max] after mapping,
                                 // supported Kind; typed errors (ErrBadLimits, ErrServoRange, …)
func (c Chain) DH() []dh.Config // Index from Kind; Theta = Offset
func (c Chain) Planar3() ([3]planar.Config, error) // only if c is yaw + two parallel pitch joints
func (c Chain) Model(opts ...Option) (kintypes.Bidirectional, error) // analytic leg solver when the shape matches, DLS otherwise
func (c Chain) ToServo(q, servoDeg []float32) error   // radians → servo degrees (clamped, error on limit)
func (c Chain) FromServo(servoDeg, q []float32) error
func (c Chain) CheckZeroPose(want vec.Vector3D, tol float32) error // FK(0) vs a measured foot position: catches sign/offset mistakes
func FromProto(*kinematics.JointsConfig) (Chain, error)           // with explicit units: radians and metres
func (c Chain) Proto() *kinematics.JointsConfig

// Legged layer
type LegGeometry struct { Coxa, Femur, Tibia, FemurOffset, TibiaOffset float32 } // offsets for bent links
func CoxaFemurTibia(name string, g LegGeometry, lim [3][2]float32, servo [3]ServoMap) Chain
type Mount struct { Leg Chain; Hip vec.Vector3D; Yaw float32 } // hip pose in the body frame
type Body  struct { Mounts []Mount; Neutral []vec.Vector3D }   // neutral foot positions (body frame)
func Hexapod(front, middle, rear, halfWidth float32, g LegGeometry) Body
func Quadruped(halfLength, halfWidth float32, g LegGeometry) Body
func (b Body) FootInLeg(i int, bodyPose Pose, footWorld vec.Vector3D) vec.Vector3D // world → body → hip frame
func (b Body) SolveLegs(bodyPose Pose, feet []vec.Vector3D, q [][]float32) []error // per-leg typed errors
```

Design rules:
- No allocation in `Solve*` or `ToServo`: preallocated slices and fixed arrays, ESP32/TinyGo friendly.
- Radians internally; degrees only in `ServoMap`.
- Errors instead of silent clamps.
- Analytic coxa–femur–tibia IK as the default for legs (exact, O(1)); DLS for arbitrary chains.

## 6. Hexapod/quadruped gap analysis

DESIGN and IMPLEMENTATION_PLAN were treated as claims and checked against the code.

| Capability | Exists? | Evidence |
|---|---|---|
| Leg FK/IK (3-DOF) | yes, with defects | [[planar-3dof-leg-kinematics]], [[dh-jacobian-pseudoinverse-ik]] (E1–E9) |
| Joint config / servo mapping / validation | **no** | §5; `drivers/manipulator/main.go:333` |
| Body → leg transforms (hip mounts, body roll/pitch/yaw/height) | **no** | grep: no hip-mount types; `BodyPose` proto only (`state.proto:18`) |
| Neutral stance / foot placement / reach check | **no** | — |
| Stride generation from body twist (foothold = neutral + v·T_stance/2 + ω×r) | **no** | `gait/support` takes `DesiredPose` from the caller |
| Gait patterns: tripod/wave/ripple (6 legs); walk/trot (4 legs) | **no**, types only | [[gait-phase-scheduler]] |
| Phase timing / duty factor / phase offsets | **no**, `GaitTemplate.DutyCycles` and `PhaseMap` declared only | `gait/types/contracts.go:155-196` |
| Swing trajectory | partial, clearance bug | [[gait-support-endpoint-planner]] (E16) |
| Stance trajectory | partial: linear to the desired pose; frame unspecified | `path_planner.go:245-251` |
| Body pose integration / estimation from feet | stub | [[gait-rigidbody-planner]] (E20) |
| Support polygon / stability margin / CoM | **no** (DESIGN claim) | grep: none |
| Terrain / contact adaptation | partial: touchdown height hook (heights / contact-only) | `path_planner.go:183-208` |
| Gait transitions | **no** (types only) | `TransitionPolicy` |
| Servo output timing (synchronous 18-channel update, fixed rate, per-joint VAJ) | partial | VAJ1D buggy (E14/E15); firmware 100 ms tick, 3 servos, `time.Since` dt (`drivers/manipulator/main.go:173-186`, `:361-366`) |
| Simulation / tests for a walking cycle | **no** | gait packages have no tests |

## 7. Proposed tickets (failing test first)

The vault uses one ID scheme: these T-IK ids are mapped here once, and every other note refers to the ER tickets.

| ID | Filed as | Title | Failing test to write first | Fix |
|---|---|---|---|---|
| T-IK-01 | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] | VAJ1D sign/limits | 0→0.5 ends within 1e-3; 0→100 ends within 1e-3; FD \|a\| ≤ 1.05·maxA; reversal never exceeds the reversal position + stopping distance | `vaj1d.go:59` `x1<0`; remove velocity snap `:92-96`; signed state; drop `motion.go:427` workaround |
| T-IK-02 | [[ER-017-dh-planar-ik-robustness\|ER-017]] | DH IK for n>3 | 4-link planar and 6-DOF reachable targets converge (probe G1/G2) | `DampedLeastSquares` (`mat/pseudo_inverse.go:48`) or SVD of Jᵀ |
| T-IK-03 | [[ER-017-dh-planar-ik-robustness\|ER-017]] | Robust DH IK | hexleg realistic workspace, nominal seed ≥ 95 % within 100 iterations; F1 returns `ErrJointLimit` plus best-effort q | DLS with adaptive λ, step cap, limit-aware (clamp-and-zero-column), relative σ cutoff; preallocated buffers |
| T-IK-04 | [[ER-021-mat-vec-quaternion-qr-inverse\|ER-021]] | Quaternion fix (owner [[mat-vec]]) | Rz(90°) → `[0 0 √½ √½]`; `Orientation(Quaternion(R)) == R` | `mat.go:626-648`; pick one layout repo-wide |
| T-IK-05 | [[ER-017-dh-planar-ik-robustness\|ER-017]] | Analytic leg IK errors | unreachable → `ErrUnreachable`; foot behind the yaw axis solved or `ErrJointLimit`; probe R set 500/500 exact | signed reach, angle wrap, typed errors (`planar3dof.go:122-144`) |
| T-IK-06 | [[ER-017-dh-planar-ik-robustness\|ER-017]] | Config validation | `dh.New` with a zero-value `Config` returns an error (or `Validate()` does) | `New` returns `(…, error)` or a separate `Validate` |
| T-IK-07 | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] | Joint/leg helper package | hexleg `Chain.DH()` FK == `Planar3()` FK; `ToServo`/`FromServo` round trip; `Validate` catches bad limits and servo range | §5 API |
| T-IK-08 | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] | Swing profile | apex == SwingHeight ± 1 mm; touchdown z monotonic non-increasing; lift starts at the stance height | rewrite `arcShape`/targets (Bezier from [[interpolation]]) |
| T-IK-09 | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] | Gait phase scheduler | tripod: always exactly 3 stance legs, offsets {0, ½}; wave: ≥5 stance legs; trot: diagonal pairs | implement `GaitScheduler` with presets |
| T-IK-10 | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] | Stride / foothold + body→leg | straight walk: stance feet fixed in the world (< 1 mm drift per cycle); turn in place | foothold generator + `Body.FootInLeg` |
| T-IK-11 | [[ER-037-body-pose-static-stability\|ER-037]] | Static stability | CoM projection inside the stance polygon with margin m; wave gait margin > tripod margin | support polygon + margin helper |
| T-IK-12 | [[ER-038-wheeled-motion-fixes\|ER-038]] | Path-tracking signs | probe M cases: path on the left ⇒ ω > 0; across the ±π seam ⇒ ω > 0 | `motion.go:364-371` |
| T-IK-13 | [[ER-038-wheeled-motion-fixes\|ER-038]] | steer4 geometry | wheel heading equals the rigid-body velocity direction at each wheel (< 1e-4 rad) | one reference point |
| T-IK-14 | [[ER-039-manipulator-integration\|ER-039]] | Manipulator integration | host unit test of the mapping function: `cmd` config → `Chain` (rad, m) → servo degrees | use T-IK-07 in `drivers/manipulator`; honour `config.Motion` |
| T-IK-15 | [[ER-037-body-pose-static-stability\|ER-037]] | gait/rigidbody | `Advance` integrates a constant twist into the analytic pose; `EstimateFromEndpoints` recovers a known transform | quaternion integration, Kabsch |

## 8. Decisions for the owner

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | Leg IK method | (a) analytic coxa–femur–tibia + DLS fallback; (b) DLS for everything | (a): exact, O(1), no allocation, no seed dependence |
| D2 | Quaternion layout | `[x y z w]` (`vec.Quaternion` methods, DH) vs `[w x y z]` (proto, `motion/rigidbody`) | One layout in Go (`[x y z w]`, matching `vec.Quaternion`); convert at the proto boundary |
| D3 | Units on the wire | radians/metres vs degrees/mm | radians/metres in `JointsConfig`; degrees only in the servo map |
| D4 | Gait architecture | DESIGN's generic template/role bindings vs a minimal phase scheduler with presets | Minimal scheduler first (T-IK-09), keeping DESIGN types where cheap |
| D5 | Where IK runs | host (send joint angles) vs MCU | MCU only with an FPU (ESP32/ESP32-S3); XIAO SAMD21 (Cortex-M0+, no FPU) [INFERENCE from the part's datasheet class] should get joint targets |

## 9. Unverified

- TinyGo/firmware behaviour (no TinyGo, no hardware).
- Real servo timing and jitter.
- Behaviour of `drivers/manipulator` on a board.
- Thruster allocation for configurations other than the hover tests.
- `motion/rigidbody` with lookahead ≥ 1.
- The IK cost on ESP32.

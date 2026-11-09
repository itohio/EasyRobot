---
type: algorithm
status: partial
tags: [easyrobot, kinematics, dh, fk]
workstream: W3
module: "[[kinematics]]"
summary: "partial (orientation quaternion inverted)"
source: [x/math/control/kinematics/joints/dh/config.go, x/math/control/kinematics/joints/dh/denavithartenberg.go, x/math/mat/mat.go]
tests: [x/math/control/kinematics/joints/dh/denavithartenberg_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 ./x/math/control/kinematics/joints/dh → ok, 7 tests/15 subtests (2026-11-09); probe sections A,H in [[ER-R2-ik-review]]: position exact to 1.5e-8 m, orientation quaternion is the inverse rotation"
---
# DH forward kinematics

**What it does:** Chains one homogeneous transform per joint from a Denavit–Hartenberg table and returns the end-effector position plus an orientation quaternion.

**Where:** `dh/config.go:48-88` (`Config.CalculateTransform`), `dh/denavithartenberg.go:77-97` (`Forward`), `:121-143` (`forwardInternal`). Quaternion from `x/math/mat/mat.go:626-648` (`Matrix.Quaternion`).

**Inputs → outputs:** `state` DOF×1 joint values (radians for revolute `Index 0`, length units for prismatic `Index 3`; `Index 1/2` vary α / r) → `destination` 7×1 `[x y z qx qy qz qw]`. Convention: **standard (distal) DH**, `A_i = Rz(θ_i)·Tz(d_i)·Tx(r_i)·Rx(α_i)`, matrix written out at `config.go:71-86`. `H0i[0]=I`, `H0i[i+1]=H0i[i]·A_i`. Joint values are clamped to `[Min,Max]` and written back (`:126`).

## Completeness review

- Complete: yes for position; orientation output present but wrong (below).
- Tested: `TestDenavitHartenberg_Forward` (4 cases: zero pose, 90° base, 3-link, α=π/2 with d offset; positions only), `TestDenavitHartenberg_ForwardPopulatesTransforms` (H0i equals chained `CalculateTransform`), `TestDenavitHartenberg_JointLimits`, `TestDenavitHartenberg_PrismaticJoint`. Run: `go test -count=1 -json ./x/math/control/...` → pass (2026-11-09). No test checks the quaternion rows.
- Correct: **position yes** — probe section A: hexapod leg (39/45/100 mm) at 5 hand-computed poses, max error 1.5e-8 m; equals `planar.New3DOF` FK to 3.0e-8 m over 10 000 random poses; float64 independent DH reference agrees (Jacobian section). **Orientation no** — probe H: a single revolute joint at θ=+90° returns `[0 0 0.7071 −0.7071]` (= Rz(−90°)); θ=30° returns `[0 0 −0.2588 0.9659]`; α=+90° returns `[0.7071 0 0 −0.7071]`. `Matrix.Quaternion` (`mat.go:631-643`) uses the row-vector (transposed) formula, so it returns the conjugate. Layout `[x y z w]` also differs from `proto/types/control/manipulator.proto` and `motion/rigidbody` (`[w x y z]`).
- Other: zero-value `Config{}` has `Min=Max=0`, so every joint clamps to 0 (probe F2: FK(0.5,0.5,−1) = stretched pose). `View()` at `:139` copies the matrix into a heap `mat.Matrix` every call.

## TODOs

- [ ] Fix rotation→quaternion (conjugate) in `Matrix.Quaternion`; add test Rz(90°)→[0,0,√½,√½] — x/math/mat/mat.go:626 (owner [[mat-vec]]) #p1 #ER-021
- [ ] Add quaternion assertions to `TestDenavitHartenberg_Forward` — denavithartenberg_test.go:29 #p2
- [x] Reject or warn on `Min==Max==0` configs (zero-value limits clamp all joints) — `dh.New` returns `ErrInvalidConfig` via `Config.Validate` since [[ER-017-dh-planar-ik-robustness|ER-017]] #p2
- [ ] Avoid per-call heap copy from `H0i[n].View()`; read rotation from the fixed `Matrix4x4` — denavithartenberg.go:139 #p3
- [ ] Pick one quaternion layout repo-wide ([x y z w] vs [w x y z]) — manipulator.proto vs denavithartenberg.go:75 #p2

## Links

- Module: [[kinematics]] · Jacobian: [[dh-geometric-jacobian]] · IK: [[dh-jacobian-pseudoinverse-ik]] · Review: [[ER-R2-ik-review]] · [[W3-legged-ik]] · quaternion conversion defect (mat-vec slice): [[vector-and-quaternion]], [[homogeneous-transforms]] (DH uses only `Matrix4x4.Mul`, `Col3D`, `Matrix.Quaternion`; not `HomogenousFrom*`/`Inverse`)

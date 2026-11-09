---
type: algorithm
status: tested
tags: [easyrobot, kinematics, legged, servo, config]
workstream: W3
module: "[[kinematics]]"
summary: "One joint/leg declaration → validated DH or analytic leg model, servo degrees, body→hip transforms, allocation-free multi-leg IK."
source: [x/math/control/kinematics/chain/chain.go, x/math/control/kinematics/chain/legged.go]
tests: [x/math/control/kinematics/chain/chain_test.go, x/math/control/kinematics/chain/legged_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race -count=1 ./x/math/control/kinematics/chain → ok (2026-11-09); tests failed first against stub signatures (tmp/easyrobot/er018/chain-failing.txt) [[ER-018-leg-helpers-gait-scheduler|ER-018]]"
source_revision: 1bf6234e676daa87d3e9eb191bb1dc2420f05ff9
---
# Leg/chain configuration, servo mapping and body→hip transforms

**What it does:** one declaration of a joint chain (DH parameters, limits, speed limits, servo mapping) produces the solver configs, validates them, converts joint angles to servo degrees and back, and for legged bodies places each leg's hip on the body and solves all legs per tick. Implements IK review §5 (evidence: `~/src/docs/evidence/ER-R2-ik-review.md#5. Joint-configuration helpers (proposed API sketch)`, outside the repo). Radians and metres everywhere; degrees only inside `ServoMap`.

**Where:** `x/math/control/kinematics/chain/chain.go`, `legged.go`.
- `Joint{Name, Kind, A, Alpha, D, Offset, Min, Max, MaxVel, MaxAcc, Servo}`; `ServoMap{Channel, ZeroDeg, Sign, Ratio, MinDeg, MaxDeg}`: servoDeg = ZeroDeg + Sign·Ratio·deg(q); MinDeg = MaxDeg = 0 means a 0…180° servo.
- `Chain.Validate()` (finite values, Min < Max, |Sign| = 1, Ratio > 0, servo travel covers [Min, Max]; errors `ErrLimits`, `ErrServoMap`, `ErrServoRange`, all wrapping `kintypes.ErrInvalidConfig`), `DH()`, `Planar3()` (yaw + two parallel pitch joints only; Femur, Tibia > 0 and Coxa ≥ 0, else `ErrInvalidConfig` — no silent switch to DH), `Model(...Option)` (analytic [[planar-3dof-leg-kinematics]] when the shape matches, damped DH ([[dh-jacobian-pseudoinverse-ik]]) otherwise), `ToServo`/`FromServo` (no allocation; a NaN joint returns an error instead of NaN degrees), `CheckZeroPose`.
- Legged: `LegGeometry{Coxa, Femur, Tibia}`, `CoxaFemurTibia`, `DefaultLegLimits()` (probe R), `Mount{Leg, Hip, Yaw}`, `Body{Mounts, Neutral}` with `Clone()`, `Hexapod(front, middle, rear, halfWidth, g)` (hip x coordinates, hips point radially, left femur/tibia servos Sign −1, servos centred on 90°), `Quadruped`, `Pose{X, Y, Z, Roll, Pitch, Yaw}` with `ToWorld`/`ToBody`, `Body.FootInLeg`/`FootInWorld`/`BodyToLeg`/`LegToBody`, `Body.NewLegSolver` → `LegSolver.SolveLegs(pose, feet, q, errs)`. `NewLegSolver` deep-copies the body and `Body()` returns a copy; a `LegSolver` is not safe for concurrent use.
- Leg order matches [[gait-phase-scheduler]] (left front → back, then right).

## Completeness review

- Complete: no — `FromProto`/`Proto` (send the declaration over `JointsConfig` with units) and bent-link `LegGeometry` offsets from the §5 sketch are not implemented.
- Tested: `TestHexlegDHMatchesPlanar3` (DH and analytic FK agree < 1e-6 m over 2000 random poses); `TestServoRoundTrip` (incl. Sign −1, 0 allocations); `TestValidate` (19 cases); `TestFootInLegInvertsForwardTransform` (vs an independent float64 transform, < 2e-6); `TestSolveLegsReproducesJointAngles` (0 allocations); `TestSolveLegsDHFallback`, `TestModelPicksSolver`, `TestCheckZeroPose`, layout tests; roster-review tests `TestLegGeometryValidated` (6 cases, failed first), `TestLegSolverCopiesBody`, `TestBodyClone`, `TestToServoNaNJoint` (mutant C1 killed).
- Correct: yes on these tests.

## TODOs

- [ ] `FromProto`/`Proto` for `JointsConfig` with explicit units, offset, sign and servo channel — chain/chain.go #p2
- [ ] Bent-link leg geometry (femur/tibia offsets) — chain/legged.go #p3

## Links

- Module: [[kinematics]] · [[planar-3dof-leg-kinematics]] · [[dh-jacobian-pseudoinverse-ik]] · [[servo-angle-mapping-and-array]] · [[gait-foothold-planner]] · [[W3-legged-ik]]

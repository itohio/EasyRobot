---
type: algorithm
status: tested
tags: [easyrobot, kinematics, planar, ik, legged]
workstream: W3
module: "[[kinematics]]"
summary: "Closed-form coxa–femur–tibia FK/IK; exact over the whole workspace incl. foot behind the yaw axis; typed errors (ER-017)."
source: [x/math/control/kinematics/joints/planar/planar3dof.go, x/math/control/kinematics/joints/planar/config.go]
tests: [x/math/control/kinematics/joints/planar/planar3dof_test.go, x/math/control/kinematics/joints/planar/leg_ik_test.go, x/math/control/kinematics/joints/planar/config_test.go]
test_result: pass
complete: no
correct: yes
evidence: "go test -race -count=1 ./x/math/control/kinematics/... → ok (2026-11-09). Before (tmp/easyrobot/er017/ik/before-*.txt): probe R 422/500, ±π limits 1486/2000, wrong poses with err=nil. After: 500/500 and 2000/2000 exact; review: 30 000 random in-limit targets over 300 limit sets, 0 failures [[ER-017-dh-planar-ik-robustness|ER-017]]"
---
# Planar 3-DOF leg kinematics (coxa–femur–tibia)

**What it does:** closed-form FK/IK for a yaw joint followed by two pitch joints in a vertical plane — exactly a hexapod/quadruped leg.

**Where:** `planar/planar3dof.go` (`New3DOF`, `Forward`, `Backward`); `planar/config.go` (`Config{Min, Max, Length}`, `Validate`).

**Inputs → outputs:** joints `[a0 a1 a2]` radians (a0 coxa yaw about +z, a1 femur pitch, a2 tibia **relative** to femur; positive = up) → `[x y z roll pitch yaw]` with `x' = l0 + l1·cos a1 + l2·cos(a1+a2)`, `z = l1·sin a1 + l2·sin(a1+a2)`, `x = x'·cos a0`, `y = x'·sin a0`; `pitch = a1+a2`, `yaw = a0`. Equivalent to a DH chain with α₁=+π/2, a=(l0,l1,l2) (probe A: FK agree to 3e-8 m; `chain.TestHexlegDHMatchesPlanar3` < 1e-6 m).

**IK (since ER-017):** four candidates — coxa forward and reversed (foot behind the yaw axis, signed reach), each with knee down first, then knee up; angles are wrapped into [Min, Max]; the first in-limit candidate is returned. The solution depends on the target only; the `state` seed is ignored. Errors: `ErrUnreachable` (outside the annulus or on the yaw axis; best effort stretches from the femur pivot straight at the target), `ErrJointLimit` (reachable only past a limit; best effort is the nearest of several clamped candidates), `ErrNonFinite` (NaN/Inf target, controls untouched). The written pose is always in limits; `Backward` does not allocate. One instance per goroutine/leg. `New3DOF` validates configs (Min < Max, femur and tibia lengths > 0).

## Completeness review

- Complete: no — no per-joint zero offset/sign in `planar.Config` (handled by the leg chain helpers of ER-018), no caller choice of knee branch.
- Tested: `TestPlanar3DOF_ProbeRWorkspace` (500/500 exact incl. the 78 feet behind the yaw axis), `TestPlanar3DOF_FullRangeExact` (±π limits, 2000/2000, incl. the review's minimal reproduction), `TestPlanar3DOF_TypedErrors`, `TestPlanar3DOF_NonFiniteTarget`, review fixes `TestPlanar3DOF_KneeBranches` (knee-up-only limits 500/500, knee-down tried first), `TestPlanar3DOF_BestEffortQuality` (stretch direction; `ErrJointLimit` residual ≤ a 61³ in-limit grid minimum), `TestPlanar3DOF_BoundaryRoundTrips` (full stretch, full fold, joints exactly at limits), `TestPlanar3DOF_BackwardNoAllocations`, `TestConfig_Validate`, `TestNew_RejectsInvalidConfig`, plus the original forward/backward/round-trip tests. Mutants P03, P10, P12 and zero tolerances are killed.
- Correct: yes.

## TODOs

- [x] Return `ErrUnreachable` / `ErrJointLimit` instead of clamping silently — planar3dof.go `Backward` #p1 #ER-017
- [x] Handle foot behind the yaw axis (signed reach, or pick the coxa solution inside limits) — planar3dof.go `Backward` #p1 #ER-017
- [x] Wrap angles into [Min,Max] (±2π) before clamping — planar3dof.go #p2
- [x] Knee-branch selection (knee up/down tried in order) — planar3dof.go #p2
- [x] Tests: random realistic workspace round trip, unreachable target error, foot-behind-axis case (ER-017) — leg_ik_test.go #p2
- [x] Best-effort pose closest to the target on errors (not least-violating candidate) — planar3dof.go `closestInLimits` #p3
- [ ] Fix units in cmd/manipulator config (radians, metres) — cmd/manipulator/main.go:346 #p1 #ER-039

## Links

- Module: [[kinematics]] · [[planar-2dof-arm-kinematics]] · [[dh-jacobian-pseudoinverse-ik]] · [[ER-R2-ik-review]] · [[W3-legged-ik]]

---
type: algorithm
status: tested
tags: [easyrobot, kinematics, planar]
workstream: W3
module: "[[kinematics]]"
summary: "Yaw joint plus one pitch link: FK and closed-form IK."
source: [x/math/control/kinematics/joints/planar/planar2dof.go]
tests: [x/math/control/kinematics/joints/planar/planar2dof_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test planar → ok (2026-11-09): FK 3 cases, IK targets (2,0,0)/(0,2,0), round trip, joint limits"
---
# Planar 2-DOF yaw + pitch kinematics

**What it does:** Yaw joint plus one pitch link: FK and closed-form IK.

**Where:** `planar/planar2dof.go` (`New2DOF(cfg [2]Config) (*p2d, error)`, which rejects invalid configs such as zero-width limits since [[ER-017-dh-planar-ik-robustness|ER-017]]; `Forward`; `Backward`).

**Inputs → outputs:** `[a0 a1]` radians → `[x y z 0 a1 a0]`; `x' = l0 + l1·cos a1`, `z = l1·sin a1`. IK: `a0 = atan2(y,x)`, `a1 = atan2(z, √(x²+y²) − l0)`.

## Completeness review

- Complete: yes for its scope (2 joints have a 2-D reachable set; a 3-D target is generally not reachable).
- Tested: `TestPlanar2DOF_Forward` (zero, 90° base, 45° elbow), `TestPlanar2DOF_Backward` (reachable targets), `TestPlanar2DOF_RoundTrip`, `TestPlanar2DOF_JointLimits`, `TestPlanar2DOF_NonFiniteTarget` (NaN/Inf → `ErrNonFinite`, controls untouched; failed first with NaN controls and nil error). Pass (2026-11-09).
- Correct: yes for reachable targets (tests). Off-surface targets return the direction-only solution with `err=nil`; limits clamp silently.

## TODOs

- [ ] Report distance-to-reachable-set or `ErrUnreachable` for targets off the l1 circle — planar2dof.go:118 #p3

## Links

- Module: [[kinematics]] · [[planar-3dof-leg-kinematics]]

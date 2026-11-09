---
type: algorithm
status: tested
tags: [easyrobot, motion, path]
workstream: [W3, W6]
module: "[[motion-and-gait]]"
summary: "Arc-length parameterised piecewise-linear path with optional per-waypoint quaternion (slerp) and…"
source: [x/math/control/motion/rigidbody/path_planner_base.go, x/math/control/motion/rigidbody/path_planner_factory.go, x/math/control/motion/rigidbody/path_planner_position.go, x/math/control/motion/rigidbody/path_planner_position_velocity.go, x/math/control/motion/rigidbody/path_planner_pose.go, x/math/control/motion/rigidbody/path_planner_pose_velocity.go]
tests: [x/math/control/motion/rigidbody/motion_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test motion/rigidbody → ok (2026-11-09): TestSetWaypointMatrixPosition/Pose/PositionVelocity/PoseVelocity, TestForwardStraightPath"
---
# Polyline path sampler

**What it does:** Arc-length parameterised piecewise-linear path with optional per-waypoint quaternion (slerp) and linear/angular velocity hints: `Length`, `Sample(s)`, `Project(p)`, `Curvature(s)` (three-point circumradius), `Signature` (FNV-style hash to detect path changes).

**Where:** `path_planner_base.go` (helpers: `searchSegment` :120, `curvature2D` :144, `projectPoint` :194, `interpolateQuaternion` :216); `path_planner_factory.go:14-32` (waypoint matrix rows 3/6/7/14 → planner); four planners (`path_planner_position.go:37` `Sample`, `:76` `Project`, `:92` `Curvature`; same shape in the other three files).

**Inputs → outputs:** waypoints (≥2) → samples (position, unit tangent, orientation, hints).

## Completeness review

- Complete: yes.
- Tested: the four `TestSetWaypointMatrix*` tests check sample positions/orientations/hints; pass (2026-11-09).
- Correct: yes for tested layouts; curvature formula checked by hand (`2·|den|/(abc)` = 4·Area/(abc)). Quaternion layout here is `[w x y z]` (`identityQuaternion` = {1,0,0,0}, `:36`) while `vec.Quaternion` methods assume `[x y z w]` — `Slerp` is layout-agnostic, `yawFromQuaternion` is internally consistent, but mixing with [[dh-forward-kinematics]] output would be wrong.

## TODOs

- [ ] Use the repo-wide quaternion layout once decided — path_planner_base.go:36 #p2

## Links

- Module: [[motion-and-gait]] · [[rigidbody-path-follower]]

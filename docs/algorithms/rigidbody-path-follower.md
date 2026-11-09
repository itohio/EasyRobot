---
type: algorithm
status: partial
tags: [easyrobot, motion, planner, path-following]
workstream: [W3, W6]
module: "[[motion-and-gait]]"
summary: "partial (VAJ path bypassed for short lookahead)"
source: [x/math/control/motion/rigidbody/motion.go]
tests: [x/math/control/motion/rigidbody/motion_test.go, x/math/control/motion/rigidbody/toy_triangle_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test motion/rigidbody → ok, 10 tests (2026-11-09) incl. closed-loop sinusoid (deviation ≤0.5 m, yaw ≤0.55 rad); depends on VAJ1D which is wrong for moves ≥1 unit"
---
# Rigid-body path follower (Motion.Forward)

**What it does:** Pure-pursuit-like preview: project the pose on the path, advance progress toward `progress + lookahead` with a jerk-limited profile, limit speed by curvature (`v ≤ √(a_lat/κ)`) and stopping distance, and rate-limit yaw toward the path tangent (or waypoint orientation / velocity hints).

**Where:** `motion/rigidbody/motion.go:187-232` (`Forward`), `:234-300` (`planForward`), `:413-449` (`advanceProfile`), `:451-462` (`advanceYaw`), `:464-482` (`limitSpeed`).

**Inputs → outputs:** state 6×1 `[x y z yaw speed t]`, optional controls ≥2×1 `[linear angular effort…]` → next state (a point on the path, not a dynamics integration) and controls.

## Completeness review

- Complete: no — the generic planner in `motion/IMPLEMENTATION_PLAN.md` (any `Bidirectional` model, cost terms) is not implemented; this planner is unicycle-only (yaw + speed).
- Tested: `TestForwardStraightPath`, `TestForwardRequiresPath`, `TestSetWaypointMatrix*` (4 layouts), `TestMotionMetadata`, `TestTriangularChassisFollowsSinusoid` (closed loop with [[rigid-body-velocity-wrench-model]]: max deviation ≤ 0.5 m, yaw error ≤ 0.55 rad over 200 steps). Pass (2026-11-09).
- Correct: unknown. Lookahead < 1 (all tests use 0.5) skips VAJ entirely (`:427`, a workaround for the [[vaj1d-jerk-limited-profile]] bug); with lookahead ≥ 1 progress is fed through the faulty profile. `Project` is a global nearest-segment search each tick (`path_planner_position.go:76`), so self-crossing paths can jump.

## TODOs

- [ ] Re-test with lookahead ≥ 1 after the VAJ fix; drop `delta < 1` branch — motion.go:427 #p2
- [ ] Windowed projection (search near last progress) — motion/rigidbody/path_planner_position.go:76 #p3

## Links

- Module: [[motion-and-gait]] · [[polyline-path-sampler]] · [[path-tracking-pid-controller]] · [[vaj1d-jerk-limited-profile]]

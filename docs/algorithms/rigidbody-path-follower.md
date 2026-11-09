---
type: algorithm
status: tested
tags: [easyrobot, motion, planner, path-following]
workstream: [W3, W6]
module: "[[motion-and-gait]]"
summary: "Unicycle path follower; jerk-limited progress to the path end at MaxSpeed (ER-018); lateral speed cap only reported"
source: [x/math/control/motion/rigidbody/motion.go]
tests: [x/math/control/motion/rigidbody/motion_test.go, x/math/control/motion/rigidbody/toy_triangle_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test -race -count=1 ./x/math/control/motion/rigidbody → ok (2026-11-09). TestForwardStraightPathReachesCruiseSpeed failed after the VAJ workaround removal (peak 0.887/1.88/1.63/4.19 vs MaxSpeed 2/10; 23.44 s vs optimum 12.2 s) and passes (peak = MaxSpeed, 11.98 s and 3.36 s) [[ER-018-leg-helpers-gait-scheduler|ER-018]]"
---
# Rigid-body path follower (Motion.Forward)

**What it does:** project the pose on the path, advance progress along it with a jerk-limited profile ([[vaj1d-jerk-limited-profile]]) whose target is the path end and whose acceleration limit is min(MaxAcceleration, MaxDeceleration) (VAJ1D has one symmetric limit, so it also accelerates at that rate), limit the reported speed by curvature (`v ≤ √(a_lat/κ)`, curvature read at `progress + LookaheadDistance`) and stopping distance, and rate-limit yaw toward the path tangent (or waypoint orientation / velocity hints).

**Where:** `motion/rigidbody/motion.go`: `Forward`, `planForward`, `advanceProfile(total)`, `advanceYaw`, `limitSpeed`.

**Inputs → outputs:** state 6×1 `[x y z yaw speed t]`, optional controls ≥2×1 `[linear angular effort…]` → next state (a point on the path, not a dynamics integration) and controls.

## Completeness review

- Complete: no — the generic planner in `motion/IMPLEMENTATION_PLAN.md` (any `Bidirectional` model, cost terms) is not implemented; this planner is unicycle-only (yaw + speed).
- Tested: `TestForwardStraightPath`, `TestForwardStraightPathReachesCruiseSpeed` (20 m closed loop, two limit sets × lookahead 0.5 and 2: actual per-tick displacement speed peaks at MaxSpeed (2.00005, 10.00004 m/s) and matches the reported speed, arrival ≤ 1.15× time-optimal), `TestForwardPathEndRespectsMaxDeceleration` (0.5003 m/s² for MaxDeceleration 0.5; failed first at 2.0), `TestForwardArcCapsSpeedAtLookahead` (reported speed obeys the lateral cap on an arc, 10 % tolerance for the polyline curvature estimate), `TestForwardRequiresPath`, `TestSetWaypointMatrix*` (4 layouts), `TestMotionMetadata`, `TestTriangularChassisFollowsSinusoid`. Mutants R1 (curvature at progress) and R3 (lateral cap off) are killed.
- Correct: unknown. Since ER-018 the `delta < 1` branch that bypassed VAJ is gone and progress always goes through the profile. Known gap: `limitSpeed` (lateral-acceleration and deceleration caps) lowers only the reported speed; the next position still comes from the unlimited profile, so the two can disagree on curves. `Project` is a global nearest-segment search each tick (`path_planner_position.go:76`), so self-crossing paths can jump.

## TODOs

- [x] Re-test with lookahead ≥ 1 after the VAJ fix; drop `delta < 1` branch — motion.go `advanceProfile` #p2
- [ ] Apply the curvature/stopping speed cap to the profile, not only to the reported speed — motion.go `limitSpeed` #p2
- [ ] Windowed projection (search near last progress) — motion/rigidbody/path_planner_position.go:76 #p3

## Links

- Module: [[motion-and-gait]] · [[polyline-path-sampler]] · [[path-tracking-pid-controller]] · [[vaj1d-jerk-limited-profile]]

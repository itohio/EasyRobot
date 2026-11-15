---
type: algorithm
status: partial
tags: [easyrobot, filter, slam, ekf, lidar]
workstream: W8
module: "[[filters]]"
summary: "Zero Jacobian, no motion model"
source: [x/math/filter/slam/slam.go]
tests: [x/math/filter/slam/slam_test.go]
test_result: pass
complete: no
correct: no
evidence: "slam 12/12 pass (shape checks); probe slamprobe: estimate stays at (1.83,2.13) for 20 updates vs truth (2.03,2.03)"
---
# EKF ray-cast localization ("SLAM" filter)

**What it does:** Estimates a 2-D pose `[x, y, θ]` against an occupancy grid: EKF whose measurement model is ray-casting the map for a fixed set of beam angles (numerical Jacobian), optionally followed by log-odds map updates ([[log-odds-occupancy-mapping]]). Not joint SLAM: map and pose are updated separately.

**Where:** `x/math/filter/slam/slam.go`: options :22-77, `SLAM` :84-121, `New` :133-196, `initEKF` :199-239 (static motion model :203-207, ray-cast `h` :211-215, Q=0.01·I, R=0.1·I), `UpdateMeasurement` :387-427, `Update` (Filter interface, input 2×N [angles; distances]) :457-472, `GetResiduals` :496-501.

**Inputs → outputs:** beam angles (rad, robot frame), distances (m), map `matTypes.Matrix` (occupied > 0.5), resolution, origin → pose as `mat.Matrix3x3` homogeneous transform.

## Completeness review

- Complete: no. No prediction/odometry input (`f` is identity, `Predict` never called in `UpdateMeasurement`); measurement angles passed at runtime are ignored by the EKF (uses construction-time `rayDirs`, :214) and not checked against them; no data association/outlier rejection; no loop closure.
- Tested: 12 tests — ray-cast helpers (`TestRayCast_*`), construction, set-pose, update (asserts pose within ±10 m and 2 expected distances, `slam_test.go:202-235`), reset, residuals length. Run: go test -count=1 -json ./x/math/filter/slam (2026-11-09) → ok, 12 pass.
- Correct: **no** (probe: 4×4 m walled room, res 0.05 m, 6 beams, exact distances from truth (2.03, 2.03, 0), start (1.83, 2.13)): estimate unchanged at (1.830, 2.130) after 20 updates. Causes: (1) ray-cast distance is piecewise constant in pose, so central differences with ε=1e-5 (:229 → `ekalman` default) give a zero Jacobian → zero gain; (2) inherited covariance collapse of [[extended-kalman-filter]] (`ekalman.go:459`).
- Embedded: `h` allocates a distance vector per call (:212) and is called 2n+1 times per update; `getMapMatrix` clones non-`mat.Matrix` maps on every call (:243-261); `GetResiduals` allocates (:498).

## TODOs

- [ ] Analytic/finite-difference Jacobian with ε ≥ one cell (or likelihood-field/scan-matching model) — slam.go:229 #p1 #backlog
- [ ] Add motion model with odometry input and call Predict — slam.go:203-207,387 #p1 #backlog
- [ ] Use runtime beam angles or validate them — slam.go:214,391 #p2
- [ ] End-to-end localization test (converges from offset start) #p1 #backlog
- [ ] Remove per-call allocations in `h`/`getMapMatrix` #p2

## Links

- Module: [[filters]]; ray casting in [[grid-maps]]; lidar driver: [[lidar]]; workstream [[W8-lidar]]

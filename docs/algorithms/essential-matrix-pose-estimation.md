---
type: algorithm
status: stub
tags: [easyrobot, vision, gocv, odometry, pose]
workstream: [W5]
module: "[[vision]]"
summary: "imports broken types"
source: [x/vision/odometry/pose/processor.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "go/types check with FakeImportC against gocv v0.42.0 (tmp/easyrobot/io/typecheck, 2026-11-09): imports broken x/vision/types; code read: Process always returns 0"
---
# Relative pose from matched features (essential matrix)

**What it does (intended):** from `MatchedFeatures` and camera intrinsics, estimate E with RANSAC (5-point), recover R, t → `Pose{Rotation, Translation, Inliers}` — the core of monocular visual odometry.

**Where:** `New` `processor.go:35-50`; `Process` :58-127 (copies points into Mats, then TODO :110-117 and `return 0`); `extractMatchedPoints` :131-150.

## Completeness review

- Complete: **stub** — never writes a pose; the comment says gocv lacks `findEssentialMat`/`recoverPose`. Confirmed: gocv v0.42.0 has no `FindEssentialMat`, `FindFundamentalMat` or `RecoverPose`, but it has `FindHomography`, `EstimateAffine2D`, `SolvePnP`, `TriangulatePoints`, `UndistortPoints` (`calib3d.go`). `Reset` replaces the preallocated 1000-row point Mats with empty ones (:153-180), so after any `Reset` every call takes the "resize" path.
- Tested: none. Correct: no.
- W5 relevance: monocular VO needs this; options are an 8-point/5-point implementation in Go on `x/math/mat` (no OpenCV, embeddable) or a custom cgo binding.

## TODOs

- [ ] Implement E estimation (normalised 8-point + RANSAC in Go, or bind `findEssentialMat`/`recoverPose`) — x/vision/odometry/pose/processor.go:110-117 #p2
- [ ] Keep preallocated buffers across `Reset` — x/vision/odometry/pose/processor.go:153-180 #p3

## Links

- Module: [[vision]]; input [[vision-feature-matching]]; math [[mat-vec]]; [[W5-feature-tensor-experiments]]

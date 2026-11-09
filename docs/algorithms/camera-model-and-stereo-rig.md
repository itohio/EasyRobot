---
type: algorithm
status: partial
tags: [easyrobot, vision, gocv, calibration, stereo]
workstream: [W4, W5]
module: "[[vision]]"
summary: "14 errors"
source: [x/vision/types/camera.go, x/vision/types/stereo_camera.go, x/vision/types/types.go, x/vision/types/mask.go]
tests: []
test_result: build-fail
complete: no
correct: unknown
evidence: "go/types check with FakeImportC against gocv v0.42.0 (tmp/easyrobot/io/typecheck, 2026-11-09): x/vision/types 14 type errors"
---
# Camera model and stereo rig types (`x/vision/types`)

**What it does:** shared data types for all vision processors (`KeypointsWithImage`, `DescriptorsWithImage`, `MatchedFeatures`, `Pose`, `Displacement`, `ImageWithMask`) plus a pinhole `Camera` (K, distortion, R, P) and `StereoCamera` (two cameras, R, T, E, F, Q; baseline; disparity→depth).

**Where:** `types.go` (data types), `mask.go`; `Camera` `camera.go:14-196` (`ProjectPoints` :141-153, `BackprojectPoints` :157-165, `UndistortPoints` :169-181 — all return "not yet implemented"); `StereoCamera` `stereo_camera.go:21-211`.

## Completeness review

- Complete: **no**, and **the package does not compile** (14 type errors): field `Q` and method `Q()` both declared (`stereo_camera.go:35`, `:181`); `NewStereoCameraFromParameters` passes `*Camera` where `NewStereoCamera` takes `Camera` (:84); struct values compared with `nil` (:108, :190, :193); `gocv.ReprojectImageTo3D` undefined (:135). Because detect/blobs, detect/features, extract/blobs, extract/features, odometry/phase, odometry/pose, transform/background and transform/histogram import this package, **none of them can build even with OpenCV installed**.
- Stubs claim gocv lacks the functions, but gocv v0.42.0 has `UndistortPoints` (`calib3d.go:208`) and `TriangulatePoints` (`calib3d.go:338`); `ProjectPoints`/backprojection are a few lines of Go.
- Data types embed `gocv.KeyPoint`/`gocv.Mat`, so nothing in the vision pipeline is usable without CGO/OpenCV (no TinyGo, no K230).
- Tested: none.

## TODOs

- [ ] Fix the 14 type errors (rename method `Q`→`QMatrix`, pointer params, drop nil compares, remove ReprojectImageTo3D use) — x/vision/types/stereo_camera.go:35,84,108,135,181,190,193 #p1 #ER-030
- [ ] Implement `ProjectPoints`, `BackprojectPoints`, `UndistortPoints` (gocv or pure Go) — x/vision/types/camera.go:141-181 #p2
- [ ] CGO-free keypoint/pose types (plain structs) so features can cross the K230→ESP32 boundary — x/vision/types/types.go #p2

## Links

- Module: [[vision]]; [[chessboard-corner-detection]]; [[stereo-disparity-to-depth]]; [[W4-k230-feature-stream]]

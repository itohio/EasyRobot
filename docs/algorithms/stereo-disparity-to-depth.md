---
type: algorithm
status: partial
tags: [easyrobot, vision, gocv, stereo, depth]
workstream: [W5]
module: "[[vision]]"
summary: "partial (no binding, ×16 scale)"
source: [x/vision/stereo/depth/processor.go, x/vision/stereo/depth/combined_processor.go, x/vision/types/stereo_camera.go]
tests: []
test_result: build-fail
complete: no
correct: no
evidence: "go/types check with FakeImportC against gocv v0.42.0 (tmp/easyrobot/io/typecheck, 2026-11-09): stereo/depth 5 errors (gocv.ReprojectImageTo3D, StereoSGBM undefined)"
---
# Disparity → 3-D points (reprojection with Q)

**What it does (intended):** `ReprojectImageTo3D(disparity, Q)` → H×W×3 float point cloud tensor; `CombinedProcessor` = SGBM + reprojection in one step; `StereoCamera.ReprojectTo3D` helper.

**Where:** `stereo/depth/processor.go:26-110` (call at :72); `combined_processor.go:31-144` (call at :98); `StereoCamera.ReprojectTo3D` `types/stereo_camera.go:129-143`.

## Completeness review

- Complete: no — **does not compile**: `gocv.ReprojectImageTo3D` does not exist in gocv v0.42.0 (ROADMAP line 128 lists it unchecked).
- Correct: **no** even if bound: SGBM disparity is CV_16S with 4 fractional bits; OpenCV treats 16-bit signed disparity as integer pixels, so depth comes out 16× too small unless the disparity is converted to float and divided by 16 first. Neither processor converts.
- In Go, reprojection is a per-pixel 4×4 multiply — straightforward to implement without OpenCV.
- Tested: none.

## TODOs

- [ ] Implement reprojection in Go (Q·[x,y,d,1]) or bind it — x/vision/stereo/depth/processor.go:72 #p1 #ER-030
- [ ] Scale CV_16S disparity by 1/16 before reprojection — x/vision/stereo/depth/combined_processor.go:98 #p1 #ER-030

## Links

- Module: [[vision]]; [[stereo-sgbm-disparity]]; [[camera-model-and-stereo-rig]]

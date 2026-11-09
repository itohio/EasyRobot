---
type: algorithm
status: unverified
tags: [easyrobot, vision, gocv, transforms, calibration]
workstream: [W5]
module: "[[vision]]"
summary: "ok"
source: [x/vision/transform/affine/processor.go, x/vision/transform/crop/processor.go, x/vision/transform/undistort/processor.go, x/vision/transform/rectify/processor.go, x/vision/transform/color/processor.go, x/vision/transform/format/image_to_mat.go, x/vision/transform/format/mat_to_image.go, x/vision/transform/format/stereo_processor.go]
tests: []
test_result: none
complete: yes
correct: unknown
evidence: "go/types check with FakeImportC against gocv v0.42.0 (tmp/easyrobot/io/typecheck, 2026-11-09): affine, crop, undistort, rectify, color, format → 0 type errors; no tests"
---
# Image transforms: affine warp, crop, undistort, rectify (remap), colour and format conversion

**What it does:** single-call OpenCV wrappers as chain processors: `WarpAffine` with a fixed 2×3 matrix; crop to rectangle list; `Undistort` with K and distortion (optional new K); `Remap` with precomputed rectification maps; `CvtColor`; `image.Image` ↔ Mat tensor; stereo pair splitter (`format/stereo_processor.go`).

**Where:** affine `Process` `affine/processor.go:54-103`; crop `crop/processor.go:40-111`; undistort `undistort/processor.go:53-103` (+ `Undistort` helper :125-142); rectify `rectify/processor.go:49-93` (+ `Rectify` :114-124); color `color/processor.go:39-84` (+ `CvtColor` :104); format files.

## Completeness review

- Complete: yes for their narrow scope. Each `Process` clones the scratch result into a new Mat per frame (allocation per frame, despite "preallocated" comments). `undistort` creates a new empty Mat every call when no new K is set and never closes it (`undistort/processor.go:82`, `:130`) → native memory leak per frame. Undistort recomputes maps every frame (`gocv.Undistort`) — `InitUndistortRectifyMap` + `Remap` would be cheaper. Rectify takes one map pair; a stereo rig needs two processor instances.
- Tested: none. Correct: unknown. All six packages type-check against gocv v0.42.0.

## TODOs

- [ ] Close or hoist the empty new-camera Mat — x/vision/transform/undistort/processor.go:82,130 #p2
- [ ] Undistort via cached maps + Remap — x/vision/transform/undistort/processor.go:53-103 #p3
- [ ] Add tests for affine/crop/undistort with synthetic images — x/vision/transform #p2

## Links

- Module: [[vision]]; [[camera-model-and-stereo-rig]]; [[chessboard-corner-detection]]

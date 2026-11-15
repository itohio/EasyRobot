---
type: algorithm
status: unverified
tags: [easyrobot, vision, gocv, features]
workstream: [W5]
module: "[[vision]]"
summary: "imports broken types"
source: [x/vision/detect/features/processor.go]
tests: [x/vision/detect/features/processor_test.go]
test_result: build-fail
complete: yes
correct: unknown
evidence: "go/types check with FakeImportC against gocv v0.42.0 (2026-11-09): package 0 own errors but imports broken x/vision/types; OpenCV missing"
---
# Keypoint detection (ORB, SIFT, FAST, GFTT, KAZE, AKAZE, BRISK)

**What it does:** thin wrapper selecting one OpenCV detector at construction and emitting `KeypointsWithImage{Image, []gocv.KeyPoint}` per frame; `Draw` overlays keypoints.

**Where:** `FeatureType` consts `processor.go:17-25`; `New` :54-98; `Process` :100-145; `Close` :152-175; `Draw`/`DrawKeypoints` :177-207.

## Completeness review

- Complete: yes for detection with default parameters; no parameter options (n features, thresholds), no mask input (`scratchMask` allocated but unused).
- Frames with zero keypoints return 0 (`processor.go:127-130`), which stops a [[processor-chain-composition]] chain for that frame silently.
- Build: imports `x/vision/types`; `x/vision/types` itself fails type-check (14 errors, `stereo_camera.go:35/181` field+method `Q`, `:84` `*Camera` passed as `Camera`, `:108/:190/:193` struct compared with nil, `:135` `gocv.ReprojectImageTo3D` undefined), so this package cannot compile even with OpenCV installed.
- Tested: 7 tests (one per detector) on `testdata/img1/000001.png` (present) asserting one output with keypoints — not run.
- Correct: unknown (delegates to OpenCV).
- W5 relevance: classic-feature baseline for visual odometry / BoW place recognition, to compare against learned feature tensors.

## TODOs

- [ ] Detector parameter options (nfeatures, FAST threshold, …) — x/vision/detect/features/processor.go:54-98 #p3
- [ ] Emit empty keypoint sets instead of dropping frames — x/vision/detect/features/processor.go:127-130 #p2

## Links

- Module: [[vision]]; next stage [[vision-descriptor-extraction]]; [[W5-feature-tensor-experiments]]

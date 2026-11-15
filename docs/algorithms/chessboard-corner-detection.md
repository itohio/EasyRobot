---
type: algorithm
status: unverified
tags: [easyrobot, vision, gocv, calibration]
workstream: [W4, W5]
module: "[[vision]]"
summary: "2 (fixture `left04.jpg` missing)"
source: [x/vision/detect/corners/processor.go]
tests: [x/vision/detect/corners/processor_test.go]
test_result: not-run
complete: yes
correct: unknown
evidence: "go/types check with FakeImportC against gocv v0.42.0 (2026-11-09): 0 errors; OpenCV missing, test fixture left04.jpg absent"
---
# Chessboard corner detection (calibration)

**What it does:** finds an inner-corner grid with `FindChessboardCornersSB` (adaptive threshold + normalise), refines with `CornerSubPix` (11×11, 30 iter / 1e-3), outputs an N×1 CV_32FC2 corner tensor; helpers for drawing and planar object points. Used by `cmd/calibrate` (mono and stereo calibration).

**Where:** `New` `processor.go:31-45`; `Process` :47-127; `Detect` (returns `[]Point2f`, checks count) :193-262; `GenerateObjectPoints` :270-285 (unit squares, z=0); `Draw` :287; aliases `Detector`/`NewDetector` for `cmd/calibrate`.

## Completeness review

- Complete: yes for chessboard calibration targets. `CornerSubPix` after the SB detector is redundant (SB already sub-pixel) but harmless; object points are in square units — caller must scale by square size [INFERENCE: check cmd/calibrate].
- Tested: `TestProcessor_DetectCorners` asserts one N×1 output; fixture `x/marshaller/testdata/stereo_calib/left04.jpg` is **missing** (only `left01.jpg` exists) → would skip. Not run.
- Correct: unknown. Type check passes.

## TODOs

- [ ] Point the test at an existing fixture (`left01.jpg`) — x/vision/detect/corners/processor_test.go:19,113 #p2
- [ ] Take square size in `GenerateObjectPoints` — x/vision/detect/corners/processor.go:270-285 #p3

## Links

- Module: [[vision]]; consumers [[camera-model-and-stereo-rig]], [[vision-geometric-transforms]]

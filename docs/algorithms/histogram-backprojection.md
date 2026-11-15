---
type: algorithm
status: partial
tags: [easyrobot, vision, gocv, segmentation, color]
workstream: [W6]
module: "[[vision]]"
summary: "4 errors"
source: [x/vision/transform/histogram/processor.go]
tests: []
test_result: build-fail
complete: no
correct: unknown
evidence: "go/types check with FakeImportC against gocv v0.42.0 (2026-11-09): 4 type errors in CalculateHistogram (NewScalar arity, InRange Mat vs Scalar)"
---
# Histogram back-projection (HSV colour model → mask)

**What it does:** BGR→HSV, `CalcBackProject` of one or more caller-supplied H-S histograms, disc filter, 50×50 blur, threshold 50, optional invert and OR-combination → `ImageWithMask`; `CalculateHistogram` builds a normalised histogram from an image + mask + HSV range (colour-object tracking, EasyVision port).

**Where:** options `processor.go:50-90`; `New` :95-110; `Process` :116-226; `CalculateHistogram` :264-292 (compile errors at :274).

## Completeness review

- Complete: no — **does not compile**: `gocv.NewScalar` needs 4 float64 (given 3 float32) and `gocv.InRange` takes Mats, not Scalars (`processor.go:274`; gocv has `InRangeWithScalar`). Colour space hard-coded to HSV (TODO :146). With `combineMasks`, masks 2..n are never released (:196-212). `CalculateHistogram` leaks the un-normalised `hist` Mat (:283-288).
- Also imports `x/vision/types`; `x/vision/types` itself fails type-check (14 errors, `stereo_camera.go:35/181` field+method `Q`, `:84` `*Camera` passed as `Camera`, `:108/:190/:193` struct compared with nil, `:135` `gocv.ReprojectImageTo3D` undefined), so this package cannot compile even with OpenCV installed.
- Tested: none. Correct: unknown.

## TODOs

- [ ] Use `InRangeWithScalar` with 4-component float64 scalars — x/vision/transform/histogram/processor.go:274 #p1 #ER-030
- [ ] Release merged masks and intermediate hist — x/vision/transform/histogram/processor.go:196-212,283-288 #p3

## Links

- Module: [[vision]]; [[vision-blob-detection]]

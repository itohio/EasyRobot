---
type: algorithm
status: stub
tags: [easyrobot, vision, gocv, matching]
workstream: [W5]
module: "[[vision]]"
summary: "ok"
source: [x/vision/match/features/processor.go, x/vision/match/corners/processor.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "go/types check with FakeImportC against gocv v0.42.0 (tmp/easyrobot/io/typecheck, 2026-11-09): 0 errors; code read — matches are computed and discarded"
---
# Feature matching (FLANN kNN) and corner matching

**What it does (intended):** match descriptors of two frames (kNN k=2 for Lowe's ratio test) and emit matched pairs for [[essential-matrix-pose-estimation]].

**Where:** `match/features/processor.go`: `New` :21-25, `Process` :34-74; `match/corners/processor.go`: `Process` :29-40 (TODO :36).

## Completeness review

- Complete: **no / stub.**
  - `match/features` runs `KnnMatch` and then **discards the matches**, writing the first *descriptor* tensor to the output ("placeholder", `processor.go:66-72`). No ratio test, no distance filter, no cross-check.
  - Output type is `Tensor`, but pose estimation consumes `visiontypes.MatchedFeatures` — the stages cannot be chained.
  - Default `FlannBasedMatcher` (KD-tree) does not accept binary descriptors (ORB/BRISK/AKAZE, CV_8U); OpenCV throws unless descriptors are converted to float or an LSH index is used [INFERENCE from OpenCV behaviour; not run]. No brute-force Hamming matcher.
  - `match/corners` returns 0 unconditionally (`processor.go:36-38`).
- Tested: no tests.
- Correct: no (by construction).

## TODOs

- [ ] Emit `MatchedFeatures` with ratio-test-filtered `DMatch` list — x/vision/match/features/processor.go:66-72 #p1 #ER-030
- [ ] BFMatcher(Hamming) for binary descriptors; FLANN only for float — x/vision/match/features/processor.go:21-25 #p2
- [ ] Implement or delete corner matcher — x/vision/match/corners/processor.go:36-38 #p3

## Links

- Module: [[vision]]; [[vision-descriptor-extraction]]; [[essential-matrix-pose-estimation]]

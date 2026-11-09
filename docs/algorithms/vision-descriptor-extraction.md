---
type: algorithm
status: partial
tags: [easyrobot, vision, gocv, features, descriptors]
workstream: [W5]
module: "[[vision]]"
summary: "partial (ignores input keypoints)"
source: [x/vision/extract/features/processor.go]
tests: [x/vision/extract/features/processor_test.go]
test_result: build-fail
complete: no
correct: unknown
evidence: "go/types check with FakeImportC against gocv v0.42.0 (tmp/easyrobot/io/typecheck, 2026-11-09): imports broken x/vision/types; OpenCV missing"
---
# Feature descriptor extraction (ORB, SIFT, KAZE, AKAZE, BRISK)

**What it does:** given `KeypointsWithImage`, produces `DescriptorsWithImage{Image, Keypoints, Descriptors tensor}` via `DetectAndCompute`.

**Where:** `New` `processor.go:47-100` (closures per type); `Process` :103-162; TODOs :61, :133.

## Completeness review

- Complete: **no** — the input keypoints are ignored: every extractor calls `DetectAndCompute` and re-detects (`processor.go:61` TODO "Filter to match provided keypoints", :133 "Compute-only path"), so a detect→extract chain detects twice and the descriptor set does not correspond to the upstream keypoints. gocv v0.42 exposes `Compute` on these types [INFERENCE: not checked per type]. Descriptor Mat leaks if `FromMat` fails (:144-147).
- Build: `x/vision/types` itself fails type-check (14 errors, `stereo_camera.go:35/181` field+method `Q`, `:84` `*Camera` passed as `Camera`, `:108/:190/:193` struct compared with nil, `:135` `gocv.ReprojectImageTo3D` undefined), so this package cannot compile even with OpenCV installed.
- Tested: 5 extractor tests + 2 more on `img1/000001.png` (present) — not run.
- Correct: unknown.

## TODOs

- [ ] Use `Compute(src, keypoints)` so extraction honours upstream keypoints — x/vision/extract/features/processor.go:61,133 #p1 #ER-030
- [ ] Close descriptors on error — x/vision/extract/features/processor.go:144-147 #p3

## Links

- Module: [[vision]]; [[vision-keypoint-detection]]; [[vision-feature-matching]]

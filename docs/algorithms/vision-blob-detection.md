---
type: algorithm
status: partial
tags: [easyrobot, vision, gocv, blobs]
workstream: [W6]
module: "[[vision]]"
summary: "extract/blobs 2 errors; both import broken types"
source: [x/vision/detect/blobs/processor.go, x/vision/extract/blobs/processor.go]
tests: [x/vision/detect/blobs/processor_test.go]
test_result: build-fail
complete: no
correct: unknown
evidence: "go/types check with FakeImportC against gocv v0.42.0 (tmp/easyrobot/io/typecheck, 2026-11-09): extract/blobs 2 type errors; both packages import broken x/vision/types; OpenCV missing so no run"
---
# Blob detection (SimpleBlobDetector on inverted, blurred, thresholded image/mask)

**What it does:** `detect/blobs`: grey → invert → box blur (default 30×30) → binary threshold at 50 → `SimpleBlobDetector` (area 100…160000 px) → `KeypointsWithImage`. `extract/blobs`: the same pipeline applied to an `ImageWithMask.Mask` (e.g. from [[background-subtraction-mog2]] or [[histogram-backprojection]]). Port of EasyVision's blob step; useful for coloured-marker / obstacle-blob tracking.

**Where:** `detect/blobs/processor.go`: `New` :61-84, `NewWithArea` :86-106, `Process` :108-168 (pipeline :150-156), `Draw` :170; `extract/blobs/processor.go`: `New` :52-73, `NewWithArea` :75-93, `Process` :95-147.

**Inputs → outputs:** `chain.Processor[KeypointsWithImage, Tensor]` (detect) / `[KeypointsWithImage, ImageWithMask]` (extract); GoCV-backed tensors only.

## Completeness review

- Complete: no.
  - **Compile error**: `extract/blobs/processor.go:78-79` passes `int` to `SetMinArea/SetMaxArea(float64)`.
  - Options `WithMinArea`, `WithMaxArea` (`detect/blobs/processor.go:44-58`) and `WithBlobParams` (`extract/blobs/processor.go:43-47`) are no-ops.
  - Threshold (50) and polarity (always inverted, i.e. dark blobs) are hard-coded.
  - Both import `x/vision/types`; `x/vision/types` itself fails type-check (14 errors, `stereo_camera.go:35/181` field+method `Q`, `:84` `*Camera` passed as `Camera`, `:108/:190/:193` struct compared with nil, `:135` `gocv.ReprojectImageTo3D` undefined), so this package cannot compile even with OpenCV installed.
- Tested: `processor_test.go` (3 tests) needs `x/marshaller/testdata/detect_blob.png`, **which is not in the repo** → would skip even with OpenCV. Not run.
- Correct: unknown.

## TODOs

- [ ] Fix int→float64 area arguments — x/vision/extract/blobs/processor.go:78-79 #p1 #ER-030
- [ ] Add `detect_blob.png` test fixture — x/vision/detect/blobs/processor_test.go:20 #p2

## Links

- Module: [[vision]]; [[W6-decision-trees]] (camera obstacle cues)

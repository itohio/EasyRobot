---
type: algorithm
status: partial
tags: [easyrobot, vision, gocv, odometry, fft]
workstream: [W5]
module: "[[vision]]"
summary: "Incorrect (no conjugate, no normalisation)"
source: [x/vision/odometry/phase/processor.go]
tests: [x/vision/odometry/phase/processor_test.go]
test_result: build-fail
complete: no
correct: no
evidence: "go/types check with FakeImportC against gocv v0.42.0 (tmp/easyrobot/io/typecheck, 2026-11-09): imports broken x/vision/types; pure-Go mirror tmp/easyrobot/io/phasemirror (go run .) → repo math returns (4,16)/(1,-11)/(5,14) for true shifts (3,0)/(0,5)/(4,-2); textbook returns exact shifts"
---
# Phase-correlation frame-to-frame displacement

**What it does (intended):** estimates integer pixel translation `(dx, dy)` between consecutive frames: grey → float64 → zero-padded DFT; cross-power spectrum with the previous frame; inverse DFT; argmax with wrap-around → `Displacement{DX, DY, PeakValue}`. Warm-up skip (default 5 frames).

**Where:** `New` `processor.go:60-93` (DFT size `GetOptimalDFTSize(size-1)` :62-63); `Process` :102-166; `processImage` :170-192; `phaseCorrelation` :196-236; `Reset`/`Close` :239-260.

## Completeness review

- Complete: no.
  - **Wrong spectrum product**: `gocv.MulSpectrums(cur, last, dst, 0)` (`processor.go:198`) multiplies without conjugating `last` (gocv v0.42 calls `cv::mulSpectrums(a,b,c,flags)` with default `conjB=false`, `core.cpp:1145`). That is convolution, not correlation; and there is **no magnitude normalisation**, so it is not phase correlation either.
  - Wrap-around uses the image size, but the correlation surface is periodic in the padded DFT size; the result is cropped to image size (:213-217) and negative shifts beyond `dft−img` are lost.
  - `GetOptimalDFTSize(N−1)` can return N−1 < N for some widths (e.g. 641 → 640), making the ROI copy in `processImage` (:185-188) exceed the buffer.
  - No windowing (Hanning), no sub-pixel peak, no rotation/scale (log-polar). gocv already provides `PhaseCorrelate(src1, src2, window)` with sub-pixel output (`imgproc.go:2016`).
  - Imports `x/vision/types`; `x/vision/types` itself fails type-check (14 errors, `stereo_camera.go:35/181` field+method `Q`, `:84` `*Camera` passed as `Camera`, `:108/:190/:193` struct compared with nil, `:135` `gocv.ReprojectImageTo3D` undefined), so this package cannot compile even with OpenCV installed.
- Tested: 3 tests (shifted image within ±1 px, no-shift, warm-up). They need `testdata/stereo_calib/left11.jpg`, **which is missing** → would skip. Not run.
- Correct: **no.** Mirror probe `~/src/tmp/easyrobot/io/phasemirror/main.go` reimplements the same arithmetic in pure Go (32×32 random image, circular shifts): repo-style product gives (4,16), (1,−11), (5,14) for true shifts (3,0), (0,5), (4,−2); the textbook normalised conj product gives the exact shifts. The mirror omits OpenCV padding [INFERENCE: padding does not fix the missing conjugate].

## TODOs

- [ ] Wrap and crop in DFT size, not image size — x/vision/odometry/phase/processor.go:213-233 #p1 #ER-030
- [ ] `GetOptimalDFTSize(N)` not `N-1` — x/vision/odometry/phase/processor.go:62-63 #p2
- [ ] Fix test fixture (left11.jpg missing) so the tests run — x/vision/odometry/phase/processor_test.go:41 #p2

## Links

- Module: [[vision]]; FFT alternatives without OpenCV: [[dsp-core]]; [[W5-feature-tensor-experiments]]

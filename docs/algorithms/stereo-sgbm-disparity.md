---
type: algorithm
status: partial
tags: [easyrobot, vision, gocv, stereo]
workstream: [W5]
module: "[[vision]]"
summary: "partial (no gocv binding)"
source: [x/vision/stereo/matching/processor.go, x/vision/stereo/depth/combined_processor.go]
tests: []
test_result: build-fail
complete: no
correct: unknown
evidence: "go/types check with FakeImportC against gocv v0.42.0 (tmp/easyrobot/io/typecheck, 2026-11-09): stereo/matching 3 errors (gocv.StereoSGBM, NewStereoSGBM, StereoSGBMModeSGBM3Way undefined); gocv ROADMAP lists stereo as not bound"
---
# Stereo disparity (semi-global block matching)

**What it does (intended):** compute a disparity map from a rectified `StereoPair` with OpenCV SGBM (3-way), output as a tensor.

**Where:** `New` `stereo/matching/processor.go:26-50`; `Process` :54-100; same SGBM set-up duplicated in `stereo/depth/combined_processor.go:31-58`.

## Completeness review

- Complete: no — **does not compile**: gocv v0.42.0 has no `StereoSGBM`/`StereoBM` binding (type check: `undefined: gocv.StereoSGBM` at `processor.go:19,28,39`; `go doc gocv.io/x/gocv.StereoSGBM` fails). Parameters are fixed (numDisparities 16 → very short range; block 3; P1/P2 assume 3 channels; uniqueness/speckle 0 → noisy). Output is SGBM's CV_16S fixed-point (×16) passed on unscaled.
- Tested: none. Correct: unknown.

## TODOs

- [ ] Provide SGBM via a custom cgo binding, a newer gocv, or a Go block matcher — x/vision/stereo/matching/processor.go:19-50 #p1 #ER-030
- [ ] Expose SGBM parameters; convert disparity to float32 /16 — x/vision/stereo/matching/processor.go:26-50 #p2

## Links

- Module: [[vision]]; [[stereo-disparity-to-depth]]; rectification [[vision-geometric-transforms]]; capture [[multi-source-frame-sync]]

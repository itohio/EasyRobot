---
type: algorithm
status: partial
tags: [easyrobot, vision, gocv, segmentation]
workstream: [W6]
module: "[[vision]]"
summary: "4 errors"
source: [x/vision/transform/background/processor.go]
tests: []
test_result: build-fail
complete: no
correct: no
evidence: "go/types check with FakeImportC against gocv v0.42.0 (tmp/easyrobot/io/typecheck, 2026-11-09): 4 type errors (interface method Apply returns Mat, gocv returns error; impossible type switch; unused learningRate)"
---
# Background subtraction (MOG2) → foreground mask

**What it does (intended):** MOG2 foreground mask with a learning phase of `maxBackgroundNum` frames, disc-kernel filter, threshold 100 → `ImageWithMask` (feeds [[vision-blob-detection]]).

**Where:** `New` `processor.go:67-96`; `Process` :101-179; `Reset`/`Close` :191-216; `subtractor` interface :35-37.

## Completeness review

- Complete: no — **does not compile**: the private `subtractor` interface requires `Apply(Mat,*Mat) Mat` but gocv returns `error` (`processor.go:87`, `:145`, `:211`). `learningRate` is computed and never used (:135) — gocv has `ApplyWithLearningRate`. MOG and GMG options leave `subtractor = nil`, so `Process` returns 0 forever (:84, :90). `scratchMask` is a zero `gocv.Mat` (never `NewMat`) passed to `Apply` [INFERENCE: likely crash at runtime]. Unnormalised `Filter2D` with a 5×5 disc saturates (acts like dilation).
- Also imports `x/vision/types`; `x/vision/types` itself fails type-check (14 errors, `stereo_camera.go:35/181` field+method `Q`, `:84` `*Camera` passed as `Camera`, `:108/:190/:193` struct compared with nil, `:135` `gocv.ReprojectImageTo3D` undefined), so this package cannot compile even with OpenCV installed.
- Tested: none. Correct: no.

## TODOs

- [ ] Fix interface (`Apply(...) error`), use `ApplyWithLearningRate` — x/vision/transform/background/processor.go:35-37,135-150 #p1 #ER-030
- [ ] Initialise `scratchMask` with `gocv.NewMat()` — x/vision/transform/background/processor.go:44,67-96 #p1 #ER-030
- [ ] Remove or implement MOG/GMG options — x/vision/transform/background/processor.go:84,90 #p3

## Links

- Module: [[vision]]; [[vision-blob-detection]]

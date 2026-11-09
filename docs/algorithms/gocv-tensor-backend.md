---
type: algorithm
status: unverified
tags: [easyrobot, tensor, gocv, opencv, vision]
workstream: W9
module: "[[tensor]]"
summary: "93 stub methods panic; no tests; OpenCV missing"
source: [x/math/tensor/gocv/tensor.go, x/math/tensor/gocv/manipulation.go, x/math/tensor/gocv/convert.go, x/math/tensor/gocv/accessor.go, x/math/tensor/gocv/options.go, x/math/tensor/gocv/errors.go, x/math/tensor/gocv/stubs_activations.go, x/math/tensor/gocv/stubs_convolutions.go, x/math/tensor/gocv/stubs_dropout.go, x/math/tensor/gocv/stubs_elementwise.go, x/math/tensor/gocv/stubs_math.go, x/math/tensor/gocv/stubs_normalizations.go, x/math/tensor/gocv/stubs_pooling.go]
tests: []
test_result: build-fail
complete: no
correct: unknown
evidence: "go test -count=1 ./x/... (2026-11-09): x/math/tensor/gocv build-fail (pkg-config: opencv4 not found); go vet ./x/math/tensor/gocv same (2026-11-09); no _test.go files; read-only review"
---
# gocv.Mat tensor backend

**What it does:** Wraps an OpenCV `gocv.Mat` as a `types.Tensor` for image pipelines (shape `[H, W, C]`), giving zero-copy `Data()` access to the Mat buffer and conversions Mat ↔ `image.Image`. All math/NN ops are unsupported.

**Where:** `x/math/tensor/gocv/tensor.go:33` `NewImage(rows, cols, channels)`, `:68` `FromMat`, `:120-180` dtype mapping (`CV8UC1..4` ↔ `UINT8`, `CV32FC1..4` ↔ `FP32`, anything else `ErrUnsupportedDepth`), `:279-308` `Data()` (slice over Mat bytes via `dataUint8`/`dataFloat32` `:491-511`, `unsafe`), `:339-437` `At/SetAt` (float64), `:440` `Elements`, `:473` `Release` (atomic refcount handle); `manipulation.go:12-245` Clone, Copy, Reshape, Transpose, Permute (identity only), Fill, FillFunc (11 `panicUnsupported`: Slice, BroadcastTo, Pad, Unpad, non-trivial cases); `convert.go:13-63` `ToMat`, `FromImage`, `ToImage`; stubs: **93 methods** across `stubs_*.go` all `panicUnsupported` (`errors.go:19-21`).

**Inputs → outputs:** `gocv.Mat` (8-bit or 32-bit float, 1–4 channels) ↔ `types.Tensor` UINT8/FP32 `[H,W,C]`; `image.Image` via RGBA conversion (`convert.go:49`).

## Completeness review

- Complete: **no** — accessor-only backend; 93 + 11 panicking methods; no int16/float64 Mats.
- Tested: no tests in package; cannot build here (OpenCV missing). Consumers: `cmd/calibrate`, `cmd/cr30`, `cmd/display`, `cmd/spectrometer`, `drivers/lidar/display.go`, `x/marshaller/gocv/*` (read only).
- Correct: unknown (no build, no test).

**Observations (read-only):** `Data()` panics on accessor errors instead of returning nil (`tensor.go:287-299`); `Data()` returns `data[:length]` without checking Mat continuity (`gocv.Mat.DataPtr*` requires continuous Mat [INFERENCE]) — ROI/submat views would be misread; `At/SetAt` go through float64.

**Embedded suitability:** not applicable to ESP32 (CGo + OpenCV). K230 Linux possible only with an OpenCV build.

## TODOs

- [ ] Add build-tagged tests (`//go:build gocv`) and a CI job with OpenCV — x/math/tensor/gocv #p2
- [ ] Check `mat.IsContinuous()` before zero-copy `Data()` — x/math/tensor/gocv/tensor.go:279 #p2
- [ ] Return errors instead of panics in `Data()` — x/math/tensor/gocv/tensor.go:287 #p3

## Links

- Module: [[tensor]]
- Related: [[gorgonia-graph-backend]], [[vision]], [[camera-io]]

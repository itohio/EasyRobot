---
type: algorithm
status: unverified
tags: [easyrobot, camera, gocv, recording, protobuf]
workstream: [W4, W5]
module: "[[camera-io]]"
summary: "not run"
source: [x/marshaller/gocv/marshaller.go, x/marshaller/gocv/manifest.go, x/marshaller/gocv/file_writer.go, x/marshaller/gocv/display_writer.go, x/marshaller/gocv/sink.go, x/marshaller/gocv/dnn.go, x/marshaller/gocv/FILE_FORMAT.md]
tests: [x/marshaller/gocv/manifest_test.go, x/marshaller/gocv/streams_test.go, x/marshaller/gocv/stream_display_manual_test.go]
test_result: not-run
complete: unknown
correct: unknown
evidence: "OpenCV not installed; typecheck probe 0 errors for x/marshaller/gocv (2026-11-09)"
---
# GoCV sinks: frame-stream recording, display, protobuf manifest, DNN loading

**What it does:** `Marshal(w, FrameStream)` consumes frames and writes each tensor as an image file (`fileWriter`) and/or shows it in HighGUI windows (`displayWriter` with its own GUI event loop and key/mouse callbacks); a protobuf `FrameStreamManifest` describes sources and sync mode so a recording can be replayed by `Unmarshal`. Single `Mat`/`image.Image`/`Tensor` values are encoded with `cv.IMEncode`; DNN models are loaded from bytes (ONNX/Caffe/TF/Darknet via temp files).

**Where:** dispatch `marshaller.go:44-83`; `writeFrameStream` :120-335; `tensorToMat` :336-346; manifest ↔ config `manifest.go:107-140`, `writeManifest`/`readManifest` :156-233; `fileWriter` `file_writer.go:13-103`; `displayWriter` `display_writer.go:94-749` (global GUI loop :229); `loadNetFromBytes` `dnn.go:13-55`; format spec `FILE_FORMAT.md`.

## Completeness review

- Complete: unknown — code is substantial (display writer 749 LOC), but `writeTensor` requires a GoCV-backed tensor (`tensorToMat` fails for eager tensors, `marshaller.go:336-345`), so NN outputs must be converted first. Recording stores per-frame image files plus manifest — no timestamps-per-frame index for odometry datasets beyond metadata [INFERENCE from FILE_FORMAT.md].
- Tested: `manifest_test.go` (manifest round trip, format detection, multiple sources, sync mode) and `streams_test.go` — not run (OpenCV missing).
- Correct: unknown.
- W5 relevance: the recording/replay path for building datasets (feature tensors, VO sequences) on the desktop.

## TODOs

- [ ] Accept eager tensors in `writeTensor` (copy into Mat) — x/marshaller/gocv/marshaller.go:336-345 #p2
- [ ] Run manifest_test/streams_test on an OpenCV host — x/marshaller/gocv/manifest_test.go #p2

## Links

- Module: [[camera-io]]; [[gocv-frame-sources]]; [[W5-feature-tensor-experiments]]

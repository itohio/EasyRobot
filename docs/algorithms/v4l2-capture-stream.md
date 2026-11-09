---
type: algorithm
status: unverified
tags: [easyrobot, camera, v4l2, linux, capture]
workstream: [W4]
module: "[[camera-io]]"
summary: "unverified (MJPEG/NV12 shapes wrong, blocking receive)"
source: [x/marshaller/v4l/device.go, x/marshaller/v4l/stream.go, x/marshaller/v4l/tensor.go, x/marshaller/v4l/unmarshaller.go, x/marshaller/v4l/marshaller.go, x/marshaller/v4l/options.go, x/marshaller/v4l/types.go]
tests: [x/marshaller/v4l/example_test.go]
test_result: build-fail
complete: no
correct: unknown
evidence: "GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build ./x/marshaller/v4l → go4vl/v4l2 undefined types (needs cgo + linux headers) (2026-11-09); //go:build linux"
---
# V4L2 capture stream (`x/marshaller/v4l`, go4vl)

**What it does:** Linux-only camera access without OpenCV: opens `/dev/videoN` with go4vl, sets pixel format/resolution/fps, queries controls (mapped to shared names), streams mmap buffers and wraps each frame in a pooled `uint8` tensor `[H,W,C]`. Also marshals device info / stream configs.

**Where:** `NewDevice`/`Open` `device.go:23-113`, `queryDeviceInfo` :125; `newV4LStream` `stream.go:33-124`; `Start`/`Stop` :126-164; `captureLoop` :189-235; `createTensorFromFrame` :238-262; controller :532-573; `PooledTensor` `tensor.go:21-198`, `defaultTensorFactory` :199-203; `Unmarshaller.Unmarshal` `unmarshaller.go:59-83`, frame-stream bridge :206-304.

## Completeness review

- Complete: no.
  - `captureLoop` blocks on `<-s.stream.GetOutput()` outside the `select` (`stream.go:203`): `Stop()` cannot interrupt a stalled camera, and once go4vl closes the channel the loop spins producing empty tensors (nil frame, no `ok` check).
  - Channel count is guessed from FOURCC (`stream.go:243-258`): **MJPEG frames are wrapped undecoded as 3-channel `[H,W,3]`** although the buffer is compressed JPEG (:252-255); NV12 is declared 1 channel but the buffer is 1.5·W·H bytes (:248-249); YUYV `[H,W,2]`. Shape and byte length disagree for MJPEG/NV12.
  - `defaultTensorFactory` wraps the driver frame slice and `Release()` returns it to the uint8 pool (`tensor.go:35-44, 167-174`), mixing foreign buffers into the pool [INFERENCE: depends on go4vl buffer ownership].
  - Dropped frames release the tensor silently (`stream.go:230-232`); no drop counter.
- Tested: `example_test.go` needs `/dev/video*`; not built on darwin and excluded by roster rules.
- Correct: unknown (no build). Cross-compile with `CGO_ENABLED=0` fails inside go4vl (cgo-only types), so even a type check needs a linux cgo toolchain.
- W4 relevance: the only OpenCV-free capture path; closest model for a K230/Linux-side capture, but MJPEG decode and YUV→RGB conversion are missing.

## TODOs

- [ ] Receive frames inside the `select` with `ok` check — x/marshaller/v4l/stream.go:203 #p1 #backlog
- [ ] Decode MJPEG (image/jpeg) and convert YUYV/NV12 to RGB or expose planar shapes correctly — x/marshaller/v4l/stream.go:243-258 #p1 #backlog
- [ ] Copy driver buffers into pool-owned memory before handing out tensors — x/marshaller/v4l/tensor.go:35-44 #p2
- [ ] Count dropped frames in metadata — x/marshaller/v4l/stream.go:230-232 #p3

## Links

- Module: [[camera-io]]; [[multi-source-frame-sync]]; [[gocv-frame-sources]]; [[W4-k230-feature-stream]]

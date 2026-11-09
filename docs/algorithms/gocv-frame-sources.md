---
type: algorithm
status: unverified
tags: [easyrobot, camera, gocv, opencv, capture]
workstream: [W4, W5]
module: "[[camera-io]]"
summary: "not run (OpenCV missing)"
source: [x/marshaller/gocv/loader_device.go, x/marshaller/gocv/loader_video.go, x/marshaller/gocv/loader_image.go, x/marshaller/gocv/loader_filelist.go, x/marshaller/gocv/loader_mat.go, x/marshaller/gocv/streams.go, x/marshaller/gocv/codec.go, x/marshaller/gocv/device_enumeration.go]
tests: [x/marshaller/gocv/streams_test.go, x/marshaller/gocv/device_enumeration_test.go]
test_result: not-run
complete: yes
correct: unknown
evidence: "OpenCV not installed: package cannot build (baseline-2026-11-08 §4); go/types check with FakeImportC (tmp/easyrobot/io/typecheck) → 0 type errors (2026-11-09)"
---
# GoCV frame sources: camera (`VideoCapture`), video file, image list, custom `.mat`

**What it does:** turns a source spec into a `sourceStream` that yields GoCV-backed tensors: camera index via `cv.OpenVideoCapture` with FOURCC → width/height/fps set-up, video files, glob/file lists and single images (decoded by OpenCV), and a custom binary `.mat` file. `Unmarshal(r, *FrameStream, WithVideoDevice…/WithPath…)` wires them into a channel ([[multi-source-frame-sync]]).

**Where:** `newVideoDeviceLoader` `loader_device.go:177-219` (FOURCC first, read-back check :184-201; resolution/fps :204-212), `Next` :221-274 (per-frame `cv.NewMat`, best-effort retry sleeps 10/5 ms :233-249, metadata `device/timestamp/frame_index/filename` :257-266), camera controls `cameraController` :23-129, `pixelFormatToFOURCC` :138-175; `newVideoFileLoader` `loader_video.go:23-86`; `fileListLoader` `loader_filelist.go`; `imageLoader`/`loadImageFile`/`loadMatFile` `loader_image.go:21-122`; `readCustomMat` `loader_mat.go:15-68`; `buildStreams` `streams.go:5-38`; `matToTensor` `codec.go:92-102`; `enumerateVideoDevices` `device_enumeration.go:16-66` (probes `/dev/video*`, linux-style paths).

**Inputs → outputs:** options `WithVideoDevice(id,w,h)`, `WithVideoDeviceEx(id,w,h,fps,fourcc)`, `WithPath(glob|file)`, `WithFrameRate`, `WithPixelFormat` → `types.Frame{Index, Timestamp, Metadata, Tensors: [Mat tensor HxWxC uint8 BGR]}`.

## Completeness review

- Complete: yes for desktop capture. Gaps: `matToTensor` ignores the requested dtype (`codec.go:100`, "reserved"); timestamp is host `time.Now()` after `Read` (`loader_device.go:259`), not the driver timestamp; `buildStreams` leaks already-opened captures when a later source fails (`streams.go:13-33`); `Read` failure without best-effort silently ends the stream (`loader_device.go:233-241`).
- Tested: not run — OpenCV missing. `streams_test.go` (glob streams, multi-glob, marshal frame stream to images, generated stream) uses repo images; `TestCameraController` opens a real camera (skips without one) — would be skipped by the roster rules anyway.
- Correct: unknown. Go-level type check passes (`typecheck` probe: `x/marshaller/gocv: 0 type errors`).
- W4 relevance: this is the desktop/PC capture path (USB webcams via OpenCV). It is not usable on K230/ESP32 (CGO + OpenCV); the K230 side needs its own capture and only the wire format can be shared.

## TODOs

- [ ] Close opened streams when `buildStreams` fails — x/marshaller/gocv/streams.go:13-33 #p2
- [ ] Honour `DestinationType` in `matToTensor` (uint8→fp32 normalisation for NN input) — x/marshaller/gocv/codec.go:92-102 #p2
- [ ] Use capture timestamps (`CAP_PROP_POS_MSEC`) when available — x/marshaller/gocv/loader_device.go:259 #p3
- [ ] Run streams_test on a host with OpenCV — x/marshaller/gocv/streams_test.go #p2

## Links

- Module: [[camera-io]]; [[multi-source-frame-sync]]; [[v4l2-capture-stream]]; [[W4-k230-feature-stream]]

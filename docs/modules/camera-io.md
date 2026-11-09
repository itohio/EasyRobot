---
type: module
group: "IO, camera, vision"
status: unverified
tags: [easyrobot, camera, gocv, v4l2, capture]
workstream: [W4, W5]
packages: [x/marshaller/gocv, x/marshaller/v4l, x/marshaller/types]
test_result: not-run
evidence: "x/marshaller/gocv: OpenCV not installed (baseline-2026-11-08 §4), go/types check with FakeImportC → 0 errors (2026-11-09); x/marshaller/v4l: //go:build linux, GOOS=linux CGO_ENABLED=0 build fails inside go4vl (cgo-only) (2026-11-09)"
---
# camera-io — camera capture, recording and display (`x/marshaller/gocv`, `x/marshaller/v4l`)

**Purpose:** get frames from cameras, video files and image sequences into tensors and out to files/windows, behind the [[marshallers]] contract (`Unmarshal(r, *types.FrameStream, opts...)`, `Marshal(w, FrameStream)`). Two back ends: OpenCV via GoCV (desktop, any OS with OpenCV) and V4L2 via go4vl (Linux, no OpenCV).

**Packages:**

| Package | LOC (non-test/test) | Tags / native deps | Role |
|---|---|---|---|
| `x/marshaller/gocv` | 3702 / 1080 | CGO + OpenCV 4 (pkg-config `opencv4`); `stream_display_manual_test.go` tag `gocv_display` | camera/video/image sources, multi-source sync, image/file/display sinks, protobuf manifest, DNN loading; docs `SPEC.md`, `DESIGN.md`, `SOLID_PLAN.md`, `FILE_FORMAT.md` |
| `x/marshaller/v4l` | 2194 / 227 | `//go:build linux`, CGO (go4vl v0.3.0) | device enumeration, controls, mmap streaming, pooled uint8 tensors, multi-stream; docs `README.md`, `SPEC.md` |
| `x/marshaller/types` (camera parts) | `camera.go` 237, `display.go` 395 | none | `CameraInfo`, `VideoFormat`, `ControlInfo`, `CameraController`, `CameraDevice`, `CameraStream`, `CameraOption`; display window/event types |

## Public API

- GoCV: `gocv.NewUnmarshaller(opts...)`, `gocv.NewMarshaller(opts...)`; options `WithPath(glob)`, `WithVideoDevice(id,w,h)`, `WithVideoDeviceEx(id,w,h,fps,fourcc)`, `WithFrameRate`, `WithPixelFormat`, display options (window title, key/mouse callbacks, event loop), DNN options; values `cv.Mat`, `image.Image`, `types.Tensor`, `types.FrameStream`, `[]types.CameraInfo`, `cv.Net`.
- V4L2: `v4l.New(opts...)` (unmarshaller), `v4l.NewMarshaller`, `NewDevice(path)`, `GetAllDevices()`, `NewMultiStream`; options `WithDevicePath(s)`, `WithResolution`, `WithFrameRate(Fraction)`, `WithPixelFormat`, `WithBufferCount`, `WithControls`, `WithBufferPool`, `WithTensorFactory`, `WithBestEffort`, `WithSequential`.
- Shared camera interfaces (`types/camera.go:87-137`): `CameraDevice.Open(opts) → CameraStream{Start, Stop, Controller, Close}`, `CameraController{Controls, Get/SetControl(s)}`.

Flow: options → source specs → `sourceStream`s ([[gocv-frame-sources]], [[v4l2-capture-stream]]) → `newFrameStream` ([[multi-source-frame-sync]]) → `types.FrameStream{C <-chan Frame}` → consumer; sinks via `Marshal` ([[gocv-frame-stream-sink-and-manifest]]).

## Algorithms

<!-- vault:algorithms h=27924c558f65 -->

Counts: 0 tested · 0 implemented · 0 partial · 0 stub · 4 unverified (4 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[gocv-frame-sources]] | unverified | not-run | not run (OpenCV missing) |
| [[gocv-frame-stream-sink-and-manifest]] | unverified | not-run | not run |
| [[multi-source-frame-sync]] | unverified | not-run | unverified (goroutine leak, no time alignment) |
| [[v4l2-capture-stream]] | unverified | build-fail | unverified (MJPEG/NV12 shapes wrong, blocking receive) |

<!-- /vault:algorithms -->

## Build targets and dependencies

- GoCV path: CGO + OpenCV (desktop/RPi-class Linux). Not TinyGo, not ESP32, not the K230 bare-metal SDK. The baseline lists 33 packages blocked by missing OpenCV (incl. `x/marshaller/gocv`, all of `x/vision`).
- V4L2 path: Linux + CGO (go4vl). Plausible on a Linux SBC or a Linux-running K230 (the K230 SDK ships Linux on the big core [INFERENCE, not checked]); MJPEG decode and YUV→RGB must be added first.
- Neither path runs on ESP32. For [[W4-k230-feature-stream]] the ESP32 side only needs the *receiving* codec (tensor wire format, see [[marshaller-tensor-proto-conversion]]); capture stays on the K230.
- Goroutines: one producer per stream (gocv), capture loop + sync loop (v4l), a global GUI event loop for display.

## Tests and evidence

- `x/marshaller/gocv`: not built — OpenCV missing. Go-level type check (custom `go/types` checker with `FakeImportC`, `~/src/tmp/easyrobot/io/typecheck/main.go`, run `./tc . github.com/itohio/EasyRobot/x/marshaller/gocv`) → `0 type errors` (2026-11-09). Tests: `manifest_test.go` (4), `streams_test.go` (4, repo images), `device_enumeration_test.go` (camera — skipped by roster rules), `stream_display_manual_test.go` (GUI, tag).
- `x/marshaller/v4l`: `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go build ./x/marshaller/v4l` → `go4vl/v4l2/types.go: undefined: Capability, IOType, BufType` (cgo-only types). `example_test.go` needs `/dev/video*`.

## API notes (ease of use, composability)

- **Cameras as `Unmarshal` into `*FrameStream`** reuses the marshaller contract, but the `io.Reader` is ignored and all configuration rides on options — `Unmarshal(nil, &stream, gocv.WithVideoDevice(0, 640, 480))` reads oddly. A dedicated `Open(ctx, spec) (FrameStream, error)` would be clearer; the shared `CameraDevice`/`CameraStream` interfaces exist (`types/camera.go:105-130`) but `CameraStream` has **no frame accessor** (Start/Stop/Controller/Close only), so they cannot be used on their own.
- **Two back ends, two option sets:** gocv `WithVideoDevice(id,w,h)` vs v4l `WithVideoDevice(id,w,h)` (different packages, different `Option` types: v4l's `Option` is its own interface with `Apply(*v4l.Options)`, `options.go:14-17`, plus `ApplyCamera` adapters). Switching back end means rewriting option code. Even inside v4l, `WithVideoDevice` returns `v4l.Option` while `WithVideoDeviceEx` returns `types.Option` (`v4l/options.go:231, :240`).
- **Frame shape is implicit:** tensors are HxWxC uint8 BGR (gocv) or whatever the FOURCC implies (v4l; MJPEG mislabelled). No pixel-format field in `Frame` other than metadata maps (`format`, `device` keys); consumers must know the source.
- **Errors as frames** (`Metadata["error"]`, empty tensors) are easy to miss; a `Frame.Err` field would be safer.
- **Ownership:** `WithRelease` + pooled tensors (v4l) is the right idea; GoCV tensors adopt Mats and must be released by the consumer — not documented at the call site.
- **No bridge** from `FrameStream` to `x/chain` processors used by [[vision]]; every app hand-writes the loop.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Add frame access to `CameraStream` (`Frames() <-chan Frame`) or drop the interface — x/marshaller/types/camera.go:117-130 #p2
- [ ] Unify gocv/v4l device options in `types` (`types.WithCameraResolution` etc. already exist at camera.go:186-238) — x/marshaller/v4l/options.go:14-17 #p3
- [ ] Run gocv tests on an OpenCV host; v4l example on a Linux box (owner approval for camera access) — x/marshaller/gocv/streams_test.go #p2

## Inferred objectives (proposals)

- **Proposal ([[W4-k230-feature-stream]]):** keep camera capture on the K230 (vendor SDK or V4L2 if Linux), define the feature-tensor frame (`shape`, `dtype`, `scale/zero-point`, `timestamp`, `seq`) as a protobuf/fixed binary message, and reuse only that codec on the ESP32; the gocv/v4l packages serve as the desktop reference recorder and replay source.
- **Proposal ([[W5-feature-tensor-experiments]]):** use `gocv` recording + manifest to build replayable datasets (raw frames + feature tensors + timestamps) for visual-odometry and place-recognition experiments.
- **Proposal:** add a `FrameStream → chain.Processor` adapter so camera sources compose directly with [[vision]] processors.

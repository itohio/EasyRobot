---
type: module
group: "IO, camera, vision"
status: partial
tags: [easyrobot, vision, gocv, opencv, odometry, stereo]
workstream: [W5, W4, W6]
packages: [x/vision/types, x/vision/detect/blobs, x/vision/detect/corners, x/vision/detect/features, x/vision/extract/blobs, x/vision/extract/features, x/vision/match/corners, x/vision/match/features, x/vision/odometry/phase, x/vision/odometry/pose, x/vision/stereo/depth, x/vision/stereo/matching, x/vision/transform/affine, x/vision/transform/background, x/vision/transform/color, x/vision/transform/crop, x/vision/transform/format, x/vision/transform/histogram, x/vision/transform/rectify, x/vision/transform/undistort]
test_result: build-fail
evidence: "OpenCV not installed → no build/run (baseline-2026-11-08 §5). Go-level type check against gocv v0.42.0 with go/types FakeImportC (2026-11-09): 6 of 20 packages have type errors, 8 more import the broken x/vision/types; phase-correlation math mirror shows wrong results"
---
# vision — `x/vision/**` (GoCV processors)

**Purpose:** OpenCV-backed image processing stages implementing `chain.Processor` ([[processor-chain-composition]]): detect (blobs, chessboard corners, keypoints), extract (blobs from masks, descriptors), match (features, corners), odometry (phase correlation, essential-matrix pose), stereo (SGBM disparity, reprojection), transforms (affine, background subtraction, colour, crop, format, histogram back-projection, rectify, undistort), plus shared types and a camera model. Port in progress of the owner's Python EasyVision (`x/vision/WORK.md` tracks tasks; `SPEC.md` self-described as outdated; `CHAIN_PROCESSOR_DESIGN.md`). Only `cmd/calibrate` imports it (`detect/corners`).

**Packages** (non-test LOC / test LOC): types 477/0; detect/blobs 230/200, detect/corners 312/134, detect/features 209/197; extract/blobs 191/0, extract/features 188/267; match/corners 52/0, match/features 87/0; odometry/phase 262/279, odometry/pose 205/0; stereo/depth 256/0, stereo/matching 120/0; transform/affine 122, background 218, color 112, crop 123, format 267, histogram 295, rectify 126, undistort 144 (no tests). Total ≈ 4,000 LOC + 1,077 test LOC. All need CGO + OpenCV 4 (`gocv.io/x/gocv v0.42.0`) and `x/math/tensor/gocv` (Mat-backed tensors).

## Public API

Each stage: `New(...) *Processor`, `Process(dst buffer.Writer[OUT], src buffer.Reader[IN]) int`, `Reset()`, `Close()` (frees Mats), sometimes `Draw(...)`. Sample types (`x/vision/types/types.go`): `KeypointsWithImage`, `DescriptorsWithImage`, `MatchedFeatures`, `Pose`, `Displacement`, `ImageWithMask(s)`; `transform/format.StereoPair`, `ImageWithTensor`; `crop.CropRectangles`; `rectify.RectificationMaps`. Intended VO chain: `detect/features` → `extract/features` → `match/features` → `odometry/pose`; stereo chain: `rectify` ×2 → `stereo/matching` → `stereo/depth`.

## Algorithms

<!-- vault:algorithms h=32c99e1600e7 -->

Counts: 0 tested · 0 implemented · 8 partial · 2 stub · 3 unverified (13 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[background-subtraction-mog2]] | partial | build-fail | 4 errors |
| [[camera-model-and-stereo-rig]] | partial | build-fail | 14 errors |
| [[histogram-backprojection]] | partial | build-fail | 4 errors |
| [[phase-correlation-odometry]] | partial | build-fail | Incorrect (no conjugate, no normalisation) |
| [[stereo-disparity-to-depth]] | partial | build-fail | partial (no binding, ×16 scale) |
| [[stereo-sgbm-disparity]] | partial | build-fail | partial (no gocv binding) |
| [[vision-blob-detection]] | partial | build-fail | extract/blobs 2 errors; both import broken types |
| [[vision-descriptor-extraction]] | partial | build-fail | partial (ignores input keypoints) |
| [[essential-matrix-pose-estimation]] | stub | none | imports broken types |
| [[vision-feature-matching]] | stub | none | ok |
| [[chessboard-corner-detection]] | unverified | not-run | 2 (fixture `left04.jpg` missing) |
| [[vision-geometric-transforms]] | unverified | none | ok |
| [[vision-keypoint-detection]] | unverified | build-fail | imports broken types |

<!-- /vault:algorithms -->

## Build targets and dependencies

- CGO + OpenCV only; data types embed `gocv.Mat`/`gocv.KeyPoint`, so nothing here runs under TinyGo, on ESP32, or on the K230 RTOS side. Every `Process` allocates (input slice, output slice, cloned result Mat) despite "preallocated" comments.
- **Compile status with OpenCV installed (Go-level):** would build: corners, match/corners, match/features, transform/{affine,color,crop,format,rectify,undistort}, `cmd/calibrate`. Would **not** build: types, extract/blobs, stereo/depth, stereo/matching, transform/background, transform/histogram, and — through `x/vision/types` — detect/blobs, detect/features, extract/features, odometry/phase, odometry/pose. Root causes: APIs not bound in gocv v0.42.0 (`StereoSGBM`, `ReprojectImageTo3D`; ROADMAP line 128), signature drift (`BackgroundSubtractorMOG2.Apply` returns `error`, `NewScalar` arity, `InRange` takes Mats, `SetMinArea(float64)`), and Go errors in `types/stereo_camera.go`.

## Tests and evidence

- Not run: OpenCV missing (baseline-2026-11-08 §5 excludes all `x/vision/*`).
- Type check: (go/types with `FakeImportC`, package files from `go list -deps -json` run in the repo with `GOFLAGS=-mod=readonly`; command `./tc . <pkgs…>`), 2026-11-09: errors per package — types 14, extract/blobs 2, stereo/depth 5, stereo/matching 3, transform/background 4, transform/histogram 4; all others 0 own errors. Caveat: C-side errors cannot be seen this way; dependents of a broken package show 0 own errors but cannot compile.
- Fixtures: of the images the tests reference under `x/marshaller/testdata/`, only `img1/000001.png` exists; `detect_blob.png`, `stereo_calib/left04.jpg`, `stereo_calib/left11.jpg` are missing (only `left01.jpg` present), so blob, corner and phase tests would skip even on an OpenCV host.
- Mirror probe (pure Go, `go run .`): repo phase-correlation arithmetic returns wrong shifts; see [[phase-correlation-odometry]].
- gocv API lookups: `go doc gocv.io/x/gocv.<Symbol>` from the module cache (no network).

## API notes (ease of use, composability)

- **Uniform stage shape** (`chain.Processor` + `Reset`/`Close`) makes stages swappable and chainable; `Draw` helpers are handy for debugging.
- **Silent failure:** every error path returns 0 (wrong tensor type, empty Mat, OpenCV error, zero keypoints), indistinguishable from "no output yet"; inside a `Chain` this aborts the frame without a trace. An error return or `LastErr()` is needed.
- **Type mismatches across stages:** `match/features` outputs `Tensor` while `odometry/pose` wants `MatchedFeatures`; `extract/features` re-detects instead of using the incoming keypoints — the advertised VO chain cannot be assembled.
- **Ownership is manual:** stages clone Mats into adopted tensors; consumers must `Release` them; histograms/Q/K matrices are borrowed references the caller must keep alive. Not documented per stage.
- **Two-input stages** (matching) read two items from one reader, so pairing previous/current frames is left to the caller; a stateful "previous frame" stage (like phase) is more convenient.
- **No option structs:** parameters (SGBM disparities, blob thresholds, detector settings) are hard-coded; several `With…` options are no-ops.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] CI job on a host with OpenCV so vision stops rotting — x/vision #p2

## W4 / W5 relevance

- **W4 (K230 camera → feature stream → ESP32):** capture lives in [[camera-io]] (gocv `VideoCapture`, v4l2), not here. From `x/vision`, only the camera model/calibration (`detect/corners` + `cmd/calibrate`, [[camera-model-and-stereo-rig]]) matters: intrinsics are needed to turn feature-map coordinates into rays. The vision types are CGO-bound and cannot be the wire format.
- **W5 (experiments on feature tensors: VO, place recognition, depth):** relevant baselines are [[vision-keypoint-detection]] / [[vision-descriptor-extraction]] (classic ORB/SIFT features for BoW place recognition and as a teacher/comparison), [[phase-correlation-odometry]] (cheap dense VO baseline — currently wrong), [[essential-matrix-pose-estimation]] (monocular VO — stub), and the stereo pair [[stereo-sgbm-disparity]] / [[stereo-disparity-to-depth]] (depth supervision — does not compile). None of these is usable today without fixes; WORK.md's own plan (BoW vocabulary, topological map) is unstarted.

## Inferred objectives (proposals)

- **Proposal:** decide whether `x/vision` stays OpenCV-based (desktop tooling, dataset generation, calibration) or whether VO/place-recognition baselines move to pure Go on [[tensor]]/[[mat-vec]] so they can run next to the NN code on K230/ESP32. Recommendation: keep OpenCV for calibration and dataset tooling, implement the W5 baselines that must run on-device (phase correlation via [[dsp-core]] FFT, essential matrix on `x/math/mat`) in pure Go.
- **Proposal:** first unblock compilation (types + signature drift) and add an OpenCV CI host, then fix phase correlation; that yields one verified VO baseline quickly.
- **Proposal:** define CGO-free feature types (keypoints, descriptors as tensors, poses as `mat`) shared by classic and learned pipelines.

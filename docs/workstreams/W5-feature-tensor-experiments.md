---
type: workstream
status: proposed
tags: [easyrobot, workstream, vision, nn, odometry]
workstream: W5
source: [x/vision/, x/math/nn/layers/, x/math/learn/, x/math/tensor/eager/conv.go, x/math/tensor/types/convolutions.go]
evidence: "source reading + grep (2026-11-09); find . -name '*.ipynb' → none; gocv v0.42.0 module cache checked for FindEssentialMat/RecoverPose → absent; vision packages not built (OpenCV missing, baseline)"
---
# W5 — Experiments on the feature tensor

**Goal:** experiments on the feature tensor (streamed on to an ESP32 board over SPI): YOLO head, BoW place recognition, unsupervised visual odometry and depth with a feature-metric teacher → trained small networks; ideation research first, then EasyRobot. Input = the K230 backbone's last feature tensor from [[W4-k230-feature-stream]].

**Notebooks:** notebooks are expected to be uploaded; none are in the repo (`find . -name '*.ipynb' -not -path './.git/*'` → none).

## What exists

| Capability needed | In the repo | Status / evidence |
|---|---|---|
| Classical features (for BoW, VO baselines) | `x/vision/detect/features`, `x/vision/extract/features`: ORB, SIFT, AKAZE, KAZE, BRISK, FAST, GFTT via GoCV (`gocv.New{ORB,SIFT,AKAZE,KAZE,BRISK,FastFeatureDetector,GFTTDetector}`) | [[vision]]; not built (OpenCV missing); 1 test file each, not run |
| Matching | `x/vision/match/features` (FLANN, `gocv.NewFlannBasedMatcher`); `x/vision/match/corners` | FLANN not built/run; **corner matching is a stub** (`x/vision/match/corners/processor.go:36` TODO, returns 0) |
| Visual odometry | `x/vision/odometry/phase` (DFT phase correlation → image displacement, preallocated Mats); `x/vision/odometry/pose` | phase: not built/run; **pose is a stub**: fills point Mats then returns 0 (`x/vision/odometry/pose/processor.go:110–124`); gocv v0.42.0 has no `FindEssentialMat`/`FindFundamentalMat`/`RecoverPose` (only `EstimateAffine2D`, `FindHomography` in `calib3d.go`, module cache) |
| Depth | `x/vision/stereo/{matching,depth}` (StereoSGBM + `ReprojectImageTo3D`) | stereo only; no monocular depth; not built/run |
| Calibration / geometry | `x/vision/transform/{undistort,rectify,affine}`, `x/vision/detect/corners` (chessboard), `cmd/calibrate` (mono/stereo) | not built/run |
| Conv-net layers | `x/math/nn/layers`: Conv1D, Conv2D, MaxPool2D, AvgPool2D, GlobalAvgPool2D, BatchNorm2D, Dense, Dropout, ReLU/Sigmoid/Tanh/Softmax, Flatten/Reshape/Squeeze/Unsqueeze/Transpose/Pad/PadReflect/Concatenate, LSTM, GPT attention/embedding/encoding (constructors `New*` in `layers/*.go`); tensor ops include `Conv2DTransposed`, `Im2Col`/`Col2Im` (`x/math/tensor/types/convolutions.go:26–65`), eager `DepthwiseConv2D` (NCHW, `x/math/tensor/eager/conv.go:1247`), activations ReLU6/LeakyReLU/Swish/GELU (`tensor/types/activations.go`) | [[nn]], [[tensor]]; `x/math/nn/layers` 91 pass / 2 fail incl. **`TestDense_NumericalGradientCheck`** (baseline §4) — training correctness is not established; no depthwise **layer** (only the tensor op) |
| Training | `x/math/learn`: SGD, Adam, train step, post-training quantisation, MNIST loader | [[learn]]; 26 pass, 1 skip (baseline) |
| Model import | TFLite wrapper (linux, cgo), OpenCV DNN from bytes | [[tflite]], [[camera-io]]; not built here |
| Detection head pieces (anchors/DFL decode, NMS) | **none** | `grep -rlE "NMS\|NonMax"` under `x/` → none |
| BoW pieces (k-means vocabulary, TF-IDF, inverted index, binary descriptor distance) | **none** | `grep "KMeans\|TFIDF\|Vocabulary\|BagOfWords"` → none; Hamming distance only in the line-code package |
| Self-supervised VO/depth pieces (warping/grid-sample, SSIM/photometric or feature-metric loss, pose network, depth decoder with upsampling) | **none** in [[nn]]/[[tensor]] | no grid-sample/bilinear warp op (`grep "GridSample\|Bilinear"` hits only `x/math/filter/iir`); upsampling exists only in [[interpolation]] (`x/math/interpolation/upsample.go`), not as an NN layer |
| Feature-tensor dataset/recordings | **none** | depends on W4's recorder; `core.Frame` sink drops tensors (`x/marshaller/gocv/sink.go:185–210`) |

## Gaps

1. No end-to-end training stack is proven: the Dense numerical gradient check fails, so backprop through deeper conv stacks is unverified ([[nn]]).
2. No detection decode/NMS, no BoW vocabulary/retrieval, no essential-matrix/RANSAC pose, no differentiable warping or photometric/feature-metric loss.
3. All classical vision is GoCV-bound and unbuilt on this host; pose estimation is blocked on gocv bindings or a pure-Go 5-/8-point + RANSAC implementation.
4. No dataset path: feature tensors are neither recorded nor loadable as a dataset.
5. Layout mismatch risk: eager tensors are NCHW; TFLite/K230 maps are typically NHWC [INFERENCE].

## TODOs

Algorithm defects are tracked in the algorithm and module notes and rolled up by ticket in [[todo]]; this list holds workstream-level items only.

- [ ] Fix `TestDense_NumericalGradientCheck` before any training experiment — `x/math/nn/layers` (baseline §4) #p1 #ER-030
- [ ] Add detection decode + NMS utilities (if the YOLO head is trained/run in Go) — none exist #p2
- [ ] Add bilinear warp / grid-sample and an upsampling layer (needed for depth decoders and warping losses) — none exist; `x/math/interpolation/upsample.go` is non-NN #p2

## Inferred objectives (proposals)

- **Proposal:** run W5 training in Python notebooks first (the owner's notebooks), with Go limited to (a) recording/loading feature tensors and (b) running small heads at inference — the Go training stack is not yet proven correct.
- **Proposal:** share one frozen backbone across the three experiments: P5-level map for BoW place recognition (global pooled or VLAD/NetVLAD-style descriptor), P3–P5 for the YOLO head, and the same map as the feature-metric teacher target for VO/depth — so the K230 streams one tensor set.
- **Proposal:** a pure-Go geometry module (essential matrix, RANSAC, triangulation) under `x/math` rather than in GoCV-bound `x/vision`, so it also runs on embedded targets and gives VO a non-learned baseline.

## Research questions for ideation

1. **YOLO head on a frozen backbone:** which head (YOLOv8 decoupled/DFL head vs SSD-lite) trains well on a frozen int8 backbone map, and does a single P5 map suffice at the robot's object sizes, or are P3/P4 required (which costs ~7× SPI bandwidth, see W4 table)?
2. **BoW place recognition on CNN features:** classic visual words over local CNN descriptors (k-means vocabulary + TF-IDF + inverted index) vs global descriptors (GeM/NetVLAD) computed on the ESP32 or host — which fits the recorded-tensor setting and memory budget?
3. **Unsupervised VO + depth with a feature-metric teacher:** what loss (feature-metric reconstruction à la FeatDepth, vs photometric + SSIM) works when only the backbone's low-resolution map is available (7×7 or 10×10)? Is the stride too coarse for depth, i.e. must W4 also stream an earlier, higher-resolution map?
4. **Teacher/student split:** which network runs where — teacher on the host from recordings, student (pose/depth heads) on K230 or ESP32 — and what minimum head size meets the budget?
5. **Evaluation without ground truth:** how to score VO/depth/place recognition on the owner's own recordings (loop closures, wheel odometry, lidar scans from [[W8-lidar]] as weak ground truth)?
6. **Notebook interop:** which export path from the owner's notebooks into Go (TFLite via `x/marshaller/tflite`, ONNX via OpenCV DNN, or native `types.nn.Model` protobuf) should be the contract?

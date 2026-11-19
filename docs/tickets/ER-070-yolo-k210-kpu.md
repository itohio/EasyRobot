---
id: ER-070
title: "Fast detection on the K210's KPU: SSD-MobileNetV1 through the Go KPU compiler (YOLOX-nano stays for the Raspberry Pi)"
type: ticket
status: ready
kind: implementation
owner: EasyRobot
workstream: [K210-nn, W5]
milestone: YOLO-K210
tags: [easyrobot, ticket, k210, kpu, ssd, detection, nn, compiler]
created: 2026-11-19
updated: 2026-11-19
---
# ER-070 — Fast detection on the K210's KPU

[[tickets/index|Ticket index]] · [[roadmap#8. Next|roadmap §8]] · [[kpu-compiler]] · [[k210-kpu]]

**Outcome:** the stream of [[ER-069-yolo-k210-tinygo-cpu|ER-069]] (the UnitV's camera image with detections, drawn by a static web page) becomes video. **Owner's decision (2026-11-19):** "yes. faster detector."
- The K210 runs **SSD-MobileNetV1 COCO** (TensorFlow Lite's quantised object-detection model, Apache-2.0) on its KPU, through this repository's own Go compiler (`drivers/kpu/compile`), and accepts its lower accuracy (COCO mAP ≈ 18–21 against YOLOX-nano's 25.8).
- YOLOX-nano stays as it is (bit-exact on the K210's CPU, ER-069) for the Raspberry Pi.
- Not in scope: training, other models or downloads, other boards.

Each slice records its measured numbers. This line is the headline of the K210 write-up.

## The model (not in the repository)

- `detect.tflite` from `coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip` (download.tensorflow.org, Apache-2.0; SHA-256 `e4b118e5…` for the `.tflite`):
  - 300×300×3 uint8 in, uint8 per-tensor asymmetric weights;
  - 34 CONV_2D, 13 DEPTHWISE_CONV_2D, 13 RESHAPE, 2 CONCATENATION, 1 LOGISTIC;
  - then the custom op `TFLite_Detection_PostProcess` (anchor decode + NMS) on `concat` [1917×4] box encodings, `concat_1` [1917×91] class scores (after LOGISTIC) and an `anchors` constant [1917×4] (uint8); COCO labels in `labelmap.txt`.
- Never committed; examples and tests take it from a path given at build or test time.
- Today `tfl.Load` stops at the first uint8 CONCATENATION (`BoxPredictor_0/Reshape`). The backbone (MobileNetV1, ReLU6, 1×1/3×3, TF SAME stride 2) is the family the compiler already lowers exactly in one KPU segment.

## What the KPU path costs (measured, ER-069 A and the K210 speed check)

| Measurement | Value |
|---|---|
| KPU compute, MobileNetV1-0.25 at 256×256 (~53 M MAC) | 4.2 ms (≈ 12.6 G MAC/s) |
| KPU compute, one 1×1 conv 104×104×32→64 | 1.14 ms (≈ 19 G MAC/s) |
| CPU upload + read-back per segment boundary | 13–17 ms (for tensors of 104×104×32 to 52×52×64) |
| One direct kmodel inference of MobileNetV1-0.25 at 256 (KPU + upload + read-back) | 7.42 ms |
| The KPU's input limit | 256 rows per layer |

**Estimate for SSD-MobileNetV1 at 256×256** [INFERENCE]: ~0.9 G MAC → ~45–70 ms of KPU compute, plus one input upload and the small head read-backs → **~10–15 frames/s** of detection.

**Why not YOLOX-nano on the KPU:** its export is range-split (1–14-channel slices summed by 188 ADDs) and keeps SiLU as separate LOGISTIC + MUL ops, so the compiler gives 442 one-layer segments. Each segment pays 13–17 ms of copies: ≈ 10–12 s per frame as the compiler stands, ~1–2 frames/s even after weeks of graph passes. On the K210's CPU it runs 25.2 s per frame at `-opt=2`.

## Slices (host first; each: red tests where host-testable, verify, one commit)

| Slice | Scope | Check (and the numbers it records) |
|---|---|---|
| **S0** prerequisite | Commit the paused compiler work that sits uncommitted in the shared checkout (`drivers/kpu/compile/{conv.go,model.go,asym_test.go,fuse_test.go,model_test.go}`, `drivers/kpu/emulated.go`, `drivers/kpu/kmodel/{emulate.go,emulate_test.go}`, `docs/architecture/{k210-kpu,kpu-compiler}.md`): a fused 2×2/4×4 average pool rounds half up on the silicon as TFLite uint8 does (`TiesUp`), so the compiler stops declining it and the emulator models it. SSD has no average pool, so this is not needed for SSD. It lands first because S2–S4 edit the same package. Rebased on `main`, comments without evidence names, as its own commit(s) | the package's tests (including the silicon-derived cases) pass at `main` + the change; `make test-host` no new failures |
| **S1** host: load and run | `x/math/nn/qnet/tfl` loads `detect.tflite`: uint8 CONCATENATION and LOGISTIC; the graph is cut at `TFLite_Detection_PostProcess` (raw box regressions and class scores become the outputs), and its anchors and options are read from the file. An SSD decoder plus NMS in Go (`x/vision/detect/ssd`, reusing `detect/nms`), no allocation after setup | against LiteRT's reference kernels on ≥ 4 images: every layer bit-exact (as YOLOX-nano was), and the decoded detections equal the custom op's 4 outputs. Records: layers, MACs, weight bytes, host ms |
| **S2** host: what the KPU runs | Input 256×256 (`compile.Resize`, extended to RESHAPE/CONCATENATION). The anchors are regenerated for 256 by an SSD anchor generator that first reproduces the file's 1,917 anchors at 300 (stored there as a uint8 tensor) within one quantisation step. Then `compile.Compile` on the emulated KPU | which layers run on the KPU and which stay on the CPU, and why: heads smaller than 4×4, reshapes, concatenations, logistic, decode/NMS. Segment count; per layer ≤ 1 LSB against qnet's CPU run; detections at 256 against LiteRT at 300 on ≥ 20 images (accuracy cost stated). Kmodel size and the board memory budget (model + code + buffers within 6 MiB) |
| **S3** board: one frame | The kmodel and one test image on the UnitV, run by `kpu.Runner` (camera off) | KPU output per layer equal to the host emulation (within the known 1-LSB silicon cases); detections equal the host's; memory fits; ms per stage |
| **S4** speed and stream | The detector behind ER-069's `objects.go` interface; camera → letterbox to 256 → KPU → decode → stream. The KPU memory plan: the KPU now owns the 2 MiB AI SRAM, so `kpu.LendMemory` is not used on this path | frames/s end to end, ms per stage (capture, upload, KPU, read-back, decode, send); the link (115200 vs 1.5 Mbaud with resend; RGB565 vs JPEG) chosen by the measured detect time |

## Acceptance evidence required later

- [ ] S0: the paused compiler work committed, verified.
- [ ] S1: bit-exact layers and equal detections against LiteRT; host numbers.
- [ ] S2: KPU/CPU split, per-layer error, detection agreement at 256, memory budget.
- [ ] S3: board equals host on one frame; memory fits; stage times.
- [ ] S4: live stream frames/s and stage times.

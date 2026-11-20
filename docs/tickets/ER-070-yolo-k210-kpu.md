---
id: ER-070
title: "Fast detection on the K210's KPU: SSD-MobileNetV1 through the Go KPU compiler (YOLOX-nano stays for the Raspberry Pi)"
type: ticket
status: in-progress
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
  - 34 CONV_2D, 13 DEPTHWISE_CONV_2D, 13 RESHAPE, 2 CONCATENATION, 1 LOGISTIC; the layer widths are MobileNetV1 at depth 0.75 (24 … 768 channels): 4,088,592 B of weights, 728 M MACs at 300;
  - then the custom op `TFLite_Detection_PostProcess` (anchor decode + NMS) on `concat` [1917×4] box encodings, `concat_1` [1917×91] class scores (after LOGISTIC) and an `anchors` constant [1917×4] (uint8); COCO labels in `labelmap.txt`.
- Never committed; examples and tests take it from a path given at build or test time.
- `tfl.LoadDetection` loads it (S1). The backbone (MobileNetV1, ReLU6, 1×1/3×3, TF SAME stride 2) is the family the compiler already lowers exactly in one KPU segment.

## What the KPU path costs (measured, ER-069 A and the K210 speed check)

| Measurement | Value |
|---|---|
| KPU compute, MobileNetV1-0.25 at 256×256 (~53 M MAC) | 4.2 ms (≈ 12.6 G MAC/s) |
| KPU compute, one 1×1 conv 104×104×32→64 | 1.14 ms (≈ 19 G MAC/s) |
| CPU upload + read-back per segment boundary | 13–17 ms (for tensors of 104×104×32 to 52×52×64) |
| One direct kmodel inference of MobileNetV1-0.25 at 256 (KPU + upload + read-back) | 7.42 ms |
| The KPU's input limit | 256 rows per layer |

**Estimate for SSD-MobileNetV1 at 256×256** [INFERENCE]: ~506 M MAC (backbone and extra layers 445.6 M from `compile.Kmodel`, heads ~60 M) → ~27–40 ms of KPU compute, plus one input upload and the head read-backs at the six branch points → **~10–15 frames/s** of detection.

## Memory budget (S1, measured on the host; the board checks it in S3)

The UnitV has 6 MiB of main SRAM (6,291,456 B) for the program image, the heap and the stack. The KPU reads the weights from main memory by address. Its own 2 MiB holds the activations: the 256×256 graph's largest layer needs 1.18 MB for input and output, and `compile.Kmodel` accepted the chain.

| Item | Bytes | Source |
|---|---|---|
| Weights in the KPU's form: backbone + extra layers (34 layers, one kmodel at 256) | 3,100,816 | `compile.Kmodel` |
| The 12 head convolutions (sum of one-layer kmodels) | 1,071,296 | `compile.Kmodel` |
| Last extra layer (1×1 input at 256; SAME stride 2 needs an even input, so it stays on the CPU) | ~42,000 | its int8 weights + bias |
| Program without a model | ~173,000–211,000 | ER-069 B camera build 1,749,744 B minus its embedded YOLOX model (1,577,040 B); the image build 2,307,248 B minus model and test image (2,096,208 B) |
| Buffers: camera frame, 256×256 input, scores (1,284 anchors × 91), decode, stream | ~700,000 | [INFERENCE] |
| **Total** | **~5.1 MB of 6.29 MB** | about 1.2 MB spare |

It fits only if the weights are held once. `kpu.New` needs the kmodel 128-byte aligned. An embedded Go string has no such alignment, and `AlignedCopy` would hold a second 4.2 MB copy (≈ 9.3 MB: does not fit). So S3 keeps the kmodel out of the program image, in its own flash region, and reads it once at boot into one aligned buffer, as MaixPy does; or it embeds it at a guaranteed 128-byte alignment.

**Fallbacks if the board's budget fails** (each would need the owner's word where marked):

| Fallback | Memory effect | Cost | New download / training |
|---|---|---|---|
| Page the weights from flash: each layer's weights read into one reused buffer before it runs | 4.2 MB → ~0.7 MB (the largest layer: 663,552 B) | reads 4.2 MB of flash a frame: ~0.1–0.2 s at SPI speeds [INFERENCE], so ~3–5 frames/s instead of ~10–15; the compiler emits per-layer weight addresses | no / no |
| Keep part of the weights in the KPU's 2 MiB beside the activations (~0.8 MB free at 256) | −0.8 MB | that the KPU fetches weights from its own RAM is not verified; compiler work | no / no |
| Smaller input (224 or 192) | none for the weights (they do not depend on the input); activations and KPU time shrink | more missed small objects | no / no |
| Narrower backbone (SSD-MobileNetV1 at depth 0.5 or 0.25) | weights ÷ ~2.2 or ÷ ~9 | a different model, lower accuracy | yes: a published quantised model, if one exists, needs a download; otherwise training (owner) |

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

- [x] S0: the paused compiler work committed as two changes. The Go model of the KPU rounds a fused average pool half up, as the K210 does (on the board only half up reproduces the pooled output: 131,072 of 131,072 and 1,024 of 1,024); the compiler fuses TFLite uint8 average pools again (150 chains: 50,714 outputs equal, 6 at 1 LSB; the fused suite has none at 2 LSB).
- [x] S1: `tfl.LoadDetection` + `x/vision/detect/ssd`. `detect.tflite`: 63 layers (34 conv, 13 depthwise, 13 reshape, 2 concat, 1 logistic), 728 M MACs, 4 088 592 B weights, 1 620 000 B activations, 1 917 anchors; every layer bit-exact with LiteRT's reference kernels on 4 images and the 10 detections each equal the post-process operator's; host (Apple M5) ~160 ms a frame, decode 0.3–0.6 ms, no allocation. Test: `testdata/ssdtail.tflite` (64 inputs). Two things matched only once found: LiteRT's uint8 concatenation rescale is one fused multiply-add, and its optimised kernels detect differently from its reference kernels. Memory budget above: ~5.1 MB of 6.29 MB if the weights are held once.
- [x] S2, anchors: `ssd.Anchors(ssd.MobileNetV1Anchors(maps))` (`x/vision/detect/ssd/anchors.go`) is TF Object Detection API's SSD anchor generator. For the head maps at 300 (19, 10, 5, 3, 2, 1) its boxes quantise to the export's uint8 anchors tensor byte for byte (1,917 rows; the test pins the tensor's SHA-256); the largest difference from the stored values is 0.0032, under half a quantisation step. At 256 the head maps are 16, 8, 4, 2, 1, 1: 1,284 anchors. A resized graph keeps the trained anchors of its remaining cells, in pixels (`ssd.CropAnchors`: the 300 grid's cells inside each 256 map, × 300/256), because the run with its sampling grid kept (`compile.ResizeAligned`) is the 300 model on the top-left 256×256 pixels; a grid generated for the 256 maps moves the boxes (10 of LiteRT's 25 detections matched, against 16 with the kept anchors).
- [ ] S2, the rest: KPU/CPU split, per-layer error, detection agreement at 256, memory budget, and the form of the model the board loads (one copy in memory).
- [ ] S3: board equals host on one frame; memory fits; stage times.
- [ ] S4: live stream frames/s and stage times.

---
type: research
status: draft
owner: ideation
tags: [easyrobot, research, ideation, vision, navigation, k210, kpu, esp32s3, tinygo, odometry, depth, place-recognition]
workstream: [W4, W5, W9]
created: 2026-11-10
updated: 2026-11-10
---

# Vision navigation on the UnitV (K210) + XIAO ESP32-S3: shared frozen frontend, heads, topological map, split inference

Vision navigation. Refines [[W4-k230-feature-stream]] and [[W5-feature-tensor-experiments]]. Builds on [[2026-11-09-k210-camera-nn]] and [[2026-11-09-on-device-learning]].

**Scope:** research, not redesign. The owner has working notebooks for P1 (odometry) and P2 (depth) with "great results", so this note doesn't redesign them. It asks what the literature and existing systems offer for **his** architecture on **his** hardware, and where the risks are.

**Status:** recommendations. Nothing was built or run for this note.

**Labels:** observed (read in our evidence or code this session), reported (a paper or project page says so), inferred, `#verify`.

**Network (read-only):** 8 web searches (ViNG/GNM/NoMaD; ViNG's arXiv id; compact VPR descriptors; NetVLAD; Monodepth2/SfMLearner; DBoW2; Zhan/Shu feature losses; learned-VO generalisation) and 3 page reads (arXiv abstracts of NoMaD 2310.07896 and ViNG 2012.09812; NetVLAD on ar5iv.labs.arxiv.org). No datasets or models downloaded.

## Verdict

1. **Frontend:** *revised in §1a after the owner ruled out nncase.* Originally: a **MobileNetV1 α 0.5 (or MobileNetV2 α ≤ 0.5) at QVGA on the K210 KPU**, cut at stride 16.
   - The KPU takes inputs **≤ 320×240** and **1×1/3×3 kernels only**. That rules out VGA on the KPU (tile it or use it only offline), EfficientNet-Lite (5×5 depthwise) and MobileNetV3 (h-swish, squeeze-excite).
   - Our own nncase run (K8) shows the official MobileNet `.tflite` fails ("DepthwiseConv2d not supported") until it is rebuilt with explicit PAD + VALID. That rebuilt path compiles (mbnet025 kmodel v3, 507,576 B).
   - **Risk:** no KPU MobileNet timing exists yet (K8c pending).
2. **Split K210 → ESP32 is the right shape, and the only one that's fast enough.**
   - **Frontend on the S3:** MobileNetV1-0.25 at **96²** takes **208 ms** with PIE (observed), so a QVGA α 0.5 frontend would take seconds.
   - **Frontend on the K210 CPU:** one hart does about **24–30 MMAC/s** int8 (observed, K7), so the CPU can't run it either. It has to run on the KPU.
   - **The binding constraint is the link,** not compute: the UnitV's Grove port has two signals (UART only). The rate actually achieved, to the Mac at 1.5 Mbaud paced with chunk resend, is about **64 KB/s**. **A learned 1×1 bottleneck on the K210** (96 → 16 channels at 20×15, 4.8 KB/frame) gives about 13 fps at that rate.
   - The K210 ↔ ESP32 link itself is **untested**.
3. **Heads:**
   - **VO:** a depthwise head on feature pairs, the owner's P1 as is, with **metric scale from the training labels**.
   - **Depth:** the owner's P2. Self-supervised, SfMLearner → Monodepth2 → feature-metric losses.
   - **Place recognition:** a GeM/NetVLAD-style head with a PCA/learned projection to **32–128 int8 numbers per place**.
   - **Hazards:** for the first win, a **per-column free-space head** rather than full segmentation. People come from the existing person-detect model; cliffs from a cheap sensor.
   
   All of them read the same frozen features.
4. **Trusting odometry:** VO edges plus place-code matches, accepted only after **temporal consistency** (k frames) and **geometric verification** (VO between the current frame and the candidate node's stored features), as in DBoW2 / ORB-SLAM. The ESP32 can run this.
5. **Map and goals:**
   - **Robot:** a topological map on the ESP32 (nodes = keyframe place code + a small stored feature grid; edges = VO transform + traversal success).
   - **PC:** goal search ("show an object or scene", or click). It needs **per-cell descriptors** stored at each node, because global place codes don't see small objects.
   - ViNG/GNM/NoMaD show the shape (a graph of images plus a goal-conditioned controller), but their policies are GPU-sized. We borrow the structure, not the models.
6. **First win:** obstacle avoidance from a free-space head on the ESP32, plus a topological map of place codes, recorded on the robot.
   - **Kill criteria:** KPU frontend ≥ 5 fps, link ≥ 10 fps for the bottleneck tensor, free-space column accuracy ≥ 85 % on held-out recordings, ≤ 1 collision per 10 min in a test room.

## 1. Shared frozen frontend

**Hardware facts** (observed unless labelled):

| | K210 (UnitV) | XIAO ESP32-S3 Sense |
|---|---|---|
| Camera | **OV2640** (SCCB 0x30, PID 0x26) via DVP, 63 ms/frame | **OV3660** via LCD_CAM, 11.1 fps sensor, 5.4 fps QVGA YUYV over USB with 0 errors in 5 min; high-res JPEG up to 2048×1536 in 0.2–0.3 s |
| NN compute | KPU (1×1/3×3 conv + BN + act + pool, input ≤ 320×240, feature maps in the 2 MiB AI SRAM) — tiny kmodels run bit-close (32,749/32,768 within 1 LSB), **no MobileNet timing**; CPU one hart 24–30 MMAC/s int8 | qi8 + PIE: conv 3×3 1.77 cyc/MAC, pointwise 3.8, depthwise 26.6 (≈ 135 / 63 / 9 MMAC/s at 240 MHz, inferred); MobileNetV1-0.25 96² 208 ms |
| Memory | 6 MiB + 2 MiB AI SRAM | 512 KB SRAM; 8 MB PSRAM **not yet working in TinyGo** (driver built, two bugs found in review, board test pending) |

**Candidates:**

| Frontend | KPU fit | Loader fit (S3/PC) | Verdict |
|---|---|---|---|
| **MobileNetV1 α 0.5** | 3×3 depthwise + 1×1, ReLU(6), no residual adds; already compiled on our nncase path (α 0.25, K8) | yes (all ops supported) | **recommended first** |
| MobileNetV2 α 0.35–0.5 | 3×3/1×1, residual Add, ReLU6; the first expansion at α 1.0 QVGA is 160×120×96 int8 = **1.84 MB**, over the 2 MiB AI SRAM with its input (inferred); α ≤ 0.5 halves it | yes (Add supported) | second, if V1 features are weak for VO |
| MobileNetV3-small | h-swish, squeeze-excite (global pool + MUL broadcast) — not in kmodel v3's KPU set (CPU fallback) | HARD_SWISH missing | no |
| EfficientNet-Lite0 | **5×5 depthwise** — the KPU only does 1×1/3×3 | ok | no (KPU) |
| ResNet-8/18-style | KPU-friendly 3×3 | ok | heavier per MAC for VO-grade features; keep for comparison |

**Where to cut** (MobileNetV1, input 320×240; channels for α 0.5, inferred):

| Cut | Tensor (H×W×C) | int8 bytes | Use |
|---|---|---|---|
| stride 8 (after conv block 5) | 40×30×128 | 153,600 | depth/segmentation skip features (PC or later) |
| **stride 16** (after block 11) | **20×15×256** | 76,800 | **recommended cut**: VO, free space, place codes |
| stride 32 (after block 13) | 10×8×512 | 40,960 | place codes; too coarse for depth |

- **Cut at stride 16 with a learned 1×1 bottleneck** to 16–32 channels, run on the KPU: 20×15×16 = **4,800 B** or ×32 = 9,600 B per frame.
- **VGA:** the KPU can't take 640×480 in one pass (input limit). VGA stays an offline/training option (owner: "maybe try out vga too for better robustness"), or a 4-tile pass on the KPU at a quarter of the frame rate.
- **The ESP32-S3** gets only the bottleneck tensor and runs the heads. A frontend on the S3 is out of reach: a QVGA α 0.5 frontend is about 195 MMAC to stride 16 (inferred from MobileNetV1-0.5's 149 MMAC at 224² scaled by area), seconds per frame at our measured rates.

## 1a. Frontend revisit: no nncase; TFLite or the owner's own graphs (amended 2026-11-10)

**Requirement:** nncase models are not used; models are loaded as TFLite or as EasyRobot's own marshalled model graphs. What matters is which backbone is more efficient and suits the final layers; both depthwise-conv and dense final output layers are candidates, chosen by efficiency and accuracy.

**Consequence:** nncase is out as a runtime and toolchain for the owner's models. The models run through our TFLite loader / `qnet` on the S3 and the K210 CPU, and on the KPU only through a Go KPU compiler we'd write ourselves.

### 1a.1 Ranking the frontends

**How the numbers were made:**
- **Shapes:** computed for 320×240 (QVGA) and 160×120 (QQVGA), cut at stride 16. Params exclude the classifier.
- **Peak activation:** the largest layer's input + output, int8.
- **Time** (inferred): the MACs by layer kind × **our measured cycles per MAC**.
  - S3 with PIE at 240 MHz: first conv 10.3, depthwise 26.6, pointwise 3.80, Add 51 cycles per element.
  - K210 CPU, one hart at 390 MHz: 21.4 / 29.7 / 14.3.
  - The model reproduces the measured MobileNetV1-0.25 96² times exactly (208 ms S3, 316 ms K210), because its rates came from those runs. It is a cost model for our kernels, not new measurements.
- **ImageNet top-1 at 224²** is the only published accuracy proxy (reported, MobileNet V1/V2 papers and the TF-Slim checkpoint table, from memory, not re-read this session `#verify`). Nobody publishes accuracy at QVGA, and what matters is head accuracy on our recordings (§1a.4).

| Frontend (cut s16) | Output QVGA | MMAC QVGA | Params | Peak act. QVGA | ImageNet top-1 | S3 PIE QVGA / QQVGA | K210 CPU QVGA / QQVGA | Loader ops | KPU (own compiler) |
|---|---|---|---|---|---|---|---|---|---|
| **V1 0.25** | 20×15×128 | 55 | 108 k | 461 KB | 50.6 % | **1.6 s / 0.41 s** | 2.3 s / 0.60 s | all present | maps fully (conv, DW, pointwise, ReLU6) |
| V1 0.5 | 20×15×256 | 198 | 411 k | 922 KB | 63.7 % | 4.6 s / 1.2 s | 7.9 s / 2.1 s | all | maps fully |
| V1 1.0 | 20×15×512 | 750 | 1.6 M | 1.84 MB | 70.6 % | 14.8 s | 28.8 s | all | 1.84 MB peak, at the AI SRAM limit |
| **V2 0.35** | 20×15×32 | **64.7** | **82 k** | 1.15 MB | 60.3 % | **2.45 s / 0.60 s** | 3.03 s / 0.74 s | all (ADD present) | residual Adds on the CPU (a KPU↔CPU round trip per block); **at QVGA the first expanded depthwise needs row-band tiling** (§1a.3) |
| **V2 0.5** | 20×15×48 | 104 | 147 k | 1.15 MB | 65.4 % | 3.4 s / 0.87 s | 4.6 s / 1.2 s | all | as above |
| V2 1.0 | 20×15×96 | 321 | 534 k | 2.30 MB | 71.8 % | 8.4 s | 13.3 s | all | over 2 MiB at QVGA |
| MobileNetV3, EfficientNet-Lite | — | — | — | — | — | — | — | HARD_SWISH missing (V3); 5×5 DW (Lite) | no (h-swish/SE; 5×5) |

**Reading the table:**
- **Per MAC, V1 is faster on our kernels.** V2 spends more of its MACs in depthwise (15–19 % vs 3–12 %), our slowest kernel, and adds 0.12–0.31 M element Adds per frame.
- **Per unit of accuracy, V2 wins at width 0.5.** V2 0.5 is faster than V1 0.5 (3.4 s vs 4.6 s) and more accurate (65.4 vs 63.7 %). **V2 0.35 costs ≈ 54 % more time than V1 0.25 on the S3** (2.45 vs 1.59 s at QVGA) for about 10 points more ImageNet accuracy.
- **Correction:** my first V2 0.35 row (49 MMAC, 73 k) used plain 8-rounding instead of Keras's `_make_divisible` rule, whose "never more than 10 % below" clause makes the stem 16 channels. A recomputation, calibrated against TF-Slim's published 59 / 97 / 300 M for the full V2 at 224², gives 64.7 MMAC and 82 k params; the row above uses it. For our int8 path, TF-Slim's **quantized** V1 checkpoints score 48.0 / 60.7 / 70.1 % (0.25 / 0.5 / 1.0), a better accuracy proxy than the float column.
- **Memory:** no QVGA frontend fits the S3's 512 KB SRAM with an arena: V1 0.25 peaks at 461 KB and V2 at 1.15 MB, its first expanded layer of 160×120×48. QVGA on the S3 needs PSRAM, which isn't working yet. At **QQVGA**, V1 0.25 (115 KB) and V2 0.35 (288 KB) fit in SRAM.
- **Speed on the CPUs:** no MobileNet frontend reaches the 5 fps of §7 on either CPU. The best is about **2.4 fps** (V1 0.25 at QQVGA on the S3). The K210's second hart (untested) could roughly halve its time.
- **An S3-only option now exists** (no K210, no link): the XIAO's own OV3660 → V1 0.25 (≈ 2.4 fps) or V2 0.35 (≈ 1.7 fps) at QQVGA → heads. It is slow but needs no link and no KPU compiler. That makes it a good way to run the owner's experiments first.
- **Plain 3×3 conv nets** (the owner's own graphs): our S3 PIE 3×3 conv runs at **1.77 cycles/MAC**, the fastest kernel we have. Per C→C layer, the separable form beats full conv by about 1.4× at C = 32, 2× at C = 64 and 2.8× at C = 128 (inferred from the per-kind cycles). Narrow plain-conv stems are competitive, so they are worth an nnbench row.
- **QVGA vs QQVGA:** QVGA's extra resolution matters most for VO (sub-cell motion). QQVGA is about 4× cheaper.
- **Recommendation:** shortlist **V1 0.25 and V2 0.35** (and V2 0.5 when accuracy is short) at QQVGA first, then QVGA. Decide by head accuracy per millisecond on our recordings (§1a.4), not ImageNet.
  - **V2 for the CPU path;**
  - **V1 if the KPU compiler (§1a.3) becomes the runtime**, because V1 maps onto the KPU with no CPU round trips.
- **Head architecture** follows from §1a.2.
- **Unchanged:** the KPU still can't take EfficientNet-Lite or MobileNetV3, and the stride-16 cut and the bottleneck still hold.

### 1a.2 Two head styles on our kernels

**Cycles per MAC on the S3 with PIE** (measured for the conv kinds; **FC is an estimate**):

| Layer kind | Cycles/MAC |
|---|---|
| depthwise | 26.6 (measured) |
| pointwise (1×1) | 3.80 (measured) |
| FC (PIE dot) | **≈ 0.6 estimate, not measured.** Our closest numbers are dot-row microbenchmarks: 0.60 (older loop) and 0.19 (shipped fused loop, data in SRAM). nnbench has no FC layer of head size |
| conv 3×3 | 1.77 (measured) |

**Depthwise is cheap in MACs and parameters but the most expensive kernel per MAC.** Dense and pointwise work is cheap per MAC but heavy in parameters. **For large dense layers, the weight bytes, not the MACs, likely set the time** (inferred): the 4800→64 FC below has 307 KB of weights that stream from flash or PSRAM, not SRAM.

**Example: a VO head** on a pair of 20×15×C features (inferred from those rates; the shapes are illustrative; the FC parts are estimates):

| Head | C = 16 | C = 32 | C = 128 | Weights |
|---|---|---|---|---|
| **dw-style:** concat → pw 2C→64 → dw3×3 s2 → pw 64→64 → dw3×3 s2 → pw 64→6 → GAP | 1.0 M MAC, 21 ms | 1.6 M, 31 ms | 5.3 M, 90 ms | ~15 k |
| **dense:** concat → pw 2C→16 → flatten (4,800) → FC 4800→64 → FC 64→6 | 0.46 M, **3.3 ms + weight streaming** | 0.61 M, 5.7 ms + streaming | 1.5 M, 20 ms + streaming | **~307 k** (flash/PSRAM); the FC's MACs are < 1 ms, so its time depends on the flash/PSRAM read rate (unmeasured; an nnbench row, §1a.4) |
| GAP + dense: concat → pw 2C→128 → GAP → FC 128→6 | 1.2 M, 20 ms | 2.5 M, 39 ms | 9.8 M, 156 ms | ~33 k at C = 128 |

- **Expectation, to be checked by measurement:**
  - dense heads on a channel-reduced grid are the fastest on the S3 and keep spatial position, which VO needs;
  - dw heads are smaller and generalise better per parameter, but cost about 7× more cycles per MAC;
  - GAP + dense throws away position: good for place codes, poor for VO and per-column free space.
- **On the K210 CPU** the gap is smaller (depthwise 29.7 vs pointwise 14.3 cycles/MAC), so dw heads cost relatively less there.
- **Every op both styles need is in the loader:** CONV_2D, DEPTHWISE_CONV_2D, CONCATENATION, MEAN, FULLY_CONNECTED, RESHAPE.

### 1a.3 KPU without nncase: what our Go runner can express

- **Today:** `drivers/kpu` replays layer arguments taken from nncase-built kmodel files. `kmodel.Args` copies the 12 u64 words and patches the weight, BN and activation addresses. It is **not** a compiler.
- **Without nncase,** we need a **Go KPU compiler**: TFLite or the owner's marshalled graph → per-layer 12-word arguments, weights, BN and activation tables, and the KPU memory layout (`Layout`: group, row lines, channel lines). The per-layer argument format is the one the runner already decodes.

**Verified layer model** (from the SDK's `kpu_layer_argument_t` and the rc5 converter sources; this replaces my earlier summary).

**What one KPU layer can be:**

| Feature | What a Go compiler can set |
|---|---|
| Kernel | **1×1 or 3×3** only |
| Conv kind | regular or **depthwise** (a flag), for both kernel sizes |
| Stride | **always stride 1** with SAME padding (`pad_value`). Stride 2/4 exists **only through the pool stage** |
| Bias, requantisation | the per-output-channel BN table (`norm_mul` 24 bit, `norm_add` 32 bit, `norm_shift` 4 bit) |
| Zero points | uint8 asymmetric inputs and weights; **weights have one scale per layer, not per channel** |
| Activation | a 16-segment piecewise-linear table after BN, clamped to 0–255. **ReLU, ReLU6, identity and LeakyReLU are exact;** sigmoid and h-swish only approximate (not bit-exact with TFLite) |
| Pool (after the activation) | none; max or avg 2×2/2; max or avg 4×4/4; **"left-top" 2×2/2 = subsampling (stride 2)** |
| Sizes | width ≤ 1024, height ≤ 512, channels ≤ 1024 (also the bit widths in our `kmodel/layout.go`) |
| Memory | 2 MiB AI SRAM in 64-byte lines; a layer's input **and** output must both be resident; a row wider than 32 takes ⌈W/64⌉ lines |
| Not available | elementwise Add/Mul, concat as an op, 5×5 or dilated kernels, native stride 2, transposed conv / upsampling, global pool, FC (nncase runs FC as a 1×1 conv on a padded 4×4 map, or on the CPU) |

**MobileNet blocks against it**:

| Block | KPU mapping | Fits? |
|---|---|---|
| V1 stem: 3×3 conv /2 + ReLU6 | 3×3 regular conv (stride 1) + ReLU6 table + left-top pool | **yes** |
| V1 block /1: dw 3×3 → pw 1×1 | two layers | **yes** |
| V1 block /2 | depthwise 3×3 at stride 1 (full resolution), then the 1×1 with the left-top pool (1×1 commutes with subsampling) | **yes**: 4× the depthwise compute, cheap on the KPU |
| V1 α 0.5 at QVGA | largest input + output ≈ 1.1 MB | **yes** |
| V2 expand, depthwise, linear project | regular 1×1; depthwise as V1; 1×1 with an identity table | **yes** |
| **V2 residual Add** | no KPU op. CPU round trip: read both tensors out of KPU RAM, requantised Add, upload back. **~15–20 ms per Add** for a 60×80×16 tensor (inferred, scaled from K8b's ~700 µs per 3 KB). An untested alternative folds the Add into the projection conv via a channel-contiguous layout, limited by the one weight scale per layer | **CPU per block** |
| **V2 at QVGA**, the first expanded depthwise (120×160×48) | input + output ≈ 2.21 MB > 2 MiB | **only with row-band tiling** (horizontal bands, 1-row halo, edge rows discarded); without tiling V2 fits up to about **256×256 / 224×288** |
| V3 h-swish / squeeze-excite; EfficientNet-Lite 5×5 | approximate activation / CPU-only SE / no exact 5×5 | no |

**Consequences for the shortlist (§1a.1):**
- **V1 is the KPU-native choice:** every block maps, with no CPU round trips and no tiling at QVGA.
- V2 costs about 0.25 M element Adds per frame at QVGA as CPU round trips, plus row tiling of its first expansion. Prefer V2 only on the CPU path, or on the KPU at ≤ 256×256 (e.g. 224×288 or QQVGA).

**Training rules for bit-exact KPU runs** (also in §6):
1. **Stride-2 layers as explicit PAD(1,1) + VALID, not TF/Keras SAME.**
   - The KPU's left-top pool keeps pixel (2i, 2j) of the stride-1 SAME output: windows centred on even pixels. TF SAME at stride 2 on even sizes centres on odd pixels.
   - A SAME-trained model runs shifted by one pixel on the KPU: a small accuracy change, and not bit-exact with the owner's TFLite.
   - Our K8 rebuild already used PAD + VALID.
2. **Keep every stride-2 input even** (odd sizes such as 15 → 8 don't map; nncase refused them). QVGA is even down to stride 16, which matches our cut.
3. **Per-tensor (per-layer) weight quantisation** for the layers that go on the KPU (inferred from the one-scale-per-layer field). TFLite's default per-channel int8 weights would have to be re-quantised, losing accuracy.
4. **Activations from the exact set** (ReLU, ReLU6, identity, LeakyReLU) for KPU layers.

**Numerics:**
- TFLite int8 (symmetric per-channel weights, TFLite rounding) ≠ KPU uint8 with per-layer weight scale and its own BN and activation rounding, so the compiler converts. The training rules above keep that conversion small.
- KPU results won't be bit-exact with `qnet` unless the model follows the rules. Validate layer by layer within ±2 LSB, the K8b precedent (32,749 of 32,768 within 1 LSB).

**Effort** (inferred): register semantics from the SDK's `kpu.h` (documented layer by layer in k210's `EasyRobot/docs/architecture/k210-kpu.md`, uncommitted) and our K8 runner; a few weeks. V2 adds CPU Add + Download/Upload and row-band tiling. It is the critical path for a K210 frontend at useful fps, because the K210 CPU is 0.6 s per frame even at QQVGA.

### 1a.4 How to measure (nnbench rows)

**Frontends:**
- V1 0.25, V2 0.35, V2 0.5 (plus one narrow plain-conv stem from the owner), cut s16, at QQVGA and QVGA;
- targets: **S3 PIE**, S3 generic, **K210 CPU**, Mac;
- per row: ms, ms per layer kind, peak arena (`PlanMemory`), weight bytes.
- The S3 runs QVGA only after the PSRAM board test (E6).

**Heads**, on precomputed 20×15×C and 10×8×C inputs, C ∈ {16, 32, 128}: the three head styles above for VO (6 outputs), per-column free space (20×4 bins) and the place code (64 outputs, L2 norm in Go). Same targets.

**KPU rows,** once the Go compiler exists: the same frontends, ms per layer, ±LSB vs `qnet`.

**Accuracy** (the deciding column): fp32 TFLite on the PC over our W2 recordings:
- VO translation/rotation error per metre;
- free-space column accuracy;
- place recall@1.

Then int8 through the loader, checked to match fp32 within tolerance. The pick is the best accuracy per S3 (or K210) millisecond for each head.

## 2. Heads on precomputed features

### 2.1 Visual odometry (P1, the owner's notebook)

- **Shape:** two stacked feature tensors (t, t+1) → depthwise-separable layers → global average pool → FC 6 (or 3 for planar: x, y, yaw on a wheeled robot).
- **Labels:** ORB matches on high-res frames, from essential matrix or PnP with known intrinsics (owner).
- **Scale:**
  - monocular ORB pose is up to scale;
  - metric scale needs a known baseline: wheel odometry, IMU, known camera height, or stereo for training;
  - a planar robot with a fixed camera height can recover scale from the ground plane;
  - the head inherits whatever scale the labels carry.
- **Known risk** (reported): learned pose regressors generalise poorly across domains, e.g. outdoor-trained models fail indoors, and ground-robot data lacks pitch and roll (TartanVO, CoRL 2020). Train and test on our own robot's recordings, not public datasets alone.

### 2.2 Self-supervised depth (P2, the owner's notebook)

Primary papers (reported, abstracts):

| Paper | What it adds |
|---|---|
| SfMLearner, Zhou et al., CVPR 2017, arXiv 1704.07813 | depth net + pose net trained by view synthesis (photometric reconstruction) |
| Monodepth2, Godard et al., ICCV 2019, arXiv 1806.01260 | per-pixel minimum reprojection (occlusions), auto-masking of static pixels, full-resolution multi-scale |
| Zhan et al., CVPR 2018, arXiv 1803.03893 | **deep-feature reconstruction**: the warp loss in a feature space beats photometric loss for depth and VO |
| Shu et al., ECCV 2020, arXiv 2007.10603 | a **learned feature-metric loss** for textureless regions |
| Bian et al., CVPR 2020 ("without PoseNet") | scale-consistent depth/pose; better indoor generalisation |

**Fit with a frozen frontend:**
- The depth decoder reads the frontend's features. The loss needs **images**, or features, at both frames, so recordings must keep frames.
- The decoder needs upsampling and usually skip features (stride 8 or 4). Those are big over the link. **Recommendation:** run depth on the **PC** (it gets features plus frames) or as a coarse 40×30 map later. The robot's obstacle avoidance uses the cheaper free-space head (§2.4).

### 2.3 Similarity, place recognition and "BoW"

| Option | Size per place | Robustness | Fit |
|---|---|---|---|
| **DBoW2** (ORB + binary vocabulary; Gálvez-López & Tardós, T-RO 2012) | sparse word histogram, ~hundreds of words | proven in ORB-SLAM loop closure, with temporal consistency + geometric check | needs ORB on frames; the robot sees features, not frames, so a poor fit on-robot (fine on the PC) |
| **BoW over CNN cells** (k-means over the 20×15 cell descriptors) | sparse histogram | the owner's "bag of words" on our own features | good, and explainable |
| **NetVLAD / GeM + PCA** (Arandjelović et al., CVPR 2016) | NetVLAD reports state-of-the-art compact **256-D** descriptors (reported) | strong | **good**: a few int8 numbers per place |
| learned compact code (a projection head trained with triplet/contrastive loss on our own revisits) | **32–128 int8 B** | tuned to our rooms | **recommended**: "places encoded in fewer features" (owner) |

- **How few numbers:** start at 128 int8 (128 B). Reduce to 64 or 32 while recall@1 on our own revisit set stays within 5 points.
- Storage: 8 MB PSRAM (once working) holds tens of thousands of codes; even 512 KB SRAM holds ~2,000 at 128 B.

### 2.4 Hazards: segmentation vs cheaper heads

| Hazard | Option | Recommendation |
|---|---|---|
| obstacles in the path | **per-column free-space head** (for each of 20 columns, the distance bin to the first obstacle; trained from the depth model's output or from bumper/ToF events) | **first win**: 20 bytes out, tiny head |
| cliffs, edges | monocular cliff detection is hard (the floor ends where texture continues) | an IR/ToF cliff sensor; a learned edge class later |
| people, faces, hands | the existing person-detect / VWW models (they run bit-exact in our loader on the Mac); a face detector on the KPU | separate small models first; a shared-frontend seg head later |
| full semantic segmentation | a decoder like depth's; needs labelled data | later, on the PC first |

**One frontend can feed all heads** (multi-task on frozen features is standard). The cost is the decoder (§2.2), not the sharing.

## 3. Trusting odometry (kidnapped robot, loop closure)

**What systems do** (reported):
- DBoW2 / ORB-SLAM: candidates from the place database, **temporal consistency** (the same match across consecutive frames), then **geometric verification** before accepting a loop.
- ViNG: plans over a graph of past images with a learned reachability/distance; **graph pruning** removes unreliable edges.

**Simple scheme for us:**
1. **Each frame:** VO predicts the pose relative to the current node, and the place head gives the top-k nodes by code distance.
2. **Consistent:** the VO-predicted node is in the top-k → trust VO.
3. **Inconsistent for k = 3 frames in a row,** with one node winning with a clear margin (best/second distance ratio < 0.8) → **verify**: run the VO head between the current features and that node's **stored feature grid**. Accept the relocalisation if the predicted transform is plausible, i.e. inside the frame and the residual is small.
4. **VO residual high** (the feature-metric reconstruction error of the current pair over a threshold) → mark VO untrusted and slow down.
5. **Edge bookkeeping:** every accepted relocalisation or traversal adds or confirms an edge. Edges that repeatedly fail are pruned (the ViNG idea).

All of this is integer arithmetic over small tensors, so it fits the ESP32.

## 4. Topological map and goals

- **Node:** a keyframe with its place code (32–128 B), its stored bottleneck grid (20×15×16 = 4.8 KB; or 10×8×32 = 2.5 KB) for geometric checks and goal search, a timestamp, and visit counts.
- **Edge:** a VO transform, its uncertainty, and traversal success/failure.
- **New node:** when the code distance to the current node exceeds a threshold, or VO has moved more than about 0.5 m or 30°.

**Goal-directed systems** (reported):

| System | Idea | Size |
|---|---|---|
| **ViNG** (Shah et al., arXiv 2012.09812) | topological graph of past images + learned controller to a goal image; waypoint proposal, graph pruning, negative mining; offline data only | GPU |
| **ViKiNG** (Shah & Levine, RSS 2022) | kilometre-scale, with geographic hints | GPU |
| **GNM / ViNT / NoMaD** (general-navigation-models; NoMaD arXiv 2310.07896) | cross-robot foundation policies; NoMaD = one diffusion policy for goal-reaching and exploration | Transformer + diffusion: far beyond the ESP32 |

**What we take:** the graph-of-images structure, goal = an image, and following edge by edge with a local controller.

**PC vs robot:**

| PC (Go host or Python first) | Robot (ESP32-S3) |
|---|---|
| map viewer; the owner clicks a node, an object or a region | builds nodes and edges; stores codes and grids |
| **goal search:** the PC runs the same frozen frontend (+ bottleneck) on the shown image or crop; a **scene** goal matches place codes, an **object** goal matches the crop embedding against **per-cell** grids at each node; ranked candidates come back to the owner | follows the path edge by edge: steer toward the next node (code distance shrinking + VO heading), with free-space avoidance always on |
| path planning (Dijkstra/A* over edges weighted by traversal success and length) | relocalisation (§3); reports progress and failures |
| the depth autoencoder and big decoders | obstacle avoidance |

**Object goals need per-cell descriptors.** A global place code averages away a mug on a table. Storing the 20×15 grid per node (4.8 KB) lets the PC search cells. This is where "plenty of space in RAM" (owner) matters: 8 MB PSRAM ≈ 1,500 nodes with grids (inferred).

## 5. Split inference K210 → ESP32

**Link** (observed, K210 → Mac via an FT232BM at 1.5 Mbaud):

| Mode | Effective |
|---|---|
| paced 256 B / 2 ms with chunk resend | ~64 KB/s (QVGA Y8 at 0.86 fps, 0 frames lost in 100) |
| unpaced | loses data; no RTS/CTS available |

K210 ↔ ESP32 directly is **untested**. Without the FTDI chip, and with the S3's UART RX in DMA mode, higher rates are plausible (both chips are rated to ~5 Mbaud on paper, `#verify`), but plan on 64 KB/s until measured.

| Payload per frame | Bytes | fps at 64 KB/s | fps at 200 KB/s (target) |
|---|---|---|---|
| stride-16 raw, α 0.5 (20×15×256) | 76,800 | 0.8 | 2.6 |
| stride-32 raw (10×8×512) | 40,960 | 1.6 | 4.9 |
| **bottleneck 20×15×16** | **4,800** | **13** | 42 |
| bottleneck 10×8×32 | 2,560 | 25 | 78 |
| place code + free-space columns only | ~150 | link not limiting | — |

**Compression options:**

| Option | Effect | Cost |
|---|---|---|
| int8 (already) | 4× vs fp32 | none |
| channel pruning (keep channels with high variance / importance for the heads) | 2–4× | weaker features |
| **learned 1×1 bottleneck** (trained with the heads, frontend frozen; runs on the KPU) | 8–16× | needs head training through it |
| entropy coding (delta + RLE on sparse ReLU maps) | 1.5–3× (inferred) | K210 CPU time; variable size |

**Framing:** reuse the W4 chunk header (magic, seq, len, CRC-32) with NAK/resend, as K5c already does.

**Latency budget per frame** (inferred, `#verify`):

| Stage | Time |
|---|---|
| capture | 63 ms |
| KPU frontend | unknown (K8c) |
| bottleneck transfer at 64 KB/s | 75 ms |
| S3 heads (free space + place code) | 20–60 ms |

≈ 0.2 s end to end, about 5 fps, which is enough for slow obstacle avoidance.

## 6. Training pipeline

1. **Record** on the robot.
   - **Sensors:** co-mount the UnitV and the XIAO. The K210 can't stream raw QVGA frames fast (0.87 fps Y8), but it can stream its features. The XIAO captures high-res JPEG for ORB pseudo-labels (0.2–0.3 s per frame) and QVGA frames for the photometric losses.
   - **Sync:** timestamps over a shared serial line, or an LED flash at the start.
   - **Domain gap:** OV2640 vs OV3660, a risk for the frontend's features. Mitigation: train heads on **K210-produced features** where possible, and check the gap on a paired set.
2. **Freeze the frontend;** precompute its features once, in fp32 for training and, for validation, **KPU-exact** via our Dockerised nncase simulator (K8).
3. **Train heads** (and the bottleneck) on the features. Quantisation-aware or noise-augmented, so the int8 KPU features don't surprise the heads.
4. **QVGA first, then VGA** (owner): VGA as an offline teacher or augmentation, since the robot's KPU path is QVGA.
5. **fp32 TFLite** check on the PC (owner's step), then **int8 TFLite** (`RoundTFLite`, which our loader matches bit-exactly for Conv2D/DW/Add, observed), then our qi8 loader on the S3.
6. **If the frontend will run on the KPU** (§1a.3), train it with stride-2 layers as PAD(1,1) + VALID, even sizes at every stride-2 input, per-tensor weight quantisation and KPU-exact activations, so the KPU, `qnet` and the owner's TFLite agree.

**Ops each head needs vs our loader** (supported in `x/math/nn/qnet/tfl`, observed: CONV_2D, DEPTHWISE_CONV_2D, FULLY_CONNECTED, AVERAGE/MAX_POOL_2D, RESHAPE, SOFTMAX, ADD, LOGISTIC, MUL, MEAN, QUANTIZE, DEQUANTIZE, CONCATENATION, PAD/PADV2, RESIZE_NEAREST_NEIGHBOR; fused RELU/RELU6):

| Head | Needs | Missing |
|---|---|---|
| VO | concat (pair), conv/DW, mean, FC | none (keep yaw/translation as FC outputs) |
| free space (per column) | conv/DW, mean or max over rows, FC or 1×1 conv | none |
| place code | conv, mean (GeM ≈ mean of p-th powers: do the power in Go, or use plain mean), FC, **L2 normalisation** | **L2_NORMALIZATION** (do it in Go after the graph) |
| NetVLAD-style | conv 1×1, softmax over clusters, residual aggregation | **custom aggregation** (in Go); BATCH_MATMUL if exported as a graph |
| depth / segmentation decoder | upsampling + conv + sigmoid; skip concat | **RESIZE_BILINEAR** (train with nearest instead), **TRANSPOSE_CONV**, **DEPTH_TO_SPACE**; ARG_MAX for segmentation labels |
| frontends we said no to | h-swish, SE | HARD_SWISH, LEAKY_RELU (rejected today), SUB |

## 7. First win: basic obstacle avoidance and mapping

**Platform:** the wheeled avoider (owner question; the EasyRobot roadmap's first wheeled robot). Both camera boards are mounted on it.

**Revised 2026-11-10 after a usability review**: the stop path, the robot↔PC telemetry, calibration, the recording jobs, the baseline ladder, feature versions, run tracking and map persistence are now steps. W0 follows §1a (no nncase). The cross-cutting rules and the radio choice are in §7a.

| Step | Done when | Kill / rethink if |
|---|---|---|
| W0 Frontend (§1a) | **S3-only:** V1 0.25 or V2 0.35 at QQVGA on the XIAO through `qnet`, from its own camera. **Or K210:** the same frontend through the Go KPU compiler (§1a.3), within ±2 LSB of `qnet` | S3 < 2 fps; K210 < 5 fps → smaller α or input (§1a.1) |
| W1 Link (K210 path only) | K210 → ESP32 UART with W4 framing **and the feature version id in every chunk header** (§7a); the bottleneck tensor arrives CRC-clean for 10 min | < 10 fps for 4.8 KB frames → 10×8×32 bottleneck or a higher baud after measuring |
| **W1b Telemetry and view** (G3/G4) | **USB serial on the bench** (XIAO USB-CDC). One telemetry record per frame: stage times (capture / frontend / link / heads), link CRC errors and resends, the 20 free-space columns, the chosen column and speed, node id, trust state (§3), stop reason. Map events and dumps on the same channel, with W4 framing and `camframe`'s resend. A **`camframe` overlay view** draws the columns and decision on the XIAO's QVGA frame. An **e-stop key** on the PC. **LED codes:** UnitV green = frontend running, red blink = link error; XIAO green = driving, amber = stopped by safety, red blink = lost (§7a) | the owner can't tell a bad head from a slow link from a bad controller on a recorded bench run |
| **W2a Calibration** (G6) | `cmd/calibrate` **mono** for each camera (OV2640 and OV3660 intrinsics) and **stereo** for the rig's extrinsics, saved under a **calibration id**. Pose labels from the XIAO frames are transformed into the K210 camera's frame before training | reprojection error above `cmd/calibrate`'s usual threshold, or the rig not rigid (re-check after each mount change) |
| W2 Data (G2, G5) | 1 h in 2–3 rooms: K210 features + XIAO frames (QVGA + high-res) + wheel ticks if any + bumper events. **Driven by hand:** RC through `cmd/elrs-rx` (its failsafe is the stop), or PC keys over W1b on the bench, at the speeds the autonomous runs will use. **One session = one folder + manifest** (§7a) outside every repo. **Sync:** an LED flash at start and end, seen by both cameras, measures drift | measured drift > 1 frame; storage rate too low (see §7a "Storage") |
| **W2b Baseline ladder** (G7) | the owner's notebook numbers at each rung (§7a): (a) his notebook on his data; (b) on our recordings, in the format his notebook reads; (c) on our frontend's features (fp32, then KPU-exact or `qnet` int8); (d) through the bottleneck. Same metrics each time (VO RPE/ATE; depth abs-rel; free-space column accuracy) | a drop at a rung is investigated at that rung before moving on |
| W3 Free-space head | per-column free distance (labels from the owner's depth model on XIAO frames, or bumper/ToF), int8 on the S3, **scored on the fixed held-out set** with a run record (§7a) | column accuracy < 85 % on the held-out room |
| **W3s Stop path** (G1, **prerequisite of W4**) | (1) **stale-data stop:** motors off when the newest free-space output is older than **300 ms**, or after N link CRC errors in a row (K210 path); (2) **e-stop:** a button on the robot, the PC key (W1b) and the RC failsafe (`elrs-rx`); (3) **bumper** contact → stop and back off; (4) a configured **speed cap** (default 0.2 m/s); (5) **floor-only runs** until a cliff sensor exists. **Done when:** pulling the K210's UART lead (or killing the head on the S3-only path) stops the wheels within 300 ms, on a bench test with the wheels up | any stop path that doesn't trip in the bench test |
| W4 Avoid | reactive controller (steer to the freest column, slow on low clearance) on the S3, **only with W3s passed**, floor only, under the speed cap | > 1 collision per 10 min in the test room at 0.2 m/s |
| W5 Map (G10, G12) | a place code per keyframe (start: mean-pooled bottleneck + PCA from the host, 64 B); nodes and edges from wheel/VO on the S3. **Persisted:** save/load to XIAO flash or microSD with the feature version id, `reset` per room, `dump` to the PC. **Lost** (§3) for > 10 s → stop, report `lost since 14:02, last node 12`, optional look-around; trust state on the LED and in telemetry | revisit recall@1 < 70 % in one room → a trained code head (§2.3) |
| W6 (after the win; G11) | VO head from the owner's notebook; relocalisation (§3); PC goal search (§4). **The robot's map copy is authoritative;** the PC viewer edits it through commands (name a place, merge nodes, delete an edge, reset). **Goal states:** confirm candidate, en route (node k of n), arrived, cancelled, unreachable (edges failing), not in the map ("seen nowhere; best 0.41 at node 12"), with a cancel from the PC and an explicit "explore or give up" choice for a not-found goal | — |

This slice still uses none of the unbuilt pieces (no depth decoder, no PSRAM). It proves the frontend, the split (or the S3-only path), the link and one head end to end, now with a way to stop, watch, record and compare.

## 7a. Cross-cutting rules for the plan

**Feature version id** (G8):
- **What it is:** a short hash of frontend + bottleneck weights (and input size). It appears in:
  - the link's chunk header;
  - the head files;
  - every dataset manifest;
  - every map node.
- **Mismatches are refused:**
  - at boot the ESP32 reports `K210 sends features v7; heads expect v6: flash matching heads` and doesn't drive;
  - a map of another version opens read-only with "re-map needed";
  - training refuses to mix feature sets of different versions.

**Matched releases** (G15): one make target per board takes a **release**: the feature version plus the head versions (and the K210 firmware/kmodel or KPU program). The last good pair is recorded so it can be re-flashed in one command.

**Recording sessions** (G5):
- **Location:** `_data/vision-nav/<date>-<room>/`, outside every repo. Home recordings contain people's faces, so they are never committed and never sent to a remote model.
- **Manifest** `session.yaml`: room, date, lighting, platform, rig, firmware ids, feature version, calibration id, sync method, measured drift, operator, duration, file list with sizes.
- **Storage rate:** the XIAO's SD card over SPI measured **65–75 KB/s** (observed, M3), about 250 MB per hour. 1 h of high-res JPEG at 3–5 fps is several GB (an estimate), so **SD alone can't hold the high-res stream**. Options:
  - high-res at ~0.3 fps on SD (enough for sparse ORB pose anchors), QVGA at full rate;
  - a faster SD path (4-bit SDMMC or a higher SPI clock, `#verify`);
  - **tethered recording to the PC over USB-CDC** (observed bursts ~0.9 MB/s, 114 KB/s end to end in E3) on a cart or slow drive.
  
  Pick per session; the manifest records which.

**Run tracking** (G9):
- one **fixed held-out set per room**, chosen before training and never trained on;
- a **run record** per training run: frontend/bottleneck/head versions, data sessions, metrics, who ran it (Colab or the Mac), and the artifact location under `_data/vision-nav/runs/`;
- a results table per head, so going *up* the frontend ladder (V1 → V2, QQVGA → QVGA) has a number, e.g. RPE on the same held-out set within the time budget.

**Who decides the frontend** (G14): after ladder rung **W2b(c)**, the owner picks the frontend from the table (his notebook's frontend vs the §1a shortlist). VGA stays a training-time option, because of the KPU input limit and the CPU times in §1a.

**Robot hardware prerequisites for W2–W4** (G13), owner decisions alongside the platform choice:
- bumper or ToF (stop path, labels);
- a cliff sensor, or floor-only runs;
- wheel encoders, if VO scale should come from wheels;
- an e-stop button;
- a battery that lasts at least one recording session.

**Radio for untethered runs** (G3), an **owner decision**. The board work so far is under "no Wi-Fi" (owner, 2026-11-09).

| Option | Throughput | Latency | Status in our stack | Fits |
|---|---|---|---|---|
| **USB-CDC (bench, tethered)** | ~0.9 MB/s burst, 114 KB/s end to end (observed, E3) | ms | works today | telemetry, overlay, map dumps, e-stop key, recording; only on the bench or a cart |
| Wi-Fi (UDP/TCP) | Mbit/s class (inferred) | ms to tens of ms | no ESP32-S3 Wi-Fi driver in our TinyGo stack (`#verify` upstream); against the current "no Wi-Fi" rule | live frames, overlay, map, goals |
| ESP-NOW (to a second ESP32 on the PC's USB) | ~1 Mbit/s class, ~250 B packets (reported by Espressif docs, from memory `#verify`) | a few ms | same radio-stack gap as Wi-Fi | telemetry, map events, small frames |
| **ELRS telemetry** via `cmd/elrs-rx` (CRSF) | tens to hundreds of B/s (inferred; depends on packet rate and telemetry ratio) | tens of ms | `elrs-rx` exists (pico, esp32-coreboard, SAMD XIAO targets; an S3 target is `#verify`); RC control + failsafe already there | **control, e-stop, status line**; not frames or maps |
| BLE | tens of KB/s (inferred) | ~tens of ms | not in our stack (`#verify`) | telemetry |

**Recommendation:**
- **now:** USB on the bench, plus ELRS for driving, failsafe and a status line on untethered runs;
- **later:** decide on a second radio (Wi-Fi or ESP-NOW) when live frames or the map are needed off the bench. That needs the owner to lift "no Wi-Fi" for this robot, and a TinyGo radio driver.


## 8. TinyGo path (order)

1. **K210:** the KPU runner on a MobileNet kmodel (K8c), plus the camera → KPU → UART pipeline in TinyGo. The runner exists (`drivers/kpu`); the camera path exists (`drivers/camera`, `sensor/ov2640`).
2. **Link:** W4 framing over UART on both boards (ER-031's idea, over UART instead of SPI).
3. **ESP32-S3:** heads through the qi8 loader (`x/math/nn/qnet/tfl`), with PIE kernels for conv/DW/FC (built). The free-space head first, then the place code (L2 normalisation in Go).
4. **ESP32-S3:** map, relocalisation and controller in plain Go (integer, small). PSRAM after its board test (E6) for larger maps.
5. **VO head** (the owner's notebook → int8 TFLite → loader).
6. **PC side in Go:** map viewer, goal search (the same frontend via qi8 on the host, or fp32), planner.
7. Later: depth / segmentation decoders (loader ops: bilinear or transpose conv), K210 second hart for pre/post-processing.

## Knowledge map

| Quadrant | Item | Next step, owner |
|---|---|---|
| Known knowns | KPU limits (≤ 320×240, 1×1/3×3), our nncase path (PAD+VALID rebuild compiles), K210 CPU 24–30 MMAC/s, S3 PIE rates and 208 ms for MobileNetV1-0.25 96², link ~64 KB/s at 1.5 Mbaud paced, loader op set, OV2640/OV3660 sensors | — |
| Known unknowns | KPU MobileNet fps on camera frames | k210, K8c |
| Known unknowns | K210 ↔ ESP32 UART rate without FTDI | a link test (W1) |
| Known unknowns | XIAO PSRAM on silicon | E6 board test |
| Known unknowns | Feature domain gap OV2640 vs OV3660 | W2 paired recordings |
| Unknown knowns | The user's notebooks: frontend, cut layer, losses, label scale | the user shares them |
| Unknown knowns | The robot platform and whether it has wheel encoders (scale for VO) | owner |
| Unknown unknowns | Whether learned VO from a frozen classification frontend holds up indoors on our robot | W6 with the owner's notebook |

## Decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | Frontend (revised, §1a): no nncase. Shortlist MobileNetV1 0.25 and MobileNetV2 0.35 (V2 0.5 if accuracy is short) at QQVGA then QVGA, cut s16; pick by head accuracy per ms in nnbench (§1a.4). V2 for the CPU path, V1 if a Go KPU compiler becomes the runtime. Run the owner's experiments S3-only at QQVGA first (~2 fps) while the KPU compiler is decided | Yes |
| 1b | Build a Go KPU compiler (TFLite or owner graph → KPU layer arguments, conv 1×1/3×3, DW, BN, PWL activation, pooling; Add/FC/GAP on the CPU), validated layer by layer against `qnet` (±2 LSB) | Yes, if a K210 frontend at useful fps is wanted; otherwise S3-only |
| 1c | Heads: benchmark dw-style, dense-on-grid and GAP+dense for each head; expect dense-on-grid for VO and free space on the S3, GAP+dense for place codes | Yes |
| 2 | Split: K210 = frontend + bottleneck; ESP32-S3 = heads, map, controller; PC = goal search, depth, viewer | Yes |
| 3 | Link: UART over Grove with W4 framing and resend; plan on 64 KB/s until W1 measures the direct link | Yes |
| 4 | First win = §7 (free-space head + avoidance + place-code map) with its kill criteria; depth and segmentation wait | Yes |
| 5 | Place codes: learned 32–128 int8 B per place, per-cell grids stored at nodes for object goals | Yes |
| 6 | Recording rig: UnitV + XIAO co-mounted (features + high-res frames, time-synced) | Yes; needs the owner's platform choice |
| 7 | Training: frozen frontend, precompute features (fp32 + KPU-exact validation), heads + bottleneck trained jointly; QVGA first; fp32 TFLite → int8 TFLite → qi8 loader | Yes |
| 8 | Loader additions only when a head needs them: L2 norm in Go now; RESIZE_BILINEAR/TRANSPOSE_CONV for decoders later | Yes |
| 9 | Radio for untethered runs (§7a): USB on the bench now, plus ELRS for drive, e-stop and status; Wi-Fi or ESP-NOW later only if the owner lifts "no Wi-Fi" for this robot | **Owner's call**; I'd start with USB + ELRS |
| 10 | Stop path W3s is a hard prerequisite of W4 (300 ms stale-data stop, e-stop button + PC key + RC failsafe, bumper, speed cap, floor only); done when pulling the K210 UART lead stops the wheels within 300 ms | Yes |
| 11 | Telemetry and overlay (W1b) before any autonomous run; LED codes as defined | Yes |
| 12 | Recording: sessions under `_data/vision-nav/` with a manifest, LED-flash sync with a drift check, calibration id from `cmd/calibrate` (W2a); SD rate limits high-res to sparse anchors unless tethered | Yes |
| 13 | Baseline ladder W2b (the owner's notebook → our recordings → our features → bottleneck) and run tracking with fixed held-out sets; the owner picks the frontend after rung (c) | Yes |
| 14 | Feature version id in link headers, head files, manifests and map nodes; mismatches refused; matched-release flashing with a recorded last-good pair | Yes |
| 15 | Robot hardware prerequisites for W2–W4 (bumper/ToF, cliff sensor or floor-only, encoders, e-stop button, battery for one session) | Owner's call, with the platform choice |

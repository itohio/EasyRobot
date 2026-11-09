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

Brief: `~/src/docs/tasks/O-007-vision-nav-research.md`, objective O-007 (`docs/objectives/O-007-vision-navigation.md`, the owner's words verbatim). Refines [[W4-k230-feature-stream]] and [[W5-feature-tensor-experiments]]. Builds on [[2026-11-09-k210-camera-nn]] and [[2026-11-09-on-device-learning]].

**Scope:** research, not redesign. The owner has working notebooks for P1 (odometry) and P2 (depth) with "great results", so this note doesn't redesign them. It asks what the literature and existing systems offer for **his** architecture on **his** hardware, and where the risks are.

**Status:** recommendations. Nothing was built or run for this note.

**Labels:** observed (read in our evidence or code this session), reported (a paper or project page says so), inferred, `#verify`.

**Network (read-only):** 8 web searches (ViNG/GNM/NoMaD; ViNG's arXiv id; compact VPR descriptors; NetVLAD; Monodepth2/SfMLearner; DBoW2; Zhan/Shu feature losses; learned-VO generalisation) and 3 page reads (arXiv abstracts of NoMaD 2310.07896 and ViNG 2012.09812; NetVLAD on ar5iv.labs.arxiv.org). No datasets or models downloaded.

## Verdict

1. **Frontend:** use a **MobileNetV1 α 0.5 (or MobileNetV2 α ≤ 0.5) at QVGA on the K210 KPU**, cut at stride 16.
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
| Memory | 6 MiB + 2 MiB AI SRAM | 512 KB SRAM; 8 MB PSRAM **not yet working in TinyGo** (driver built, two bugs found by verify, board test pending) |

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

**Platform:** the wheeled avoider (owner question in O-007; the EasyRobot roadmap's first wheeled robot). Both camera boards are mounted on it.

| Step | Done when | Kill / rethink if |
|---|---|---|
| W0 KPU frontend (K8c) | MobileNetV1 α 0.5 QVGA cut at stride 16 + bottleneck runs on the KPU from camera frames; outputs within 2 LSB of the simulator | < 5 fps, or features off by more than a few LSB → try α 0.25 / MobileNetV2 α 0.35, or 160×120 input |
| W1 Link | K210 → ESP32 UART with W4 framing; the bottleneck tensor arrives CRC-clean for 10 min | < 10 fps for 4.8 KB frames → 10×8×32 bottleneck or a higher baud after measuring |
| W2 Data | 1 h of recordings in 2–3 rooms: K210 features + XIAO frames (QVGA + high-res) + wheel ticks if any + bumper events | sync drift > 1 frame |
| W3 Free-space head | per-column free distance (labels from the owner's depth model on XIAO frames, or bumper/ToF), int8 on the S3 | column accuracy < 85 % on a held-out room |
| W4 Avoid | reactive controller (steer to the freest column, slow on low clearance) on the S3 | > 1 collision per 10 min in the test room at 0.2 m/s |
| W5 Map | a place code per keyframe (start: mean-pooled bottleneck + PCA from the host, 64 B); nodes and edges from wheel/VO on the S3; graph dumped to the PC | revisit recall@1 < 70 % in one room → a trained code head (§2.3) |
| W6 (after the win) | VO head from the owner's notebook; relocalisation (§3); PC goal search (§4) | — |

This slice uses none of the unbuilt pieces (no depth decoder, no PSRAM). It proves the frontend, the split, the link and one head end to end.

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
| Known unknowns | K210 ↔ ESP32 UART rate without FTDI | k210 + esp32cam, a link test (W1) |
| Known unknowns | XIAO PSRAM on silicon | esp32cam, E6 board test |
| Known unknowns | Feature domain gap OV2640 vs OV3660 | W2 paired recordings |
| Unknown knowns | The owner's notebooks: frontend, cut layer, losses, label scale | owner shares them (`~/src/docs/research/vision-nav/`) |
| Unknown knowns | The robot platform and whether it has wheel encoders (scale for VO) | owner |
| Unknown unknowns | Whether learned VO from a frozen classification frontend holds up indoors on our robot | W6 with the owner's notebook |

## Decisions for Leader

| # | Decision | Recommendation |
|---|---|---|
| 1 | Frontend: MobileNetV1 α 0.5 at QVGA on the KPU, cut at stride 16, with a learned 1×1 bottleneck (16–32 ch) on the KPU; MobileNetV2 α ≤ 0.5 as second choice; no MobileNetV3/EfficientNet-Lite (KPU limits) | Yes, subject to the owner's notebook's frontend |
| 2 | Split: K210 = frontend + bottleneck; ESP32-S3 = heads, map, controller; PC = goal search, depth, viewer | Yes |
| 3 | Link: UART over Grove with W4 framing and resend; plan on 64 KB/s until W1 measures the direct link | Yes |
| 4 | First win = §7 (free-space head + avoidance + place-code map) with its kill criteria; depth and segmentation wait | Yes |
| 5 | Place codes: learned 32–128 int8 B per place, per-cell grids stored at nodes for object goals | Yes |
| 6 | Recording rig: UnitV + XIAO co-mounted (features + high-res frames, time-synced) | Yes; needs the owner's platform choice |
| 7 | Training: frozen frontend, precompute features (fp32 + KPU-exact validation), heads + bottleneck trained jointly; QVGA first; fp32 TFLite → int8 TFLite → qi8 loader | Yes |
| 8 | Loader additions only when a head needs them: L2 norm in Go now; RESIZE_BILINEAR/TRANSPOSE_CONV for decoders later | Yes |

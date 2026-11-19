---
type: moc
status: active
tags: [easyrobot, moc]
created: 2026-11-08
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# EasyRobot — knowledge vault

Home map of the EasyRobot Go repo ([itohio/EasyRobot](https://github.com/itohio/EasyRobot), the owner's maintained fork of the archived [foxis/EasyRobot](https://github.com/foxis/EasyRobot); MIT). Notes describe HEAD `abb5c9c` unless a note's `source_revision` says otherwise. How the vault works and how to keep it true after a fix: [[conventions]]. Regenerate with `make docs`, check with `make docs-check`.

## Next — the owner's order (2026-11-19)

Copied from [[roadmap#8. Next|roadmap §8]], which has the critical path and state of each line.

| # | Outcome | Tickets |
|---|---|---|
| 1 | YOLO on the K210: the UnitV streams its camera image with detections over USB-C to a static web page that draws them, live | [[ER-069-yolo-k210-tinygo-cpu\|ER-069]] (CPU: one frame to the page, then a continuous stream), then [[ER-070-yolo-k210-kpu\|ER-070]] (speed to video) |
| 2 | YOLO on the ESP32-S3: the same detections and log from the XIAO S3's camera, with optimised kernels | [[ER-071-yolo-esp32s3\|ER-071]] |
| 3 | Motion estimators as bricks (eskf, Mahony) for IMU-only dead reckoning | committed; nothing open |
| 4 | Water: a faster 2-D liquid engine, then a 3-D multi-liquid engine | [[ER-061-sim-liquid-2d\|ER-061]], then [[ER-064-sim-liquid-3d\|ER-064]] |

<!-- vault:status h=4820323d2099 -->

State: 75 tested · 29 implemented · 137 partial · 10 stub · 13 unverified (264 algorithms)

<!-- /vault:status -->

## Start here

- Architecture: [[overview]] (layers, dependency diagram) · [[packages]] (all 200 package dirs) · [[data-flow]] · [[dndm-and-protobuf]] · [[build-targets]] · [[source-inventory|tensor/graph source inventory]]
- Work: [[todo]] (generated, by ticket) · [[tickets/index|ticket index]] · [[roadmap]] · [[known-unknown|known/unknown map]]
- Reviews: IK and Jacobian review · catalog review · knowledge engine v1 · vault maintenance · host baseline 2026-11-08

## Workstreams

| # | Note | State in one line |
|---|---|---|
| W1 | [[W1-knowledge-vault]] | This vault and its generator (`docs/tools/vault`). |
| W2 | [[W2-code-optimisation]] | Asm/BCE experiments exist; the faster GEMM loop order found in `x/iter` is not ported; amd64-only asm breaks arm64. |
| W3 | [[W3-legged-ik]] | FK positions and the Jacobian are correct; the pose quaternion is inverted; IK solvers fail silently or stall; gait scheduling, strides and timing not written. |
| W4 | [[W4-k230-feature-stream]] | No K230 code; camera capture, SPI abstraction and tensor serialisation exist in pieces; no SPI slave, no SD recording. |
| W5 | [[W5-feature-tensor-experiments]] | Vision baselines (phase correlation, pose, matching) are broken or stubs; notebooks not in the repo. |
| W6 | [[W6-decision-trees]] | Hand-written closure trees evaluate; no learning, no plain-data tree, no MCU executor. |
| W7 | [[W7-filters-codecs-voice]] | Noise figures come from modem acquisition/sync and test design, not symbol detection; RS is a placeholder; CLI coding path crashes. |
| W8 | [[W8-lidar]] | LD06 and XWPFTB drivers never emit a correct scan; RP2040 firmware does not compile; lidar model unidentified. |
| W9 | [[W9-go-nn-framework]] | Eager tensors pass tests but views, norm gradients and GPT layers are wrong; Qwen 0.5B is 31× ESP32-S3 PSRAM even at int4. |
| W10 | [[knowledge-engine]] | `x/knowledge` (own module): ingest + hybrid lexical/graph retrieval for a consuming service (v1). |

## Modules

Algorithm notes per module by status (generated from the algorithm notes; `tested` needs passing value tests).

<!-- vault:modules h=b1270d34110e -->

| Module | tested | implemented | partial | stub | unverified | total |
|---|---|---|---|---|---|---|
| **Control and planning** | | | | | | |
| [[graph-and-planning]] | 7 | 1 | 9 | 0 | 0 | 17 |
| [[kinematics]] | 7 | 5 | 2 | 0 | 0 | 14 |
| [[motion-and-gait]] | 5 | 0 | 2 | 1 | 0 | 8 |
| [[navigation]] | 5 | 0 | 0 | 0 | 0 | 5 |
| [[pid-control]] | 1 | 0 | 0 | 0 | 0 | 1 |
| **Signal, codecs, audio** | | | | | | |
| [[audio-io]] | 1 | 1 | 1 | 0 | 1 | 4 |
| [[audio-modem-cli]] | 0 | 0 | 2 | 0 | 0 | 2 |
| [[bit-buffer]] | 2 | 0 | 2 | 0 | 0 | 4 |
| [[dsp-core]] | 4 | 0 | 7 | 0 | 0 | 11 |
| [[filters]] | 2 | 2 | 11 | 1 | 0 | 16 |
| [[line-coding-and-fec]] | 2 | 2 | 6 | 1 | 0 | 11 |
| [[modulation]] | 0 | 4 | 14 | 2 | 0 | 20 |
| [[peg-protocol-parser]] | 0 | 0 | 2 | 0 | 0 | 2 |
| **Tensor and NN** | | | | | | |
| [[iter-workers]] | 0 | 1 | 3 | 0 | 0 | 4 |
| [[learn]] | 2 | 2 | 1 | 0 | 0 | 5 |
| [[mat-vec]] | 5 | 0 | 5 | 0 | 0 | 10 |
| [[nn]] | 2 | 3 | 10 | 0 | 0 | 15 |
| [[primitive-kernels]] | 3 | 0 | 3 | 1 | 0 | 7 |
| [[tensor]] | 6 | 1 | 11 | 0 | 1 | 19 |
| [[tflite]] | 0 | 0 | 0 | 0 | 1 | 1 |
| [[type-conversion]] | 0 | 0 | 1 | 1 | 0 | 2 |
| **Math and maps** | | | | | | |
| [[colorscience]] | 2 | 3 | 6 | 1 | 0 | 12 |
| [[gps-geodesy]] | 0 | 1 | 3 | 0 | 0 | 4 |
| [[grid-maps]] | 3 | 2 | 2 | 0 | 0 | 7 |
| [[interpolation]] | 5 | 0 | 3 | 0 | 0 | 8 |
| [[simulation-algorithms]] | 3 | 0 | 0 | 0 | 0 | 3 |
| **Devices and apps** | | | | | | |
| [[actuators]] | 0 | 1 | 4 | 0 | 1 | 6 |
| [[apps]] | 0 | 0 | 0 | 0 | 0 | 0 |
| [[crsf-elrs]] | 3 | 0 | 5 | 0 | 0 | 8 |
| [[device-io]] | 0 | 0 | 0 | 0 | 0 | 0 |
| [[lidar]] | 0 | 0 | 3 | 0 | 0 | 3 |
| [[nmea]] | 1 | 0 | 1 | 0 | 0 | 2 |
| [[sensors]] | 1 | 0 | 4 | 0 | 2 | 7 |
| **IO, camera, vision** | | | | | | |
| [[camera-io]] | 0 | 0 | 0 | 0 | 4 | 4 |
| [[marshallers]] | 3 | 0 | 5 | 0 | 0 | 8 |
| [[pipeline-and-chain]] | 0 | 0 | 1 | 0 | 0 | 1 |
| [[store-and-plugin]] | 0 | 0 | 0 | 0 | 0 | 0 |
| [[vision]] | 0 | 0 | 8 | 2 | 3 | 13 |
| **Knowledge engine** | | | | | | |
| [[knowledge-engine]] | 0 | 0 | 0 | 0 | 0 | 0 |
| **All** | 75 | 29 | 137 | 10 | 13 | 264 |

<!-- /vault:modules -->

The knowledge engine is its own Go module and is catalogued in [[knowledge-engine]].

## Open tickets

Generated from the ticket notes; "Open p1" counts the distinct open `#p1` TODOs tagged with the ticket. The defects themselves are in the linked notes and in [[todo]].

<!-- vault:tickets h=0500b0190531 -->

| Ticket | Title | Status | Workstream | Open p1 |
|---|---|---|---|---|
| [[ER-002-cpu-graph-semantics\|ER-002]] | Establish CPU graph and tensor semantics | ready | W9 | 0 |
| [[ER-003-operator-capabilities\|ER-003]] | Map actual operator dtype and layout capabilities | backlog | W9 | 0 |
| [[ER-004-device-quantisation-contract\|ER-004]] | Define device buffer transfer and quantisation ownership | blocked | W9 | 0 |
| [[ER-005-xtensa-feasibility\|ER-005]] | Investigate the exact Xtensa acceleration target | backlog | W9 | 0 |
| [[ER-006-xtensa-kernel-adapter\|ER-006]] | Implement one verified Xtensa kernel adapter | blocked | W9 | 0 |
| [[ER-007-k230-feasibility\|ER-007]] | Investigate K230 NPU compiler and runtime feasibility | backlog | W4 | 0 |
| [[ER-008-k230-subgraph-adapter\|ER-008]] | Integrate one verified K230 NPU subgraph | blocked | W4 | 0 |
| [[ER-009-k510-feasibility\|ER-009]] | Investigate K510 NPU independently | backlog | W9 | 0 |
| [[ER-010-k510-subgraph-adapter\|ER-010]] | Integrate one verified K510 NPU subgraph | blocked | W9 | 0 |
| [[ER-011-local-small-llm-reference\|ER-011]] | Select and establish a small local CPU LLM reference | backlog | W9 | 0 |
| [[ER-012-decoder-coverage\|ER-012]] | Map selected decoder operators and state to EasyRobot | backlog | W9 | 0 |
| [[ER-013-attention-kv-cpu-fixture\|ER-013]] | Prove a tiny CPU attention and KV graph fixture | blocked | W9 | 0 |
| [[ER-014-small-llm-native-graph\|ER-014]] | Run a bounded small local decoder through EasyRobot graph | blocked | W9 | 0 |
| [[ER-015-measured-backend-comparison\|ER-015]] | Measure approved backend costs without hidden fallback | blocked | W9 | 0 |
| [[ER-016-serialization-boundaries\|ER-016]] | Prove graph tensor serialization and resume boundaries | blocked | W9 | 0 |
| [[ER-019-filter-correctness\|ER-019]] | Filter correctness | backlog | W7 | 15 |
| [[ER-020-eager-views-newas-norm-gradients\|ER-020]] | Eager tensor views, NewAs aliasing and norm gradients | backlog | W9 | 38 |
| [[ER-021-mat-vec-quaternion-qr-inverse\|ER-021]] | mat/vec: quaternion from matrix, QR and inverse tolerance | backlog | W3, W9 | 8 |
| [[ER-022-audio-modem-acquisition\|ER-022]] | Audio modem acquisition, sync, drift and BER tests | blocked | W7 | 54 |
| [[ER-023-coding-fixes\|ER-023]] | Coding: GF(256) RS, CRC catalogue, line codes, LFSR, Gray, PEG | backlog | W7 | 17 |
| [[ER-024-lidar-decoder-fixes\|ER-024]] | Lidar decoder fixes | blocked | W8 | 9 |
| [[ER-025-device-fixes\|ER-025]] | Device fixes: encoder, PCA9685, servo array, NMEA, serial | backlog | W3, W8 | 12 |
| [[ER-026-build-green\|ER-026]] | Build green: host part, then TinyGo stage | ready | all | 11 |
| [[ER-027-marshaller-fixes\|ER-027]] | Marshallers: proto small ints, json/yaml ints, graph node data | backlog | W4 | 10 |
| [[ER-028-graph-fixes\|ER-028]] | Graph: cycle checks, detached handles, RRT*, decision-tree Balance | backlog | W6 | 17 |
| [[ER-029-nn-layers-fixture-harness\|ER-029]] | NN layer fixes and reference-fixture harness | backlog | W9 | 21 |
| [[ER-030-vision-compile-phase-correlation\|ER-030]] | Vision compile fixes and phase-correlation maths | backlog | W5 | 12 |
| [[ER-031-spi-slave-feature-framing\|ER-031]] | SPI slave interface and feature-frame SPI framing | backlog | W4 | 0 |
| [[ER-032-flat-decision-tree-executor\|ER-032]] | Plain-data decision tree and flat executor | backlog | W6 | 0 |
| [[ER-033-cart-trainer-episode-logger\|ER-033]] | CART trainer and episode logger | backlog | W6 | 0 |
| [[ER-034-watcher-crash-test\|ER-034]] | Knowledge engine: fsnotify watcher, crash-mid-ingest test, missing-target index | backlog | W10 | 0 |
| [[ER-035-chunkers\|ER-035]] | Knowledge engine: py/c/ts/proto chunkers | backlog | W10 | 0 |
| [[ER-036-find-service\|ER-036]] | Knowledge engine: expose find to a consuming service | backlog | W10 | 0 |
| [[ER-037-body-pose-static-stability\|ER-037]] | Body pose integration and static stability | backlog | W3 | 0 |
| [[ER-038-wheeled-motion-fixes\|ER-038]] | Wheeled motion fixes: path-tracking signs and steer4 geometry | backlog | W3, W6 | 3 |
| [[ER-039-manipulator-integration\|ER-039]] | Manipulator integration: units and helpers in drivers/cmd manipulator | backlog | W3 | 1 |
| [[ER-042-k210-led-buttons\|ER-042]] | UnitV WS2812 LED and buttons | backlog | K210-camera | 0 |
| [[ER-043-k210-camera-identity\|ER-043]] | UnitV camera identity via DVP SCCB | ready | K210-camera | 0 |
| [[ER-044-k210-frame-over-uart\|ER-044]] | UnitV one camera frame over UART | ready | K210-camera | 0 |
| [[ER-045-k210-native-int8-benchmark\|ER-045]] | Native int8 conv/dot/GEMV benchmark on one K210 hart | blocked | K210-nn | 0 |
| [[ER-046-k210-kpu-kmodel-runner\|ER-046]] | KPU kmodel v3 run through a pure-Go runner | ready | K210-nn | 0 |
| [[ER-047-k210-feature-tensor-out\|ER-047]] | UnitV camera → KPU → feature tensor stream | blocked | K210-camera, K210-nn, W4 | 0 |
| [[ER-048-sim-proto-contract\|ER-048]] | Sim S1: `types/sim` protobuf contract — Command, State, ModelLoad | proposed | W3, W6 | 0 |
| [[ER-049-sim-go-interface\|ER-049]] | Sim S1: Go `Sim` interface and a backend conformance suite | proposed | W3, W6 | 0 |
| [[ER-050-sim-lockstep-client\|ER-050]] | Sim S1: lockstep client — `Sim` over length-prefixed protobuf on a Unix socket or TCP | proposed | W3, W6 | 0 |
| [[ER-051-sim-mujoco-sidecar\|ER-051]] | Sim S1: MuJoCo sidecar — load MJCF, step in lockstep, fill State, headless with an optional viewer | proposed | W3, W6 | 0 |
| [[ER-052-sim-robot-description\|ER-052]] | Sim S1: robot description structs — a wheeled base with masses and motor torque and speed | proposed | W3, W6 | 0 |
| [[ER-053-sim-mjcf-exporter\|ER-053]] | Sim S1: MJCF exporter — wheeled description → MJCF bytes with named actuators and sensors | proposed | W3, W6 | 0 |
| [[ER-054-sim-sensor-models\|ER-054]] | Sim S1: sensor models — IMU noise and wheel-encoder quantisation from ground truth | proposed | W3, W6 | 0 |
| [[ER-055-sim-kinematic-fake\|ER-055]] | Sim S1: kinematic fake backend — a pure-Go differential integrator implementing `Sim` for CI | proposed | W3, W6 | 0 |
| [[ER-056-sim-determinism-speed\|ER-056]] | Sim S1: determinism and speed checks with the note's kill criteria | proposed | W3, W6 | 0 |
| [[ER-057-sim-wheeled-controller-adapter\|ER-057]] | Sim S1: `simbase` — the sim implementation of `wheeled.Base` and `Clock` over `Sim` | proposed | W3, W6 | 0 |
| [[ER-058-sim-s1-wheeled-run\|ER-058]] | Sim S1: wheeled run, the house — a differential robot driven through `wheeled.Base` with logged results, teleop and kill criteria | proposed | W3, W6 | 0 |
| [[ER-059-sim-robot-base-and-clock\|ER-059]] | Sim S1: robot-facing `wheeled.Base` interface and `Clock` — the seam between sim and real robot | proposed | W3, W6 | 0 |
| [[ER-060-sim-wheel-odometry\|ER-060]] | Sim S1: wheel odometry — encoder ticks → 2D pose from the robot description | proposed | W3, W6 | 0 |
| [[ER-061-sim-liquid-2d\|ER-061]] | 2D liquid engine: PIC/FLIP particles on a grid, fixed point — x/math/fluid/flip | in-progress |  | 1 |
| [[ER-062-sim-game-of-life\|ER-062]] | Game of Life, bit-packed (port from a badge demo) — x/math/automata/life | blocked |  | 0 |
| [[ER-064-sim-liquid-3d\|ER-064]] | 3D liquid automaton driven by the IMU's continuous gravity vector | proposed |  | 0 |
| [[ER-065-sim-fluid-dynamics\|ER-065]] | Fluid dynamics: Stam stable fluids, 2D first — x/math/fluid | proposed |  | 0 |
| [[ER-066-ray-casting-dda\|ER-066]] | Ray casting: grid DDA traversal and a 2.5D column helper — x/math/grid | proposed |  | 0 |
| [[ER-067-strapdown-motion-path\|ER-067]] | Strapdown integrator with ZUPT for the badge's 3D motion path — x/math/nav/strapdown | active |  | 1 |
| [[ER-068-startup-calibration\|ER-068]] | Startup calibration of the motion sensor at rest — x/math/nav/calib | active |  | 0 |
| [[ER-069-yolo-k210-tinygo-cpu\|ER-069]] | YOLOX-nano on the UnitV in TinyGo: camera frames with detections streamed over USB to a static web page (CPU path) | active | K210-nn, W5 | 0 |
| [[ER-070-yolo-k210-kpu\|ER-070]] | YOLO on the K210's KPU: graph passes so the compiler runs long KPU segments, then fps on the UnitV | proposed | K210-nn, W5 | 0 |
| [[ER-071-yolo-esp32s3\|ER-071]] | YOLOX-nano on the ESP32-S3 in TinyGo: first run, then optimised kernels, detections from camera frames | blocked | W2, W5, W9 | 0 |

<!-- /vault:tickets -->

## Ownership and limits

Algorithm notes were written from reading, tests and throwaway probes outside the repo; vision, TFLite, V4L2 and all firmware paths are judged by reading and type-checking only (no OpenCV, no TinyGo, no hardware).

**Optional reference, not integration proof:** a separate local-model experiment outside this repo can inform memory-conditioned control later. Nothing here couples it to EasyRobot.

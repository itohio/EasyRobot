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

## Next (ranked) — proposed, owner to approve

From the roadmap's no-hardware demo checkpoints ([[roadmap#8. Next|roadmap §8]]). None of these needs hardware, TinyGo or a download.

| # | Step | Tickets | Demo checkpoint |
|---|---|---|---|
| 1 | ~~Knowledge engine v1 committed after verify, with the `go.mod` tidy as its own commit~~ — committed | KE-1 (report (evidence: `~/src/docs/evidence/KE-1.md`, outside the repo)) | M-g: KE-1 bench table vs the ripgrep baseline |
| 2 | ~~Leg IK and gait scheduling with strides, body→leg transforms, gait→IK→servo glue~~ — done, see report (evidence: `~/src/docs/evidence/ER-017-018.md`, outside the repo) | [[ER-017-dh-planar-ik-robustness\|ER-017]], [[ER-018-leg-helpers-gait-scheduler\|ER-018]] | M-a: host tripod simulation, stance feet fixed in the world — passes |
| 3 | Voice-modem composition fixes (stale `tc` test port first), then BER harness, synchroniser, framer, FEC | ER-AUDIO-1, then [[ER-022-audio-modem-acquisition\|ER-022]], [[ER-023-coding-fixes\|ER-023]] (with [[ER-019-filter-correctness\|ER-019]]'s PLL/AGC) | M-e: BER-vs-Eb/N0 curve on a channel model |
| 4 | Trusted reference: eager views and aliasing, graph fixes, existing-layer fixes and the reference-fixture harness | [[ER-002-cpu-graph-semantics\|ER-002]], [[ER-020-eager-views-newas-norm-gradients\|ER-020]], [[ER-028-graph-fixes\|ER-028]], then [[ER-029-nn-layers-fixture-harness\|ER-029]] | M-f: eager fixtures pass |
| 5 | Decision trees on the host: plain-data tree, flat executor, CART trainer, episode logger | [[ER-032-flat-decision-tree-executor\|ER-032]], [[ER-033-cart-trainer-episode-logger\|ER-033]] | M-d: host-trained tree classifies a recorded or synthetic obstacle set |

The sixth checkpoint, M-b's synthetic recorder round trip, follows: it needs [[ER-027-marshaller-fixes|ER-027]] and [[ER-031-spi-slave-feature-framing|ER-031]]. Full order, accepted order changes and the T-IK mapping: [[tickets/index|ticket index]].

<!-- vault:status h=244d66d9ba5b -->

State: 66 tested · 29 implemented · 138 partial · 10 stub · 13 unverified (256 algorithms)

<!-- /vault:status -->

## Start here

- Architecture: [[overview]] (layers, dependency diagram) · [[packages]] (all 200 package dirs) · [[data-flow]] · [[dndm-and-protobuf]] · [[build-targets]] · [[source-inventory|tensor/graph source inventory]]
- Work: [[todo]] (generated, by ticket) · [[tickets/index|ticket index]] · [[roadmap]] · [[known-unknown|known/unknown map]]
- Reviews: IK and Jacobian review (evidence: `~/src/docs/evidence/ER-R2-ik-review.md`, outside the repo) · ER-R2 report (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · KE-1 knowledge engine (evidence: `~/src/docs/evidence/KE-1.md`, outside the repo) · vault maintenance (ER-docs-fix) (evidence: `~/src/docs/evidence/ER-docs-fix.md`, outside the repo) · host baseline 2026-11-08 (evidence: `~/src/docs/evidence/baseline-2026-11-08.md`, outside the repo)

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
| W10 | [[knowledge-engine]] | `x/knowledge` (own module): ingest + hybrid lexical/graph retrieval for Abugaga (KE-1). |

## Modules

Algorithm notes per module by status (generated from the algorithm notes; `tested` needs passing value tests).

<!-- vault:modules h=604f7b1df024 -->

| Module | tested | implemented | partial | stub | unverified | total |
|---|---|---|---|---|---|---|
| **Control and planning** | | | | | | |
| [[graph-and-planning]] | 7 | 1 | 9 | 0 | 0 | 17 |
| [[kinematics]] | 7 | 5 | 2 | 0 | 0 | 14 |
| [[motion-and-gait]] | 5 | 0 | 2 | 1 | 0 | 8 |
| [[pid-control]] | 1 | 0 | 0 | 0 | 0 | 1 |
| **Signal, codecs, audio** | | | | | | |
| [[audio-io]] | 1 | 1 | 1 | 0 | 1 | 4 |
| [[audio-modem-cli]] | 0 | 0 | 2 | 0 | 0 | 2 |
| [[bit-buffer]] | 2 | 0 | 2 | 0 | 0 | 4 |
| [[dsp-core]] | 4 | 0 | 7 | 0 | 0 | 11 |
| [[filters]] | 2 | 2 | 12 | 1 | 0 | 17 |
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
| **Devices and apps** | | | | | | |
| [[actuators]] | 0 | 1 | 4 | 0 | 1 | 6 |
| [[apps]] | 0 | 0 | 0 | 0 | 0 | 0 |
| [[crsf-elrs]] | 3 | 0 | 5 | 0 | 0 | 8 |
| [[device-io]] | 0 | 0 | 0 | 0 | 0 | 0 |
| [[lidar]] | 0 | 0 | 3 | 0 | 0 | 3 |
| [[nmea]] | 1 | 0 | 1 | 0 | 0 | 2 |
| [[sensors]] | 0 | 0 | 4 | 0 | 2 | 6 |
| **IO, camera, vision** | | | | | | |
| [[camera-io]] | 0 | 0 | 0 | 0 | 4 | 4 |
| [[marshallers]] | 3 | 0 | 5 | 0 | 0 | 8 |
| [[pipeline-and-chain]] | 0 | 0 | 1 | 0 | 0 | 1 |
| [[store-and-plugin]] | 0 | 0 | 0 | 0 | 0 | 0 |
| [[vision]] | 0 | 0 | 8 | 2 | 3 | 13 |
| **Knowledge engine** | | | | | | |
| [[knowledge-engine]] | 0 | 0 | 0 | 0 | 0 | 0 |
| **All** | 66 | 29 | 138 | 10 | 13 | 256 |

<!-- /vault:modules -->

The knowledge engine is its own Go module and is catalogued in [[knowledge-engine]].

## Open tickets

Generated from the ticket notes; "Open p1" counts the distinct open `#p1` TODOs tagged with the ticket. The defects themselves are in the linked notes and in [[todo]].

<!-- vault:tickets h=0e48eda73e07 -->

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
| [[ER-019-filter-correctness\|ER-019]] | Filter correctness | backlog | W7 | 17 |
| [[ER-020-eager-views-newas-norm-gradients\|ER-020]] | Eager tensor views, NewAs aliasing and norm gradients | backlog | W9 | 38 |
| [[ER-021-mat-vec-quaternion-qr-inverse\|ER-021]] | mat/vec: quaternion from matrix, QR and inverse tolerance | backlog | W3, W9 | 8 |
| [[ER-022-audio-modem-acquisition\|ER-022]] | Audio modem acquisition, sync, drift and BER tests | blocked | W7 | 54 |
| [[ER-023-coding-fixes\|ER-023]] | Coding: GF(256) RS, CRC catalogue, line codes, LFSR, Gray, PEG | backlog | W7 | 17 |
| [[ER-024-lidar-decoder-fixes\|ER-024]] | Lidar decoder fixes | blocked | W8 | 9 |
| [[ER-025-device-fixes\|ER-025]] | Device fixes: encoder, PCA9685, servo array, NMEA, serial | backlog | W3, W8 | 13 |
| [[ER-026-build-green\|ER-026]] | Build green: host part, then TinyGo stage | ready | all | 11 |
| [[ER-027-marshaller-fixes\|ER-027]] | Marshallers: proto small ints, json/yaml ints, graph node data | backlog | W4 | 10 |
| [[ER-028-graph-fixes\|ER-028]] | Graph: cycle checks, detached handles, RRT*, decision-tree Balance | backlog | W6 | 17 |
| [[ER-029-nn-layers-fixture-harness\|ER-029]] | NN layer fixes and reference-fixture harness | backlog | W9 | 21 |
| [[ER-030-vision-compile-phase-correlation\|ER-030]] | Vision compile fixes and phase-correlation maths | backlog | W5 | 12 |
| [[ER-031-spi-slave-feature-framing\|ER-031]] | SPI slave interface and feature-frame SPI framing | backlog | W4 | 0 |
| [[ER-032-flat-decision-tree-executor\|ER-032]] | Plain-data decision tree and flat executor | backlog | W6 | 0 |
| [[ER-033-cart-trainer-episode-logger\|ER-033]] | CART trainer and episode logger | backlog | W6 | 0 |
| [[ER-034-ke2-watcher-crash-test\|ER-034]] | KE-2: fsnotify watcher, crash-mid-ingest test, missing-target index | backlog | W10 | 0 |
| [[ER-035-ke3-chunkers\|ER-035]] | KE-3: py/c/ts/proto chunkers | backlog | W10 | 0 |
| [[ER-036-ke4-abugaga-find\|ER-036]] | KE-4: Abugaga find (abg.svc.find, roster_find, abu find) | backlog | W10 | 0 |
| [[ER-037-body-pose-static-stability\|ER-037]] | Body pose integration and static stability | backlog | W3 | 0 |
| [[ER-038-wheeled-motion-fixes\|ER-038]] | Wheeled motion fixes: path-tracking signs and steer4 geometry | backlog | W3, W6 | 3 |
| [[ER-039-manipulator-integration\|ER-039]] | Manipulator integration: units and helpers in drivers/cmd manipulator | backlog | W3 | 1 |
| [[ER-042-k210-led-buttons\|ER-042]] | UnitV WS2812 LED and buttons | backlog | K210-camera | 0 |
| [[ER-043-k210-camera-identity\|ER-043]] | UnitV camera identity via DVP SCCB | ready | K210-camera | 0 |
| [[ER-044-k210-frame-over-uart\|ER-044]] | UnitV one camera frame over UART | ready | K210-camera | 0 |
| [[ER-045-k210-native-int8-benchmark\|ER-045]] | Native int8 conv/dot/GEMV benchmark on one K210 hart | blocked | K210-nn | 0 |
| [[ER-046-k210-kpu-kmodel-runner\|ER-046]] | KPU kmodel v3 run through a pure-Go runner | blocked | K210-nn | 0 |
| [[ER-047-k210-feature-tensor-out\|ER-047]] | UnitV camera → KPU → feature tensor stream | blocked | K210-camera, K210-nn, W4 | 0 |

<!-- /vault:tickets -->

## Ownership and limits

The `easyrobot` peer of the personal mp roster owns `docs/**`. Algorithm notes were written from reading, tests and throwaway probes under `~/src/tmp/easyrobot/`; vision, TFLite, V4L2 and all firmware paths are judged by reading and type-checking only (no OpenCV, no TinyGo, no hardware).

**Optional reference, not integration proof:** `experience-learning-agent` is a separate local-model experiment in the personal workspace; its `docs/idea.md` and `docs/abstraction.md` can inform memory-conditioned control later. Nothing here couples it to EasyRobot.

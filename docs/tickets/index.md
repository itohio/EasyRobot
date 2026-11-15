---
type: ticket
status: active
tags: [easyrobot, ticket, index]
created: 2026-11-08
updated: 2026-11-12
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# EasyRobot tickets

[[index|Home]] · [[roadmap]] · [[known-unknown|Unknown map]] · Host baseline · [[conventions]]

Goal (2026-11-08): bring EasyRobot up to date so it can be improved further.

One ticket-ID scheme: `ER-NNN`, one file each in `tickets/`. The earlier IK proposals and the knowledge-engine slices after v1 are absorbed into these tickets (IK: the table at the end). TODO lines tag their ticket as `#ER-0xx` or `#backlog` ([[conventions#TODO lines|TODO lines]]). States: `done` = acceptance evidenced in the ticket; `ready` = dependencies met and acceptance concrete, waiting for scheduling; `backlog` = proposed, not execution permission; `blocked` = prerequisites or owner input unmet. `proposed` = written and verified as a plan, not yet scheduled (moves to `ready` when it is scheduled).

## All tickets

| Ticket | Title | Kind | State | Owner | Workstream | Milestone | Depends |
|---|---|---|---|---|---|---|---|
| [[ER-001-reference-environment\|ER-001]] | Discover a reproducible CPU reference environment | discovery | done (darwin/arm64 host) | EasyRobot | all | all | none |
| [[ER-002-cpu-graph-semantics\|ER-002]] | Establish CPU graph and tensor semantics | implementation | ready | EasyRobot | W9 | M-f | ER-001 ✓ |
| [[ER-003-operator-capabilities\|ER-003]] | Map actual operator dtype and layout capabilities | discovery | backlog | EasyRobot | W9 | M-f | ER-001 ✓ |
| [[ER-004-device-quantisation-contract\|ER-004]] | Define device buffer transfer and quantisation ownership | design | blocked | unassigned-personal-profile | W9 | M-f | ER-002, ER-003 |
| [[ER-005-xtensa-feasibility\|ER-005]] | Investigate the exact Xtensa acceleration target | discovery | backlog | unassigned-personal-profile | W9 | M-f | ER-001 ✓, ER-003 |
| [[ER-006-xtensa-kernel-adapter\|ER-006]] | Implement one verified Xtensa kernel adapter | implementation | blocked | unassigned-personal-profile | W9 | M-f | ER-004, ER-005 |
| [[ER-007-k230-feasibility\|ER-007]] | Investigate K230 NPU compiler and runtime feasibility | discovery | backlog | unassigned-personal-profile | W4 | M-b | ER-001 ✓, ER-003 |
| [[ER-008-k230-subgraph-adapter\|ER-008]] | Integrate one verified K230 NPU subgraph | implementation | blocked | unassigned-personal-profile | W4 | M-b | ER-004, ER-007 |
| [[ER-009-k510-feasibility\|ER-009]] | Investigate K510 NPU independently | discovery | backlog | unassigned-personal-profile | W9 | none | ER-001 ✓, ER-003 |
| [[ER-010-k510-subgraph-adapter\|ER-010]] | Integrate one verified K510 NPU subgraph | implementation | blocked | unassigned-personal-profile | W9 | none | ER-004, ER-009 |
| [[ER-011-local-small-llm-reference\|ER-011]] | Select and establish a small local CPU LLM reference | discovery | backlog | unassigned-personal-profile | W9 | M-f | ER-001 ✓ |
| [[ER-012-decoder-coverage\|ER-012]] | Map selected decoder operators and state to EasyRobot | discovery | backlog | unassigned-personal-profile | W9 | M-f | ER-003, ER-011 |
| [[ER-013-attention-kv-cpu-fixture\|ER-013]] | Prove a tiny CPU attention and KV graph fixture | implementation | blocked | unassigned-personal-profile | W9 | M-f | ER-002, ER-012 |
| [[ER-014-small-llm-native-graph\|ER-014]] | Run a bounded small local decoder through EasyRobot graph | implementation | blocked | unassigned-personal-profile | W9 | M-f | ER-004, ER-011, ER-013 |
| [[ER-015-measured-backend-comparison\|ER-015]] | Measure approved backend costs without hidden fallback | measurement | blocked | unassigned-personal-profile | W9 | M-f | ER-002, ER-003 |
| [[ER-016-serialization-boundaries\|ER-016]] | Prove graph tensor serialization and resume boundaries | implementation | blocked | unassigned-personal-profile | W9 | none | ER-001 ✓, ER-002 |
| [[ER-017-dh-planar-ik-robustness\|ER-017]] | DH and planar IK robustness | implementation | ready | EasyRobot | W3 | M-a | none |
| [[ER-018-leg-helpers-gait-scheduler\|ER-018]] | Leg/joint helpers, gait scheduler, strides and gait→IK→servo glue | implementation | ready | EasyRobot | W3 | M-a | ER-017 (parallel ok) |
| [[ER-019-filter-correctness\|ER-019]] | Filter correctness | implementation | backlog | EasyRobot | W7 | M-e | none (PLL/AGC before ER-022) |
| [[ER-020-eager-views-newas-norm-gradients\|ER-020]] | Eager tensor views, NewAs aliasing and norm gradients | implementation | backlog | EasyRobot | W9 | M-f | with ER-002 |
| [[ER-021-mat-vec-quaternion-qr-inverse\|ER-021]] | mat/vec: quaternion from matrix, QR and inverse tolerance | implementation | backlog | EasyRobot | [W3, W9] | M-a | none |
| [[ER-022-audio-modem-acquisition\|ER-022]] | Audio modem acquisition, sync, drift and BER tests | implementation | blocked | EasyRobot | W7 | M-e | voice-modem composition fixes, ER-019 (PLL/AGC), ER-023; owner channel choice |
| [[ER-023-coding-fixes\|ER-023]] | Coding: GF(256) RS, CRC catalogue, line codes, LFSR, Gray, PEG | implementation | backlog | EasyRobot | W7 | M-e | voice-modem composition fixes |
| [[ER-024-lidar-decoder-fixes\|ER-024]] | Lidar decoder fixes | implementation | blocked | EasyRobot | W8 | W8 | owner lidar ID + capture; ER-026 stage 2 |
| [[ER-025-device-fixes\|ER-025]] | Device fixes: encoder, PCA9685, servo array, NMEA, serial | implementation | backlog | EasyRobot | [W3, W8] | M-a | none (devices: ER-026 stage 2) |
| [[ER-026-build-green\|ER-026]] | Build green: host part, then TinyGo stage | implementation | ready (stage 1) / blocked (stage 2) | EasyRobot | all | all | stage 1 none; stage 2 TinyGo approval |
| [[ER-027-marshaller-fixes\|ER-027]] | Marshallers: proto small ints, json/yaml ints, graph node data | implementation | backlog | EasyRobot | W4 | M-b | none |
| [[ER-028-graph-fixes\|ER-028]] | Graph: cycle checks, detached handles, RRT*, decision-tree Balance | implementation | backlog | EasyRobot | W6 | M-d | with ER-002 |
| [[ER-029-nn-layers-fixture-harness\|ER-029]] | NN layer fixes and reference-fixture harness | implementation | backlog | EasyRobot | W9 | M-f | ER-002, ER-020, ER-028 |
| [[ER-030-vision-compile-phase-correlation\|ER-030]] | Vision compile fixes and phase-correlation maths | implementation | backlog | EasyRobot | W5 | M-c | none (OpenCV approval for gocv parts) |
| [[ER-031-spi-slave-feature-framing\|ER-031]] | SPI slave interface and feature-frame SPI framing | implementation | backlog | EasyRobot | W4 | M-b | ER-027, ER-023 |
| [[ER-032-flat-decision-tree-executor\|ER-032]] | Plain-data decision tree and flat executor | implementation | backlog | EasyRobot | W6 | M-d | ER-028 |
| [[ER-033-cart-trainer-episode-logger\|ER-033]] | CART trainer and episode logger | implementation | backlog | EasyRobot | W6 | M-d | ER-032 |
| [[ER-034-watcher-crash-test\|ER-034]] | Knowledge engine: fsnotify watcher, crash-mid-ingest test, missing-target index | implementation | backlog | EasyRobot | W10 | M-g | knowledge engine v1 |
| [[ER-035-chunkers\|ER-035]] | Knowledge engine: py/c/ts/proto chunkers | implementation | backlog | EasyRobot | W10 | M-g | knowledge engine v1 |
| [[ER-036-find-service\|ER-036]] | Knowledge engine: find call in a consuming service (API, client call, CLI) | implementation | backlog | consuming service | W10 | M-g | knowledge engine v1 |
| [[ER-037-body-pose-static-stability\|ER-037]] | Body pose integration and static stability | implementation | backlog | EasyRobot | W3 | M-a | ER-018 |
| [[ER-038-wheeled-motion-fixes\|ER-038]] | Wheeled motion fixes: path-tracking signs and steer4 geometry | implementation | backlog | EasyRobot | [W3, W6] | M-d | none |
| [[ER-039-manipulator-integration\|ER-039]] | Manipulator integration: units and helpers in drivers/cmd manipulator | implementation | backlog | EasyRobot | W3 | none | ER-017, ER-018 |
| [[ER-040-k210-unitv-target\|ER-040]] | UnitV board target for TinyGo (build only) | implementation | done | k210 | K210-camera | K210-M0 | none |
| [[ER-041-k210-flash-uart-hello\|ER-041]] | UnitV flashing path and UART hello | hardware | done | k210 | K210-camera | K210-M1 | ER-040 |
| [[ER-042-k210-led-buttons\|ER-042]] | UnitV WS2812 LED and buttons | implementation | backlog | k210 | K210-camera | K210-M2 | ER-041 |
| [[ER-043-k210-camera-identity\|ER-043]] | UnitV camera identity via DVP SCCB | implementation | ready | k210 | K210-camera | K210-M3 | ER-041 |
| [[ER-044-k210-frame-over-uart\|ER-044]] | UnitV one camera frame over UART | implementation | ready | k210 | K210-camera | K210-M4 | ER-043 |
| [[ER-045-k210-native-int8-benchmark\|ER-045]] | Native int8 conv/dot/GEMV benchmark on one K210 hart | measurement | blocked | k210 | K210-nn | K210-M5 | ER-041; int8 kernels |
| [[ER-046-k210-kpu-kmodel-runner\|ER-046]] | KPU kmodel v3 run through a pure-Go runner | implementation | ready | k210 | K210-nn | K210-M6 | ER-041; nncase approval |
| [[ER-047-k210-feature-tensor-out\|ER-047]] | UnitV camera → KPU → feature tensor stream | implementation | blocked | k210 | [K210-camera, K210-nn, W4] | K210-M7 | ER-044, ER-046 |
| [[ER-048-sim-proto-contract\|ER-048]] | Sim S1: `types/sim` protobuf contract — Command, State, ModelLoad | design + implementation | proposed | EasyRobot | [W3, W6] | M-d | none |
| [[ER-049-sim-go-interface\|ER-049]] | Sim S1: Go `Sim` interface and a backend conformance suite | implementation | proposed | EasyRobot | [W3, W6] | M-d | ER-048 |
| [[ER-050-sim-lockstep-client\|ER-050]] | Sim S1: lockstep client — `Sim` over length-prefixed protobuf on a Unix socket or TCP | implementation | proposed | EasyRobot | [W3, W6] | M-d | ER-048, ER-049 |
| [[ER-051-sim-mujoco-sidecar\|ER-051]] | Sim S1: MuJoCo sidecar — load MJCF, step in lockstep, fill State, headless with an optional viewer | implementation | proposed | EasyRobot | [W3, W6] | M-d | ER-048, ER-050, ER-053; download at S1 start |
| [[ER-052-sim-robot-description\|ER-052]] | Sim S1: robot description structs — a wheeled base with masses and motor torque and speed | implementation | proposed | EasyRobot | [W3, W6] | M-d | none |
| [[ER-053-sim-mjcf-exporter\|ER-053]] | Sim S1: MJCF exporter — wheeled description → MJCF bytes with named actuators and sensors | implementation | proposed | EasyRobot | [W3, W6] | M-d | ER-052 |
| [[ER-054-sim-sensor-models\|ER-054]] | Sim S1: sensor models — IMU noise and wheel-encoder quantisation from ground truth | implementation | proposed | EasyRobot | [W3, W6] | M-d | ER-048 |
| [[ER-055-sim-kinematic-fake\|ER-055]] | Sim S1: kinematic fake backend — a pure-Go differential integrator implementing `Sim` for CI | implementation | proposed | EasyRobot | [W3, W6] | M-d | ER-049, ER-052 (ER-038 if first) |
| [[ER-056-sim-determinism-speed\|ER-056]] | Sim S1: determinism and speed checks with the note's kill criteria | measurement | proposed | EasyRobot | [W3, W6] | M-d | ER-050, ER-051, ER-053, ER-055 |
| [[ER-057-sim-wheeled-controller-adapter\|ER-057]] | Sim S1: `simbase` — the sim implementation of `wheeled.Base` and `Clock` over `Sim` | implementation | proposed | EasyRobot | [W3, W6] | M-d | ER-049, ER-052, ER-054, ER-055, ER-059, ER-038 |
| [[ER-058-sim-s1-wheeled-run\|ER-058]] | Sim S1: wheeled run, the house — a differential robot driven through `wheeled.Base` with logged results, teleop and kill criteria | measurement | proposed | EasyRobot | [W3, W6] | M-d | ER-050, ER-051, ER-053, ER-054, ER-056, ER-057, ER-059, ER-060, ER-038 |
| [[ER-059-sim-robot-base-and-clock\|ER-059]] | Sim S1: robot-facing `wheeled.Base` interface and `Clock` — the seam between sim and real robot | design + implementation | proposed | EasyRobot | [W3, W6] | M-d | none |
| [[ER-060-sim-wheel-odometry\|ER-060]] | Sim S1: wheel odometry — encoder ticks → 2D pose from the robot description | implementation | proposed | EasyRobot | [W3, W6] | M-d | ER-052, ER-055, ER-057, ER-059, ER-038 |

ER-015 additionally requires the completed ticket for each optional path it measures. ER-005/007/009 keep Xtensa, K230 and K510 independent and **UNVERIFIED**. ER-011's external CPU reference is not ER-014's native graph execution. ER-026 is one ticket with two stages: stage 1 host (ready), stage 2 TinyGo (blocked on owner approval).

## Order

Agreed order:

1. **Knowledge engine v1**
2. **[[ER-017-dh-planar-ik-robustness|ER-017]] / [[ER-018-leg-helpers-gait-scheduler|ER-018]]** — M-a host simulation
3. **Voice-modem composition fixes** (first step the stale `tc` test port, O3)
4. **[[ER-002-cpu-graph-semantics|ER-002]] / [[ER-020-eager-views-newas-norm-gradients|ER-020]] / [[ER-028-graph-fixes|ER-028]]** — trusted reference
5. **[[ER-026-build-green|ER-026]] stage 1** (host)
6. **Code-optimisation skill drafts** (W2)

Then, next with no hardware: [[ER-029-nn-layers-fixture-harness|ER-029]] + fixture harness; [[ER-027-marshaller-fixes|ER-027]] + [[ER-031-spi-slave-feature-framing|ER-031]] host recorder; [[ER-019-filter-correctness|ER-019]]; [[ER-022-audio-modem-acquisition|ER-022]] / [[ER-023-coding-fixes|ER-023]]; [[ER-034-watcher-crash-test|ER-034]] / [[ER-035-chunkers|ER-035]] (+ [[ER-036-find-service|ER-036]]); [[ER-032-flat-decision-tree-executor|ER-032]] / [[ER-033-cart-trainer-episode-logger|ER-033]]. Everything else needs owner input, approvals or hardware ([[roadmap#6. What needs the owner|roadmap §6]]).

The ranked "Next" list lives on the home [[index]] (proposed — owner to approve), taken from [[roadmap#8. Next|roadmap §8]].

## Order changes accepted (O1–O9)

From [[roadmap#2.1 The agreed order, and where it should change|roadmap §2.1]], accepted:

| # | Change | Landed in |
|---|---|---|
| O1 | Strides/footholds with body→leg transforms and gait→IK→servo glue join ER-018 | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] scope items 4–5 |
| O2 | Homogeneous helpers and unpivoted 4×4 inverse pulled from ER-021 into the ER-017/018 window | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] item 7; removed from [[ER-021-mat-vec-quaternion-qr-inverse\|ER-021]] |
| O3 | Stale `tc` test port is the voice-modem composition fixes' first step | composition fixes; excluded from [[ER-026-build-green\|ER-026]] |
| O4 | ER-022/023 straight after the voice-modem composition fixes | [[ER-022-audio-modem-acquisition\|ER-022]], [[ER-023-coding-fixes\|ER-023]] dependencies; ER-022 blocked on channel choice |
| O5 | ER-029 + reference-fixture harness right after ER-002/020/028 | [[ER-029-nn-layers-fixture-harness\|ER-029]]; Order above |
| O6 | ER-026 split into host and TinyGo parts | [[ER-026-build-green\|ER-026]] stages 1 and 2 (one ticket) |
| O7 | ER-027 and ER-019 queued; ER-019 PLL/AGC before ER-022; ER-027 before M-b | [[ER-019-filter-correctness\|ER-019]] stage A; [[ER-027-marshaller-fixes\|ER-027]] → [[ER-031-spi-slave-feature-framing\|ER-031]] |
| O8 | Ask for user inputs in one batch | not a ticket |
| O9 | M-d does not wait for M-b | [[ER-032-flat-decision-tree-executor\|ER-032]], [[ER-033-cart-trainer-episode-logger\|ER-033]] have no M-b dependency |

## Absorbed IK proposals

Each earlier IK proposal maps to exactly one ticket.

| Topic | Ticket |
|---|---|
| VAJ1D sign/limits | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] |
| DH IK for n>3 | [[ER-017-dh-planar-ik-robustness\|ER-017]] |
| Robust DH IK | [[ER-017-dh-planar-ik-robustness\|ER-017]] |
| Quaternion fix | [[ER-021-mat-vec-quaternion-qr-inverse\|ER-021]] |
| Analytic leg IK errors | [[ER-017-dh-planar-ik-robustness\|ER-017]] |
| Config validation | [[ER-017-dh-planar-ik-robustness\|ER-017]] |
| Joint/leg helper package | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] |
| Swing profile | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] |
| Gait phase scheduler | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] |
| Stride / foothold + body→leg | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] |
| Static stability | [[ER-037-body-pose-static-stability\|ER-037]] |
| Path-tracking signs | [[ER-038-wheeled-motion-fixes\|ER-038]] |
| steer4 geometry | [[ER-038-wheeled-motion-fixes\|ER-038]] |
| Manipulator integration | [[ER-039-manipulator-integration\|ER-039]] |
| gait/rigidbody | [[ER-037-body-pose-static-stability\|ER-037]] |

## Knowledge engine (W10)

Design: [[2026-11-09-knowledge-engine]]; module [[knowledge-engine]].

| Slice | Ticket | State |
|---|---|---|
| v1: interfaces, stores, md/Go ingestion, hybrid retrieval, benchmark | none | review |
| Watcher: fsnotify watcher, crash-mid-ingest test, missing-target index | [[ER-034-watcher-crash-test\|ER-034]] | backlog |
| Chunkers: py/c/ts/proto chunkers | [[ER-035-chunkers\|ER-035]] | backlog |
| Find service: a find call in a consuming service (API, client call, CLI) | [[ER-036-find-service\|ER-036]] (consuming service) | backlog; first after v1 |
| Embeddings experiment (LSA vs one approved static model) | unnumbered until the model download is approved | proposed (design §7) |

## Build and test debt found by ER-001

Filed: darwin/arm64 build and vet findings F1–F5, F7 are [[ER-026-build-green|ER-026]] stage 1; F6 (`tc` tests) is in the voice-modem composition fixes. Failing-test packages are covered by ER-019 (filters), ER-023 (coding), ER-024 (lidar), ER-025 (nmea), ER-029 (nn/layers); colorscience, gps, peg-adjacent and plugin failures stay `#backlog` (baseline §4). The `go:generate tinygo flash` directive (removal held for an owner decision) makes `make gen` unsafe.

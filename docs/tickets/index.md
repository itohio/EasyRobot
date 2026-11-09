---
type: ticket
status: active
tags: [easyrobot, ticket, index]
created: 2026-11-08
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# EasyRobot tickets

[[index|Home]] · [[roadmap]] · [[known-unknown|Unknown map]] · [[baseline-2026-11-08|Host baseline]] · [[conventions]]

Owner instruction (2026-11-08, verbatim): "properly update dndm, easyrobot projects so i can start improving on them".

One ticket-ID scheme: `ER-NNN`, one file each in `tickets/`. The ER-R2 proposals T-IK-01…15 and the knowledge-engine slices KE-2…KE-4 are absorbed below (`maps:` in each ticket's frontmatter). TODO lines tag their ticket as `#ER-0xx` or `#backlog` ([[conventions#TODO lines|TODO lines]]). States: `done` = acceptance evidenced in the ticket; `ready` = dependencies met and acceptance concrete, waiting for scheduling; `backlog` = proposed, not execution permission; `blocked` = prerequisites or owner input unmet. Owner `easyrobot` is the EasyRobot peer of the personal mp roster.

## All tickets

| Ticket | Title | Kind | State | Owner | Workstream | Milestone | Depends |
|---|---|---|---|---|---|---|---|
| [[ER-001-reference-environment\|ER-001]] | Discover a reproducible CPU reference environment | discovery | done (darwin/arm64 host) | easyrobot | all | all | none |
| [[ER-002-cpu-graph-semantics\|ER-002]] | Establish CPU graph and tensor semantics | implementation | ready | easyrobot | W9 | M-f | ER-001 ✓ |
| [[ER-003-operator-capabilities\|ER-003]] | Map actual operator dtype and layout capabilities | discovery | backlog | easyrobot | W9 | M-f | ER-001 ✓ |
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
| [[ER-017-dh-planar-ik-robustness\|ER-017]] | DH and planar IK robustness | implementation | ready | easyrobot | W3 | M-a | none |
| [[ER-018-leg-helpers-gait-scheduler\|ER-018]] | Leg/joint helpers, gait scheduler, strides and gait→IK→servo glue | implementation | ready | easyrobot | W3 | M-a | ER-017 (parallel ok) |
| [[ER-019-filter-correctness\|ER-019]] | Filter correctness | implementation | backlog | easyrobot | W7 | M-e | none (PLL/AGC before ER-022) |
| [[ER-020-eager-views-newas-norm-gradients\|ER-020]] | Eager tensor views, NewAs aliasing and norm gradients | implementation | backlog | easyrobot | W9 | M-f | with ER-002 |
| [[ER-021-mat-vec-quaternion-qr-inverse\|ER-021]] | mat/vec: quaternion from matrix, QR and inverse tolerance | implementation | backlog | easyrobot | [W3, W9] | M-a | none |
| [[ER-022-audio-modem-acquisition\|ER-022]] | Audio modem acquisition, sync, drift and BER tests | implementation | blocked | easyrobot | W7 | M-e | ER-AUDIO-1, ER-019 (PLL/AGC), ER-023; owner channel choice |
| [[ER-023-coding-fixes\|ER-023]] | Coding: GF(256) RS, CRC catalogue, line codes, LFSR, Gray, PEG | implementation | backlog | easyrobot | W7 | M-e | ER-AUDIO-1 |
| [[ER-024-lidar-decoder-fixes\|ER-024]] | Lidar decoder fixes | implementation | blocked | easyrobot | W8 | W8 | owner lidar ID + capture; ER-026 stage 2 |
| [[ER-025-device-fixes\|ER-025]] | Device fixes: encoder, PCA9685, servo array, NMEA, serial | implementation | backlog | easyrobot | [W3, W8] | M-a | none (devices: ER-026 stage 2) |
| [[ER-026-build-green\|ER-026]] | Build green: host part, then TinyGo stage | implementation | ready (stage 1) / blocked (stage 2) | easyrobot | all | all | stage 1 none; stage 2 TinyGo approval |
| [[ER-027-marshaller-fixes\|ER-027]] | Marshallers: proto small ints, json/yaml ints, graph node data | implementation | backlog | easyrobot | W4 | M-b | none |
| [[ER-028-graph-fixes\|ER-028]] | Graph: cycle checks, detached handles, RRT*, decision-tree Balance | implementation | backlog | easyrobot | W6 | M-d | with ER-002 |
| [[ER-029-nn-layers-fixture-harness\|ER-029]] | NN layer fixes and reference-fixture harness | implementation | backlog | easyrobot | W9 | M-f | ER-002, ER-020, ER-028 |
| [[ER-030-vision-compile-phase-correlation\|ER-030]] | Vision compile fixes and phase-correlation maths | implementation | backlog | easyrobot | W5 | M-c | none (OpenCV approval for gocv parts) |
| [[ER-031-spi-slave-feature-framing\|ER-031]] | SPI slave interface and feature-frame SPI framing | implementation | backlog | easyrobot | W4 | M-b | ER-027, ER-023 |
| [[ER-032-flat-decision-tree-executor\|ER-032]] | Plain-data decision tree and flat executor | implementation | backlog | easyrobot | W6 | M-d | ER-028 |
| [[ER-033-cart-trainer-episode-logger\|ER-033]] | CART trainer and episode logger | implementation | backlog | easyrobot | W6 | M-d | ER-032 |
| [[ER-034-ke2-watcher-crash-test\|ER-034]] | KE-2: fsnotify watcher, crash-mid-ingest test, missing-target index | implementation | backlog | easyrobot | W10 | M-g | KE-1 |
| [[ER-035-ke3-chunkers\|ER-035]] | KE-3: py/c/ts/proto chunkers | implementation | backlog | easyrobot | W10 | M-g | KE-1 |
| [[ER-036-ke4-abugaga-find\|ER-036]] | KE-4: Abugaga find (abg.svc.find, roster_find, abu find) | implementation | backlog | babyai | W10 | M-g | KE-1 |
| [[ER-037-body-pose-static-stability\|ER-037]] | Body pose integration and static stability | implementation | backlog | easyrobot | W3 | M-a | ER-018 |
| [[ER-038-wheeled-motion-fixes\|ER-038]] | Wheeled motion fixes: path-tracking signs and steer4 geometry | implementation | backlog | easyrobot | [W3, W6] | M-d | none |
| [[ER-039-manipulator-integration\|ER-039]] | Manipulator integration: units and helpers in drivers/cmd manipulator | implementation | backlog | easyrobot | W3 | none | ER-017, ER-018 |

ER-015 additionally requires the completed ticket for each optional path it measures. ER-005/007/009 keep Xtensa, K230 and K510 independent and **UNVERIFIED**. ER-011's external CPU reference is not ER-014's native graph execution. ER-026 is one ticket with two stages: stage 1 host (ready), stage 2 TinyGo (blocked on owner approval).

## Order

Agreed by Leader (roster `decisions.md`, "EasyRobot ER-R2", "Knowledge engine", "EasyRobot roadmap"):

1. **KE-1** (roster task)
2. **[[ER-017-dh-planar-ik-robustness|ER-017]] / [[ER-018-leg-helpers-gait-scheduler|ER-018]]** — M-a host simulation
3. **ER-AUDIO-1** (roster task; first step the stale `tc` test port, O3)
4. **[[ER-002-cpu-graph-semantics|ER-002]] / [[ER-020-eager-views-newas-norm-gradients|ER-020]] / [[ER-028-graph-fixes|ER-028]]** — trusted reference
5. **[[ER-026-build-green|ER-026]] stage 1** (host)
6. **ER-R3** (skill drafts, roster task)

Then, next with no hardware: [[ER-029-nn-layers-fixture-harness|ER-029]] + fixture harness; [[ER-027-marshaller-fixes|ER-027]] + [[ER-031-spi-slave-feature-framing|ER-031]] host recorder; [[ER-019-filter-correctness|ER-019]]; [[ER-022-audio-modem-acquisition|ER-022]] / [[ER-023-coding-fixes|ER-023]]; [[ER-034-ke2-watcher-crash-test|ER-034]] / [[ER-035-ke3-chunkers|ER-035]] (+ [[ER-036-ke4-abugaga-find|ER-036]], babyai); [[ER-032-flat-decision-tree-executor|ER-032]] / [[ER-033-cart-trainer-episode-logger|ER-033]]. Everything else needs owner input, approvals or hardware ([[roadmap#6. What needs the owner|roadmap §6]]).

The ranked "Next" list lives on the home [[index]] (proposed — owner to approve), taken from [[roadmap#8. Next|roadmap §8]].

## Order changes accepted (O1–O9)

From [[roadmap#2.1 The agreed order, and where it should change|roadmap §2.1]], accepted by Leader:

| # | Change | Landed in |
|---|---|---|
| O1 | Strides/footholds with body→leg transforms and gait→IK→servo glue join ER-018 | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] scope items 4–5 |
| O2 | Homogeneous helpers and unpivoted 4×4 inverse pulled from ER-021 into the ER-017/018 window | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] item 7; removed from [[ER-021-mat-vec-quaternion-qr-inverse\|ER-021]] |
| O3 | Stale `tc` test port is ER-AUDIO-1's first step | ER-AUDIO-1 (roster task); excluded from [[ER-026-build-green\|ER-026]] |
| O4 | ER-022/023 straight after ER-AUDIO-1 | [[ER-022-audio-modem-acquisition\|ER-022]], [[ER-023-coding-fixes\|ER-023]] dependencies; ER-022 blocked on channel choice |
| O5 | ER-029 + reference-fixture harness right after ER-002/020/028 | [[ER-029-nn-layers-fixture-harness\|ER-029]]; Order above |
| O6 | ER-026 split into host and TinyGo parts | [[ER-026-build-green\|ER-026]] stages 1 and 2 (one ticket) |
| O7 | ER-027 and ER-019 queued; ER-019 PLL/AGC before ER-022; ER-027 before M-b | [[ER-019-filter-correctness\|ER-019]] stage A; [[ER-027-marshaller-fixes\|ER-027]] → [[ER-031-spi-slave-feature-framing\|ER-031]] |
| O8 | Ask the owner for inputs in one batch | Leader (roster `decisions.md`, owner questions); not a ticket |
| O9 | M-d does not wait for M-b | [[ER-032-flat-decision-tree-executor\|ER-032]], [[ER-033-cart-trainer-episode-logger\|ER-033]] have no M-b dependency |

## T-IK mapping

From [[ER-R2-ik-review#7. Proposed tickets (failing test first)|IK review §7]]; each T-IK id maps to exactly one ticket.

| T-IK | Title | Ticket |
|---|---|---|
| T-IK-01 | VAJ1D sign/limits | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] |
| T-IK-02 | DH IK for n>3 | [[ER-017-dh-planar-ik-robustness\|ER-017]] |
| T-IK-03 | Robust DH IK | [[ER-017-dh-planar-ik-robustness\|ER-017]] |
| T-IK-04 | Quaternion fix | [[ER-021-mat-vec-quaternion-qr-inverse\|ER-021]] |
| T-IK-05 | Analytic leg IK errors | [[ER-017-dh-planar-ik-robustness\|ER-017]] |
| T-IK-06 | Config validation | [[ER-017-dh-planar-ik-robustness\|ER-017]] |
| T-IK-07 | Joint/leg helper package | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] |
| T-IK-08 | Swing profile | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] |
| T-IK-09 | Gait phase scheduler | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] |
| T-IK-10 | Stride / foothold + body→leg | [[ER-018-leg-helpers-gait-scheduler\|ER-018]] |
| T-IK-11 | Static stability | [[ER-037-body-pose-static-stability\|ER-037]] |
| T-IK-12 | Path-tracking signs | [[ER-038-wheeled-motion-fixes\|ER-038]] |
| T-IK-13 | steer4 geometry | [[ER-038-wheeled-motion-fixes\|ER-038]] |
| T-IK-14 | Manipulator integration | [[ER-039-manipulator-integration\|ER-039]] |
| T-IK-15 | gait/rigidbody | [[ER-037-body-pose-static-stability\|ER-037]] |

## Knowledge engine (W10)

Design: [[2026-11-09-knowledge-engine]]; module [[knowledge-engine]].

| Slice | Ticket | State |
|---|---|---|
| KE-1: interfaces, stores, md/Go ingestion, hybrid retrieval, benchmark | none (roster task) | review — [[KE-1]] |
| KE-2: fsnotify watcher, crash-mid-ingest test, missing-target index | [[ER-034-ke2-watcher-crash-test\|ER-034]] | backlog |
| KE-3: py/c/ts/proto chunkers | [[ER-035-ke3-chunkers\|ER-035]] | backlog |
| KE-4: Abugaga `abg.svc.find`, `roster_find`, `abu find` | [[ER-036-ke4-abugaga-find\|ER-036]] (owner babyai) | backlog; first after KE-1 |
| KE-5: embeddings experiment (LSA vs one approved static model) | unnumbered until the model download is approved | proposed (design §7) |

## Build and test debt found by ER-001

Filed: darwin/arm64 build and vet findings F1–F5, F7 are [[ER-026-build-green|ER-026]] stage 1; F6 (`tc` tests) is ER-AUDIO-1. Failing-test packages are covered by ER-019 (filters), ER-023 (coding), ER-024 (lidar), ER-025 (nmea), ER-029 (nn/layers); colorscience, gps, peg-adjacent and plugin failures stay `#backlog` ([[baseline-2026-11-08#4. Findings (project code, darwin/arm64)|baseline §4]]). The `go:generate tinygo flash` directive (held decision #13) makes `make gen` unsafe.

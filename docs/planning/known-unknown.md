---
type: planning
title: EasyRobot known and unknown map
status: draft
created: 2026-11-08
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
tags: [easyrobot, planning]
---
# Known/unknown map

[[source-inventory|Source evidence]] · [[roadmap|Roadmap]] · [[tickets/index|Tickets]]

All runtime claims are **UNVERIFIED**. The owner's accelerator-gap ideas are hypotheses, not confirmed defects. A missing named adapter in this pinned source does not establish impossibility or a performance deficit.

## Known knowns — static evidence

| Fact | Evidence | Implication |
|---|---|---|
| Correct owner project and MIT root licence | Source module, root licence and itohio fork/foxis parent metadata; pin recorded in index | This is the source being planned, not another EasyRobot. Dependency terms still apply. |
| Multiple graph meanings exist | Generic callbacks, tensor Gorgonia graph, deprecated pipeline sources | Choose the actual execution path before extending an abstraction. |
| Core tensor views expose strides/offset/backing data | `x/math/tensor/types/core.go`, eager implementation | Buffer ownership, aliasing and backend layout need explicit tests. |
| Backend coverage is not uniform | Gorgonia/GoCV stub files, skipped Conv2D test source, Linux TFLite wrapper | No blanket graph/operator or embedded-support claim. |
| Sibling dndm replacement and native dependencies exist | `go.mod` | Pin dependencies and discover a CPU reference package set before attempting a reproducible run. |

## Known unknowns — resolve before choosing implementation

| ID | Unknown | Hypothesis/status | How it is resolved later | Ticket |
|---|---|---|---|---|
| KU-01 | Which source/dependency/native-runtime setup works on the reference host? | **Resolved for darwin/arm64 (2026-11-08):** CPU core builds CGO-free and passes its tests; OpenCV absent; darwin port gaps recorded (baseline (evidence: `~/src/docs/evidence/baseline-2026-11-08.md`, outside the repo)) | Linux/TinyGo/vision hosts only if the owner asks. | ER-001 |
| KU-02 | Graph cycles/error propagation, tensor-view lifetime and repeat execution correctness | Static risks and unexecuted tests, not observed failures | Small deterministic scalar/tensor fixtures and alias/release/reuse checks. | ER-002, ER-016 |
| KU-03 | Backend operator/dtype/layout support and actual fallback | Interface wider than backend coverage | Per-operation supported/rejected/copy/fallback matrix and numerical fixtures. | ER-003 |
| KU-04 | Quantisation/device memory/allocator/transfer contract | Dtype tags insufficient by themselves | Document scales, packed layouts, alignment, buffers, synchronization and ownership. | ER-004 |
| KU-05 | Exact Xtensa chip/ISA and acceleration path | **UNVERIFIED owner hypothesis**: important pieces may be missing | Identify board/chip, compiler, SIMD/DSP libraries and supported kernels; measure one selected kernel only after authorisation. | ER-005, ER-006 |
| KU-06 | K230 compiler/runtime, operators and memory | **UNVERIFIED owner hypothesis**; no named adapter evidenced | Pin actual board, SDK/nncase/runtime and model format; choose one supported non-LLM subgraph before an adapter. | ER-007, ER-008 |
| KU-07 | K510 compiler/runtime, operators and memory | **UNVERIFIED**, independent of K230 | Separate board/toolchain/operator/transfer discovery; don't reuse K230 conclusions by name. | ER-009, ER-010 |
| KU-08 | Small local Qwen/similar artifact and CPU footprint | No complete decoder found; model not selected | Pin repository/revision/licence/tokenizer/quantisation/runtime; local CPU reference before native graph comparison. | ER-011 |
| KU-09 | Decoder operators and state actually needed | No end-to-end Qwen support established | Map exact selected architecture, attention/mask/RoPE/norm/gather/KV/cache/sampling needs; compare intermediate values. | ER-012, ER-013 |
| KU-10 | Does an EasyRobot graph generate the same next-token logits/results as the chosen local reference? | UNVERIFIED | CPU-only fixed fixtures, cache/reset handling and per-layer comparison before model demo. | ER-014 |
| KU-11 | Which offload/copy/allocation costs dominate? | No measured baseline | Separate compile/load/prefill/decode/transfer and cold/warm runs; record actual fallback, memory and power where observable. | ER-015 |

## Unknown knowns — hidden assumptions to challenge

- “Xtensa” names one uniform accelerator: different chips/ISA extensions, DSP libraries, caches and toolchains change feasibility.
- K230 and K510 share support because their names/vendor are related: require separate evidence.
- INT8/INT48 tags mean a quantised decoder or NPU format exists: quantisation parameters and packing are separate.
- Graph interfaces imply every backend implements every method: explicit panic/stub and format limits disprove that shortcut.
- TinyGo device files mean numerical/NN packages fit the same target: bus APIs and full dependency/toolchain support are different.
- A local external Qwen runtime proves EasyRobot graph execution: it is the comparator, not the integration.
- Serializing tensors or callbacks means execution can be resumed safely: views, closures, ownership and native handles may not survive.
- NPU speedup dominates wall-clock latency: layout conversion, transfer, unsupported-op fallback and KV memory can erase it.

## Unknown unknowns — look for surprises without inventing results

Potential surprises: native ABI/dependency drift, target compiler rejection, unsupported mixed precision/layouts, alias/release corruption, unbounded cache growth, power/thermal throttling, nondeterministic decode, stale calibration or hidden fallback. Preserve minimal fixtures, source/toolchain pins and manifests; when authorised, use fault/round-trip/edge-case checks before performance claims. No such checks were run in this planning assignment.

## Decision ledger for the personal implementer

Unresolved: reference host/runtime; exact Xtensa chip; actual K230/K510 boards/toolchains; first kernel/subgraph; local model artifact and comparison precision; numerical tolerances/context budget; whether to add a native graph decoder after the external baseline. Keep these open instead of guessing a supported NPU LLM. Only the owner accepts/parks implementation options; all tickets remain unimplemented.

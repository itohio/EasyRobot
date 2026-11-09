---
type: architecture
title: EasyRobot source inventory and graph tensor architecture
status: source-inspected-runtime-unverified
created: 2026-11-08
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
tags: [easyrobot, architecture, tensor]
---
# Source inventory and architecture

[[index|Planning index]] · [[known-unknown|Unknowns]] · [[tickets/index|Tickets]]

All source claims refer to `abb5c9cca7f756c632548d9c7be2436b46dc2e4d` in [the owner's repository](https://github.com/itohio/EasyRobot/tree/abb5c9cca7f756c632548d9c7be2436b46dc2e4d). The clone was clean before documentation. Inspection was read-only, with graph/tensor and platform source slices inspected separately. Main entry paths and core backend contracts were spot-checked directly. No dependency implementation, compiler, runtime or hardware support is inferred from an import.

## 1. Evidence levels

- **Declared:** README, DESIGN, interface or source comment promises a feature.
- **Source-present:** an implementation path exists; it may still be incomplete, incorrect or unbuildable.
- **Static limitation:** an explicit stub, panic, build restriction, omitted contract or bounded search finding.
- **Runtime verified:** none when this inventory was written. Later host build/test results: baseline-2026-11-08.md (evidence: `~/src/docs/evidence/baseline-2026-11-08.md`, outside the repo). A test file is not a test result.

## 2. Repository map

| Area | Actual source | What it is / limit |
|---|---|---|
| Generic graph structures/search | `x/math/graph/graph.go`, `matrix_graph.go`, `grid_graph.go` | Graph algorithms and storage; not a tensor compiler. |
| Callback expression evaluation | `x/math/graph/expression_graph.go` | Generic callback computation and dependency traversal, distinct from model inference. |
| Tensor API | `x/math/tensor/types/` | Core metadata plus broad math/activation/normalisation/convolution interfaces and graph capabilities. An interface method is not universal backend support. |
| Eager/native numeric path | `x/math/tensor/eager/`, `x/math/primitive/fp32/` | Slice-backed tensors and FP32-oriented kernels; candidate CPU reference. Compilation/import feasibility still untested. |
| Lazy tensor graph | `x/math/tensor/gorgonia/` | Wraps Gorgonia ExprGraph/TapeMachine; selected operations implemented, many graph methods panic. |
| Image tensor ownership | `x/math/tensor/gocv/` | OpenCV Mat clone/adopt/share handles; not a general model backend. |
| Inference import | `x/marshaller/tflite/`, `x/marshaller/gocv/` | Runtime/model wrappers and converters; native library and format limits matter. |
| Serialization | `x/marshaller/{graph,yaml,json,proto}/`, `proto/types/`, generated `types/` | Typed graph/tensor representations; round-trip data does not automatically preserve executable closures, VM state or views. |
| Concurrency/iteration | `x/workers/`, `x/iter/` | Build-selected single/multithreaded numerical paths; worker policies and ownership need validation. |
| Composition | `x/chain/` | Synchronous processor chains. The deprecated channel-based `x/pipeline/` was deleted in ER-cleanup (evidence: `~/src/docs/evidence/ER-cleanup.md`, outside the repo). |
| Devices/examples | `x/devices/`, `drivers/`, `cmd/` | Host/TinyGo/platform source and examples; not proof of complete embedded support or acceleration. |
| Design/dependencies | `README.md`, `DESIGN.md`, `go.mod`, `Makefile` | README/design mix past plans with implementation; `DESIGN.md`'s `pkg/core` hierarchy is not the actual current `x/` tree. |

## 3. Three graph meanings — keep them separate

### A. Generic callback expression graph

Source: `x/math/graph/expression_graph.go:17–57,101–177,180–230,245–262`.

1. `NewGenericExpressionGraph` constructs the structural graph and operation map; `AddNode` attaches a callback and `AddEdge` links dependencies.
2. `Compute(start, inputs...)` resolves the requested/root node and asks `buildEvaluationOrder` for the relevant order.
3. `collectReachableNodes` traverses outgoing edges; the ordering helper queues nodes whose outgoing dependency count is zero, then releases their parents.
4. `computeOnce` calls `evaluateNode` with computed child outputs; `ExpressionFunction`/`EvaluateExpression` supplies the value and the requested result is returned from the value map.

This evaluates callbacks. It does not select tensor kernels. The ordering helper returns its produced order without an explicit full-cycle-count check; cycle/error behaviour needs a regression fixture before claiming complete DAG validation. Missing inputs or unevaluated requested nodes return errors. No graph execution was run here.

### B. Tensor lazy graph

Source: `x/math/tensor/types/graph.go:15–79,81–111`; `x/math/tensor/gorgonia/graph.go:37–84,86–128,137–173`; operations in `graph_operations.go`.

1. `NewExpressionGraph` creates a Gorgonia graph. `New`/`NewConstant` create typed placeholders/constants while Building.
2. GraphTensor operations record backend nodes, rather than run the eager FP32 kernels immediately.
3. `Compile` creates a TapeMachine and moves to Compiled. Graph mutation after compilation is rejected/panics by the relevant method.
4. `Compute` locks the graph, moves to Executing, invokes `vm.RunAll`, resets the VM and returns to Compiled; `GraphTensor.Data` reads output backing data.

Gorgonia owns backend scheduling. The local `ExpressionGraph` interface does not take a context/deadline; cancellation of a native graph run is not established. Mutex protection of the graph's lifecycle is not proof that all aliases, snapshots or device transfers are safe.

### C. Legacy streaming pipelines

`x/pipeline/pipeline.go` and `step.go` connected step channels and started steps with a context. They were never the tensor graph compiler or the scalar callback evaluator, and were deleted in ER-cleanup (evidence: `~/src/docs/evidence/ER-cleanup.md`, outside the repo) (deprecated, no importers; dndm is the transport).

## 4. Tensor data and ownership

| Contract | Source | Consequence |
|---|---|---|
| Dtype, shape/rank/size, strides, contiguity, offset, raw data, element access, Release | `x/math/tensor/types/core.go:7–82` | Row-major default and explicit views are representable; raw data is mutable. Accelerator device/transfer ownership is not specified by this interface. |
| Dtype tags include FP32/FP16/INT8/INT48/UINT8 | `types/dtype.go:15–32` under tensor | Tags alone do not supply scales/zero-points, per-channel quantisation, calibration, packed-weight layout, DMA or alignment. |
| Typed allocations/pools and clamped casts | `x/math/tensor/types/dtype.go:107–166` | Conversion is not quantisation-aware dequantisation. Supported tags still need backend/path-specific coverage tests. |
| Eager storage uses slices/strides/offset | `x/math/tensor/eager/tensor.go:13–38,56–68` | `FromArray`/`FromFloat32` adopt the caller's slice; views can alias it. Do not mutate/free shared backing while another consumer uses it. |
| Broad eager numerical kernels | `x/math/tensor/eager/{math,linalg,conv,normalizations,activations}.go` | FP32 is a candidate baseline; operation coverage is not an accelerator benchmark. Higher-rank/layout support must be checked per operation. |
| GoCV shared ownership handle | `x/math/tensor/gocv/tensor.go:14–26,65–107` | Clone/adopt/share differ. OpenCV memory lifecycle and `Release` must be included in interop tests. |
| `use_mt` build selection | `x/workers/execute_st.go`, `execute_mt.go`; `x/iter/iterators_st.go`, `iterators_mt.go` | Concurrency is build-dependent; shared backing and worker failure semantics are future validation, not assumed thread safety. |

## 5. Backend implementation versus declarations

| Backend | Source-present | Important limit / evidence |
|---|---|---|
| Eager/native | Arithmetic, views/broadcast, MatMul, activations/normalisation and other FP32 kernels with test sources | Many numerical paths are FP32-specific; eager softmax's implemented rank cases are limited. No tests/builds run. |
| Gorgonia | Construction, compile/compute/reset, selected arithmetic/reductions/activations/MatMul and convolution/pooling graph constructors | `graph_tensor_stubs.go` contains unimplemented normalisation/layout/other methods. `graph_test.go:144–146` explicitly skips graph Conv2D for tensor-format investigation. A constructor is not verified numerical coverage. |
| GoCV | Mat metadata/data and image ownership wrapper | `stubs_math.go`, `stubs_activations.go` and other stubs reject unsupported generic operations. Requires native OpenCV/GoCV; no TinyGo model support inferred. |
| TFLite | Linux-tagged interpreter wrapper and model import code | `x/marshaller/tflite/model.go:1,98–126,172–227`: single input/output; input/output copy; returned output is eager FP32. Native runtime/delegate and quantisation behaviour remain unverified. |

`go.mod:1–21` records Go 1.25, GoCV, go-tflite and Gorgonia plus `replace github.com/itohio/dndm => ../dndm`. A reproducible setup must pin that sibling revision and native dependencies. README's TensorFlow/backend headings are not evidence of a working equivalent backend. No SDK, Go module or model was downloaded.

## 6. Serialization is not executable-state checkpointing

YAML/JSON graph/tensor paths and `x/marshaller/graph` provide structural representations and reflection-based reconstruction. Generic callback closures need operation registration; serializing a function name does not capture executable code. Protobuf tensor conversion exists in `x/marshaller/proto/convert.go`; a graph metadata proto exists, but a schema alone does not prove graph round-trip dispatch. Eager strides/offset/views, ownership, graph compilation and backend handles are not assumed preserved by a serialized shape/data record. ER-016 plans explicit round-trip and failure checks.

## 7. Named accelerator and LLM search scope

Source inspectors searched the pinned repository and `x/` implementation paths for Xtensa/ESP-DSP/ESP32, K230/K510/KPU/nncase/Kendryte, build tags/CGO/TinyGo, Qwen/Llama/GGUF/safetensors/tokenizer/BPE, attention/RoPE/KV cache/sampling and quantisation terms. These bounded source searches found no named Xtensa SIMD/ESP-DSP, K230/K510 NPU adapter, or complete Qwen decoder path. They are not proof that reusable generic primitives are absent or that hardware cannot support them.

- TinyGo ESP32/device-bus source and an amd64 assembly experiment exist; neither is an Xtensa accelerator implementation or a portable embedded build result.
- K230 and K510 are distinct **UNVERIFIED** target investigations. No board, SDK/compiler version, NPU operator table or transfer budget has been validated.
- Generic MatMul/Softmax/RMSNorm declarations and some eager implementations exist. No tokenizer/weight-format/attention/RoPE/KV-cache/sampler composition is established as a working local LLM.

## Inspection boundary

Only source/docs were inspected. Native dependencies and transitive runtime behaviour were not audited end to end. No SDK/compiler/model compatibility, correctness, latency, throughput, memory, power or zero-copy claim was measured. Use the [[known-unknown|unknown map]] and per-ticket procedures to turn each important question into evidence later.

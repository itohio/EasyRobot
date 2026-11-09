---
type: module
group: "Tensor and NN"
status: partial
tags: [easyrobot, tensor, nn, math]
workstream: [W9, W2, W5]
packages: [x/math/tensor, x/math/tensor/types, x/math/tensor/eager, x/math/tensor/gorgonia, x/math/tensor/gocv]
test_result: pass
---
# tensor

**Purpose:** a backend-neutral N-d tensor interface (`types.Tensor`, 119 methods) with an eager, slice-backed FP32 implementation (the CPU reference used by [[nn]]), a lazy Gorgonia graph backend, and a GoCV `Mat` wrapper. Kernels delegate to [[primitive-kernels]]; iteration to [[iter-workers]]; dtype casts to [[type-conversion]]; shapes/strides to `x/math/shape` ([[shape-strides-iteration]]).

Architecture-level context (three meanings of "graph", ownership, backend limits) is in [[source-inventory]] §3–6; not repeated here.

**Packages (non-test LOC):**

| Package | Files | LOC | Notes |
|---|---|---|---|
| `x/math/tensor` | 2 | 281 | facade: `New`, `FromFloat32`, `FromArray`, `NewShape`, `ZerosLike`, Xavier init, `SharedTensor` (atomic refcount views, `shared_view.go`) |
| `x/math/tensor/types` | 12 | 1169 | interfaces `Core, Manipulation, ElementWise, Math, Normalizations, Activations, Convolutions, Pooling, Dropout`; `DataType` tags (INT64, C128, C64, FP64, INT32, FP32, INT, INT16, FP16, INT8, INT48, UINT8); `ExpressionGraph`/`GraphTensor`/`BackendCapabilities` |
| `x/math/tensor/eager` | 8 | 7926 | `tensor.go` (storage, views), `math.go` (2564), `linalg.go`, `linalg_helpers.go`, `conv.go` (2098), `activations.go`, `normalizations.go`, `initialization.go` |
| `x/math/tensor/gorgonia` | 6 | 3391 | wraps `gorgonia.org/gorgonia` ExprGraph/TapeMachine + `tensor.Dense` |
| `x/math/tensor/gocv` | 13 | 1454 | OpenCV Mat handle wrapper; does not build here (OpenCV missing) |

Declared docs (claims, not evidence): `tensor/SPEC.md` (1509 lines), `TENSOR_TF_ALIGNMENT_PLAN.md` (1272), `BENCHMARKS.md`, `types/SPEC.md`, `eager/BENCHMARK_REPORT.md`, `eager/SOFTMAX_MULTIDIM_PLAN.md`, gorgonia `STATUS.md`/`FINAL_SUMMARY.md`/`ARCHITECTURE.md`/…, gocv `IMPLEMENTATION_PLAN.md`.

## Public API

- Construction: `tensor.New(dtype, shape)`, `FromFloat32(shape, data)` / `FromArray[T]` (adopt the caller's slice), `NewAs`, `ZerosLike`, `XavierUniform/Normal`.
- Element access: `At(indices...) float64`, `SetAt(v float64, indices...)`, `Elements()` iterator, `Data() any`, `DataWithOffset() any`, `Strides`, `Offset`, `IsContiguous`.
- Ops: every op takes `dst` first (`t.Add(dst, other)`, `t.MatMul(dst, b)`, `t.Conv2D(dst, kernel, bias, stride, pad)`); `dst == nil` allocates (or, for some ops, writes in place — see API notes). Views: `Slice`, `Reshape`, `Transpose/Permute` are zero-copy when `dst == nil`.
- Graph: `gorgonia.NewExpressionGraph()` → `eg.New/NewConstant` → ops record nodes → `Compile()` → `Compute()`.

## Algorithms

<!-- vault:algorithms h=64c441877656 -->

Counts: 6 tested · 1 implemented · 11 partial · 0 stub · 1 unverified (19 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[conv2d-kernel]] | tested | pass | NCHW, matches float64 naive (1.9e-6) and adjoint test; eager test checks shapes only |
| [[conv2d-transposed-kernel]] | tested | pass | adjoint of Conv2D by probe |
| [[dropout-kernel]] | tested | pass | mask statistics verified by probe |
| [[im2col-col2im]] | tested | pass | — |
| [[shape-strides-iteration]] | tested | pass | scalar size inconsistency |
| [[softmax-kernel]] | tested | pass | correct rank ≤ 2 contiguous; no rank-4 softmax (attention), no LogSoftmax |
| [[conv-variants-depthwise-group-dilated-3d]] | implemented | pass | forward correct by equivalence probes; no gradients, no `dst` |
| [[activation-kernels]] | partial | pass | Softplus(100)=+Inf; no grads for ReLU6/Leaky/ELU/Softplus/Swish/GELU; GELU tanh-approx only; flat… |
| [[batch-norm]] | partial | pass | per-element not per-channel stats; no running stats/inference mode; dx wrong for γ ≠ 1; nil-dst… |
| [[conv1d-kernel]] | partial | pass | forward correct; `Conv1DKernelGrad` wrong for Cout > 1 (`primitive/fp32/tensor.go:349-370`… |
| [[gorgonia-graph-backend]] | partial | pass | 32/121 Tensor and 71/74 GraphTensor stub methods panic; `Conv2d` gets stride as kernel shape… |
| [[instance-and-group-norm]] | partial | pass | forward correct; dx and GN dγ/dβ wrong |
| [[layer-norm]] | partial | pass | forward correct; dx wrong for γ ≠ 1 (maxAbsErr 3.43) |
| [[pooling-kernels]] | partial | pass | AdaptiveAvgPool uses floor end (≠ PyTorch); MaxPool backward splits ties; −1e30 init |
| [[rms-norm]] | partial | pass | forward correct (Qwen-style, last dim, γ only); dx wrong for γ ≠ 1 and O(D²) per row |
| [[tensor-elementwise-broadcast-reduce]] | partial | pass | all kernels get `Data()` without offset → wrong on `Slice` views (Add/Sum/ReLU probes); no… |
| [[tensor-matmul]] | partial | pass | contiguous correct; views ignored (maxAbsErr 72 on a slice view, 16 on a transpose view… |
| [[tensor-views-strides]] | partial | pass | views correct through `At/SetAt` only; `Copy`, same-rank `Reshape` of non-contiguous… |
| [[gocv-tensor-backend]] | unverified | build-fail | 93 stub methods panic; no tests; OpenCV missing |

<!-- /vault:algorithms -->

## Build targets and dependencies

- eager/types/facade: pure Go, CGO-free, single-threaded by default; `chewxy/math32`; generic dtype dispatch via `any` + type switches (82 in eager); one `unsafe` use for `ID()` (`tensor.go:80`). Builds for linux/riscv64, linux/arm, linux/386; **fails for js/wasm** via `x/types/st` (see [[type-conversion]]).
- Gorgonia: large, reflection-heavy dependency tree; host only; not TinyGo-suitable [INFERENCE].
- GoCV: CGO + OpenCV; excluded on this host.
- Embedded: float64 in `At/SetAt` and scalar parameters; panics with `fmt.Sprintf` on errors (403 `panic(` in eager); `dst == nil` paths allocate. See [[W9-go-nn-framework]] §(a).

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/math/tensor/...` (2026-11-09): `tensor/eager` 112 pass; `tensor/types` 8 pass; `tensor/gorgonia` 30 pass, `TestGraphConv2D` skipped (`graph_test.go:144-146`); `gorgonia_xor_test` 1 pass; `tensor` builds, no Test funcs (benchmarks only); `tensor/gocv` build failed (OpenCV missing).
- **Green suite, wrong results:** no test feeds a view (`Slice`, `Transpose`) into a compute op, and normalisation/conv-variant tests check shapes only. Value probes by the kernel and infra slices (`~/src/tmp/easyrobot/tensor/{kernelnotes,InfraNotes}/`) found the defects listed above.
- Benchmarks (`go test -run '^$' -bench 'BenchmarkMatMul_(Eager|GorgoniaGraph)_(Medium|Large)' -benchtime 2s ./x/math/tensor/`, shared host): eager 512×512 209 ms (1.3 GFLOP/s) vs Gorgonia 15.8 ms (17 GFLOP/s). Details: [[W2-code-optimisation]].

## API notes (ease of use, composability)

- **Good:** one interface across backends; `dst`-first ops make allocation explicit and let layers pre-allocate; zero-copy views exist; `SharedTensor` refcounting for shared buffers.
- **Fat interface (119 methods):** every backend must implement everything; GoCV and Gorgonia satisfy it with panicking stubs, so "implements `Tensor`" says nothing about capability. `BackendCapabilities` only describes eager vs graph. Smaller capability interfaces (`MatMuler`, `Conv2Der`, …) plus type assertions would make support explicit.
- **Views are a trap:** `Data()` returns the whole backing slice; most ops use it and silently ignore `offset` (and some ignore strides), so a view composes correctly only with `At/SetAt`. Either every op honours `DataWithOffset()`+strides or ops must materialise contiguous inputs (`Contiguous()` does not exist).
- **`dst == nil` semantics differ by op:** some allocate, some (`NewAs` in normalisations) alias and overwrite the input.
- **Argument order inconsistencies:** `Softmax(dim, dst)` vs `dst`-first elsewhere; `DepthwiseConv2D/GroupConv2D/DilatedConv2D/Conv3D/Conv1DTransposed` have no `dst`.
- **Errors are panics:** shape/dtype mismatches panic; layers above return errors, so the error model is mixed.
- **float64 scalar API on an FP32 framework** (`At`, `AddScalar`, eps): costly on ESP32 and lossy for INT64.
- **Dtype coverage is uneven:** tags exist for 12 dtypes, but kernels are FP32-only and element access misses INT32/INT64/INT/UINT8.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Value tests for all norms, conv variants, activation grads (numerical gradient checks) — eager/*_test.go #p2
- [ ] Split the 119-method interface into capability interfaces — x/math/tensor/types/tensor.go #p2
- [ ] Mark gorgonia STATUS/FINAL_SUMMARY docs as aspirational — x/math/tensor/gorgonia/STATUS.md #p3

## Inferred objectives (proposals)

- **Proposal:** make eager the single trusted CPU reference: fix views, add a reference-fixture test suite (numpy/PyTorch outputs incl. gradients and views), then use it to validate Gorgonia, TFLite, and future ESP32/K230 kernels (ER-002/003).
- **Proposal:** introduce a typed `Tensor[float32]`-style fast path (or an `fp32`-only embedded facade) that avoids `any` dispatch and float64 scalars, for TinyGo.
- **Proposal:** retire or quarantine the Gorgonia backend unless it becomes the host performance path (it is 13× faster for MatMul today); decide explicitly.

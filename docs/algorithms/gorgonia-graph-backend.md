---
type: algorithm
status: partial
tags: [easyrobot, tensor, gorgonia, graph]
workstream: W9
module: "[[tensor]]"
summary: "32/121 Tensor and 71/74 GraphTensor stub methods panic; `Conv2d` gets stride as kernel shape…"
source: [x/math/tensor/gorgonia/tensor.go, x/math/tensor/gorgonia/graph.go, x/math/tensor/gorgonia/graph_tensor.go, x/math/tensor/gorgonia/graph_operations.go, x/math/tensor/gorgonia/graph_tensor_stubs.go, x/math/tensor/types/graph.go]
tests: [x/math/tensor/gorgonia/tensor_test.go, x/math/tensor/gorgonia/conversion_test.go, x/math/tensor/gorgonia/graph_test.go, x/math/tensor/gorgonia/example_test.go, x/math/tensor/gorgonia/gorgonia_xor_test/xor_integration_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09): x/math/tensor/gorgonia 30 pass + 1 skip (TestGraphConv2D), gorgonia_xor_test 1 pass; gorgonia.org/gorgonia v0.9.18 Conv2d signature read from module cache"
---
# Gorgonia tensor + graph backend

**What it does:** Adapter making `gorgonia.org/tensor` (`*tensor.Dense`) and Gorgonia `ExprGraph`/`TapeMachine` satisfy EasyRobot's `types.Tensor` and `types.ExecutionGraph` interfaces. Two layers: an eager-ish `Tensor` wrapper and a lazy `GraphTensor` built inside an `ExpressionGraph` (build → compile → run). Inventory overview lives in [[source-inventory]]; this note covers implementation status and defects only.

**Where:**
- Eager wrapper `x/math/tensor/gorgonia/tensor.go` (1940 lines, 121 methods): `New` `:24`, `ToEagerTensor` `:61-107`, `FromEagerTensor` `:110`, element-wise, scalar, reductions, MatMul, activations, Copy with dtype conversion.
- Graph `x/math/tensor/gorgonia/graph.go:37-173` (`NewExpressionGraph`, `New`, `NewConstant`, `Compile`, `Run`, 2 panics), `graph_tensor.go` (19 funcs, 14 panics: build-state guards), `graph_operations.go` (31 funcs: Add/Sub/Mul/Div, scalar ops, Square/Sqrt/Exp/Log/Pow/Abs/Negative, MatMul `:424-449`, Sum `:452-476`, ReLU/Sigmoid/Tanh/LeakyReLU/Softmax, Conv2D `:552-587`, MaxPool2D `:590-604`), `graph_tensor_stubs.go` (74 funcs).

**Implemented vs panicking (counted 2026-11-09):**
| part | methods | `not (yet) implemented` panics |
|---|---|---|
| `Tensor` (tensor.go) | 121 | **32** (BroadcastTo, Pad, Unpad, 6 scalar comparisons, Where, Max, Min, MatMulTransposed, MatVecMulTransposed, Norm, L2Normalize, AddScaled, ScatterAdd, Batch/Layer/RMS/Instance/Group norm fwd+grad (10), ReLU/Sigmoid/Tanh/Softmax Grad) |
| `GraphTensor` stubs | 74 | **71** (`graph_tensor_stubs.go`) |
| `GraphTensor` ops | 31 | 2 (`Copy` graph-time `:75`, `Fill` `:84`) + MatMul rank≠2 `:445` |

**Inputs → outputs:** `types.Tensor` with `DataType` FP32/FP64/… mapped to Gorgonia dtypes; graph tensors carry an EasyRobot-side `shape.Shape` that is computed by the adapter, not read back from Gorgonia.

## Completeness review

- Complete: **no** — see counts; no gradients for norms/activations in eager wrapper; graph supports only 2-D MatMul.
- Tested: 30 pass (`TestAdd`, `TestMatMul`, `TestSum`, `TestTranspose`, `TestReshape`, conversion round-trips, `TestGraphBasicOperations`, `TestGraphMatMul`, `TestGraphActivations`, `TestGraphMaxPool2D`, `TestGraphReuseExecution`, 7 `Example*`), value-checking; XOR integration trains via Gorgonia. `TestGraphConv2D` skipped (`graph_test.go:144-146`, "requires proper tensor format").
- Correct: **no** for conv/pool/sum shapes:
  - `graph_operations.go:565` calls `gorgonia.Conv2d(im, filter, kernelShape, pad, stride, dilation)` (v0.9.18 `nn.go:211`) with **`stride` passed as `kernelShape`** and stride hard-coded `[1,1]` → any stride≠1 is ignored and kernel shape is wrong unless kernel==stride. Probable cause of the skipped test.
  - `:583-584` Conv2D and `:600-601` MaxPool2D set output shape = input shape (TODO) → downstream EasyRobot shape metadata wrong.
  - `:474-475` Sum along dims keeps input shape (TODO).
  - `:576` bias add without broadcast over [N,C,H,W] [INFERENCE: fails for bias shape [C]].

**Declared (repo docs, claims):** `STATUS.md`, `FINAL_SUMMARY.md`, `IMPLEMENTATION_SUMMARY.md`, `GRAPH_IMPLEMENTATION.md` describe graph API as complete/"ready"; counts above contradict.

**Embedded suitability:** none — Gorgonia pulls reflection-heavy `tensor.Dense`, `float64` scalar APIs, CGo-free but large; not TinyGo-compatible [INFERENCE]. Native-Go reference/back-end only.

## TODOs

- [ ] Fix Conv2d argument order (pass kernel H×W as kernelShape, real stride) and enable TestGraphConv2D — x/math/tensor/gorgonia/graph_operations.go:565, graph_test.go:146 #p1 #ER-020
- [ ] Compute conv/pool/sum output shapes (or read `node.Shape()`) — x/math/tensor/gorgonia/graph_operations.go:474,583,600 #p1 #ER-020
- [ ] Batched MatMul in graph — x/math/tensor/gorgonia/graph_operations.go:445 #p2
- [ ] Implement or explicitly mark unsupported the 32 eager + 71 graph stubs (capability query instead of panic) — x/math/tensor/gorgonia/tensor.go:585, graph_tensor_stubs.go:13 #p2
- [ ] Update STATUS/FINAL_SUMMARY to match reality — x/math/tensor/gorgonia/STATUS.md #p3

## Links

- Module: [[tensor]]
- Inventory: [[source-inventory]]
- Related: [[gocv-tensor-backend]], [[tflite-model-import]]

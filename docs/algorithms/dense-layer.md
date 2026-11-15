---
type: algorithm
status: implemented
tags: [easyrobot, nn, layers]
workstream: [W9, W2]
module: "[[nn]]"
summary: "backward exact by central-difference probe; `TestDense_NumericalGradientCheck` uses float32…"
source: [x/math/nn/layers/dense.go]
tests: [x/math/nn/layers/dense_test.go, x/math/nn/layers/dense_comprehensive_test.go]
test_result: fail
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → 20/21 Dense tests pass, TestDense_NumericalGradientCheck fails (test defect); probe (2026-11-09) central diff matches ≤2e-6"
---
# Dense (fully connected) layer

**What it does:** `y = x·W + b` for one sample or a batch, with backward for `dX`, `dW`, `db`.

**Where:** `x/math/nn/layers/dense.go:31-69` (`NewDense`), `:72-122` (`Init`), `:126-164` (`Forward`), `:168-205` (`computeLinear`), `:209-344` (`Backward`), `:347-364` (`OutputShape`), `:367-466` (`Weight/Bias/SetWeight/SetBias`).

**Inputs → outputs:** input `[in]` or `[B,in]` FP32 → `[out]` or `[B,out]`. `W` is `[in,out]` (`ParamWeights`), `b` is `[out]` (`ParamBiases`). Kernels used ([[tensor]]): 1D `MatVecMulTransposed` + `AddScaled` (:177-183); batch `MatMul` + `BroadcastTo` + `AddScaled` (:188-198). Backward: `MatMulTransposed` for `dW = xᵀ·g` (:252, :310) and `dX = g·Wᵀ` (:279, :338); `db = Copy(g)` (:265) or `Sum(g, axis 0)` (:323).

## Completeness review

- Complete: yes. Forward and backward exist for 1D and 2D input. Only rank 1 and rank 2 inputs are accepted (:78-88), so a `[B,T,D]` sequence has to be reshaped first.
- Tested: `TestDense_Forward/ForwardAccuracy/ForwardBatch/ComputeOutput` compare outputs with hand-computed values. `TestDense_Backward*` (incl. `_Comprehensive`, `_BackwardBatch`, `_BackwardGradientComputation`) compare dW/db/dX with expected values. `TestDense_SetCanLearn_DisablesWeightBiasGradients` and `TestDense_WeightBiasInitializationAndCloning` cover flags and init. In `go test -count=1 ./x/... (2026-11-09)` all pass except **`TestDense_NumericalGradientCheck` (dense_comprehensive_test.go:253) FAIL**: numerical=1.049042 vs analytical=1.0 for dW[0], dW[1], db; 3.051758 vs 3.0 for dW[4], dW[5].
- Correct: yes. The failure is a **test defect**. The test uses a one-sided forward difference with `epsilon=1e-5` (:297, :325) on an FP32 output sum ≈ 6.1. The FP32 ULP near 6 is 4.77e-7, so `1e-5` is quantised to 22 ULP = 1.049e-5, which gives 1.049. Probe (same weights and input) gives analytical dW=[1 1 2 2 3 3], which is exact for d(sum(xW+b))/dW = x:
  - eps=1e-5 forward diff = [1.049 1.049 2.003 2.003 3.052 3.052] (reproduces the failure)
  - eps=1e-3 central diff = [0.99993 0.99993 2.00009 1.99986 3.00002 2.99978]
  - eps=1e-2 central diff = [0.999999 0.999975 1.999998 1.999998 2.999997 3.000021]
  
  Probe `gradcheck` (random W, [4,3] batch and [3] single sample): dInput/dW/db match central difference with maxAbs ≤2.2e-6.

## Defects

- `NewDense` always creates a bias (`hasBias: true`, dense.go:46). `UseBias(false)` is silently ignored, whereas Conv1D/Conv2D honour it.
- Batch size is frozen at `Init`. Output and `biasBroadcast` are allocated for `inputShape[0]` (:98-119), so a later Forward with a different batch panics: probe gives `tensor.MatMul: destination shape mismatch: expected [4 2], got [2 2]`. Sequential models are pinned to one batch size (also models/model.go:112).
- Gradients are overwritten, not accumulated. `MatMulTransposed` writes with beta=0 (:252, :310), bias uses `Copy` / `Sum` into dst (:265, :323). Probe: dW after two Backward calls = [1 2], where accumulation would give [2 4]. This conflicts with GPTEmbedding, which accumulates (gpt_embedding.go:233), and makes gradient accumulation over micro-batches impossible.
- Stale doc comment "Weight initialization is not handled here" (:29). The constructor does Xavier-init via `Parameter.Init` (:59, :65); see [[nn-parameter-initialisation]].
- Bias is Xavier-initialised (non-zero, limit √(6/(1+out))) rather than zero. Probe: Dense(4,2) bias = [-0.81 -1.06].

## TODOs

- [ ] Fix TestDense_NumericalGradientCheck: central difference with eps=1e-2 (or float64 loss + eps=1e-3), tolerance 1e-3 — dense_comprehensive_test.go:297,325,370 #p1 #ER-029
- [ ] Honour `UseBias(false)` via `Base.BiasHint()` like Conv2D — dense.go:46 #p2
- [ ] Support batch-size change after Init (reallocate output/biasBroadcast on shape change) — dense.go:98-119 #p2
- [ ] Decide gradient semantics (accumulate vs overwrite) and apply uniformly across layers (beta=1 in MatMulTransposed) — dense.go:252,310,265,323 #p2
- [ ] Remove the stale comment; zero-init the bias by default — dense.go:29,65 #p3
- [ ] Accept rank-3 `[B,T,D]` input for transformer MLP blocks — dense.go:78-88 #p2

## Embedded suitability

Hot path allocates in batch mode only if Init was skipped. Each Forward makes a `Reshape` view (1D Backward :250-251, :273) and allocates a fresh `gradInput2D` on a shape change. FP32 only. The `Parameter` map lookup costs `map[ParamIndex]` hashing on every call. No goroutines. No generics in the layer. Float64 appears only in scalar args (`AddScaled(…,1.0)`).

## Links

- Module: [[nn]] · kernels: [[tensor]], [[mat-vec]] · training: [[learn]], [[learn-train-step]] · related: [[sequential-model-and-backprop]], [[nn-parameter-initialisation]]

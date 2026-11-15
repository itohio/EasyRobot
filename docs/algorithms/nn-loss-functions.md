---
type: algorithm
status: implemented
tags: [easyrobot, nn, loss]
workstream: W9
module: "[[nn]]"
summary: "gradients correct by probe; MSE averages, CrossEntropy sums (inconsistent scaling)"
source: [x/math/nn/loss.go, x/math/nn/losses.go, x/math/nn/validation.go]
tests: [x/math/learn/training_test.go]
test_result: none
complete: yes
correct: yes
evidence: "x/math/nn has no test files (go test -count=1 ./x/... 2026-11-09 → [no test files]); used by passing learn tests (go test -count=1 -short ./x/math/learn/, 2026-11-09); probe (2026-11-09) gradients vs central diff ≤2.7e-4, CE value exact"
---
# NN loss functions: MSE, CrossEntropy, CategoricalCrossEntropy

**What it does:** scalar loss and gradient with respect to predictions, behind the `LossFunction` interface used by `learn.TrainStep`.

**Where:**
- Interface: `x/math/nn/loss.go:8-14` (`LossFunction{Compute, Gradient}`).
- `x/math/nn/losses.go`:
  - `MSELoss` :10-47
  - `CrossEntropyLoss` :50-138
  - `CategoricalCrossEntropy` :141-277
  - free funcs `MSE` :280-300, `CrossEntropy` :303-341
  - `validateShapes` :344-363

**Inputs → outputs:** `pred`, `target` with the same shape (any rank, FP32) → `float32` loss and a gradient tensor (a scratch buffer owned by the loss and reused across calls).

| Loss | Compute | Gradient |
|---|---|---|
| MSE | mean((p−t)²) over **all** elements | 2(p−t)/N (:42-44) |
| CrossEntropy | −Σ t·log(p+1e-10), masked to t≠0 ∧ p>0; **sum, not mean** | −t/(p+ε) where p>0 else 0 (:84-97) |
| CCE fromLogits=false | same as CrossEntropy | same |
| CCE fromLogits=true | softmax over the last axis, then CE | softmax(p) − t (:198-217), valid when each target row sums to 1 |

## Completeness review

- Complete: yes for these three.
- Missing:
  - sparse/integer-label CE (the LLM next-token loss)
  - label smoothing
  - reduction option (mean/sum)
  - ignore_index / masking
  - Huber/L1, BCE-with-logits
- Tested: no tests in `x/math/nn`. Indirect coverage: learn `TestTrainStep_SimpleTraining` (checks the loss value of `y=2x+1` before and after one step) and the convergence tests all pass.
- Correct: yes. Probe `gradcheck` (central diff eps 1e-2 on `Compute`):
  - MSE maxAbs 4.7e-6
  - CrossEntropy (probs) 2.7e-4 (rel 1.3e-4; curvature of log)
  - CCE(false) 2.7e-4
  - CCE(true, logits) 1.1e-5
  - `CrossEntropy([.2 .5 .3;.6 .1 .3],[0 1 0;1 0 0])` = 1.203973 = −(ln .5 + ln .6) ✓

## Defects

- Reductions are inconsistent: MSE averages over all elements, while CE and CCE sum over batch and classes. The effective learning rate therefore scales with batch size for CE but not for MSE. #p2
- `Compute` does not validate shapes (only `Gradient` does, :21-27, :67-73). Mismatched shapes panic inside the kernels [INFERENCE]. #p3
- `MSE()` allocates two tensors per call (:287, :296). `CrossEntropy()` allocates about nine (:314-337, including `Log(nil)`, `GreaterScalar(nil)`, `Multiply(nil)`). `CategoricalCrossEntropy(true)` runs softmax twice per step (once in Compute, once in Gradient, :178, :205). #p2
- The returned gradient aliases an internal scratch tensor (`m.grad`, `c.grad`). Holding it across calls sees it overwritten. [INFERENCE] Safe in TrainStep.
- ε=1e-10 is below FP32 resolution for p≳1e-3, so it only guards p≈0. That is acceptable because p≤0 is masked.
- `validation.go:4-14` `shapesEqual` is dead code (no callers in nn). #p3

## TODOs

- [ ] Add value tests for all three losses (forward + finite-difference gradient) in x/math/nn — (no test files) #p2
- [ ] Add a reduction mode and make CE mean-over-batch consistent with MSE (or document) — losses.go:303-341 #p2
- [ ] Sparse (index-target) CE with logits for LLM fine-tuning — losses.go (absent) #p2
- [ ] Remove per-call allocations in MSE()/CrossEntropy(); cache the softmax between Compute and Gradient — losses.go:178,205,287-337 #p2
- [ ] Delete unused shapesEqual — validation.go:4-14 #p3

## Embedded suitability

`Gradient` is allocation-free after the first call. `Compute` allocates per call. Uses a float64 scalar for `MulScalar` and `At()`. No goroutines.

## Links

- Module: [[nn]] · training: [[learn]], [[learn-train-step]] · kernels: [[tensor]] · related: [[activation-layers]], [[sequential-model-and-backprop]]

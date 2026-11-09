---
type: algorithm
status: implemented
tags: [easyrobot, learn, nn]
workstream: [W9, W6]
module: "[[learn]]"
summary: "indirect (conv ≥75% acc) pass; probe matches Kingma&Ba ref to 1e-7"
source: [x/math/learn/optimizer.go]
tests: [x/math/learn/conv1d_layer_test.go, x/math/learn/conv2d_layer_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "probe vs Kingma&Ba Alg.1 reference, 3 steps: max diff 1e-7 (float32); conv tests pass (2026-11-09)"
---
# learn Adam optimizer

**What it does:** Adam (Kingma & Ba 2015, Alg. 1) with per-parameter first/second moments and bias correction.

**Where:** `x/math/learn/optimizer.go:59-209` (`Adam`, `adamState`, `NewAdam`, `(*Adam).Update`).

**Inputs → outputs:** `NewAdam(lr, beta1, beta2, epsilon float64)` — no defaults despite comment :89; panics on invalid ranges (:91-102). `Update(types.Parameter) error` mutates `param.Data` in place. State map keyed by `param.Data.ID()` (:147) = address of first data element (tensor/eager/tensor.go:74-83).

Per step: `m=β1m+(1−β1)g` (:181-182); `v=β2v+(1−β2)g²` (:186-188); `m̂=m/(1−β1^t)`, `v̂=v/(1−β2^t)` (:174-177,192-193); `θ −= lr·m̂/(√v̂+ε)` (:196-206). ε is added **after** the sqrt of the bias-corrected v̂ — matches the paper's Algorithm 1 (not the ε̂ variant inside the step-size reformulation).

## Completeness review

- Complete: yes (no AMSGrad/AdamW/decoupled weight decay; not claimed).
- Tested: no unit test of values. Used by `TestConv1D_Layer_Learning`, `TestDenseConv1D_CombinedLearning`, `TestConv2D_*` which assert ≥75% accuracy on synthetic patterns (conv1d_layer_test.go:179, conv2d_layer_test.go:230,444) — pass (lead run 2026-11-09). Also used in tagged mnist/xor tests (not run). Status `implemented` because only indirect, threshold-style tests exist.
- Correct: yes for formula — probe compares 3 steps with varying grads against a float64 reference: step3 got `[0.9792514 −1.9997133]` vs ref `[0.9792515 −1.9997133]`.

## Defects / risks

- State key = data pointer (:147): if a layer reallocates its weights (re-Init, reshape), Adam silently starts fresh state and the old entry leaks; two params aliasing one buffer share state. No way to reset/inspect state.
- Memory: allocates 8 full-size FP32 tensors per parameter (m, v + 6 scratch, :154-166) → ~9× parameter memory; on ESP32 this dominates. Only m, v are required; scratch could be one buffer. State is always `DTFP32` regardless of param dtype (:155).
- `epsilonTensor.Fill` every step (:198) — constant; wasteful and a full-size tensor just to add a scalar.
- `sync.Mutex` held for whole update (:144) → serialises per-parameter updates even across distinct params.
- `math.Pow` per step in float64 — fine on host; slow soft-float on targets without FPU-double (ESP32 has single-precision FPU only).

## TODOs

- [ ] Unit test Adam against reference values (as in probe) — x/math/learn/optimizer.go:120 #p2
- [ ] Cut scratch to ≤1 buffer; add scalar-add instead of epsilonTensor — x/math/learn/optimizer.go:154-166,198 #p2
- [ ] Key state by parameter identity (layer/param index), not data pointer; add Reset — x/math/learn/optimizer.go:147 #p2
- [ ] Doc says defaults exist but constructor has none; add `DefaultAdam()` or options — x/math/learn/optimizer.go:89 #p3

## Links

- Module: [[learn]] · Workstreams: [[W9-go-nn-framework]], [[W6-decision-trees]]

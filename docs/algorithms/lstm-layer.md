---
type: algorithm
status: partial
tags: [easyrobot, nn, layers, rnn]
workstream: W9
module: "[[nn]]"
summary: "forward wrong (all gates read `gates[0:h]`, reproduced by probe); `Backward` not implemented…"
source: [x/math/nn/layers/lstm.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no LSTM test anywhere (go test -count=1 ./x/... 2026-11-09); probes tmp/easyrobot/tensor/layernotes/{gradcheck,micro} (2026-11-09): forward ≠ manual LSTM (maxAbs 1.39e-1), matches 'all gates read gates[0:h]' hypothesis exactly"
---
# LSTM layer (single step, stateful)

**What it does:** one LSTM time step per `Forward`. Keeps h and c inside the layer between calls. Gate order is i, f, g, o.

**Where:** `x/math/nn/layers/lstm.go`: struct :17-48, `NewLSTM` :52-99, `Init` :102-188, `Forward` :192-244, `computeForward` :254-371, `ResetState/SetState/GetState` :374-414, `OutputShape` :417-436, `Backward` :439-451.

**Inputs → outputs:** `x[in]` or `x[B,in]` → `h[hid]` or `h[B,hid]`. Params: `ParamLSTMWeightIH [4h,in]`, `ParamLSTMWeightHH [4h,h]`, `ParamLSTMBias [4h]` (types/parameter.go:23-27). No sequence input `[B,T,in]`; the caller loops over time. Kernels in [[tensor]]: `MatMulTransposed`, `Add`, `BroadcastTo`, `Slice`, `Sigmoid`, `Tanh`, `Multiply`, `Copy`.

## Completeness review

- Complete: no. `Backward` returns `"LSTM.Backward: not yet implemented"` (:448-451, TODO). It cannot be trained.
- Tested: no tests exist; there is no `lstm_test.go` and no test in learn.
- Correct: **no**. Probe `micro` (2 inputs, 2 hidden, seeded weights, 2 steps, compared with a float64 reference `c=f⊙c+i⊙g, h=o⊙tanh(c)`):
  - impl step0 = [-0.06485 0.09329], reference = [0.04650 0.14538]
  - impl step1 = [-0.08475 0.21999], reference = [0.05430 0.34899]
  - Same result with batch input `[1,2]`.
  - The hypothesis "i=f=o=σ(gates[0:h]), g=tanh(gates[0:h])" reproduces the impl **exactly** (5 decimals, both steps).
- Root cause: `gates.Slice(nil, dim, k·h, h)` returns offset views (:332-335). `Tensor.Sigmoid`/`Tanh` read `types.GetTensorData` = `Tensor.Data()`, which returns the whole backing slice and **ignores the view offset** (eager/tensor.go:103-111, eager/activations.go:52-73). Every gate is therefore computed from the input-gate pre-activations. This is a [[tensor]] defect that breaks any element-wise op on `Slice` views.

## Defects

- Wrong forward: gates are computed from the wrong slice (lstm.go:332-353 together with eager/activations.go:52-73 and eager/tensor.go:103-111).
- Backward is missing (:448).
- Batch-size and 1D/2D mode are frozen at `Init`. `Forward` checks hidden-state shape equality (:228-230).
- `hasBias` is always true; there is no `UseBias` support (:67).
- Bias is Xavier-initialised (:94-95). PyTorch-style LSTMs typically use a uniform(±1/√h) init and often forget-gate bias = 1.

## TODOs

- [ ] Fix element-wise ops on Slice views: honour offset/strides in Data()/GetTensorData consumers, or materialise contiguous gate tensors — x/math/tensor/eager/activations.go:52-73, eager/tensor.go:103-111, lstm.go:332-353 #p1 #ER-029
- [ ] Add forward value tests vs a reference LSTM (multi-step, batch) — (no test) #p1 #ER-029
- [ ] Implement BPTT backward (store per-step gates/c) or document inference-only — lstm.go:439-451 #p2
- [ ] Sequence-mode Forward `[B,T,in]` → `[B,T,h]` — lstm.go:102-118 #p3

## Embedded suitability

Many scratch tensors are preallocated in `Init` (:139-185), but each step still makes four `Slice` views, which allocate stride slices (:332-335). FP32 only. No goroutines. Stateful layer: not reentrant.

## Links

- Module: [[nn]] · kernels: [[tensor]], [[mat-vec]] · related: [[sequential-model-and-backprop]], [[activation-layers]]

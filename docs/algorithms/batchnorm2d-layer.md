---
type: algorithm
status: partial
tags: [easyrobot, nn, layers, normalization]
workstream: W9
module: "[[nn]]"
summary: "computes InstanceNorm; running stats/momentum/eval mode unused"
source: [x/math/nn/layers/activations.go]
tests: [x/math/learn/mnist_convnet_test.go]
test_result: none
complete: no
correct: no
evidence: "no test in x/math/nn/layers (go test -count=1 ./x/... 2026-11-09); mnist_convnet_test.go is behind build tag mnist_test (not run); probe (2026-11-09): forward is per-instance normalisation, dX consistent with that forward (maxAbs 2.6e-5)"
---
# BatchNorm2D layer (actually InstanceNorm)

**What it does:** declared as batch normalisation over NCHW feature maps. **Actually computes instance normalisation**: per sample and per channel over H×W, followed by γ/β.

**Where:** `x/math/nn/layers/activations.go:576-756`: `NewBatchNorm2D` :597-614, `Init` :617-666, `Forward` :669-690 (`InstanceNorm2D`, :687), `Backward` :693-726 (`InstanceNorm2DGrad`, :711), `SetTrainingMode/TrainingMode/Eps/Momentum/OutputShape` :729-756.

**Inputs → outputs:** `[B,C,H,W]` → same shape. Parameters: γ = `ParamWeights` `[C]`, initialised to 1; β = `ParamBiases` `[C]`, initialised to 0 (:633-652). Kernels in [[tensor]]: `InstanceNorm2D`, `InstanceNorm2DGrad`.

## Completeness review

- Complete: no.
  - No batch statistics: `Forward` calls `InstanceNorm2D` with a comment claiming this "is the standard behavior for BatchNorm2D" (:684-687), which is false.
  - `runningMean` and `runningVar` are created (:659-663) but never updated or read.
  - `isTraining` and `momentum` are stored but unused (:729-746), so inference behaves exactly like training.
  - No `WithCanLearn`/Options parameter in the constructor (:597).
- Tested: no unit test in `x/math/nn/layers` (none in the 2026-11-09 JSON). It is used only by `x/math/learn/mnist_convnet_test.go:30-62,195-207`, which is behind `//go:build mnist_test` and needs MNIST data, so it was not run.
- Correct: no as BatchNorm. Probe `gradcheck`:
  - dX vs central difference maxAbs 2.6e-5, so the backward matches its *instance-norm* forward.
  - With sample 1/channel 0 shifted +5, the per-sample output means are 0.0000 / 0.0000. True batch norm would give ≈ −1 / +1.

## Defects

- Not batch norm: it uses instance statistics and has no running stats or eval mode (activations.go:684-687, :659-663, :729-746). A model trained with this cannot be exported to and compared with PyTorch/TFLite BatchNorm. [[tflite]] imports would need a separate folded-BN path.
- `Init` re-creates γ/β on every call, which **resets learned parameters** if `Init` is called again (e.g. `Sequential.Init` after loading weights). It also allocates a discarded Xavier tensor first (:636, :648) and hard-codes FP32 via `FromFloat32`, ignoring `WithDataType` (:642, :651).
- `Backward` always writes param grads, even when `CanLearn()` is false (:714-723). It allocates three new tensors per call (nil dsts, :711) and never calls `StoreGrad`.
- Accesses `bn.output` (the Base field) directly (:687).

## TODOs

- [ ] Implement real BatchNorm (batch mean/var over N,H,W in training, running stats with momentum, eval mode) or rename to InstanceNorm2D — activations.go:669-690 #p1 #ER-029
- [ ] Do not reinitialise γ/β in Init if already set; respect dtype — activations.go:633-656 #p2
- [ ] Add value tests (forward vs reference, eval mode, finite-difference backward) in x/math/nn/layers — (no test exists) #p2
- [ ] Respect CanLearn and preallocate grads — activations.go:711-725 #p3
- [ ] Add LayerNorm/RMSNorm layers (needed for transformer/Qwen2) — x/math/nn/layers (absent) #p2

## Embedded suitability

Backward allocates three tensors per call. `Init` allocates and discards. Forward is allocation-free. FP32 only. `eps` is float64, passed per call. No goroutines.

## Links

- Module: [[nn]] · kernels: [[tensor]] · related: [[conv2d-layer]], [[gpt-attention-layer]] (needs RMSNorm), [[tflite]]

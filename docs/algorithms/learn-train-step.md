---
type: algorithm
status: tested
tags: [easyrobot, learn, nn]
workstream: [W9, W6]
module: "[[learn]]"
summary: "training_test pass (loss value, loss <0.1·initial)"
source: [x/math/learn/training.go]
tests: [x/math/learn/training_test.go, x/math/learn/dense_layer_test.go, x/math/learn/sequential_two_layers_test.go, x/math/learn/gpt_embedding_test.go]
test_result: pass
complete: yes
correct: unknown
evidence: "go test -count=1 -json ./x/math/learn/... (slice lead, 2026-11-09) → 26 pass / 1 skip; probe (2026-11-09)"
---
# learn single-sample training step

**What it does:** one supervised step: forward → loss → loss gradient → `ZeroGrad` → backward → optimizer update; returns loss.

**Where:** `x/math/learn/training.go:13-63` (`TrainStep`).

**Inputs → outputs:** `TrainStep(layer types.Layer, opt types.Optimizer, lossFn nn.LossFunction, input, target tensor.Tensor) (float64, error)`. Validates nil layer/opt/loss and empty tensors (:14-28), wraps errors per stage. Loss is the pre-update value. Whatever batching exists is via the tensor's leading dim; no epoch loop, shuffling, mini-batch gradient accumulation, metrics, early stopping or checkpointing — tests hand-roll epochs.

## Completeness review

- Complete: yes for a single step. Training *loop* (epochs, batching, shuffling, eval) is not provided by the package.
- Tested: `TestTrainStep_NilInputs`/`_EmptyTensors` check error messages; `TestTrainStep_SimpleTraining` checks returned loss equals independently computed loss (InDelta 1e-6, training_test.go:132) and loss non-increasing after one step (:143); `TestTrainStep_MultipleSteps` asserts final loss < 0.1·initial (:213). `TestInputNotModified*` verify input tensors are not mutated by forward/backward/TrainStep (input_modification_test.go:45-184). `TestGPTEmbeddingLearning` asserts only within-cluster cosine similarity > between-cluster (gpt_embedding_test.go:294). All pass (lead run 2026-11-09).
- Correct: unknown beyond these — no gradient-check test here (backward correctness belongs to [[nn]]).

## Notes

- `ZeroGrad` before `Backward` (:52) means gradients never accumulate across calls → no gradient accumulation for micro-batching on memory-tight devices.
- Each step allocates whatever `Forward`/`Gradient` allocate (owned by [[nn]]/[[tensor]]); TrainStep itself adds none.

## TODOs

- [ ] Add `Fit`/epoch loop with shuffling, mini-batches, metrics callback — x/math/learn/training.go:16 #p2
- [ ] Optional gradient accumulation (skip ZeroGrad, update every k) for low-RAM training — x/math/learn/training.go:52 #p2
- [ ] Numerical gradient-check test through TrainStep — x/math/learn/training_test.go #p2

## Links

- Module: [[learn]] · [[nn]] · [[W9-go-nn-framework]]

---
type: algorithm
status: partial
tags: [easyrobot, nn, training]
workstream: W9
module: "[[nn]]"
summary: "`Sequential.Parameters()` merges by `ParamIndex`, so Dense→Dense returns 2 of 4 tensors…"
source: [x/math/nn/builder.go, x/math/nn/models/model.go, x/math/nn/layers/base.go, x/math/nn/types/layer.go, x/math/nn/types/model.go, x/math/nn/types/parameter.go, x/math/learn/training.go, x/math/learn/optimizer.go]
tests: [x/math/nn/layers/base_test.go, x/math/nn/types/parameter_test.go, x/math/learn/training_test.go, x/math/learn/sequential_single_layer_test.go, x/math/learn/sequential_two_layers_test.go, x/math/learn/sequential_classification_test.go, x/math/learn/input_modification_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 ./x/... (2026-11-09) → 25 TestBase_* + 4 types tests pass; go test -count=1 -short ./x/math/learn/ (2026-11-09) → 26 pass / 1 skip; probe (2026-11-09): Sequential(Dense,Dense).Parameters() returns 2 of 4 tensors"
---
# Sequential model, layer lifecycle and backprop

**What it does:** a linear stack of `types.Layer`s with a manual reverse-mode backprop (no autograd tape). Each layer caches its input and output in Forward, computes `dX` and parameter grads in Backward, and an `Optimizer` updates the parameters in place.

**Where:**
- Interfaces:
  - `types/layer.go:9-61` (`Layer`: Name, Init, Parameter(s), Forward, Backward, OutputShape, CanLearn/SetCanLearn, Input/Output, Update, ZeroGrad)
  - `types/model.go:4-7` (`Optimizer{Update(Parameter) error}`), `:12-27` (`Model`)
  - `types/parameter.go:30-34` (`Parameter{Data, Grad, RequiresGrad}`), `ZeroGrad` :115-128
- `layers/base.go`:
  - `Base` :20-33, `NewBase` :37-50 (`canLearn=false`, own RNG)
  - Options :67-163
  - Store/Alloc helpers :285-316
  - `ZeroGrad` :357-369, `Update` :425-454
- Builder: `builder.go:13-102` (`NewSequentialModelBuilder`, `AddLayer`, `WithLayers`, `Build`).
- Model: `models/model.go:14-335` (`Init` :63-96, `Forward` :101-142, `Backward` :146-180, `Parameters` :187-214, `ZeroGrad` :217-233, `Update` :238-262).
- Training glue: `x/math/learn/training.go:14-63` (`TrainStep`); optimisers `learn/optimizer.go:27-57` (SGD), `:120-209` (Adam, state keyed by `param.Data.ID()` :147).

**Lifecycle:**
1. `New*` creates and Xavier-inits the params ([[nn-parameter-initialisation]]).
2. `Build()` checks the `OutputShape` chain only. It does **not** call Init (builder.go:72-80).
3. `model.Init(shape)` calls each layer's `Init`, which preallocates outputs for that exact shape.
4. `TrainStep`: `Forward` → `loss.Compute` → `loss.Gradient` → `ZeroGrad` → `Backward` (reverse) → `Update(opt)`.
5. `Base.Update` calls `opt.Update(param)` by value for each param with Data, RequiresGrad and Grad set (base.go:436-451). This works because `Data`/`Grad` are tensor references mutated in place (SGD `Data.AddScaled(nil, Grad, −lr)`, optimizer.go:54).

## Completeness review

- Complete: no.
  - Linear stacks only: no residual/skip connections, multi-input or multi-output, so no transformer blocks.
  - No train/eval mode.
  - No mini-batch gradient accumulation (most layers overwrite grads).
  - No gradient clipping, no checkpoint save/load API in nn (marshallers exist elsewhere [INFERENCE]).
  - No parameter naming per layer.
- Tested:
  - `TestBase_*` (25) cover options, names, param storage, ZeroGrad, store/alloc. Pass.
  - `types` `Test*ParameterAccess` check params for Dense/Conv. Pass.
  - learn `TestTrainStep_*` and `TestSequential_*` (single, two layers, classification with Sigmoid/ReLU/Tanh), `TestInputNotModified*`, `TestConv{1,2}D_Layer_Learning`, `TestDenseConv{1,2}D_CombinedLearning`, `TestGPTEmbeddingLearning` all pass. Most assert loss decrease or accuracy. `TestTrainStep_SimpleTraining` checks loss values.
  - `xor_test.go` and `mnist*_test.go` sit behind build tags and were not run.
  - `x/math/nn` and `x/math/nn/models` have no tests.
- Correct: no.
  - The Forward/Backward/Update chain is verified end to end only via convergence tests, and per layer by probe `gradcheck` (Dense, Conv, Pool, activations).
  - **`Sequential.Parameters()` merges layer maps by `ParamIndex`**, so later layers overwrite earlier ones (model.go:187-214, documented in a comment at :186). Probe: Dense→Dense returns 2 entries, but 4 tensors exist. Any optimiser, serializer or quantiser that iterates `model.Parameters()` sees only the last layer.
  - `Sequential.Parameter(idx)` is promoted from the model's own empty `Base`, which contradicts the interface doc at types/layer.go:17-20.

## Defects

- Parameters() key collision (models/model.go:187-214). #p1
- `Sequential.Forward` requires `input.Shape() == m.inputShape` exactly (:112). Combined with layers freezing batch size at Init ([[dense-layer]]), a model has one fixed batch size. #p2
- Gradient semantics differ per layer: Dense/Conv/BN overwrite, GPTEmbedding/GPTEncoding accumulate. `ZeroGrad` before Backward hides this for TrainStep, but grad-accumulation loops would silently break. #p2
- `Optimizer.Update(Parameter)` takes the Parameter by value with no identity. Adam keys its state on `param.Data.ID()` (optimizer.go:147). Replacing `Data` (e.g. `SetWeight`, `SetEmbeddingTable`) silently resets Adam moments [INFERENCE]. #p2
- Errors from a layer whose `Backward` returns `nil` gradInput (GPTEmbedding, :215) would be passed to the previous layer as a nil gradient if it were not first in the stack. There is no check in model.go:168-173. #p3
- Builder panics on non-positive dims or nil layers (builder.go:26, :41, :54) instead of returning errors. #p3
- `layers/gradient_test_helper.go:1-112` (numericalGradient, checkGradientAccuracy, …) is test tooling compiled into the library and is **unused anywhere**. #p3
- `Parameter.RequiresGrad` is marked "deprecated, use layer CanLearn" (parameter.go:33), yet both flags gate updates (base.go:438, dense.go:244). #p3

## TODOs

- [ ] Make Sequential.Parameters return per-layer params (e.g. map[layerName]map[ParamIndex]Parameter or a flat slice) — models/model.go:187-214 #p1 #ER-029
- [ ] Graph/residual container (Add, skip connections) for transformer blocks — models/model.go (absent) #p2
- [ ] Remove or move gradient_test_helper.go into a _test file and use it for per-layer finite-difference tests — layers/gradient_test_helper.go:1-112 #p3
- [ ] Add tests for x/math/nn builder/models (Init, Forward, Backward, Parameters) — x/math/nn/models (no tests) #p2

## Embedded suitability

Interface dispatch per layer plus `map[ParamIndex]Parameter` lookups and copies on every Forward/Backward. `Base.Parameters()` allocates a fresh map on every call (base.go:411). Anonymous-interface type assertions run in the ZeroGrad/Update loops (model.go:196-230). Activations are cached per layer, so training memory is about 2× activations. No goroutines (Adam takes a mutex per param). TinyGo [INFERENCE: fine; uses sync/atomic and math/rand only].

## Links

- Module: [[nn]] · training: [[learn]], [[learn-train-step]], [[learn-sgd-optimizer]], [[learn-adam-optimizer]] · related: [[nn-loss-functions]], [[nn-parameter-initialisation]], [[dense-layer]], [[conv2d-layer]], [[gpt-token-embedding-layer]]

---
type: module
group: "Tensor and NN"
status: partial
tags: [easyrobot, nn, tensor]
workstream: [W9, W5]
packages: [x/math/nn, x/math/nn/layers, x/math/nn/models, x/math/nn/types, x/math/nn/qnet, x/math/nn/qnet/tfl]
test_result: fail
---
# nn

**Purpose:** a small Keras/PyTorch-style neural-network layer library on top of the [[tensor]] interface: layers with Init/Forward/Backward, losses, a Sequential model and a builder; training (optimisers, `TrainStep`) lives in [[learn]].

**Embedded int8 inference** lives apart from the float layer library: `x/math/nn/qnet` (int8 graph with residual Add, Mul, Concat, Const, Pad, resize, Mean, Logistic, Softmax and multiple outputs, activation-memory planner, runner on qi8 prepared layers, backend interface, MobileNetV1-0.25 builder) and `x/math/nn/qnet/tfl` (pure-Go TFLite int8 importer; classification and detection models such as BlazeFace) on the [[int8-nn-kernels]] in `x/math/primitive/qi8`; they import only the standard library and qi8 so they build for MCUs ([[int8-graph-runner]]).

**Packages (non-test LOC):**

| Package | Files | LOC | Content |
|---|---|---|---|
| `x/math/nn` | 4 | 493 | `LossFunction`, `MSELoss`, `CrossEntropyLoss`, `CategoricalCrossEntropy`, `SequentialModelBuilder` (`builder.go`) |
| `x/math/nn/layers` | 13 | 6075 | `Base` + options, Dense, Conv1D, Conv2D, Max/Avg/GlobalAvg pooling, ReLU/Sigmoid/Tanh/Softmax, BatchNorm2D, Dropout, LSTM, GPTEmbedding, GPTEncoding, GPTAttention, Flatten/Reshape/Squeeze/Unsqueeze/Transpose/Pad/Concatenate |
| `x/math/nn/models` | 1 | 335 | `Sequential` |
| `x/math/nn/types` | 3 | 216 | `Layer`, `Model`, `Optimizer`, `Parameter`, `ParamIndex` |

No build tags. Declared docs: `nn/SPEC.md`, `nn/REQ_OPS.md`, `layers/SPEC.md` (lists e.g. ResidualBlock/SEBlock that do not exist), `layers/OPTIMIZATION_ANALYSIS.md`, `layers/LEARNING_OPTIMIZATION_PLAN.md`, `layers/BENCHMARK_REPORT.md`.

## Public API

- `types.Layer` (15 methods): `Name, Init(inputShape), Parameter(idx), Parameters(), Forward(x), Backward(gradOut), OutputShape, CanLearn, SetCanLearn, Input, Output, Update(optimizer), ZeroGrad`. `types.Model` = Layer + `LayerCount/GetLayer`. `types.Optimizer{ Update(Parameter) error }` — implemented by `learn.SGD`, `learn.Adam`.
- `layers.Base` is embedded by every layer: parameter map keyed by `types.ParamIndex` (`ParamWeights`, `ParamBiases`, `ParamKernels`, LSTM-specific 101+), pre-allocated output/grad tensors (`AllocOutput`), stored input/output for backward, options `WithName/WithCanLearn/UseBias/WithDataType/WithRNG/WithWeights/WithBiases/WithKernels/WithParameter(s)`.
- Lifecycle: construct → `Init(inputShape)` (allocates output; shape incl. batch is frozen) → `Forward` → loss `Gradient` → `Backward` (fills `Parameter.Grad`) → `Update(opt)` → `ZeroGrad`. `learn.TrainStep(layer, opt, loss, x, y)` wraps one iteration.
- `nn.NewSequentialModelBuilder(inputShape).AddLayer(...).Build()` validates shapes via `OutputShape` and name uniqueness → `*models.Sequential` (itself a `Layer`, so models nest).

## Algorithms

Status and evidence per note (value probes by the layer slice in `~/src/tmp/easyrobot/tensor/layernotes/{densegrad,embed1d,gradcheck,micro,xavier,utilcheck}`):

<!-- vault:algorithms h=e8911f50ae2b -->

Counts: 2 tested · 3 implemented · 10 partial · 0 stub · 0 unverified (15 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[activation-layers]] | tested | pass | gradients match finite differences |
| [[int8-graph-runner]] | tested | pass | Int8 graph (layers with producers and multiple outputs: residual Add, Mul, Concat, Const, Pad, resize, Mean, Logistic, Softmax) + greedy activation-memory planner + runner on qi8 prepared layers (PIE on the S3), TFLite int8 and uint8 importer (pure Go; BlazeFace and TF's uint8 MobileNetV1-0.25 quant models load bit-exact), MobileNetV1-0.25 builder; builds for ESP32-S3 and K210. |
| [[dense-layer]] | implemented | fail | backward exact by central-difference probe; `TestDense_NumericalGradientCheck` uses float32… |
| [[nn-loss-functions]] | implemented | none | gradients correct by probe; MSE averages, CrossEntropy sums (inconsistent scaling) |
| [[nn-parameter-initialisation]] | implemented | pass | Xavier statistics correct; biases initialised randomly instead of zero |
| [[batchnorm2d-layer]] | partial | none | computes InstanceNorm; running stats/momentum/eval mode unused |
| [[conv1d-layer]] | partial | pass | kernel gradient wrong for >1 output channel (probe `micro`: got `[0 -0.5 8 11.5]` vs ref `[5 6.5 5… |
| [[conv2d-layer]] | partial | pass | input gradient wrong when `(H+2p-k) % s ≠ 0` (maxAbs 1.52); tests only 1→1 channel, stride 1 |
| [[gpt-attention-layer]] | partial | none | `Forward` always panics; `Backward` zeroes gradient; no RoPE/GQA/KV cache |
| [[gpt-positional-encoding-layer]] | partial | none | panics for seqLen ≥ 2; reversed `Copy` (`gpt_encoding.go:193,209,271`) |
| [[gpt-token-embedding-layer]] | partial | fail | 2-D path correct; 1-D path panics or overwrites the table (`gpt_embedding.go:145-147`) |
| [[lstm-layer]] | partial | none | forward wrong (all gates read `gates[0:h]`, reproduced by probe); `Backward` not implemented… |
| [[pooling-layers]] | partial | pass | Max/Avg gradients correct; `GlobalAvgPool2D.Backward` returns zeros of shape `[B,C]`… |
| [[sequential-model-and-backprop]] | partial | pass | `Sequential.Parameters()` merges by `ParamIndex`, so Dense→Dense returns 2 of 4 tensors… |
| [[shape-utility-layers]] | partial | pass | `Squeeze`/`Unsqueeze` `Backward` panic (`utility.go:427,587`); `PadReflect` zero-pads… |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Pure Go, CGO-free; default build single-threaded. Depends on [[tensor]] (eager), [[primitive-kernels]], [[type-conversion]], `chewxy/math32`.
- **`pooling_test_edgecases.go` is a production file** that imports `testing` and testify (`go list -deps ./x/math/nn/layers` lists `stretchr/testify/{assert,require}`, `gopkg.in/yaml.v3`): a TinyGo or size-sensitive build pulls test libraries; its tests never run.
- Allocation: outputs/grads pre-allocated in `Init` (good for embedded); batch size frozen at `Init`; kernels underneath may still allocate per call ([[conv2d-kernel]]).
- float64 scalars in the tensor API (eps, momentum, dropout rate) — soft-float on ESP32 (see [[W9-go-nn-framework]]).

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/math/nn/...` (2026-11-09): `nn/layers` 91 pass, 2 fail (`TestDense_NumericalGradientCheck`, `TestGPTEmbedding_ErrorCases`); `nn/types` 4 pass; `nn`, `nn/models` no test files. Same as baseline 2026-11-08.
- `go test -count=1 -short ./x/math/learn/` (layer slice, 2026-11-09): 26 pass, 1 skip — exercises Sequential + SGD/Adam end to end (see [[learn]]).
- Passing tests are weak: most layer tests use 1 channel, stride 1, all-ones kernels, and GPT/LSTM/BatchNorm have no value tests at all, which is how the defects above survive a green suite.

## API notes (ease of use, composability)

- **Good:** one `Layer` interface for layers and models (models nest); `dst`-pre-allocation pattern keeps the steady state allocation-light; functional options on most constructors; builder validates shapes before the first `Forward`.
- **Wide interface:** 15 methods incl. `Update`, `ZeroGrad`, `Input/Output`, `SetCanLearn`; inference-only wrappers ([[tflite]]) must stub training methods. A split `Forwarder` / `Trainable` would be simpler.
- **Inconsistent constructors:** `NewReLU(name)`, `NewSoftmax(name, dim)`, `NewBatchNorm2D(numFeatures, eps, momentum, name)`, `NewMaxPool2D(kh,kw,sh,sw,ph,pw)` with its own `MaxPool2DOption`, `NewDropout(name, ...DropoutOption)`, `NewDense(in, out, ...Option)`. One `Option` type and a consistent name option would make composition uniform.
- **Parameter slots by enum:** `ParamIndex` is per-layer; `GPTAttention` stores its output projection in the `ParamBiases` slot (`gpt_attention.go:81-84`), and `Sequential.Parameters()` collides across layers. Named, hierarchical parameter keys (`"block0.attn.q"`) are needed for loading external weights (safetensors/GGUF) and for optimiser state.
- **Topology:** only `Sequential`; no residual/add, branch or multi-input layers (declared in `layers/SPEC.md`, absent), so ResNet blocks and transformer blocks cannot be built.
- **Batch frozen at `Init`:** changing batch size (training vs single-sample inference, variable sequence length for LLM decode) requires re-`Init`.
- **Errors vs panics:** layers return errors, but underlying tensor ops panic on shape/dtype problems; builders mix panics (`AddLayer(nil)`) and errors.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Residual/Add and multi-input layers (needed for transformer and ResNet/MobileNet blocks) — layers/SPEC.md:494 declared only #p2
- [ ] Unify constructor/option style across layers — layers/*.go #p3

## Inferred objectives (proposals)

- **Proposal:** freeze new layer work until a reference-comparison harness exists (numpy/PyTorch-generated fixtures per layer, forward and backward, multi-channel, stride > 1), then fix layers in order of W5/W9 need: Conv2D/Depthwise/BN/GlobalAvgPool (vision heads) → RMSNorm/attention/RoPE (LLM).
- **Proposal:** introduce named parameters and a `Graph`/`Block` composition (residual, branches) before the LLM work; it also gives a natural weight-loading map.
- **Proposal:** an inference-only, float32, zero-allocation "embedded" layer subset for TinyGo (Dense, Conv2D, Depthwise, ReLU/ReLU6, GlobalAvgPool, Softmax).

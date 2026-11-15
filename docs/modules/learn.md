---
type: module
group: "Tensor and NN"
status: partial
tags: [easyrobot, learn, nn]
workstream: [W9, W6, W5]
packages: [x/math/learn, x/math/learn/datasets/mnist]
test_result: pass
---
# learn

**Purpose:** training utilities on top of [[nn]] layers and [[tensor]]: optimizers (SGD, Adam), a single-step trainer, post-training quantization helpers, and an MNIST CSV loader. Most of the package by LOC is integration tests that double as the de-facto test suite for `nn` layer learning.

**Packages:**

| Path | Files | LOC | Tags |
|---|---|---|---|
| `x/math/learn` | optimizer.go (222), training.go (63), quantization.go (419) | 704 | none |
| `x/math/learn/datasets/mnist` | loader.go | 94 | none |
| tests (untagged) | dense_layer, conv1d_layer, conv2d_layer, gpt_embedding, input_modification, sequential_{single_layer,two_layers,classification}, training, quantization | 2571 | none |
| tests (tagged, not run) | mnist_test.go, mnist_convnet_test.go (`mnist_test`); mnist_large_test.go (`mnist_large_test`); xor_test.go (`xor_test`) | 2460 | build tags |

## Public API

- `NewSGD(lr) *SGD`, `NewAdam(lr, β1, β2, ε) *Adam` → both satisfy `nn/types.Optimizer` (`Update(types.Parameter) error`); layers call it from `layer.Update(opt)`.
- `TrainStep(layer, opt, lossFn, input, target) (float64, error)` — forward/loss/backward/update.
- Quantization: `NewCalibrator(method, scheme, bits)`, `AddSample/AddTensor`, `ComputeParams() (*QuantizationParams, error)`, `QuantizeTensor`, `DequantizeTensor`, `QuantizeModel(layer, opts...) (map[string]*QuantizationParams, error)`; enums `QuantSymmetric|QuantAsymmetric|QuantPerChannel|QuantPerTensor`, `CalibMinMax|CalibPercentile|CalibKLDivergence`.
- `mnist.Load(path, max) ([]mnist.Sample, error)`.

Composition: build `nn.NewSequentialModelBuilder(...)` / layers → `Init(shape)` → loop calling `learn.TrainStep` with an optimizer and `nn.LossFunction`. Epoch/batch/shuffle/eval loops are written by hand in every test.

## Algorithms

<!-- vault:algorithms h=fbbebca42f21 -->

Counts: 2 tested · 2 implemented · 1 partial · 0 stub · 0 unverified (5 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[learn-sgd-optimizer]] | tested | pass | dense/sequential/training tests pass; probe exact |
| [[learn-train-step]] | tested | pass | training_test pass (loss value, loss <0.1·initial) |
| [[learn-adam-optimizer]] | implemented | pass | indirect (conv ≥75% acc) pass; probe matches Kingma&Ba ref to 1e-7 |
| [[mnist-csv-gz-loader]] | implemented | not-run | only tagged tests (not run) |
| [[learn-post-training-quantization]] | partial | pass | 5 pass, TestQuantizeModel skipped; per-channel/KL missing |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Pure Go, no CGO. Imports `nn`, `nn/types`, `tensor`, `x/iter`; loader uses `compress/gzip`, `encoding/csv`, `os`.
- TinyGo/ESP32 suitability for **on-device training** [INFERENCE from code, nothing built for TinyGo]:
  - Hyperparameters and quant params are float64; Adam uses `math.Pow` float64 per step (optimizer.go:174). ESP32 (Xtensa LX6/LX7) FPU is single-precision only → float64 is soft-float.
  - Optimizer state always `DTFP32` (optimizer.go:155); Adam allocates 8 full-size tensors per parameter (:154-166) ≈ 9× weights. SGD allocates nothing. On 520 KB SRAM ESP32 only tiny dense nets are trainable, realistically with SGD.
  - Adam state is a `map[uintptr]` + `sync.Mutex` — works in TinyGo but map growth allocates.
  - Quantization calibration stores every sample and insertion-sorts (quantization.go:163,298) — not device-friendly; quantized output is FP32 storage (:341), so no memory win.
  - Loader needs filesystem + gzip; host-only in practice.
  - Forward/backward allocation behaviour lives in [[nn]]/[[tensor]] (not assessed here).

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/math/learn/...` (2026-11-09) → **26 pass, 1 skip**. `datasets/mnist` has no tests.
- Skip: `TestQuantizeModel` — `t.Skip("Requires actual model implementation to test fully")`, quantization_test.go:145; test body is a placeholder.
- What tests assert:
  - Loss decrease: dense/sequential regression assert `final < initial` and w≈2±0.5, b≈1±0.5, predictions ±0.5 (dense_layer_test.go:132-153); `TestTrainStep_MultipleSteps` final < 0.1·initial (training_test.go:213).
  - Accuracy thresholds: sequential classification ≥50% (sequential_classification_test.go:172 — weak, coin-flip level); conv1d/conv2d ≥75% (conv1d_layer_test.go:179,342; conv2d_layer_test.go:230,444); GPT embedding: within-cluster similarity > between (gpt_embedding_test.go:294).
  - Tagged (read, not run): MNIST test accuracy >15% (mnist_test.go:247-249); XOR configs ≥ expectedMinAcc and "all configs 90%+" (xor_test.go:459,469). Require `datasets/mnist/*.csv.gz`.
  - Input immutability through forward/backward/TrainStep (input_modification_test.go).
- Probe (`GOFLAGS=-mod=mod go run .`, 2026-11-09): Adam 3 steps vs float64 reference diff ≤1e-7; SGD exact; asymmetric [-1.003,2] → scale 0.011776 zp 85; `QuantPerChannel` → `unsupported scheme 2`; quantized tensor dtype FP32.
- grep TODO/FIXME/`not implemented`: none in package; only placeholder skip above.

## API notes (ease of use, composability)

- Good: optimizer is a one-method interface, so custom optimizers drop in; `TrainStep` is small and errors are wrapped per stage; input tensors are guaranteed unmodified (tested).
- Missing loop layer: no `Fit`, batching, shuffling, metrics, callbacks, LR schedules, checkpointing → every test re-implements 50–100 lines of loop. A `Trainer` with options would remove most test boilerplate.
- `NewSGD`/`NewAdam` panic on bad args while everything else returns errors — inconsistent; Adam doc promises defaults that don't exist (optimizer.go:89).
- Adam state keyed by data pointer is invisible/unresettable; re-initialised layers silently lose state.
- Quantization enum mixes scheme (sym/asym) with granularity (per-tensor/per-channel); two of four values error. `QuantizeModel` returns params keyed by `fmt.Sprintf` of index — stringly typed, and doesn't quantize anything. `QuantizeTensor` returns params it was given (redundant).
- Loader takes a filename only (no `io.Reader`, no embed), fixed 28×28.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Raise classification accuracy threshold above 50% — x/math/learn/sequential_classification_test.go:172 #p3
- [ ] float32 hyperparameter path for MCU targets — x/math/learn/optimizer.go:15,64-67 #p3

## Inferred objectives (proposals)

- **proposal:** treat ESP32 as inference-only with int8 weights produced on host by this package → needs per-channel symmetric weights, int8 storage, activation calibration and an export format ([[marshallers]]), feeding [[W9-go-nn-framework]].
- **proposal:** on-device learning limited to last-layer fine-tuning with SGD (zero extra state) for behaviour adaptation in [[W6-decision-trees]].
- **proposal:** add a `Trainer` (epochs, batches, metrics callback) so experiments in [[W5-feature-tensor-experiments]] stop duplicating loops.

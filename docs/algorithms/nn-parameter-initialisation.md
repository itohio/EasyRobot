---
type: algorithm
status: implemented
tags: [easyrobot, nn, init]
workstream: W9
module: "[[nn]]"
summary: "Xavier statistics correct; biases initialised randomly instead of zero"
source: [x/math/nn/types/parameter.go, x/math/tensor/eager/initialization.go, x/math/tensor/tensor.go, x/math/nn/layers/base.go]
tests: [x/math/nn/layers/dense_comprehensive_test.go, x/math/nn/layers/conv2d_weight_bias_test.go, x/math/nn/layers/conv1d_weight_bias_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 ./x/... (2026-11-09) → Test{Dense,Conv1D,Conv2D}_WeightBiasInitializationAndCloning pass (check presence/override, not distribution); probe tmp/easyrobot/tensor/layernotes/xavier (2026-11-09): uniform ±0.0866 std 0.0499 (expect 0.0500), normal std 0.0501 (expect 0.0500), same seed reproducible"
---
# NN parameter initialisation (Xavier/Glorot, RNG)

**What it does:** creates parameter tensors at layer construction with Xavier-uniform values, unless the caller pre-set them via `WithWeights`/`WithBiases`/`WithKernels`/`WithParameter`.

**Where:**
- `x/math/nn/types/parameter.go:48-67` (`Parameter.Init`), `:71-111` (`inferFanSizes`), `:115-128` (`ZeroGrad`, lazy grad allocation).
- `x/math/tensor/eager/initialization.go:14-25` (`XavierUniform`, limit √(6/(fanIn+fanOut))), `:31-42` (`XavierNormal`, σ=√(2/(fanIn+fanOut))), `:48-64` (`…Like`). Re-exported at `x/math/tensor/tensor.go:80-109`.
- RNG: `layers/base.go:46` (default `rand.New(rand.NewSource(rand.Int63()))` per layer), `WithRNG` :102-108.

**Inputs → outputs:** `(dtype, shape, paramIdx, fanIn, fanOut, rng, requiresGrad)` → `p.Data`. Fan inference (`inferFanSizes`):
- Biases: `(1, shape[0])`
- Weights 2D: `(shape[0], shape[1])`
- Kernels: `(Cin·k…, Cout)`
- Default: `(first, last)`

## Completeness review

- Complete: yes for Xavier-uniform.
- Not wired up: `XavierNormal` (no nn caller), He/Kaiming (better for ReLU nets), zeros/ones/normal(0,0.02) presets, an init-strategy option on layers.
- Tested: the `*_WeightBiasInitializationAndCloning` tests check that params exist, have the right shape, and that pre-set tensors are kept. No test checks the distribution, and `tensor/eager` has no Xavier test. All pass.
- Correct: yes for the formulas. Probe `xavier`:
  - Uniform 500×300: min/max ±0.0866 = √(6/800); std 0.04992 (theory 0.0500).
  - Normal: std 0.05013 (theory 0.0500).
  - Two layers with the same `WithRNG` seed produce identical weights.
  - `XavierUniform(…, rng=nil)` returns nil.

## Defects

- Biases are Xavier-initialised instead of zero. Probe: Dense(4,2) bias = [−0.81 −1.06], limit √(6/(1+out)) = 1.41 for out=2. Large random biases slow early training. This affects Dense, Conv, LSTM and Attention (parameter.go:76-79; dense.go:65; conv2d.go:96; lstm.go:94). #p2
- Conv `fanOut` is inferred as `Cout` instead of `Cout·kh·kw` (parameter.go:90-101). Kernels come out larger than Glorot intends. #p3
- The default RNG is seeded from the global `rand.Int63()`, so results are non-reproducible unless `WithRNG` is passed to every layer. Each layer holds its own `*rand.Rand` (~5 KB state each in Go's math/rand [INFERENCE]), which costs RAM on ESP32 (base.go:46). #p3
- `Init` silently leaves `Data=nil` if `rng` is nil (`XavierUniform` returns nil, initialization.go:15-17). The next Forward then fails with "weight parameter not initialized". `WithRNG(nil)` is ignored, so this is only reachable via direct calls. #p3
- `XavierUniform` fills through the `Elements()` iterator with `elem.Set(float64)`, one interface call per element (:21-23). That is slow for 100M-parameter tables, though it runs once. #p3
- `BatchNorm2D.Init` calls `Parameter.Init` (Xavier) and then discards the result (activations.go:636, :648). See [[batchnorm2d-layer]].

## TODOs

- [ ] Zero-init biases by default — types/parameter.go:76-79 #p2
- [ ] Add He/Kaiming init and a per-layer init option; use fanOut=Cout·k for kernels — types/parameter.go:71-111 #p3
- [ ] Add distribution tests for XavierUniform/XavierNormal — x/math/tensor/eager/initialization.go (no test) #p3
- [ ] Share one RNG across a model (builder-level WithRNG) to save RAM and make runs reproducible — layers/base.go:46 #p3

## Embedded suitability

Runs at construction only. Uses float64 math (`math.Sqrt`, `rng.Float64`). A per-layer RNG costs RAM. Not in the hot path.

## Links

- Module: [[nn]] · kernels: [[tensor]] · related: [[sequential-model-and-backprop]], [[dense-layer]], [[conv2d-layer]], [[lstm-layer]]

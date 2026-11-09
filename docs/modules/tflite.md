---
type: module
group: "Tensor and NN"
status: unverified
tags: [easyrobot, nn, tflite, marshaller]
workstream: [W9, W4]
packages: [x/marshaller/tflite]
test_result: not-run
---
# tflite

**Purpose:** run a TensorFlow Lite flatbuffer model through the native TFLite C runtime (via `github.com/mattn/go-tflite`, CGO) and expose it as an [[nn]] `types.Model`, so a TFLite model can stand in for a native layer stack. Inference only.

**Packages:** `x/marshaller/tflite` — `model.go` 425 LOC, `unmarshaller.go` 171 LOC, `unmarshaller_test.go` 128 LOC, `INSTALL_TFLITE.md` (declared install notes). All files `//go:build linux`; needs `libtensorflowlite_c` at link time. A second test lives in the parent package: `x/marshaller/tflite_unmarshaller_test.go` (tag `tflite`, uses `testdata/mnist_model.tflite`).

## Public API

- `NewModel(modelData []byte, name string, opts ...InterpreterOption) (*Model, error)`; options `WithNumThreads(n)`, `WithErrorReporter(func(string))`.
- `Model` implements `nn/types.Model`: `Init` (validates the shape against the model's input), `Forward(input) (tensor.Tensor, error)`; `Backward`/`Update` return errors (inference only); `GetInputTensor/GetOutputTensor/Invoke` expose the raw interpreter.
- `Unmarshaller` (marshaller-registry format `"tflite"`, `Unmarshal(r, dst)` into `**Model` or `*types.Model`), `Loader` with `LoadFromFile/LoadFromReader`.
- Composition: a `*tflite.Model` can be used wherever an `nn` `Layer`/`Model` is accepted, e.g. as a feature extractor before native heads (W4/W5 idea) — on linux hosts only.

## Algorithms

<!-- vault:algorithms h=559a89c4d8d1 -->

Counts: 0 tested · 0 implemented · 0 partial · 0 stub · 1 unverified (1 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[tflite-model-import]] | unverified | build-fail | not built on darwin; needs linux + native TFLite |

<!-- /vault:algorithms -->

## Build targets and dependencies

- linux only, CGO + native TFLite C library; not available on this host (darwin) and not installed. Excluded from `go list ./...` on darwin (baseline §1).
- Not usable with TinyGo or on ESP32. On K230 Linux it would need a cross-compiled TFLite C library and CGO cross-toolchain [INFERENCE]; K230's NPU path is nncase/kmodel, not TFLite (ER-007).

## Tests and evidence

- Not built, not run: darwin host, no TFLite library (baseline 2026-11-08 §1; still true 2026-11-09). `unmarshaller_test.go` loads an MNIST model via `x/math/learn/datasets/mnist` and checks predictions; it needs model/testdata files.
- Static reading (2026-11-09) of `model.go:180-266`:
  - single input/single output only (`GetInputTensor(0)`, `GetOutputTensor(0)`, `model.go:195,214`);
  - **output is always allocated as FP32** (`model.go:220`) and filled with `CopyToBuffer` (`model.go:258`): an int8/uint8 quantised output tensor has a different byte size, so `Forward` fails or misreads instead of dequantising; `copyFromTFLiteTensor` has no `[]int8` case (`model.go:255-265`);
  - input is copied from `input.Data()` (`model.go:201`), i.e. the full backing slice, without checking the interpreter tensor's dtype or a view's offset/strides; an FP32 tensor fed to a uint8 model input fails on size;
  - `[]float64` input allocates a conversion buffer every call (`model.go:236`).

## API notes (ease of use, composability)

- Good: it plugs into the same `nn/types.Model` interface as native models; `Unmarshaller` fits the marshaller registry pattern ([[marshallers]]).
- Weak: no quantisation parameters (scale/zero-point) surfaced; FP32-only output makes quantised models (the common case on edge devices) unusable through `Forward`; multi-input/multi-output models (detectors with several heads) are not supported; `Backward`/`Update` are dead methods forced by the wide `Layer` interface.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Allocate the output tensor with the interpreter tensor's dtype (or dequantise with its scale/zero-point) instead of always FP32 — x/marshaller/tflite/model.go:220,255-265 #p1 #backlog
- [ ] Multi-input/multi-output support for detector heads — model.go:195,214 #p2
- [ ] Linux CI job with the native library to run `unmarshaller_test.go` — not run anywhere #p2

## Inferred objectives (proposals)

- **Proposal:** use TFLite on a linux host only as a *reference* for validating native kernels (same model, compare outputs), not as a deployment path for ESP32/K230.
- **Proposal:** if W5 experiments need pre-trained backbones, a pure-Go TFLite flatbuffer *weight reader* (no interpreter) feeding native layers would serve TinyGo too; scope it as a research spike.

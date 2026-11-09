---
type: algorithm
status: unverified
tags: [easyrobot, tflite, inference, marshaller]
workstream: W9
module: "[[tflite]]"
summary: "not built on darwin; needs linux + native TFLite"
source: [x/marshaller/tflite/model.go, x/marshaller/tflite/unmarshaller.go]
tests: [x/marshaller/tflite/unmarshaller_test.go]
test_result: build-fail
complete: no
correct: unknown
evidence: "linux-only (//go:build linux); not in 2026-11-09 darwin run; GOOS=linux GOARCH=arm64 CGO_ENABLED=0 go vet/build -o /dev/null ./x/marshaller/tflite → fails in github.com/mattn/go-tflite@v1.0.5 (cgo required: undefined TensorType/Status) (2026-11-09)"
---
# TFLite model import (x/marshaller/tflite)

**What it does:** Loads a `.tflite` flatbuffer into a `mattn/go-tflite` (C API) interpreter and exposes it as an inference-only `nn/types.Model` (`Forward` copies input in, invokes, copies output into a new eager FP32 tensor).

**Where:** `x/marshaller/tflite/model.go:44-96` `NewModel` (NewModel → InterpreterOptions → options → NewInterpreter → AllocateTensors → `cacheShapes`), `:99-127` `cacheShapes` (**exactly 1 input and 1 output**, else error), `:180-228` `Forward`, `:231-252` `copyToTFLiteTensor`, `:255-266` `copyFromTFLiteTensor`, `:276-340` Model interface no-ops (Backward error, Parameters empty, LayerCount 0), `:343-378` raw accessors (`GetInputTensor`, `Invoke`), `:385-425` options `WithNumThreads`, `WithErrorReporter`. `unmarshaller.go:21-171` `Unmarshaller` (format "tflite", `Unmarshal(r, dst *Model)`), `Loader` helpers (`LoadFromFile/Reader`).

**Inputs → outputs:** input tensor whose shape equals the model's input dims and whose `Data()` is `[]float32 | []float64 (converted to float32 with an extra alloc) | []int32 | []int64 | []int8 | []uint8`; output always `tensor.New(DTFP32, outputDims)`.

## Completeness review

- Complete: **no**.
  - Single input/output only (`model.go:102-110`).
  - No dtype check against the TFLite tensor type; no quantisation params: a uint8/int8-quantised model needs raw int input — float input fails with a byte-size status error, and the output path always allocates FP32 (`:220`) so an int8/uint8 output tensor fails `CopyToBuffer` (size mismatch) or is mis-read. No dequantisation via `QuantizationParams` [INFERENCE from go-tflite API].
  - Delegates are advertised (`:21,43,68`) but only threads + error reporter options exist (`:385-425`).
  - `Forward` uses `input.Data()` without honouring offset/strides of views [INFERENCE: wrong for non-contiguous tensors].
- Tested: `unmarshaller_test.go` (linux-only) loads an MNIST model and skips when TFLite is unavailable (`isTFLiteUnavailable`). Not run (darwin host; cgo + libtensorflowlite_c required).
- Correct: unknown.

**Copies per Forward:** 1 input copy (2 for float64), 1 output allocation (pool) + copy.

**Embedded suitability:** Linux + CGo + libtensorflowlite_c only — not TinyGo/ESP32 (that would need TFLM in C). Feasible on K230 Linux if the C library is cross-built (Makefile.tflite / Dockerfile.tflite present, declared).

## TODOs

- [ ] Check input/output `TensorType` and support int8/uint8 I/O with scale/zero-point (dequantise output) — x/marshaller/tflite/model.go:201,220 #p1 #backlog
- [ ] Multi-input/output models — x/marshaller/tflite/model.go:102 #p2
- [ ] Delegate options (XNNPACK/GPU/NPU) or remove claims — x/marshaller/tflite/model.go:21 #p3
- [ ] Contiguous-copy for strided input views — x/marshaller/tflite/model.go:201 #p2
- [ ] CI job (linux, cgo) running unmarshaller_test — x/marshaller/tflite/unmarshaller_test.go #p2

## Links

- Module: [[tflite]]
- Related: [[quantised-dtypes-and-fixed-point]], [[gorgonia-graph-backend]]

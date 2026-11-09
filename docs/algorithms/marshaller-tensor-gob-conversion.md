---
type: algorithm
status: tested
tags: [easyrobot, marshaller, gob, tensor, serialization]
workstream: [W9]
module: "[[marshallers]]"
summary: "4 pass; probe all dtypes ok"
source: [x/marshaller/gob/internal.go, x/marshaller/gob/convert.go, x/marshaller/gob/marshaller.go, x/marshaller/gob/unmarshaller.go]
tests: [x/marshaller/gob/marshaller_test.go, x/marshaller/gob/xor_test.go, x/marshaller/marshaller_test.go, x/marshaller/example_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "go test -count=1 -v ./x/marshaller/gob ./x/marshaller → ok, 4+6 PASS (2026-11-09); probe tmp/easyrobot/io/fmtprobe: all 8 dtypes round-trip exactly (2026-11-09)"
---
# Tensor / model ↔ Go gob

**What it does:** wraps tensors (dtype, shape, raw typed slice), parameters, layers, models and slices in gob-friendly structs and encodes with `encoding/gob`; on decode rebuilds tensors through the configured `TensorFactory`, optionally converting dtype.

**Where:** wire structs `internal.go:12-63`, `tensorToGob` :65, model/layer :77-138; decode `gobToTensor` `convert.go:12-88` (typed copy per dtype incl. INT8, INT16, UINT8, C64 :31-85), `gobToTensorWithConversion` :90-124; `Marshaller.Marshal` `marshaller.go:33-58` (per-call `localOpts` :35-38); `Unmarshaller.Unmarshal` `unmarshaller.go:35-62`.

**Inputs → outputs:** any tensor/model/slice → gob bytes (Go-only format; not readable from C/Python/MCU without a gob decoder).

## Completeness review

- Complete: yes for Go↔Go persistence of every eager dtype.
- Tested: tensor round trip, array round trip, dtype conversion, XOR model parameters (4 PASS) plus root-package `TestGobMarshallerTensor`/`TestGobMarshallerSlice` (values compared).
- Correct: yes for the dtypes probed — `fmtprobe`: FP32, FP64, INT32, INT64, INT, INT16, INT8, UINT8 2×2 tensors round-trip exactly (shape and data). The conversion path (`gobToTensorWithConversion`) uses `SetAt` and inherits the INT32/INT64/UINT8 `SetAt` panic of the eager tensor [INFERENCE from code, not probed].
- `withMetadata` merges into the map shared between the base and per-call options copy (`types.go` `withMetadata.Apply`; `localOpts := m.opts` copies the map header) — metadata leaks across calls [INFERENCE].

## TODOs

- [ ] Typed conversion in `gobToTensorWithConversion` (avoid `SetAt`) — x/marshaller/gob/convert.go:90-124 #p2
- [ ] Deep-copy `Metadata` when cloning options per call — x/marshaller/gob/marshaller.go:35 #p3

## Links

- Module: [[marshallers]]; [[marshaller-tensor-proto-conversion]]; [[nn]]

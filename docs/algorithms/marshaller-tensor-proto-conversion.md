---
type: algorithm
status: partial
tags: [easyrobot, marshaller, protobuf, tensor, serialization]
workstream: [W4, W9]
module: "[[marshallers]]"
summary: "partial (INT8/INT16 zeroed, UINT8 panic)"
source: [x/marshaller/proto/convert.go, x/marshaller/proto/marshaller.go, x/marshaller/proto/unmarshaller.go, types/core/types.pb.go]
tests: [x/marshaller/proto/marshaller_test.go, x/marshaller/proto/xor_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 -v ./x/marshaller/proto → ok, 4 PASS (2026-11-09); probe tmp/easyrobot/io/graphprobe: INT16/INT8 round-trip to zeros, UINT8 panics on unmarshal (2026-11-09)"
---
# Tensor / model ↔ protobuf conversion (`x/marshaller/proto`)

**What it does:** maps `tensor.Tensor`, NN `Parameter`/`Layer`/`Model` and numeric slices into the `types/core` protobuf messages (`Tensor{dtype string, shape []int32, data_f32|f64|i32|i64}`, `Value` oneof) and back; `proto.Message` values pass through untouched. This is the wire format closest to what a K230→ESP32 feature stream ([[W4-k230-feature-stream]]) or dndm messages ([[dndm-and-protobuf]]) would use.

**Where:** `dtypeToString`/`stringToDtype` `convert.go:12-64`; `tensorToProto` :66-124; `protoToTensor` :126-234; `parameterToProto`/`layerToProto`/`modelToProto` :236-320; `Marshaller.Marshal` `marshaller.go:35-63`, `valueToProto` :65-124, `sliceToProto` :126-172; `Unmarshaller.Unmarshal` `unmarshaller.go:38-67`.

**Inputs → outputs:** `Marshal(w, value)` → protobuf bytes; `Unmarshal(r, &dst)` with `dst` a `proto.Message`, `*types.Tensor`, model/layer or slice pointer; `WithDestinationType(dt)` converts element-wise through `SetAt(float64)`.

## Completeness review

- Complete: **no**.
  - Only FP32, FP64, INT32, INT64, INT and C64 carry data (`convert.go:85-121`). INT16, INT8, FP16, INT48 write a dtype string with **no data**; UINT8 (the image dtype) maps to `"unknown"` (`convert.go:12-37`, no UINT8 case) and `protoToTensor` then calls the factory with `DT_UNKNOWN`, which panics.
  - Empty tensor → `tensorToProto` returns `nil, nil` (:67-69) → the `Value` carries a nil tensor that `protoToTensor` rejects ("invalid tensor proto").
  - No bounds check: `INT` path writes `destData[i]` for every element of `DataI64` (:199-207) → panic if the payload is longer than the shape.
  - Per-call options are applied to the marshaller's own `opts` (`marshaller.go:37-39`, `unmarshaller.go:40-42`), so options leak into later calls and concurrent use races (gob/json/yaml/csv copy to `localOpts`).
  - Conversion path uses `Tensor.SetAt`, which panics for INT32/INT64/INT/UINT8 destinations in the eager tensor (reported to [[tensor]] owner; `x/math/tensor/eager/tensor.go:383-405`).
- Tested: `TestTensorRoundTrip` (FP32 values), `TestArrayRoundTrip`, `TestTensorTypeConversion`, `TestXORModelParametersRoundTrip` — 4 PASS. No test covers INT8/INT16/UINT8.
- Correct: **no** for small integer dtypes. Probe (`graphprobe/main.go`, 2×2 tensors filled 1..4): FP32, FP64, INT32, INT64, INT round-trip exactly; INT16 → `[0 0 0 0]`, INT8 → `[0 0 0 0]` (silent data loss), UINT8 → panic `unsupported dtype: 0`.

## TODOs

- [ ] Encode INT8/UINT8/INT16/FP16 (e.g. `data_bytes` packed, or widen to `data_i32`) and add UINT8 to the dtype map — x/marshaller/proto/convert.go:12-121 #p1 #ER-027
- [ ] Error on unsupported dtypes instead of emitting empty data — x/marshaller/proto/convert.go:85-121 #p1 #ER-027
- [ ] Copy options per call (`localOpts`) like the other marshallers — x/marshaller/proto/marshaller.go:37-39, unmarshaller.go:40-42 #p2
- [ ] Length check before copying `DataI64` into INT tensors — x/marshaller/proto/convert.go:199-207 #p2
- [ ] Table test over every dtype for all value marshallers — x/marshaller/proto/marshaller_test.go #p2

## Links

- Module: [[marshallers]]; siblings [[marshaller-tensor-json-yaml-conversion]], [[marshaller-tensor-gob-conversion]]; [[tensor]]; [[dndm-and-protobuf]]

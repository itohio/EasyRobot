---
type: algorithm
status: tested
tags: [easyrobot, marshaller, csv, tensor]
workstream: [W1]
module: "[[marshallers]]"
summary: "48 pass"
source: [x/marshaller/csv/marshaller.go, x/marshaller/csv/unmarshaller.go, x/marshaller/csv/options.go]
tests: [x/marshaller/csv/marshaller_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test -count=1 -v ./x/marshaller/csv → ok, 48 PASS (2026-11-09); probe every tensor dtype except FP32 rejected on marshal (2026-11-09)"
---
# CSV ↔ vector / matrix / tensor / slice

**What it does:** writes `vec.Vector`, `mat.Matrix`, 1-D/2-D tensors and `[]float32`/`[]string` slices as CSV rows (optional header via metadata `header=true`, dtype comment line) and parses CSV back, inferring dtype from the comment when present.

**Where:** `Marshaller.Marshal` `marshaller.go:41-84`, tensor writer switch `marshaller.go:104-105` (FP32 only), vector/matrix/slice writers below; `Unmarshaller.Unmarshal` `unmarshaller.go:43-133` (dtype comment :60-61, header :75, dispatch :105-131).

## Completeness review

- Complete: no — tensors other than FP32 are refused (`unsupported tensor dtype`, `marshaller.go:104-105`); ranks > 2 not supported.
- Tested: 48 PASS — vectors/matrices/tensors with and without header, slices of float32 and strings, round trips, reads of repo test data.
- Correct: unknown beyond tested FP32 paths. `fmtprobe`: FP32 round-trips; FP64/INT32/INT64/INT/INT16/INT8/UINT8 → marshal error (expected given the guard, but undocumented).

## TODOs

- [ ] Support all numeric dtypes in tensor CSV (format via `strconv` per type) — x/marshaller/csv/marshaller.go:104-105 #p3

## Links

- Module: [[marshallers]]

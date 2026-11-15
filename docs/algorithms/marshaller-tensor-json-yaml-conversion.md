---
type: algorithm
status: partial
tags: [easyrobot, marshaller, json, yaml, tensor, graph, serialization]
workstream: [W1, W6]
module: "[[marshallers]]"
summary: "8+8 pass; probe"
source: [x/marshaller/json/internal.go, x/marshaller/json/marshaller.go, x/marshaller/json/unmarshaller.go, x/marshaller/yaml/internal.go, x/marshaller/yaml/marshaller.go, x/marshaller/yaml/unmarshaller.go]
tests: [x/marshaller/json/marshaller_test.go, x/marshaller/json/xor_test.go, x/marshaller/yaml/marshaller_test.go, x/marshaller/yaml/xor_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 -v ./x/marshaller/json ./x/marshaller/yaml → ok, 8+8 PASS (2026-11-09); probe INT32/INT64/INT panic, UINT8 panic (2026-11-09)"
---
# Tensor / model / graph ↔ JSON and YAML

**What it does:** human-readable encodings of tensors (`{dtype, shape, data}`), NN parameters/layers/models, numeric slices and **graphs** (generic graph, tree, decision tree, expression graph with metadata) via reflection. JSON and YAML are near-identical copies (`internal.go` 789 vs 673 LOC).

**Where (JSON; YAML mirrors line-for-line within ±10):** dtype map `internal.go:101-153`; `tensorToJSON` :155; model/layer :167-229; graph capture `graphToJSON` :231, reflection paths :236-711 (special cases for `GenericGraph[string,float32]` :294 and `[any,any]` :335); metadata :742; decode `unmarshaller.go` (tensor fill via `SetAt` at `json/unmarshaller.go:162`, `yaml/unmarshaller.go:159`).

**Inputs → outputs:** `Marshal(w, value)`; `Unmarshal(r, &dst)` for `*Tensor`, slices, `*graph.GenericGraph…`, trees, decision/expression graphs (ops re-bound by option), models.

## Completeness review

- Complete: no — decoding writes every element through `Tensor.SetAt(float64, i)` (`json/unmarshaller.go:162`), which panics for INT32/INT64/INT tensors in the eager backend ([[tensor]] defect, `x/math/tensor/eager/tensor.go:383-405`); dtype map has no `uint8` (`json/internal.go:101-153`) so UINT8 tensors decode to `DT_UNKNOWN` and panic. Graph capture has hard-coded type branches (`GenericGraph[string,float32]`), so other N/E instantiations fall back to slower reflection paths. JSON/YAML duplication doubles maintenance.
- Tested: 8 tests each pass — FP32 tensor round trip, arrays, `TestGraphRoundTrip`, `TestTreeRoundTrip`, decision tree and expression graph round trips *with computation* (outputs compared), embedded document decode, XOR model parameters.
- Correct: no for integer dtypes. Probe (2×2 tensors): FP32/FP64/INT16/INT8 exact; INT32, INT64, INT → panic `tensor.setElementAtIndex: unsupported data type`; UINT8 → panic `unsupported dtype: 0`. Same results for YAML.

## TODOs

- [ ] Decode tensor data with typed copies (switch on `Data().(type)`) instead of `SetAt` — x/marshaller/json/unmarshaller.go:162, x/marshaller/yaml/unmarshaller.go:159 #p1 #ER-027
- [ ] Add `uint8` to dtype maps — x/marshaller/json/internal.go:101-153, x/marshaller/yaml/internal.go:101-153 #p1 #ER-027
- [ ] Share one reflection-based graph capture between JSON, YAML and the graph marshaller — x/marshaller/json/internal.go:231-711 #p3

## Links

- Module: [[marshallers]]; [[marshaller-tensor-proto-conversion]]; graphs [[graph-and-planning]]; [[tensor]]

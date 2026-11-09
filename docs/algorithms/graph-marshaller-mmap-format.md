---
type: algorithm
status: partial
tags: [easyrobot, marshaller, graph, storage, serialization]
workstream: [W6, W1]
module: "[[marshallers]]"
summary: "11 pass; probe shows nil node data"
source: [x/marshaller/graph/format.go, x/marshaller/graph/writer.go, x/marshaller/graph/reader.go, x/marshaller/graph/marshaller.go, x/marshaller/graph/unmarshaller.go, x/marshaller/graph/stored_graph.go, x/marshaller/graph/graph_capture.go, x/marshaller/graph/data.go, x/marshaller/graph/graph.go]
tests: [x/marshaller/graph/format_test.go, x/marshaller/graph/marshaller_e2e_test.go, x/marshaller/graph/decision_expression_e2e_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 -v ./x/marshaller/graph → ok, 11 PASS (2026-11-09); probe tmp/easyrobot/io/graphprobe: node data comes back nil, in-memory factory cannot round-trip (2026-11-09)"
---
# Graph marshaller: memory-mapped node/edge/data file format

**What it does:** persists any `x/math/graph` graph (generic, tree, decision tree, expression graph; captured by reflection) into three files — fixed-record node file, fixed-record edge file, variable-length data file — and reopens them as a lazily-read `StoredGraph` that implements the graph interfaces over `types.MappedStorage` (mmap / memory / tar).

**Where:**
- Layout constants and (de)serialisers: `format.go:8-25` (magics `GRAPHND1/GRAPHED1/GRAPHDT1`, 64-byte headers, 32-byte records), `WriteNodeHeader`/`ReadNodeHeader` :81-124, `NodeRecord` :58-63, `EdgeRecord` :66-72.
- Write path: `GraphMarshaller.Marshal` `marshaller.go:44-69` → `marshalGraph` :72-142 → `writeNodeFile` `writer.go:28-95`, `writeEdgeFile` :97, `writeDataFile` :187-302, offset back-patching :314-409.
- Capture: `captureGraph`/`visitNodes`/`visitEdges` `graph_capture.go:18-150` (reflection over `Nodes()`/`Edges()` iterators).
- Data entry codec: `serializeData`/`deserializeData` `data.go:39-300` (protobuf by registered type name, bytes, string, all ints/uints, f32/f64, c64, arrays/slices).
- Read path: `GraphUnmarshaller.Unmarshal` `unmarshaller.go:36-130` → `newStoredGraph`, `readNodeData`/`readEdgeData` `stored_graph.go:147-168`.
- Spec: `x/marshaller/graph/FILE_FORMAT.md`, `SPEC.md`.

**Inputs → outputs:** `Marshal(w io.Writer /*ignored*/, graph any, WithPath(nodes), WithEdgesPath(edges), WithLabelsPath(data))` → three storages created through the `MappedStorageFactory` given to `NewMarshaller`. `Unmarshal(nil, dst)` with `dst` ∈ `*StoredGraph | *StoredTree | *StoredDecisionTree | *StoredExpressionGraph`. Little-endian; edge cost is not stored, it is recomputed from edge data (`writer.go:15-17`).

## Completeness review

- Complete: **no**.
  - **Node data is never written.** `capturedNode` holds only `id` (`graph_capture.go:8-10`, filled at :40) and `marshalGraph` passes `nil` node entries to `writeDataFile` (`marshaller.go:106`), so every `NodeRecord.DataOffset` stays 0 and `readNodeData` returns `nil` (`stored_graph.go:152-153`). Only edge data and graph metadata persist.
  - `GraphStorage`/`GraphTransaction` (`graph.go:87-187`): `Commit`, `AddNode`, `AddEdge`, `DeleteNode`, `DeleteEdge`, `UpdateNode`, `UpdateEdge` are TODO bodies; `Commit` returns `nil` (`graph.go:120-128`), so callers think a write succeeded.
  - Header `Checksum` is never computed or checked; `CalculateChecksum` is a byte sum marked "placeholder" (`format.go:230-238`, `writer.go:68`).
  - `EdgeRecord.DataOffset` is `uint32` (`format.go:69`, truncating cast `writer.go:401`) while node offsets are `uint64`: a data file over 4 GiB corrupts edge links.
  - Spec'd defragmentation/deleted-flag handling (`FILE_FORMAT.md` §Defragmentation) is not implemented: the writer always rewrites everything.
- Tested: 11 tests pass. `format_test.go` round-trips headers/records field by field and rejects bad magic/version; `TestGraphMarshallerMatrixRoundTrip` writes a 10×10 `MatrixGraph` and compares every edge cost after `ToMatrix` (tol 1e-5) — edge data only; `TestGraphMarshallerComplex64ArrayRoundTrip` checks the data codec; decision/expression round trips are covered in [[graph-marshaller-op-registry]]. No test reads node data back.
- Correct: **no** for node payloads. Probe `~/src/tmp/easyrobot/io/graphprobe/main.go` (`GOFLAGS=-mod=mod go run .`): `GenericBinaryTree[string,float32]("root")` + left child "left" + edge cost 2.5 → after file round trip `node id=1 data=<nil>`, `node id=2 data=<nil>`, `edge data=2.5`. Edge payloads are correct in the probe and the matrix test.
- `NewMemoryMap()` cannot be used for a round trip: the factory ignores the path and returns a fresh empty storage per call (`storage/memory.go:224-230`), and `Marshal` closes storages (which nils memory buffers). Probe: unmarshal fails with `region extends beyond storage (offset: 0, length: 64, size: 0)`.

## TODOs

- [ ] Persist node data (capture `Data()` in `captureNodes`, pass node entries to `writeDataFile`) — x/marshaller/graph/graph_capture.go:8-10, marshaller.go:106 #p1 #ER-027
- [ ] Add a test that reads node data back from a stored graph — x/marshaller/graph/marshaller_e2e_test.go #p2
- [ ] Make `GraphTransaction.Commit` and friends fail loudly (`errors.ErrUnsupported`) until implemented — x/marshaller/graph/graph.go:120-187 #p1 #ER-027
- [ ] Widen `EdgeRecord.DataOffset` to uint64 (format version bump) — x/marshaller/graph/format.go:69, writer.go:401 #p2
- [ ] Compute/verify header checksum (CRC32C/xxhash) or drop the field — x/marshaller/graph/format.go:230-238 #p3

## Links

- Module: [[marshallers]]; storage back ends [[mapped-storage-file-memory]], [[tar-archive-mapped-storage]]; ops [[graph-marshaller-op-registry]]
- Graph library: [[graph-and-planning]]; workstream [[W6-decision-trees]]

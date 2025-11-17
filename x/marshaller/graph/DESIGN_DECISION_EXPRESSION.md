# Decision & Expression Graph Marshalling Plan

This document captures the design for adding tree/decision/expression graph support
inside `x/marshaller/graph` without touching `x/math/graph`.

## Goals

1. Marshal/Unmarshal the following high-level structures:
   - `GenericTree` / `GenericBinaryTree` → treated as graphs with tree metadata.
   - `DecisionTree` → persist structure + decision node/edge operations.
   - `ExpressionGraph` → persist structure + expression operations.
2. Keep `x/math/graph` unaware of persistence.
3. Provide a single marshalling backend (similar to `StoredGraph`) that can be
   wrapped as decision/expression-capable runtime graphs after loading.
4. Register decision/expression operations via options in `x/marshaller/graph`
   (`WithDecisionOp`, `WithDecisionEdge`, `WithExpressionOp`). Serialized
   artifacts store only the operation names; unmarshal fails if a registered
   name is missing.
5. Maintain SOLID principles and extend test coverage with end-to-end scenarios.

## Approach Overview

### 1. Metadata Capture Layer

- Extend the existing reflection-based capture introduced for generic graphs.
  - Do not capture the structure of the graphs/trees. Reflection is only for mapping of operations
- Add type detection helpers (`isTree`, `isDecisionTree`,
  `isExpressionGraph`). There might be multiple implementations of the trees/graphs, we care about interfaces and
  access data via interfaces as we do with general graph.
- When capturing:
  - Trees: record node list including `parentIdx`, `childIdxs`, `costs`, and
    `rootIdx`. Also include tree type for optimization - general tree with any branches v.s. binary tree.
  - Decision trees: reuse tree metadata + capture maps of `nodeID → opName` and
    `edge(parentID, childID) → criteriaName` by inspecting the private maps via
    reflection (field names are known: `nodeOps`, `edgeOps`). For functions,
    resolve `runtime.FuncForPC(fn.Pointer()).Name()` as the canonical key.
  - Expression graphs: capture node ops map + root ID + underlying base graph
    connections (already accessible via `base *GenericGraph` field).

### 2. Serialization Format Extensions

- Reuse existing node/edge record/file layout for the structural graph portion.
- Add a **Metadata Section** stored in `data.graph` as a special entry with a
  reserved type name 
- Structural nodes continue to be stored as before; data file offsets for nodes
  remain zero (trees typically have no per-node payload in current use). Edges
  keep their payload representation for costs/weights.
- add graph type into graph header whether it is a general graph, expression graph, or tree

### 3. Operation Registration API

- Introduce registries in `x/marshaller/graph` marshaller/unmarshaller config:
  - `var decisionOpRegistry = map[string]reflect.Value`
  - `var decisionEdgeRegistry = map[string]reflect.Value`
  - `var expressionOpRegistry = map[string]reflect.Value`
- Public options:
  - `WithDecisionOp(name string, fn any)` → stores reflect.Value in registry.
  - `WithDecisionEdge(name string, fn any)`.
  - `WithExpressionOp(name string, fn any)`.
- During capture, each function is resolved to a name:
  1. If the user supplied `WithDecisionOpName(fn, "customName")`, prefer that.
  2. Otherwise, attempt to derive via `runtime.FuncForPC`. (Document that
     anonymous closures are not supported).
- During unmarshal, metadata lookup uses these registries to obtain callable
  reflect values; failure to find a name results in a descriptive error.

### 4. Runtime Wrappers

- Keep `StoredGraph` for generic graphs.
- Introduce dedicated structs in `x/marshaller/graph` implementing the relevant
  interfaces by mirroring `Generic*` behaviours but backed by the stored files:
  - `StoredTree` implements `graph.Tree[any, any]` (and `graph.Graph[...]`).
  - `StoredDecisionTree` embeds `StoredTree` and implements
    `graph.DecisionTree[...]` as well as the node/edge interfaces by
    rehydrating ops from the registry.
  - `StoredExpressionGraph` implements `graph.Graph` and `ExpressionGraph`.
- These structs reuse the same `nodeRecords`, `edgeRecords`, and `dataStorage`
  primitives (split common logic from `StoredGraph` into shared helpers).

### 5. Marshal/Unmarshal Flow

1. **Marshal**
   - Detect type (generic/ tree/ decision/ expression).
   - Capture structural data + metadata.
   - Write nodes/edges via existing storage writer.
   - Serialize metadata entry into `data.graph`; store offset in new metadata
     pointer file header or maintain mapping in code.

2. **Unmarshal**
   - Load storages and metadata entry.
   - Instantiate the appropriate stored-wrapper type.
   - Rebind operations using registries (error if missing name).
   - When caller requests a concrete type (e.g. `*StoredDecisionTree`), fill it.
   - For plain `StoredGraph`, ignore metadata.

### 6. Tests

- Add unit tests for the registries (`WithDecisionOp`, etc.) ensuring duplicate
  names override, invalid signatures panic, etc.
- End-to-end tests:
  1. Decision tree: build a tiny tree with three levels and several ops and criteria; marshal & load;
     ensure `Decide` matches original outputs for test inputs.
  2. Expression graph: build DAG with simple arithmetic ops; marshal & load;
     verify `Compute` results and root ID persistence.
- Existing matrix round-trip test continues to run via generic path.

## Reuse Strategy

- No changes are made to `x/math/graph`; we interact purely through exported
  methods and reflection.
- We do not assume implementation types - we rely on interfaces. Note that DecisionTree and ExpressionGraph are supersets of Tree which is a superset of Graph. So order of type dispatch is important.
- The stored wrappers aim to mimic the public behaviour so downstream consumers
  can treat loaded structures the same as their in-memory counterparts.

## Next Steps

1. Implement operation registries and options.
2. Build metadata capture/serialization for generic trees.
3. Layer decision tree metadata + stored wrapper.
4. Layer expression graph metadata + stored wrapper.
5. Update marshaller/unmarshaller flow + tests.

## Runtime SOLID Refactor (2025-11-17)

Goal: decompose the stored graph runtime into focused components so each type has a
single responsibility and specialized wrappers interact through clear APIs.

Planned changes:
1. **Core Split** – Move `StoredGraph` definition/constructor into `stored_graph.go` and
   isolate node/edge implementations (`stored_node.go`, `stored_edge.go`).
   Provide small helper methods on `StoredGraph` (e.g. `neighborsOf(id int64)`,
   `edgeRecord(idx int)`) so other packages don’t reach into internal slices.
2. **Traversal Helpers** – Introduce lightweight structs (`edgeIterator`,
   `nodeAccessor`) to encapsulate neighbor access, removing direct map access in
   tree/decision/expression runtimes.
3. **Wrapper Updates** – Update `StoredTree`, `StoredDecisionTree`, and
   `StoredExpressionGraph` to consume the new helper APIs instead of touching
   `edgesByFrom`, `edgeRecords`, etc. This keeps them substitutable and isolated.
4. **File Organization** – Keep each concept in its own file (graph, node, edge,
   tree, decision, expression) to mirror responsibilities.

Validation: `go test ./x/marshaller/graph` must continue to pass with existing
round-trip tests.

# Stored Runtime SOLID Refactor Plan

## Goal
Refactor the graph marshaller runtime so each component follows SOLID principles:
- **SRP**: separate concerns for core storage, traversal, and specialized wrappers.
- **O/C & LSP**: allow new stored graph types without touching core logic.
- **ISP & DIP**: keep interfaces focused and inject dependencies explicitly.

## Scope
Files touched will include:
- `stored_graph_runtime.go` (split into focused files)
- `stored_graph.go`, `stored_node.go`, `stored_edge.go` (new)
- `stored_tree.go`, `stored_decision_tree.go`, `stored_expression_graph.go`
- Supporting files if helper interfaces/structs are required

## Step-by-step Plan

### 1. Core extraction
- Create `stored_graph.go` containing `StoredGraph` struct, constructor, and shared helpers (type registry, new helper adapters).
- Create `stored_node.go` and `stored_edge.go` for node/edge implementations.
- Introduce small helper methods on `StoredGraph` (`nodeRecord(id)`, `neighborsOf(id)`, `edgeRecord(index)`) to avoid exposing raw slices.
- Keep original file until all sections are moved.

### 2. Traversal helpers
- Add iterator/helper structs (e.g., `neighborIterator`, `edgeLookup`) in a dedicated file to wrap map/slice access.
- Update node/edge implementations to use helpers instead of directly touching maps.

### 3. Wrapper alignment
- Update `StoredTree`, `StoredDecisionTree`, and `StoredExpressionGraph` to rely only on exported helpers rather than internal slices/maps.
- Ensure wrappers accept explicit dependencies (e.g., pass helper interface) for better testability.

### 4. Cleanup
- Once all logic is moved, prunes unused code from the old monolithic file (in incremental steps, not wholesale deletions).

### 5. Validation
- Run `go test ./x/marshaller/graph` after each major step.
- Add/update unit tests if helper functionality warrants it.

## Deliverables
- New focused files for graph/node/edge runtimes.
- Updated wrappers adhering to helper interfaces.
- Passing test suite demonstrating unchanged behavior.

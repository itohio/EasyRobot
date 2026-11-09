---
type: algorithm
status: tested
tags: [easyrobot, graph]
workstream: W1
module: "[[graph-and-planning]]"
summary: "`MatrixGraph` exposes a `mat.Matrix` as a directed weighted graph."
source: [x/math/graph/matrix_graph.go]
tests: [x/math/graph/matrix_graph_test.go, x/math/graph/dijkstra_test.go, x/math/graph/cycles_test.go, x/marshaller/graph/marshaller_e2e_test.go]
test_result: pass
complete: yes
correct: unknown
evidence: "go test -count=1 -run 'TestToMatrix' ./x/math/graph/ → ok (2026-11-09); lead run go test -json ./x/math/graph/... → 86 pass / 1 fail (RRT*, unrelated)"
---
# Matrix graph (adjacency matrix) + ToMatrix

**What it does:** `MatrixGraph` exposes a `mat.Matrix` as a directed weighted graph. Node *i* is row *i*, and `M[i][j] > Obstacle` means an edge i→j with weight `M[i][j]`. `ToMatrix` goes the other way: it turns any `Graph` into a dense adjacency matrix.

**Where:** `x/math/graph/matrix_graph.go`:
- Types: `MatrixGraph` (L15-18), `MatrixNode` (L22-25)
- Node methods: `Neighbors`/`Edges` (L68-106), `Cost` (L124-138)
- `MatrixEdge` (L141-169); `Nodes`/`Edges`/`NumNodes`/`NumEdges` (L174-238)
- `ToMatrix` (L243-284), `MatrixSizeError` (L287-295)

**Inputs → outputs:**
- Nodes are `MatrixNode{Index, Graph}` with ID = Index.
- `ToMatrix(g, out)` maps node IDs to indices in `Nodes()` order. `out` must be N×N, otherwise it returns `*MatrixSizeError`. `out` is zeroed first, then `out[from][to] = edge.Cost()`.

## Completeness review

- Complete: yes.
- Tested:
  - `TestToMatrix_*` (8 tests): generic and grid graphs with value checks, size errors, an empty graph, arbitrary IDs, zeroing, edge costs. Re-run `go test -count=1 -run 'TestToMatrix' ./x/math/graph/` → pass.
  - `MatrixGraph` itself is exercised by `dijkstra_test.go:48` (shortest path) and `cycles_test.go:36,52` (cycle and self-loop), which pass in the lead's run, and by the marshaller e2e test.
- Correct: unknown. Semantics are fine for square matrices. Caveats:
  - The doc comment says "each column is a node" (L10, L13), but the code uses rows as source nodes. `Nodes()` yields `len(Matrix)` rows (L174), so a non-square matrix produces neighbours `j ≥ rows` that `Nodes()`/`ToMatrix` never list.
  - Any diagonal entry `> Obstacle` is a self-loop.
  - With the default `Obstacle 0`, zero and negative weights are non-edges. `ToMatrix` writes 0 for "no edge", so a real zero-cost edge cannot be represented, and duplicate edges overwrite each other.
  - Edge IDs `from*1e6+to` (L150) collide beyond 1e6 nodes.

## Embedded notes

Iteration is allocation-light: value-type nodes and edges with no maps. `ToMatrix` allocates one ID→index map. `float32` weights. The dense O(N²) matrix limits it to small graphs on MCUs. TinyGo untested (`iter.Seq`).

## TODOs

- [ ] Doc says column=node, code uses row; non-square matrices yield out-of-Nodes neighbours — x/math/graph/matrix_graph.go:10-17,174 #p3
- [ ] ToMatrix cannot represent zero-cost edges (0 = no edge) — x/math/graph/matrix_graph.go:262-283 #p3

## Links

- Module: [[graph-and-planning]]
- [[graph-dijkstra]], [[graph-cycle-detection]], [[grid-graph]]

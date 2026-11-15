---
type: algorithm
status: implemented
tags: [easyrobot, graph]
workstream: W1
module: "[[graph-and-planning]]"
summary: "AddNode copies (detached handles); first-parallel-edge cost"
source: [x/math/graph/graph.go, x/math/graph/generic.go, x/math/graph/path.go]
tests: [x/math/graph/astar_test.go, x/math/graph/dijkstra_test.go]
test_result: pass
complete: yes
correct: unknown
evidence: "used by search tests (pass); probes: AddNode copies node (returned handles detach); Cost picks first parallel edge"
---
# Generic graph (interfaces + adjacency-list storage)

**What it does:** Defines the package-wide generic interfaces (`Node[N,E]`, `Edge[N,E]`, `Graph`, `Adder/Deleter/Updater`, transactions, `Tree`, `Searcher`) and a value-array adjacency-list implementation `GenericGraph[N,E]` used by trees, decision trees, expression graphs and RRT*.

**Where:** `x/math/graph/graph.go:8-178` (interfaces), `x/math/graph/generic.go`: `GenericNode` :7-160, `GenericEdge` :166-232, `GenericGraph` :235-617 (`AddNode` :295-346, `AddEdge` :349-458, `DeleteNode` :476, `DeleteEdge` :550); `path.go:8-43` (`Path`, grid/matrix path → vectors).

**Inputs → outputs:** node data `N`, edge data `E`; edge cost = `costCalculator(from,to)` if set, else `E` converted from float32/float64/int types (generic.go:201-226), else 0. Iteration via `iter.Seq` (Go 1.23 range-over-func).

## Completeness review

- Complete: yes for add/delete/update and iteration; `Accessor` (NodeByID) not implemented by `GenericGraph`.
- Tested: indirectly by the search/tree/decision tests. Run: go test -count=1 -json ./x/math/graph/... (2026-11-09) → ok except TestRRTStar_Simple2DGrid (flaky).
- Correct: unknown overall; defects found:
  - `AddNode` appends a *copy* of a passed `*GenericNode` (generic.go:343); the caller's pointer is not the stored node and never sees its edges. This is why [[expression-graph]] handles returned by `AddNode` evaluate to 0 (probe `exprprobe`).
  - `GenericNode.Cost` returns the cost of the *first* edge to the neighbour (generic.go:107-110), and neighbours are de-duplicated, so with parallel edges searches use an arbitrary edge. probe: 31/300 random digraphs (12 nodes, 30 edges with duplicates) gave A*/Dijkstra paths costlier than Bellman-Ford; 0/300 after removing duplicate edges.
  - `DeleteNode` re-indexes every edge/neighbour list (nested loops, generic.go:476-548) — O(V·E).
  - Edge-cost conversion uses a runtime type switch on `any(e.data)` per call (generic.go:213-224).

## TODOs

- [ ] Return the stored node (`&g.nodes[idx]`) from add paths; document copy semantics — generic.go:343 #p1 #ER-028
- [ ] `Cost` should return the minimum over parallel edges (or forbid parallel edges) — generic.go:107 #p2
- [ ] Implement `Accessor` (`NodeByID`) on GenericGraph #p3

## Links

- Module: [[graph-and-planning]]

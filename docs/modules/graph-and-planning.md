---
type: module
group: "Control and planning"
status: partial
tags: [easyrobot, graph, planning, decision-tree, expression-graph]
workstream: [W6, W8, W3, W1]
packages: [x/math/graph]
test_result: flaky
---
# Graph and planning (x/math/graph)

**Purpose:** Generic graph library (Go generics + `iter.Seq`): graph/tree data structures, path search (A*, Dijkstra, BFS, DFS, RRT*), cycle detection, a kd-tree, a dataflow **expression graph**, and **decision trees** for hand-authored behaviours. Persistence (memory-mapped files, op registry) lives in `x/marshaller/graph` ([[marshallers]]).

**Packages:** `x/math/graph` — 25 source files, 6 588 LOC; 13 test files, 2 725 LOC; no build tags. Largest: tree.go 787, binary_tree.go 749, kd_tree.go 643, generic.go 617, forest.go 577, rrtstar.go 482, expression_graph.go 396, decision_tree.go 379. Deps: `x/math/mat`, `x/math/vec`. Consumers: `x/math/grid` (path search wrappers, [[grid-maps]]), `x/marshaller/{graph,json,yaml}`.

## Public API

- Interfaces (graph.go): `Node[N,E]` (ID, Data, Neighbors, Edges, Cost, Equal/Compare), `Edge[N,E]`, `Graph[N,E]`, mutation (`Adder`, `Deleter`, `Updater`, transactions), `Tree[N,E]`, `Searcher[N,E]`.
- Concrete graphs: `GenericGraph` (adjacency list), `MatrixGraph` (adjacency matrix), `GridGraph` (2-D grid, 4/8-connected), `GenericTree`, `BinaryTree`, `Forest`, `KDTree`.
- Search: `NewAStar(g,h).Search(s,g)`, `NewDijkstra`, `NewBFS`, `NewDFS` (+ `*Func` one-shots), `NewRRTStar(g, sampler, goalChecker, opts)`; `LoopDetection`, `ConnectedComponents`.
- Computation: `GenericExpressionGraph` (`AddNode(data, op)`, `AddEdge`, `Compute(start, inputs...)`), `GenericDecisionTree` (`AddChild`, `SetNodeOpByIndex`, `SetEdgeOpByIndex`, `Decide`), `VectorDecisionTree` (`AddDecisionChild`, `AddLeafChild`, `ComputeDecision`, criteria helpers).
- Composition: all algorithms take the `Graph`/`Node` interfaces, so any structure (and the stored/mmapped graphs from the marshaller) can be searched or evaluated; decision/expression graphs wrap `GenericTree`/`GenericGraph` and re-wrap nodes on iteration.

## Algorithms

<!-- vault:algorithms h=6786bcb7e963 -->

Counts: 7 tested · 1 implemented · 9 partial · 0 stub · 0 unverified (17 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[decision-tree]] | tested | pass | Evaluation + storage round trip |
| [[graph-astar]] | tested | pass | Optimal on 300 random digraphs |
| [[graph-bfs]] | tested | pass | Unweighted shortest path (fewest edges) with FIFO queue, visited-on-enqueue, predecessor map. |
| [[graph-dfs]] | tested | pass | Iterative stack search for *a* path (not shortest); neighbours pushed in reverse so the first… |
| [[graph-dijkstra]] | tested | pass | Single-pair Dijkstra with lazy-deletion binary heap; maps keyed by `Node` interface values (not… |
| [[grid-graph]] | tested | pass | Diagonal corner cutting |
| [[matrix-graph]] | tested | pass | `MatrixGraph` exposes a `mat.Matrix` as a directed weighted graph. |
| [[generic-graph]] | implemented | pass | AddNode copies (detached handles); first-parallel-edge cost |
| [[expression-graph]] | partial | pass | No cycle check; AddNode handles evaluate to 0 |
| [[graph-binary-tree]] | partial | pass | An index-based binary tree (nodes in a slice; parent, left and right are indices) with optional… |
| [[graph-cycle-detection]] | partial | pass | Misses directed 2-cycles |
| [[graph-forest]] | partial | none | MergeTrees drops subtree |
| [[graph-tree]] | partial | pass | FindNodeByData always -1 |
| [[kd-tree]] | partial | pass | NN pruning bug; kNN/range/insert missing |
| [[rrt-star]] | partial | flaky | No steer/collision/choose-parent; dead Seed |
| [[tree-insertion-strategies]] | partial | pass | Placeholder comparison |
| [[vector-decision-tree]] | partial | pass | Balance swaps branches; >2 children ignored |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Pure Go, float32 costs, no CGO/goroutines. Heavy use of generics, interface values, `iter.Seq` closures, maps keyed by ID or by interface value, and wrapper allocation per iteration (decision/expression graphs) — fine on host; on ESP32/TinyGo expect code-size growth from generic instantiation and GC churn [INFERENCE: TinyGo not run]. Stored decision/expression graphs call ops via `reflect.Value.Call` (`x/marshaller/graph/ops_registry.go:130-`), which TinyGo supports only partially [INFERENCE].
- Light structures for MCU: `GridGraph`, `MatrixGraph` (value types, no maps).

## Tests and evidence

- `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly go test -count=1 -json ./x/math/graph/...` (2026-11-09) → 86 pass, 1 fail (`TestRRTStar_Simple2DGrid`); `-count=20 -run 'TestRRTStar_Simple2DGrid$'` → 7/20 and 6/20 fail, always "Path nodes should be connected" (baseline 2026-11-08: 14/30, 16/30).
- `go test -count=1 -json ./x/marshaller/graph/...` → 11 pass (incl. `TestDecisionTreeRoundTrip`, `TestExpressionGraphRoundTrip`).
- Probes: `searchprobe` (A*/Dijkstra vs Bellman-Ford: 0/300 mismatches on simple digraphs, 31/300 with parallel edges), `exprprobe` (cycle and handle behaviour), `cycleprobe` (directed 2-cycle missed); data-structure probes by the sub-task (deleted after use, results in notes).

## API notes (ease of use, composability)

- **Good:** one set of interfaces for every structure and algorithm; searches are reusable instances with buffer reuse; decision trees and expression graphs persist to memory-mapped files with ops resolved from a registry — a clever way to ship behaviours as data.
- **Friction:**
  - Handles are unreliable: `GenericGraph.AddNode` stores a copy, so the node you hold is not the node in the graph ([[generic-graph]]); expression-graph `AddNode` returns such a handle and `Compute(handle)` silently returns zero ([[expression-graph]]). Index-based APIs (`AddChild(parentIdx)`) on trees vs node-based APIs on graphs add a second mental model.
  - Identity is inconsistent: A* keys state by `ID()`, Dijkstra/BFS/DFS/components by interface value.
  - Decision logic lives in closures → readable for hand-written behaviours, but not inspectable, not serialisable with parameters, not trainable. `VectorDecisionTree` routing depends on child *order*, which generic tree ops (`Balance`, `ReorderChildren`) freely change.
  - Expression ops receive `map[int64]Output` — operand order is lost; a slice in edge order would be simpler.
  - Errors are often swallowed (`AddEdge` returning nil for unknown nodes; RRT* returning a fallback path).
- Overlap: grid path search exists three times (`graph.GridGraph`+A*, `grid.AStar`, `grid.FastAStar`) — see [[grid-maps]].

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Data-driven split nodes + training (see [[W6-decision-trees]]) #p2
- [ ] kd-tree NN fix and kNN/range queries for lidar points ([[kd-tree]]) #p2

## Inferred objectives (proposals)

- **proposal:** make decision trees the behaviour layer for the wheeled ESP32 robot (W6): plain-data binary split nodes trained offline (CART) or online (Hoeffding), executed allocation-free on device, stored with the existing marshaller.
- **proposal:** use the expression graph as a small typed dataflow for sensor pipelines (filters → features → decision tree), once ordering/cycle errors and handles are fixed.
- **proposal:** RRT*/A* over [[grid-graph]] or a lidar occupancy grid for local planning ([[W8-lidar]]); kd-tree for scan matching.

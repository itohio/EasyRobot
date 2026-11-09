---
type: algorithm
status: tested
tags: [easyrobot, graph, planning, search]
workstream: [W1, W8]
module: "[[graph-and-planning]]"
summary: "Optimal on 300 random digraphs"
source: [x/math/graph/astar.go]
tests: [x/math/graph/astar_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "5 tests pass; searchprobe: 0/300 cost mismatches vs Bellman-Ford on simple digraphs (zero heuristic)"
---
# A* search (generic graph)

**What it does:** Best-first search with `f = g + h` over any `Graph[N,E]`, user heuristic callback, binary heap (`container/heap`) with lazy duplicates, ID-keyed maps; reusable instance keeps buffers between searches.

**Where:** `x/math/graph/astar.go`: `Heuristic` :8, `AStar` :11-21, `NewAStar` :24, `Search` :38-95, `reconstructPath` :114-136, `bindNode` (attaches `GridGraph`/`MatrixGraph` to value nodes) :150-175, heap :178-216, `AStarFunc` :220.

**Inputs → outputs:** start, goal nodes; heuristic `h(from, goal) float32` → `Path` (nil if unreachable). Cost from `Node.Cost(neighbor)`.

## Completeness review

- Complete: yes. Popped stale heap entries are not skipped (closed check only for neighbours, :76), so a node can be expanded more than once — extra work, not wrong results with a consistent heuristic. With an admissible but inconsistent heuristic the closed set prevents reopening, so optimality is not guaranteed (standard graph-search caveat).
- Tested: `TestAStar_SimplePath` (path length 4, endpoints), `ReuseInstance`, `GridGraph`, `NoPath`, interface. Run: go test -count=1 -json ./x/math/graph/... (2026-11-09) → ok except TestRRTStar_Simple2DGrid (flaky) (5 pass).
- Correct: yes for simple graphs — probe tmp/easyrobot/filters/searchprobe: 300 random 12-node digraphs, path cost equal to Bellman-Ford in all cases once parallel edges are excluded (with parallel edges see [[generic-graph]] `Cost` defect).
- Embedded: allocates a `nodeWrapper` per push and uses 4 maps; float32 costs; generic interfaces (code size under TinyGo [INFERENCE]).

## TODOs

- [ ] Skip popped nodes already in closed set — astar.go:59-66 #p3
- [ ] Optimality test with weighted graph vs Dijkstra — astar_test.go #p2
- [ ] Pool `nodeWrapper`s / index-based state for embedded use #p3

## Links

- Module: [[graph-and-planning]]; grid-specialised versions: [[grid-maps]]

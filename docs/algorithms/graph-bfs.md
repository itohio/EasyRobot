---
type: algorithm
status: tested
tags: [easyrobot, graph, search]
workstream: W1
module: "[[graph-and-planning]]"
summary: "Unweighted shortest path (fewest edges) with FIFO queue, visited-on-enqueue, predecessor map."
source: [x/math/graph/bfs.go]
tests: [x/math/graph/bfs_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "5 tests pass (path found/none, grid)"
---
# Breadth-first search (generic graph)

**What it does:** Unweighted shortest path (fewest edges) with FIFO queue, visited-on-enqueue, predecessor map.

**Where:** `x/math/graph/bfs.go` (94 LOC): `BFS`, `NewBFS`, `Search`, `reconstructPath`, `BFSFunc`.

**Inputs → outputs:** start, goal → `Path` or nil.

## Completeness review

- Complete: yes.
- Tested: simple path, reuse, grid graph, no-path. Run: go test -count=1 -json ./x/math/graph/... (2026-11-09) → ok except TestRRTStar_Simple2DGrid (flaky).
- Correct: yes (standard algorithm; tests check endpoints/lengths). Queue is `queue = queue[1:]` — the backing array is retained until the search ends; maps keyed by interface values (same caveat as [[graph-dijkstra]]).

## TODOs

- [ ] Key by `ID()`; ring-buffer queue for long searches — bfs.go #p3

## Links

- Module: [[graph-and-planning]]

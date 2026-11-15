---
type: algorithm
status: tested
tags: [easyrobot, grid, planning, dijkstra]
workstream: W8
module: "[[grid-maps]]"
summary: "Dijkstra shortest path on a cost matrix via `graph.GridGraph` + generic `graph.NewDijkstra`."
source: [x/math/grid/dijkstra.go]
tests: [x/math/grid/astar_test.go]
test_result: pass
complete: yes
correct: unknown
evidence: "TestDijkstra_* pass; probe optimal cost 0.6 where A* gave 3.1; 52483 allocs/search on 50x50"
---
# Grid Dijkstra (matrix wrapper)

**What it does:** Dijkstra shortest path on a cost matrix via `graph.GridGraph` + generic `graph.NewDijkstra`.

**Where:** `x/math/grid/dijkstra.go:22-61` (`Dijkstra`, `DijkstraOptions`); rejects start/goal on obstacle `:40-43`.

**Inputs → outputs:** same as [[grid-astar]] minus heuristic; returns `[]vec.Vector2D` {col,row} or nil.

## Completeness review

- Complete: yes (4/8-conn, diag ×√2, corner cutting allowed — inherited from `graph/grid_graph.go`).
- Tested: TestDijkstra_SimplePath/WithObstacles/DiagonalMovement (`astar_test.go`) — non-nil, endpoints/length bounds; no cost assertion. Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead) → pass.
- Correct: unknown (no value-checking test); probe found the optimal 0.6-cost path on the 0.1-cost grid where A* failed (2026-11-09).
- Embedded: heavy — 52 483 allocs per 50×50 8-conn search (probe with testing.AllocsPerRun, 50x50 all-1 grid, 8-conn, corner to corner), interface/generic graph layer.

## TODOs

- [ ] Assert path cost in tests — x/math/grid/astar_test.go #p2
- [ ] Allocation cost of generic graph Dijkstra on grids (52k allocs/search) — x/math/grid/dijkstra.go:54 #p3

## Links

- Module: [[grid-maps]] · [[graph-and-planning]]

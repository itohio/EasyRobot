---
type: algorithm
status: partial
tags: [easyrobot, grid, planning, astar]
workstream: W8
module: "[[grid-maps]]"
summary: "partial (inadmissible heuristic when cell cost < 1)"
source: [x/math/grid/astar.go]
tests: [x/math/grid/astar_test.go]
test_result: pass
complete: yes
correct: no
evidence: "grid tests 64/64 pass; probe: AStar cost 3.1 vs Dijkstra 0.6 on 0.1-cost grid → suboptimal"
---
# Grid A* (matrix wrapper)

**What it does:** A* shortest path on a cost matrix; thin wrapper that builds a `graph.GridGraph` and runs generic `graph.NewAStar`.

**Where:** `x/math/grid/astar.go:23-61` (`AStar`, `AStarOptions`), heuristics `:78-100` (`EuclideanHeuristic`, `ManhattanHeuristic`, `ZeroHeuristic`), `validateInput :63`. Neighbour/cost model lives in `x/math/graph/grid_graph.go` (`GridNode.Cost :165`, diagonal ×√2 `:179`).

**Inputs → outputs:** `mat.Matrix` (float32 rows) where cell value = cost to enter, `value <= ObstacleValue` = blocked; start/goal as (row, col); opts nil → 4-conn, Euclidean, obstacle 0. Returns `[]vec.Vector2D` as **{col,row}** start→goal, nil if none/invalid.

## Completeness review

- Complete: yes for 4/8-conn uniform-step model. No start/goal-obstacle check (probe: start on obstacle cell still returns a path), unlike `Dijkstra` (`dijkstra.go:140`). 8-conn allows corner cutting between two blocked orthogonal cells (probe: 2×2 with off-diagonal blocked → path `[[0 0] [1 1]]`). `opts.Heuristic==nil` mutates caller's options struct (`astar.go:41-43`).
- Tested: `astar_test.go` TestAStar_SimplePath/WithObstacles/NoPath/DiagonalMovement/ManhattanHeuristic/SameStartGoal/InvalidInput/CustomObstacleValue + TestHeuristic_* — assert non-nil, start cell, path length bounds, nil when blocked; **none check path cost/optimality**. Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead ) → pass.
- Correct: **no** in general. Edge cost = destination cell value (×√2 diagonal); Euclidean/Manhattan heuristics assume min cell cost ≥ 1. With costs < 1 the heuristic overestimates → inadmissible → suboptimal paths. Probe (5×5 grid of 0.1, expensive top row of 1.0): `AStar` cost 3.1 vs `Dijkstra` 0.6 (probe, GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run . (2026-11-09)). Manhattan with 8-conn is inadmissible even at cost 1 (diag step √2 < 2). Admissible pairs: 4-conn+Manhattan or 8-conn+octile, scaled by min cell cost. Tie-breaking: none (heap on f only; order depends on generic graph A*).

## TODOs

- [ ] Scale heuristic by min cell cost (or document cost ≥ 1 requirement); add octile heuristic for 8-conn — x/math/grid/astar.go:78-95 #p1 #backlog
- [ ] Reject start/goal on obstacle like Dijkstra — x/math/grid/astar.go:29 #p2
- [ ] Optional no-corner-cutting for diagonal moves — x/math/graph/grid_graph.go:97 #p2
- [ ] Tests that assert optimal path cost (compare with Dijkstra) — x/math/grid/astar_test.go #p2
- [ ] Don't mutate caller `opts` — x/math/grid/astar.go:41 #p3

## Links

- Module: [[grid-maps]] · Graph slice: [[graph-and-planning]]

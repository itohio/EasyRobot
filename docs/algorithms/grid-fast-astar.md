---
type: algorithm
status: partial
tags: [easyrobot, grid, planning, astar]
workstream: W8
module: "[[grid-maps]]"
summary: "partial (same; ~1k allocs/search)"
source: [x/math/grid/fast_astar.go]
tests: [x/math/grid/fast_test.go]
test_result: pass
complete: yes
correct: no
evidence: "fast_test pass; allocs probe ~1079 allocs/search; same heuristic issue as AStar (cost 3.1 vs 0.6)"
---
# Grid FastAStar (reusable searcher)

**What it does:** A* implemented directly on the matrix (no `graph` interfaces for nodes/edges), with a searcher object meant to be reused across queries.

**Where:** `x/math/grid/fast_astar.go:13-24` (`FastAStar`), `NewFastAStar :35`, `Search :50-102`, `getNeighbors :132`, `getCost :155` (dest cell value ×√2 diag), `fastHeuristic :165`, own heap `fastPriorityQueue :209-247`.

**Inputs → outputs:** constructor binds matrix, allowDiag, obstacle, heuristic (`graph.Heuristic[GridNode,float32]`); `Search(sr,sc,gr,gc) []vec.Vector2D` ({col,row}); nil heuristic → nil result (`:55`).

## Completeness review

- Complete: yes, but "fast" is only partially real. Reused: maps (`closedSet`, `gScore`, `cameFrom`) cleared by range-delete (`:118-130`, O(previous size)) and heap backing slice. Still allocating per expansion: `getNeighbors` new slice (`:133`), `getDirections` new `[][]int` literal (`:177`), `fastHeuristic` allocates a `GridGraph` + 2 nodes per call (`:167-174`), `&fastNodeWrapper` per push (`:97`), result path. Measured (probe tmp/easyrobot/filters/gridinterp/allocs/main.go (testing.AllocsPerRun, 50x50 all-1 grid, 8-conn, corner to corner)): FastAStar 1079 allocs/search vs AStar 1288; ~470 µs vs ~692 µs. No start/goal-obstacle check; duplicate heap entries are re-expanded (no stale check after Pop, `:72`); `fastNodeWrapper.gScore`/`index` unused for decrease-key.
- Tested: `fast_test.go` TestFastAStar_SimplePath/ReuseInstance/WithObstacles/DiagonalMovement — non-nil, endpoints, reuse gives a path again; no optimality. Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: tmp/easyrobot/filters/test-2026-11-09.json) → pass.
- Correct: **no** for cell costs < 1 (same inadmissible heuristic as [[grid-astar]]; probe cost 3.1 vs Dijkstra 0.6). Correct (optimal) for costs ≥ 1 with Euclidean [INFERENCE: standard A* argument, not probed].

## TODOs

- [ ] Heuristic admissibility for costs < 1 (scale by min cost / octile) — x/math/grid/fast_astar.go:165 #p1 #backlog
- [ ] True zero-alloc: flat []float32/[]int32 arrays indexed row*cols+col with generation stamps instead of maps; static direction tables; value-type heap; integer heuristic call without GridGraph — x/math/grid/fast_astar.go:13-24,132,165,177 #p2
- [ ] Skip stale popped entries (closed check after Pop) — x/math/grid/fast_astar.go:72 #p3
- [ ] Reject start/goal on obstacle — x/math/grid/fast_astar.go:50 #p2

## Links

- Module: [[grid-maps]]

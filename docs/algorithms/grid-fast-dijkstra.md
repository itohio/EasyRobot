---
type: algorithm
status: tested
tags: [easyrobot, grid, planning, dijkstra]
workstream: W8
module: "[[grid-maps]]"
summary: "Dijkstra directly on the matrix with reusable maps/heap; lazy-deletion heap (visited check after…"
source: [x/math/grid/fast_dijkstra.go]
tests: [x/math/grid/fast_test.go]
test_result: pass
complete: yes
correct: unknown
evidence: "TestFastDijkstra_* pass; ~7989 allocs/search 50x50"
---
# Grid FastDijkstra (reusable searcher)

**What it does:** Dijkstra directly on the matrix with reusable maps/heap; lazy-deletion heap (visited check after Pop, `:65-70`).

**Where:** `x/math/grid/fast_dijkstra.go:12-22` (`FastDijkstra`), `NewFastDijkstra :25`, `Search :39-101`, `getCost :154`, heap `:196-233`.

**Inputs → outputs:** constructor (matrix, allowDiag, obstacle); `Search(sr,sc,gr,gc) []vec.Vector2D` {col,row}; nil if blocked/invalid/start-or-goal obstacle (`:45`).

## Completeness review

- Complete: yes. Same allocation profile issues as [[grid-fast-astar]]: per-expansion neighbour slice (`:132`) and direction table (`:164`), pointer heap nodes (`:95`), map-based state. 7989 allocs/search vs 52 483 for wrapper `Dijkstra` (probe: testing.AllocsPerRun, 50x50 all-1 grid, 8-conn, corner to corner).
- Tested: `fast_test.go` TestFastDijkstra_SimplePath/ReuseInstance/WithObstacles/DiagonalMovement/NoPath — existence/endpoints/nil; no cost check. Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09) → pass.
- Correct: unknown (algorithm is textbook lazy Dijkstra; no value-checking test or probe of cost).

## TODOs

- [ ] Flat arrays + static direction table to make it actually allocation-free — x/math/grid/fast_dijkstra.go:12-22,131,163 #p2
- [ ] Cost assertion test — x/math/grid/fast_test.go #p2

## Links

- Module: [[grid-maps]]

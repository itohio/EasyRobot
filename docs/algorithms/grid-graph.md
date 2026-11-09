---
type: algorithm
status: tested
tags: [easyrobot, graph]
workstream: [W1, W8]
module: "[[graph-and-planning]]"
summary: "Diagonal corner cutting"
source: [x/math/graph/grid_graph.go]
tests: [x/math/graph/bfs_test.go, x/math/graph/dfs_test.go, x/math/graph/dijkstra_test.go, x/math/graph/astar_test.go, x/math/graph/cycles_test.go, x/math/graph/matrix_graph_test.go]
test_result: pass
complete: yes
correct: unknown
evidence: "lead run go test -count=1 -json ./x/math/graph/... → 86 pass / 1 fail (RRT*, unrelated) (2026-11-09); TestToMatrix_GridGraph re-run → pass; probe graphds: diagonal move allowed between two blocked orthogonal cells"
---
# Grid graph (occupancy/cost grid as Graph)

**What it does:** `GridGraph` adapts a 2-D `mat.Matrix` cost grid to `Graph[GridNode, float32]` so that the generic BFS, DFS, Dijkstra and A* can run on it. Each cell is a node. Its 4 neighbours (or 8 with `AllowDiag`) are edges if their value is above `Obstacle`.

**Where:** `x/math/graph/grid_graph.go`:
- `GridGraph{Matrix, AllowDiag, Obstacle}` (L14-18), `GridNode{Row, Col}` (L22-26)
- `ID` (L29-37)
- `Neighbors`/`Edges`/`NumNeighbors` (L82-164), `Cost` (L167-184)
- `GridEdge` (L187-217); `Nodes`/`Edges`/`NumNodes`/`NumEdges` (L220-302)
- `NewGridNode` (L305), and `getGridDirections` (L306-312)

**Inputs → outputs:**
- A cell is passable iff `M[r][c] > Obstacle`.
- Edge cost = value of the *destination* cell, times √2 for diagonals (L179-182).
- Node ID = `row*cols + col`.

## Completeness review

- Complete: yes as an adapter.
- Tested: it is the graph used by most search tests: `bfs_test.go`, `dfs_test.go`, `dijkstra_test.go` (6 grids), `astar_test.go:132`, `cycles_test.go`, `TestToMatrix_GridGraph`. All have `AllowDiag:false`, `Obstacle:0`, and all pass in the lead's run. **Diagonal mode is untested.**
- Correct: unknown.
  - With `AllowDiag`, diagonal moves are allowed even when both orthogonal cells are blocked (no corner-cutting check, L89-104). Probe: in a 2×2 grid with blocked (0,1) and (1,0), (0,0) still gets neighbour (1,1) at cost 1.414. A robot path can then squeeze through a wall corner.
  - Cost uses only the destination cell, so it is asymmetric. That is acceptable, but undocumented.
  - IDs assume a rectangular grid: column bounds use `len(Matrix[0])` (L91), while `Nodes()` uses `len(Matrix[row])` (L227).
  - The fallback ID `row*10000+col` (L33) applies when `graph==nil`, so the ID of the same cell depends on whether `graph` is set. That breaks ID-keyed lookups for user-built nodes without `graph`.
  - Edge ID `from*1e6+to` (L203) collides for grids over 1e6 cells, e.g. a 1000×1000 lidar occupancy map.

## Embedded notes

Value-type nodes and edges with no maps, so iteration is cheap. A `*GridGraph` pointer travels in every node. `float32` costs. Suitable for small grids; the faster dedicated grid planners are in [[grid-astar]] and [[grid-fast-astar]]. TinyGo untested (`iter.Seq`, generics).

## TODOs

- [ ] Diagonal moves cut blocked corners — x/math/graph/grid_graph.go:89-104 #p2
- [ ] No test with AllowDiag=true — x/math/graph/grid_graph.go:179 #p2
- [ ] Edge ID collides for >1e6 cells; nil-graph ID fallback differs (×10000) — x/math/graph/grid_graph.go:33,203 #p3

## Links

- Module: [[graph-and-planning]]
- [[matrix-graph]], [[graph-astar]], [[graph-dijkstra]], [[grid-astar]]

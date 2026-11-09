---
type: module
group: "Math and maps"
status: partial
tags: [easyrobot, grid, planning, lidar, slam]
workstream: W8
packages: [x/math/grid]
test_result: pass
---
# Grid maps (x/math/grid)

**Purpose:** Operations on 2-D float32 grids (`mat.Matrix`) used as cost maps / occupancy grids: path search (A*, Dijkstra), lidar ray casting for SLAM beam models, rasterisation of shapes, region extraction.

**Packages:** `x/math/grid` — 7 source files, 1844 LOC (astar.go 100, dijkstra.go 61, fast_astar.go 247, fast_dijkstra.go 233, draw.go 585, extract.go 193, raycast.go 425); tests 803 LOC (astar_test.go, draw_test.go, fast_test.go); README.md, SPEC.md. No build tags. Deps: `x/math/mat`, `x/math/vec`, `x/math/graph`, `github.com/chewxy/math32`.

## Public API

- Path search, functional: `AStar(m, sr, sc, gr, gc, *AStarOptions)`, `Dijkstra(m, …, *DijkstraOptions)` → `[]vec.Vector2D` ({col,row}). Wrap `graph.GridGraph` + generic `graph.NewAStar/NewDijkstra` (`x/math/graph/grid_graph.go`, owned by [[graph-and-planning]]). Heuristics `EuclideanHeuristic`, `ManhattanHeuristic`, `ZeroHeuristic` typed `graph.Heuristic[graph.GridNode,float32]`.
- Path search, reusable: `NewFastAStar(m, diag, obstacle, h).Search(...)`, `NewFastDijkstra(m, diag, obstacle).Search(...)` — reimplement the neighbour/cost model inline (third copy, after `graph/grid_graph.go`).
- Cost model (all four): cell value = cost to enter; `value <= obstacle` blocked; diagonal ×√2; corner cutting allowed.
- Ray casting: `NewRayDirections`, `RayCast`, `RayCastOptimized`, `RayCastAll`, `RayCastAllOptimized(dst)`, `RayProjection`, `GetOccupancyReadings`. Consumer: `x/math/filter/slam/slam.go:87,187,214,422,424` (see [[filters]]).
- Drawing: `Line`, `Path`, `HLine`, `VLine`, `GradientLine`, `Rectangle`, `Circle`, `Ellipse`, `Triangle`, `ShadedTriangle` (in place).
- Extraction: `Mask`, `ExtractRectangle`, `ExtractCircle`, `ExtractEllipse` (allocate).

## Algorithms

<!-- vault:algorithms h=a273d822a8b7 -->

Counts: 3 tested · 2 implemented · 2 partial · 0 stub · 0 unverified (7 algorithms)

| Algorithm | Status | Tests | Summary |
|---|---|---|---|
| [[grid-dijkstra]] | tested | pass | Dijkstra shortest path on a cost matrix via `graph.GridGraph` + generic `graph.NewDijkstra`. |
| [[grid-drawing-primitives]] | tested | pass | Rasterises shapes into a `mat.Matrix` in place (for map editing, inflating obstacles, path… |
| [[grid-fast-dijkstra]] | tested | pass | Dijkstra directly on the matrix with reusable maps/heap; lazy-deletion heap (visited check after… |
| [[grid-raycast]] | implemented | none | Expected range of a lidar beam in an occupancy grid (beam model for SLAM/MCL). |
| [[grid-region-extraction]] | implemented | none | Element-wise mask multiply and copying rectangular / circular / rotated-elliptical windows out of… |
| [[grid-astar]] | partial | pass | partial (inadmissible heuristic when cell cost < 1) |
| [[grid-fast-astar]] | partial | pass | partial (same; ~1k allocs/search) |

<!-- /vault:algorithms -->

## Build targets and dependencies

- Pure Go, no CGO, no reflection, no goroutines; float32 throughout (math32) — suitable for TinyGo in principle [INFERENCE: not built with TinyGo here].
- Allocations (measured, probe tmp/easyrobot/filters/gridinterp/allocs/main.go (testing.AllocsPerRun, 50x50 all-1 grid, 8-conn, corner to corner)): `AStar` 1288, `FastAStar` 1079, `FastDijkstra` 7989, `Dijkstra` 52 483 allocs per 50×50 8-conn search; `RayCastAllOptimized` 0 allocs (360 beams, caller-provided dst). FastAStar ~470 µs vs AStar ~692 µs per search on host. "Fast" variants use `map[cell]…` state + pointer heap nodes + per-expansion slices — poor for ESP32 heap/GC. `mat.Matrix` is `[][]float32` (row slices), not a flat buffer.
- `Dijkstra`/`AStar` go through generic interfaces (`graph.Node[...]`), increasing code size under TinyGo [INFERENCE].

## Tests and evidence

- go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: tmp/easyrobot/filters/test-2026-11-09.json) → `x/math/grid` 64 pass, 0 fail.
- Probes: probe tmp/easyrobot/filters/gridinterp/main.go, GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=mod go run . (2026-11-09); probe tmp/easyrobot/filters/gridinterp/allocs/main.go (testing.AllocsPerRun, 50x50 all-1 grid, 8-conn, corner to corner). Findings: A* suboptimal on 0.1-cost grid (3.1 vs 0.6); A*/FastAStar accept start on obstacle, Dijkstra rejects; diagonal corner cutting; RayCast vs RayCastOptimized differ (4.5 vs 5.0 for exact 4.5; π/4: 6.5 vs 7.07 for exact ≈6.36).
- Untested: all of raycast.go and extract.go.

## API notes (ease of use, composability)

- **Coordinate conventions clash:** search takes `(row, col)` and returns `Vector2D{col,row}`; draw/extract take `(x=col, y=row)`; raycast uses world metres + `mapGrid[y][x]` with origin at cell corner. No `GridMap` type bundling matrix + resolution + origin, so every raycast call carries 4 loose float params and world↔cell conversion is duplicated (`raycast.go:63,134,402`).
- Three copies of the neighbour/cost model (graph/grid_graph.go, fast_astar.go, fast_dijkstra.go) can drift (e.g. obstacle checks already differ).
- Options pattern inconsistent: functional `AStar(opts)` vs constructor-bound `NewFastAStar(...)`; heuristic signature requires building `graph.GridNode`s, which forces FastAStar to allocate a `GridGraph` per heuristic call (`fast_astar.go:167`).
- Occupancy semantics inverted between modules: search treats **low** values as obstacles (`<= obstacle`, cost map), raycast treats **high** values (>0.5) as occupied. A SLAM occupancy grid can't be fed to `AStar` without conversion.
- Good: drawing API is uniform (dst in, dst out, clipping, nil-safe); `RayCastAllOptimized` takes a dst buffer.

## TODOs

Algorithm defects are tracked in the algorithm notes (table above) and rolled up in [[todo]]; this list holds module-level items only.

- [ ] Tests for raycast and extraction — x/math/grid/raycast.go, x/math/grid/extract.go #p2
- [ ] Single neighbour/cost implementation shared with graph.GridGraph — x/math/grid/fast_astar.go:132-177, x/math/graph/grid_graph.go:97-180 #p3
- [ ] Grep for TODO/FIXME/panic("not implemented") in x/math/grid non-test files: none found (2026-11-09) — x/math/grid #p3

## Inferred objectives (proposals)

- **proposal:** W8 lidar occupancy mapping: add `OccupancyGrid{Data []float32; W,H int; Res, OriginX, OriginY float32}` with log-odds update + Bresenham/DDA free-space carving, share it between `filter/slam` raycast and path planning (with an occupancy→cost conversion + obstacle inflation via `Circle`).
- **proposal:** Exact DDA ray traversal (Amanatides–Woo) as the single raycast, benchmarked vs current two variants on ESP32-sized maps.
- **proposal:** Zero-alloc grid A* (flat arrays, generation counters, binary heap of int32 indices, octile heuristic, optional no-corner-cut) as the embedded planner.

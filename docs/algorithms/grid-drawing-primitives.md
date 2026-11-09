---
type: algorithm
status: tested
tags: [easyrobot, grid, raster]
workstream: W8
module: "[[grid-maps]]"
summary: "Rasterises shapes into a `mat.Matrix` in place (for map editing, inflating obstacles, path…"
source: [x/math/grid/draw.go]
tests: [x/math/grid/draw_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "41 draw tests pass (value-checking cell assertions)"
---
# Grid drawing primitives (rasterisation)

**What it does:** Rasterises shapes into a `mat.Matrix` in place (for map editing, inflating obstacles, path overlay, test fixtures).

**Where:** `x/math/grid/draw.go` — `Line :15` (integer Bresenham), `Path :73` (polyline of `vec.Vector2D`), `HLine :95`, `VLine :133`, `GradientLine :171` (linear value ramp), `Rectangle :237`, `Circle :277` (filled: dist² test; outline: midpoint), `Ellipse :361` (rotated, a/b/angle), `Triangle :439` (3 lines), `ShadedTriangle :461` (scanline with linear value interpolation).

**Inputs → outputs:** `dst mat.Matrix`, coords as **x=col, y=row** (note: pathfinding API uses (row,col)), value(s) float32 → returns `dst`. Out-of-bounds pixels clipped; nil/empty dst returned unchanged.

## Completeness review

- Complete: yes. No filled polygon other than triangle; `ShadedTriangle` allocates a 3-element slice of structs (`:470`), otherwise allocation-free.
- Tested: `draw_test.go` 41 tests (Line/HLine/VLine/GradientLine/Path/Rectangle/Circle/Ellipse/Triangle/ShadedTriangle incl. out-of-bounds, nil, degenerate, reversed) asserting specific cell values. Run: go test -count=1 -json ./x/math/{colorscience,learn,grid,interpolation}/... (2026-11-09, slice lead; raw: tmp/easyrobot/filters/test-2026-11-09.json) → pass.
- Correct: yes for tested cases (value assertions pass). Sub-pixel/anti-aliasing not provided.

## TODOs

- [ ] Document x/y vs row/col convention difference with AStar/Dijkstra — x/math/grid/draw.go:15, x/math/grid/astar.go:23 #p3

## Links

- Module: [[grid-maps]]

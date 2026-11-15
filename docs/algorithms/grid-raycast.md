---
type: algorithm
status: implemented
tags: [easyrobot, grid, lidar, raycast, slam]
workstream: W8
module: "[[grid-maps]]"
summary: "Expected range of a lidar beam in an occupancy grid (beam model for SLAM/MCL)."
source: [x/math/grid/raycast.go]
tests: []
test_result: none
complete: yes
correct: unknown
evidence: "no tests in x/math/grid; probe: RayCast 4.5 (exact 4.5), RayCastOptimized 5.0; pi/4: 6.5 vs 7.07 (exact 6.36); RayCastAllOptimized 0 allocs"
---
# Grid ray casting (lidar beam model)

**What it does:** Expected range of a lidar beam in an occupancy grid (beam model for SLAM/MCL). Two variants: fixed-step ray marching and integer Bresenham; plus a boolean line-of-sight check and endpoint occupancy sampling.

**Where:** `x/math/grid/raycast.go`
- `NewRayDirections :18` — precomputes cos/sin of beam angles (`RayDirections`).
- `RayCast :48-114` — marches 0.5 cell steps from pose, hit when cell > 0.5.
- `RayCastOptimized :118-217` — rotates precomputed dir by heading, Bresenham from start cell to end cell at maxRange, returns cell-centre-to-cell-centre distance.
- `RayCastAll :221`, `RayCastAllOptimized :239` (caller-provided `distances`, 0 allocs measured).
- `RayProjection :268-347` — bool: any cell > threshold within `length` along a Bresenham line.
- `GetOccupancyReadings :363-417` — map value at hit cell per beam.

**Inputs → outputs:** pose `vec.Vector{x,y,heading}` (m, rad), map `mat.Matrix[row=y][col=x]` (occupancy 0..1), resolution m/cell, origin (m), maxRange (m) → range in m (or maxRange). Map indexing `mapGrid[y][x]`; origin is cell (0,0) corner.

## Completeness review

- Complete: yes for stated scope. Used by `x/math/filter/slam/slam.go:187,214,422,424` (`NewRayDirections`, `RayCastAllOptimized`, `RayCastAll`).
- Tested: **no tests** in `x/math/grid` for any raycast function (grep `func Test` in grid: astar/draw/fast only). SLAM tests exercise it indirectly (not inspected here).
- Correct: unknown/approximate. Probe (2026-11-09), 10×10 map, wall at col 5, pose (0.5,0.5,0), res 1:
  - beam 0: `RayCast` 4.5 (exact 4.5), `RayCastOptimized` 5.0 (cell-centre quantisation, error up to ~1 cell).
  - beam π/4: `RayCast` 6.5, `RayCastOptimized` 7.07; exact first-wall-entry ≈ 6.36. `RayCast` step 0.5 cell can skip corner-clipped cells (not a DDA/Amanatides-Woo traversal).
  - beam π leaving map: both return maxRange — rays exiting the map return maxRange, not distance to border (`:89`, `:185`); arguably fine for "unknown".
  - empty map, maxRange 3: both 3.
- Edge cases from code: empty `mapGrid` panics (`len(mapGrid[0])` at `:67`, `:139`, `:279`); nil `rayDirs` panics in `RayCastOptimized :125`; hit in start cell returns 0; `RayCast` threshold hard-coded 0.5 (`:104`, `:189`) while `RayProjection` takes a parameter; `int()` truncation of negative end coords (`:162-163`) rounds toward zero, not floor. `GetOccupancyReadings` with `len(rayDirs) < len(rayAngles)` silently returns 0 for extra beams (probe, no panic). `RayCast` and `RayCastOptimized` disagree by up to ~0.6 cell, and SLAM uses both (`slam.go:422` vs `:424`).

## TODOs

- [ ] Unit tests for RayCast/RayCastOptimized/RayProjection against analytic distances — x/math/grid/raycast.go:48,118,268 #p2
- [ ] Exact grid traversal (Amanatides–Woo DDA) returning boundary-entry distance; unify both variants — x/math/grid/raycast.go:82,182 #p2
- [ ] Guard empty map / nil rayDirs — x/math/grid/raycast.go:67,125,139 #p3
- [ ] Configurable occupancy threshold — x/math/grid/raycast.go:104,189 #p3

## Links

- Module: [[grid-maps]] · consumer: `x/math/filter/slam` ([[filters]])

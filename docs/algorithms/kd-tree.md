---
type: algorithm
status: partial
tags: [easyrobot, graph]
workstream: [W1, W8]
module: "[[graph-and-planning]]"
summary: "NN pruning bug; kNN/range/insert missing"
source: [x/math/graph/kd_tree.go]
tests: [x/math/graph/kd_tree_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 -run 'TestKDTree' ./x/math/graph/ → ok (2026-11-09); probe graphds vs brute force: 0/1000 wrong at coords in [0,1], 74/1000 wrong at coords in [0,100]"
---
# k-d tree (GenericKDTree)

**What it does:** Static k-d tree over `vec.Vector` points, built by median split with dimension cycling `depth % k`; supports single nearest-neighbour query. Also exposes itself as a `Graph` (parent→child edges).

**Where:** `x/math/graph/kd_tree.go` — `GenericKDTree[V, E]` (L23-30), `NewGenericKDTree` (L33-60), `buildTree` (L63-118), `findMedian` (L166-194), `NearestNeighbor` / `nearestNeighborRecursive` (L197-249), `SetCost`/`GetCost` (L252-279), Graph adapter `KDTreeGraphNode`/`KDTreeGraphEdge` (L332-643).

**Inputs → outputs:** `NewGenericKDTree[V,E](points []V)`; `k` = `points[0].Len()`. `NearestNeighbor(query V) V` → stored point (zero `V` if empty). No kNN, radius/range query, insert or delete.

## Completeness review

- Complete: no — kNN, `RangeQuery`, `Insert` absent; their tests are commented out with `// TODO: Implement KNearestNeighbors, RangeQuery, and Insert methods` (`kd_tree_test.go:122-172`).
- Tested: `TestKDTree_BuildTree` (root non-nil, k=2), `TestKDTree_NearestNeighbor` (6 classic points, query (5.1,4.1) → (5,4)), `TestKDTree_GetHeight` (only `>=0`), `TestKDTree_EmptyTree`, `TestKDTree_SinglePoint`, `TestKDTree_3D` (3 points). All queries are ~0.14 from the answer, so the pruning bug cannot show. Run: `go test -count=1 -run 'TestKDTree' -v ./x/math/graph/` → pass.
- Correct: **no**. `kd_tree.go:244` prunes the far subtree with `diff*diff < bestDist`, but `bestDist` is Euclidean `Distance` (sqrt, `vec/vec.go:79-81`), not squared. Correct when `bestDist < 1` (search is only looser), wrong when `bestDist > 1`: subtrees with `sqrt(d) ≤ |diff| < d` are skipped. Probe (`tmp/easyrobot/filters/graphds/main.go`, 200 random 2-D points, 1000 queries vs brute force): scale 1 → 0 wrong, scale 100 → **74/1000 wrong**. So for lidar points in mm or cm, or map coordinates, results are wrong. Fix: compare `|diff| < bestDist` or use `DistanceSqr` throughout.
- Other defects:
  - Initial best is `1e10` (L203), so a query farther than 1e10 returns zero `V`.
  - `findMedian` is a partial selection sort, O(n²) per level (L179-190), giving an O(n² log n) build; `buildTree` allocates new left/right slices at every level (L80-81).
  - Points equal to the median on the split axis go left (L89), which is fine for queries.
  - Coordinates are read with `v.Slice(dim,dim+1).Sum()` (L149-162), which allocates a view per access in the hot loop.
  - Edge IDs `from*1e6+to` (L533) collide past 1e6 nodes.

## Embedded notes

- float32 throughout. Generic over `vecTypes.Vector`, which is an interface, so every coordinate access is a dynamic call.
- Uses maps (`nodeMap`, `costMap`) and `iter.Seq`, plus many build-time allocations. Not suitable for MCU-side lidar NN without rework (flat `[]float32`, squared distances, preallocated nodes).
- TinyGo: generics and `iter` need a recent TinyGo `[INFERENCE]`. Not checked.

## TODOs

- [ ] Fix NN pruning that mixes squared and unsquared distance (`diff*diff < bestDist`) — x/math/graph/kd_tree.go:244 #p1 #ER-028
- [ ] Add NN test with coordinates >1 vs brute force (would catch the above) — x/math/graph/kd_tree_test.go:37 #p2
- [ ] Implement kNN / radius / range query / Insert (tests commented out) — x/math/graph/kd_tree_test.go:122 #p2
- [ ] O(n²) median selection and per-level slice allocs in build — x/math/graph/kd_tree.go:166-194 #p3
- [ ] Coordinate access via Slice().Sum() allocates per call — x/math/graph/kd_tree.go:155-163 #p3

## Links

- Module: [[graph-and-planning]]

---
type: algorithm
status: partial
tags: [easyrobot, graph, planning, sampling]
workstream: [W3, W8]
module: "[[graph-and-planning]]"
summary: "No steer/collision/choose-parent; dead Seed"
source: [x/math/graph/rrtstar.go]
tests: [x/math/graph/rrtstar_test.go]
test_result: flaky
complete: no
correct: no
evidence: "TestRRTStar_Simple2DGrid failed 7/20 and 6/20 (-count=20, 2026-11-09), always 'Path nodes should be connected'"
---
# RRT* (sampling-based planner)

**What it does:** Grows a tree from start by connecting the nearest tree node to each user-sampled node, rewires nodes within a radius when the path through the new node is cheaper, returns a path when `goalChecker` accepts a new node.

**Where:** `x/math/graph/rrtstar.go`: `RRTStarGraph`/`Sampler`/`GoalChecker` types, `RRTStar` :27-39, `NewRRTStar` :60-87 (rng :71-75), `Search` :91-180 (goal inserted into node map :107; rewire call :154; fallback :171), `findNearest` :183, `connectNodes` :229, `rewire` :268-293, `getPathCostToRoot` :296, `removeParentEdge` :346, `estimateRewireRadius` :363, `ensureNodeInGraph` :387, `reconstructPath` :421.

**Inputs → outputs:** graph (Adder+Deleter), sampler, goal checker, options {RewireRadius, MaxIterations, Seed} → `Path[N,float32]`. Distance = `Node.Cost(other)`, i.e. the graph's cost calculator.

## Completeness review

- Complete: **no** — missing core RRT* steps: no *steer* (edge goes straight from nearest to the sample, any length), no collision/obstacle check hook, no choose-best-parent among near nodes (only rewire), radius not shrinking with n (γ(log n/n)^(1/d)). After `MaxIterations` it connects the nearest node directly to the goal regardless of distance (:171), so a "path" is (almost) always returned.
- Tested: `TestRRTStar_Simple2DGrid` (+ interface tests). Run: go test -count=1 -json ./x/math/graph/... (2026-11-09) → ok except TestRRTStar_Simple2DGrid (flaky); `go test -count=20 -run 'TestRRTStar_Simple2DGrid$' ./x/math/graph/` → 7/20 and 6/20 failures; every failure is the assertion "Path nodes should be connected" (edge length ≤ 2.2, `rrtstar_test.go:111`).
- Why it flakes: the test's sampler uses global `rand.Float32()` (`rrtstar_test.go:24-25`), so runs differ; the `Seed: 42` option only seeds `r.rng`, which the implementation never uses (rng set at :71-85, no other reference) — dead option. The real cause of the failing assertion is the missing steer/max-edge-length: `connectionThreshold` (test :42) is never passed to the planner.
- Correct: no (not RRT*; and goal is pre-inserted in the node map (:107) so it can become a "nearest" node before it is reached).
- Performance: `findNearest` O(n) per sample; `getPathCostToRoot` scans all nodes' edges per hop (O(n·E) per call) and is called twice per node in `rewire` → O(n²·E) per iteration; many maps/allocations per call.

## TODOs

- [ ] Add steer (max step) + `CollisionChecker(from,to)` hook; choose-best-parent; shrinking radius — rrtstar.go:91-180,268-293 #p1 #ER-028
- [ ] Use `r.rng` (give samplers access) or delete `Seed`; make the test sampler deterministic — rrtstar.go:71-85, rrtstar_test.go:24 #p1 #ER-028
- [ ] Store parent pointer + cost-to-come per node instead of scanning edges — rrtstar.go:296,346,421 #p2
- [ ] Do not insert goal into the tree before it is reached; no unconditional fallback connection — rrtstar.go:107,171 #p2

## Links

- Module: [[graph-and-planning]]; candidate use: foot/arm trajectory search in [[W3-legged-ik]]

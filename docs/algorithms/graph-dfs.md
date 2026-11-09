---
type: algorithm
status: tested
tags: [easyrobot, graph, search]
workstream: W1
module: "[[graph-and-planning]]"
summary: "Iterative stack search for *a* path (not shortest); neighbours pushed in reverse so the first…"
source: [x/math/graph/dfs.go]
tests: [x/math/graph/dfs_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "5 tests pass"
---
# Depth-first search (generic graph)

**What it does:** Iterative stack search for *a* path (not shortest); neighbours pushed in reverse so the first neighbour is explored first; visited marked on push.

**Where:** `x/math/graph/dfs.go` (103 LOC): `DFS`, `NewDFS`, `Search`, `reconstructPath`, `DFSFunc`.

**Inputs → outputs:** start, goal → `Path` or nil.

## Completeness review

- Complete: yes for reachability/path finding. Marking on push means the visit order is not true DFS preorder (a node reachable from two parents keeps the first discoverer) — fine for paths, wrong if callers want DFS trees/finish times.
- Tested: simple path, reuse, grid graph, no-path. Run: go test -count=1 -json ./x/math/graph/... (2026-11-09) → ok except TestRRTStar_Simple2DGrid (flaky).
- Correct: yes for path existence. Allocates a `neighbors` slice per expanded node.

## TODOs

- [ ] Reuse a scratch slice for neighbours — dfs.go (Search) #p3

## Links

- Module: [[graph-and-planning]]

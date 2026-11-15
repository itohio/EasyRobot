---
type: algorithm
status: partial
tags: [easyrobot, graph]
workstream: W1
module: "[[graph-and-planning]]"
summary: "Misses directed 2-cycles"
source: [x/math/graph/cycles.go]
tests: [x/math/graph/cycles_test.go]
test_result: pass
complete: yes
correct: no
evidence: "6 tests pass; probe cycleprobe: directed 2-cycle A->B->A reported as no cycle"
---
# Cycle detection and connected components

**What it does:** `LoopDetection` — recursive DFS with white/grey/black colouring from a start node, skipping the immediate parent; `ConnectedComponents` — BFS flood from each listed node over `Neighbors()`.

**Where:** `x/math/graph/cycles.go`: `LoopDetection` :5-12, `loopDetectionDFS` :14-39, `ConnectedComponents` :43-64, `findComponent` :66-85.

**Inputs → outputs:** graph + start → bool; graph + node list → `[][]Node`.

## Completeness review

- Complete: partial — only the part of the graph reachable from `start` is checked; no cycle path returned.
- Tested: `TestLoopDetection_NoCycle/WithCycle/SelfLoop` (3-cycle and self-loop on `MatrixGraph`), components single/multiple (multiple only asserts ≥1 component, `cycles_test.go:108`). Run: go test -count=1 -json ./x/math/graph/... (2026-11-09) → ok except TestRRTStar_Simple2DGrid (flaky).
- Correct: **no** for directed graphs. The parent-skip (`cycles.go:29-31`) is an *undirected* rule mixed into *directed* colouring: a directed 2-cycle A→B→A is ignored. Probe: `MatrixGraph` with A→B, B→A → `false`. Conversely, on an undirected graph stored as symmetric edges the colouring is the right tool only with parent-skip — so the function is correct for neither model in general. `ConnectedComponents` follows directed edges, so it returns reachability sets (order-dependent), not weakly connected components. Recursion depth = path length (stack growth on embedded).
- Not used by [[expression-graph]], which has its own (incomplete) ordering.

## TODOs

- [ ] Split into `HasDirectedCycle` (no parent skip) and `HasUndirectedCycle` (skip parent edge once) — cycles.go:29 #p1 #ER-028
- [ ] Iterative DFS; return the cycle nodes #p3
- [ ] Weakly connected components for directed graphs; tighten test at cycles_test.go:108 #p2

## Links

- Module: [[graph-and-planning]]

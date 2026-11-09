---
type: algorithm
status: tested
tags: [easyrobot, graph, planning, search]
workstream: W1
module: "[[graph-and-planning]]"
summary: "Single-pair Dijkstra with lazy-deletion binary heap; maps keyed by `Node` interface values (not…"
source: [x/math/graph/dijkstra.go]
tests: [x/math/graph/dijkstra_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "7 tests pass incl. TestDijkstra_ShortestPath; searchprobe 0/300 mismatches on simple digraphs"
---
# Dijkstra shortest path (generic graph)

**What it does:** Single-pair Dijkstra with lazy-deletion binary heap; maps keyed by `Node` interface values (not IDs).

**Where:** `x/math/graph/dijkstra.go`: `Dijkstra` struct, `NewDijkstra`, `Search` (visited-on-pop), `reconstructPath`, heap, `DijkstraFunc` (whole file, 164 LOC).

**Inputs → outputs:** start, goal → `Path` or nil; non-negative float32 edge costs assumed (not checked).

## Completeness review

- Complete: yes for single-pair; no single-source distance map API.
- Tested: `TestDijkstra_ShortestPath` (cheaper direct edge chosen), grid graph with/without obstacles, no-path, reuse. Run: go test -count=1 -json ./x/math/graph/... (2026-11-09) → ok except TestRRTStar_Simple2DGrid (flaky) (7 pass).
- Correct: yes on simple graphs (probe tmp/easyrobot/filters/searchprobe, 0/300 mismatches vs Bellman-Ford). Keys are interface values: works for pointer nodes that are stable (`GenericNode`) and comparable value nodes (`GridNode`, `MatrixNode`); node types that re-wrap on every `Neighbors()` call or contain non-comparable fields would break visited tracking or panic — inconsistent with A*, which keys by `ID()`.

## TODOs

- [ ] Key state by `ID()` like A* — dijkstra.go (maps in struct) #p2
- [ ] Reject negative costs; expose single-source distances #p3

## Links

- Module: [[graph-and-planning]]

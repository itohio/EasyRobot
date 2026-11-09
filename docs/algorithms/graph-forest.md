---
type: algorithm
status: partial
tags: [easyrobot, graph]
workstream: W1
module: "[[graph-and-planning]]"
summary: "MergeTrees drops subtree"
source: [x/math/graph/forest.go]
tests: []
test_result: none
complete: no
correct: no
evidence: "no tests; probe graphds (2026-11-09): MergeTrees drops tree2 subtree (4 nodes → 2), node IDs collide across trees (4 nodes, 3 distinct IDs), cost key (0→1000) aliases (1→0)"
---
# Forest (GenericForest)

**What it does:** Holds a collection of `Tree[N,E]` values under stable tree IDs, with a forest-level cost map, merge, and a `Graph` view over all nodes.

**Where:** `x/math/graph/forest.go`:
- `GenericForest` (L10-15)
- Tree management: `AddTree`/`RemoveTree` (L29-70), `FindTreeContainingNode` (L74-114), `MergeTrees` (L119-161)
- Costs: `SetCost`/`GetCost` (L178-216)
- Graph adapter: `ForestGraphNode` (L219-433), `Nodes`/`Edges`/`NumNodes`/`NumEdges` (L485-577)

**Inputs → outputs:**
- `AddTree(tree)` → tree ID.
- `MergeTrees(id1, id2)` → bool.
- `FindTreeContainingNode(nodeIdx)` → tree ID, or -1.
- Most internals need `*GenericTree`. Other `Tree` types are skipped or have ID 0.

## Completeness review

- Complete: **no**. `MergeTrees` copies only tree2's root data under tree1's root, then deletes tree2. The source says so: "For now, we'll just mark tree2 as merged… A full implementation would copy all nodes" (L150-155). Probe: merging {1} with {10→11,10→12} gives 2 nodes, expected 4.
- Tested: no tests reference `GenericForest`.
- Correct: no.
  - Node IDs come from per-tree counters, which all start at 1, so `ForestGraphNode.ID()` collides across trees (L242). Probe: 4 nodes, 3 distinct IDs. Any ID-keyed search (BFS, Dijkstra, `ToMatrix`) over a forest is therefore wrong.
  - The cost key `tree*1e6 + parent*1000 + child` (L184) aliases whenever `child ≥ 1000` or `parent ≥ 1000`. Probe: `SetCost(t,0,1000)` is read back by `GetCost(t,1,0)`.
  - `RemoveTree` deletes costs with the condition `k/1000000 == treeIdx` (L64), but the stored keys are `key/1000`. Costs are therefore never removed and get reused by the next tree with the same ID offset.
  - `FindTreeContainingNode` takes a per-tree node index, so it is ambiguous. It returns whichever tree map iteration reaches first (L80).
  - `Nodes()` iterates a map, so order is nondeterministic (L487).

## Embedded notes

Map-heavy, with type assertions to `*GenericTree` throughout. Host only. TinyGo untested.

## TODOs

- [ ] MergeTrees discards tree2's subtree (acknowledged in code) — x/math/graph/forest.go:150-155 #p1 #ER-028
- [ ] Node IDs collide across trees — x/math/graph/forest.go:242 #p1 #ER-028
- [ ] Cost composite key aliases for indices ≥1000 — x/math/graph/forest.go:184 #p1 #ER-028
- [ ] RemoveTree cost cleanup uses wrong divisor — x/math/graph/forest.go:64 #p2
- [ ] Ambiguous FindTreeContainingNode; nondeterministic Nodes order — x/math/graph/forest.go:80 #p3
- [ ] No tests for GenericForest — x/math/graph/forest.go #p2

## Links

- Module: [[graph-and-planning]]
- [[graph-tree]], [[graph-binary-tree]]

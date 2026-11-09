---
type: algorithm
status: partial
tags: [easyrobot, graph]
workstream: W1
module: "[[graph-and-planning]]"
summary: "Placeholder comparison"
source: [x/math/graph/tree_strategies.go]
tests: [x/math/graph/tree_strategies_test.go]
test_result: pass
complete: no
correct: no
evidence: "go test -count=1 -run 'TestBinarySearchTree|TestFirstAvailable|TestBalancedInsertion|TestComparisonStrategies|TestBinaryTree_Balance' ./x/math/graph/ → ok (2026-11-09); probe graphds: SimpleComparison makes every BST insert a duplicate (-1); 64 sorted inserts → height 63"
---
# Tree insertion / comparison strategies

**What it does:** Pluggable functions that choose where `Insert` places a node. There are BST descent for `GenericBinaryTree`, first-available and smallest-subtree placement for `GenericTree`, and typed comparators. These are insertion strategies, not traversal strategies; traversal lives in [[graph-bfs]] and [[graph-dfs]].

**Where:** `x/math/graph/tree_strategies.go`:
- Function types `InsertionStrategy`, `BinaryInsertionStrategy`, `ComparisonStrategy` (L6-15)
- `BinarySearchTreeStrategy` (L19-60)
- Comparators `SimpleComparison` (L63-75), `IntComparison`/`Float32Comparison`/`StringComparison` (L78-115)
- `FirstAvailableStrategy` (L118-130), `BalancedInsertionStrategy` (L133-162)

**Inputs → outputs:**
- Binary strategy `(tree, data, idx) → (parentIdx, goLeft)`. It returns `(-1,false)` for duplicates or when the tree is not a `*GenericBinaryTree`.
- N-ary strategy `→ parentIdx`.

## Completeness review

- Complete: no.
  - `SimpleComparison` is a self-declared placeholder that returns 0 for all inputs (L66-74). Used with the BST strategy, it rejects every insert as a duplicate (probe: `Insert(7)` → -1).
  - `BalancedInsertionStrategy` always descends to a leaf of the smallest subtree, so it builds chains rather than a balanced tree, and calls the O(n) `countDescendants` at every level (L150-153).
  - `FirstAvailableStrategy` just returns the parent (L129).
- Tested: `TestBinarySearchTreeStrategy_Insert`, `_DuplicatePrevention`, `_Float32`, `_String`, `TestFirstAvailableStrategy`, `TestBalancedInsertionStrategy`, `TestComparisonStrategies`, `TestBinaryTree_Balance*`, `TestGenericTree_Balance`, `*_Insert_WithoutStrategy`. Run → pass.
- Correct: the BST descent is correct for a strict total order. Probe: a 7-insert BST has height 3 and a duplicate returns -1. There is no self-balancing: 64 ascending inserts give height 63, so lookups are O(n) worst case. `correct: no` because of the placeholder comparator.
- The BST descent recurses and builds a new closure per level (L44-45, L52-53).

## TODOs

- [ ] SimpleComparison placeholder returns 0 for unequal values (BST rejects all inserts) — x/math/graph/tree_strategies.go:63-75 #p1 #ER-028
- [ ] BalancedInsertionStrategy does not balance; O(n) per level — x/math/graph/tree_strategies.go:133-162 #p2
- [ ] No self-balancing BST (AVL/RB) — degenerate on sorted input — x/math/graph/tree_strategies.go:19 #p3

## Links

- Module: [[graph-and-planning]]
- [[graph-binary-tree]], [[graph-tree]]

---
type: algorithm
status: partial
tags: [easyrobot, graph]
workstream: W1
module: "[[graph-and-planning]]"
summary: "FindNodeByData always -1"
source: [x/math/graph/tree.go]
tests: [x/math/graph/tree_strategies_test.go, x/math/graph/vector_decision_tree_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 -run 'TestGenericTree|TestFirstAvailable|TestBalancedInsertion' ./x/math/graph/ → ok (2026-11-09); probe graphds: FindNodeByData always -1; RemoveChild leaves grandchildren counted"
---
# N-ary tree (GenericTree)

**What it does:** An index-based n-ary tree (children kept as `[]int` per node) with ordered children, optional parent→child costs and a `Graph`/`Tree` adapter. It is the backing store for the decision trees and for [[graph-forest]].

**Where:** `x/math/graph/tree.go`:
- Types: `TreeNodeValue` (L7-12), `GenericTree` (L17-23)
- Adding children: `AddChild` (L66-71), `AddChildAtPosition` (L76-99)
- Reordering: `ReorderChildren` (L104-136), `SwapChildren` (L140-153)
- `Insert` (L156-170), `Balance` (L188-237), `RemoveChild` (L240-270)
- Costs (L273-300); depth, height and path (L329-394)
- `FindNodeByData` (L398-426), `dataNodeWrapper` (L429-522), Graph adapter (L525-787)

**Inputs → outputs:**
- `AddChild(parentIdx, data)` → index, or -1.
- `Balance()` only reorders children so that smaller subtrees come first. It does not restructure the tree.
- `RemoveChild` → bool.

## Completeness review

- Complete: yes for the API.
- Tested: `TestGenericTree_Balance`, `TestGenericTree_Insert_WithoutStrategy`, `TestFirstAvailableStrategy`, `TestBalancedInsertionStrategy` (`tree_strategies_test.go`); `TestGenericTree_AddChildAtPosition`, `TestGenericTree_ReorderChildren`, `TestGenericTree_SwapChildren` (`vector_decision_tree_test.go:233-300`), which check child order. Run → pass. There is no test of `FindNodeByData` or `RemoveChild`.
- Correct: no.
  - **`FindNodeByData` never finds anything.** L415 calls `node.Equal(dataNode)`, and `TreeGraphNode.Equal` type-asserts the argument to `TreeGraphNode` (L539-548), so it is always false. Probe: a tree {1→2} returns -1 for both 1 and 2. `GenericBinaryTree` calls the method in the other direction and works.
  - `RemoveChild` marks only the child as removed (L262). Its descendants stay counted by `Nodes`/`NumNodes`. Probe: `NumNodes()=2`, expected 1.
  - `NodeCount()` includes removed nodes (L316).
  - `countDescendants` is recursive and O(subtree). `balanceRecursive` calls it per child at every level, giving O(n²) on deep trees (L213), and sorts children with an O(c²) exchange sort (L218-224).
  - Edge IDs `from*1e6+to` (L686) collide beyond 1e6 nodes.

## Embedded notes

Every node allocates its own `childIdxs` slice (L50). Maps hold IDs and costs. `dataNodeWrapper.Compare` boxes values into `any` and uses a type switch (L447-502). Host-oriented; TinyGo untested.

## TODOs

- [ ] FindNodeByData always returns -1 (Equal argument order) — x/math/graph/tree.go:415 #p1 #ER-028
- [ ] RemoveChild leaves descendants live — x/math/graph/tree.go:262 #p1 #ER-028
- [ ] Add tests for FindNodeByData / RemoveChild — x/math/graph/tree.go:240 #p2
- [ ] O(n²) Balance via repeated countDescendants — x/math/graph/tree.go:195-237 #p3

## Links

- Module: [[graph-and-planning]]
- [[graph-binary-tree]], [[tree-insertion-strategies]], [[graph-forest]], [[decision-tree]]

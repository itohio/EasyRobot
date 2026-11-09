---
type: algorithm
status: partial
tags: [easyrobot, graph]
workstream: W1
module: "[[graph-and-planning]]"
summary: "An index-based binary tree (nodes in a slice; parent, left and right are indices) with optional…"
source: [x/math/graph/binary_tree.go]
tests: [x/math/graph/binary_tree_test.go, x/math/graph/tree_strategies_test.go]
test_result: pass
complete: yes
correct: no
evidence: "go test -count=1 -run 'TestBinaryTree|TestBinarySearchTree' ./x/math/graph/ → ok (2026-11-09); probe graphds: RemoveLeft leaves orphans counted (NumNodes 3, expected 1); FindNodeByData on []int panics"
---
# Binary tree (GenericBinaryTree)

**What it does:** An index-based binary tree (nodes in a slice; parent, left and right are indices) with optional edge costs. It becomes a BST when used with `BinarySearchTreeStrategy` (see [[tree-insertion-strategies]]). It also implements `Graph` and `Tree`.

**Where:** `x/math/graph/binary_tree.go`:
- Types: `BinaryTreeNodeValue` (L7-13), `GenericBinaryTree` (L18-24)
- Editing: `SetLeft`/`SetRight` (L63-100), `Insert` (L103-124), `Balance` (L134-225), `RemoveLeft`/`RemoveRight` (L228-271)
- Edge costs: `SetCost`/`GetCost` (L274-301)
- Queries: `FindPathToRoot`, `GetDepth`, `GetHeight` (L330-394), `FindNodeByData` (L398-436)
- Graph adapter (L441-749)

**Inputs → outputs:**
- `NewGenericBinaryTree[N,E](root)`.
- `SetLeft`/`SetRight` return the child index and overwrite the data of an existing child.
- `Insert(data, strategy)` → index, or -1 (a duplicate under BST).
- `Balance(cmp)` rebuilds the tree as a perfectly balanced BST from the sorted node list.

## Completeness review

- Complete: yes for the stated API, which has no BST search or delete-with-rebalance. `FindNodeByData` is a linear DFS even for a BST (L402-436).
- Tested: 13 `TestBinaryTree_*` tests (new, set left/right, depth, height values 0/1/1/2, find, remove, path to root, graph interface, neighbours, cost), plus `TestBinaryTree_Balance*` and `TestBinaryTree_Insert_WithoutStrategy` in `tree_strategies_test.go`. Run → pass. Balance only asserts `heightAfter <= heightBefore` and the node count.
- Correct: no.
  - `RemoveLeft`/`RemoveRight` mark only the direct child as removed (`parentIdx=-2`, L239/L262). Its descendants stay "live", so `Nodes()`/`NumNodes()` still report them. Probe: removing the subtree {2,3,4} from {1,2,3,4} → `NumNodes()=3`, expected 1.
  - `NodeCount()` returns `len(nodes)` including removed nodes (L317-319), which disagrees with `NumNodes()`.
  - `FindNodeByData` compares values with `any == any` (`tree.go:437-445`), so it panics for non-comparable `N`. Probe: `[]int` → `runtime error: comparing uncomparable type []int`.
  - `Balance` drops all edge costs (L160) and reassigns IDs. With `cmp == nil` it builds a "balanced" tree in pre-order, not sorted order. It sorts with an O(n²) exchange sort (L145-152).
  - Unbalanced by default: 64 ascending BST inserts give height 63 (probe).
  - Edge IDs `from*1e6+to` (L643) collide beyond 1e6 nodes.

## Embedded notes

Generic `N`/`E`. Every structure keeps maps (`nodeMap`, nested `costMap`). Recursion is used for height, find and balance. `iter.Seq` closures; no float math. Usable on the host; heavy for MCU (maps, interface boxing in `Compare`). TinyGo untested.

## TODOs

- [ ] RemoveLeft/RemoveRight leave descendants live (counted by Nodes/NumNodes) — x/math/graph/binary_tree.go:239 #p1 #ER-028
- [ ] FindNodeByData panics for non-comparable N (`any ==`) — x/math/graph/tree.go:443 #p1 #ER-028
- [ ] NodeCount counts removed nodes, inconsistent with NumNodes — x/math/graph/binary_tree.go:317 #p3
- [ ] Balance discards costs; O(n²) sort; nil comparator gives unsorted shape — x/math/graph/binary_tree.go:134-166 #p2
- [ ] Tests for removal of subtrees and balance shape/order — x/math/graph/binary_tree_test.go:119 #p2

## Links

- Module: [[graph-and-planning]]
- [[tree-insertion-strategies]], [[graph-tree]], [[graph-forest]]

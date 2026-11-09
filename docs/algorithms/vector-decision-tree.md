---
type: algorithm
status: partial
tags: [easyrobot, graph, decision-tree, behaviour]
workstream: W6
module: "[[graph-and-planning]]"
summary: "Balance swaps branches; >2 children ignored"
source: [x/math/graph/vector_decision_tree.go]
tests: [x/math/graph/vector_decision_tree_test.go]
test_result: pass
complete: yes
correct: no
evidence: "vector_decision_tree tests pass; code review: Balance() reorders true/false children; 3rd+ children ignored"
---
# Vector decision tree (binary threshold routing on feature vectors)

**What it does:** Tree over `vecTypes.Vector` inputs; internal nodes hold a `DecisionCriteria(v) bool`, leaves an `Output`. Evaluation: criterion true → child 0, false → child 1; single child taken only if true. Helper criteria: per-dimension threshold, norm threshold, dot-product threshold (all `≤ threshold`).

**Where:** `x/math/graph/vector_decision_tree.go`: node data :18-25, `NewVectorDecisionTree` :37-49, `AddDecisionChild` :57-69, `AddLeafChild` :77-89, `ComputeDecision` :95-102, `computeDecisionRecursive` :104-153, `ReorderChildren`/`SwapChildren`/`Balance` :156-168, criteria helpers :194-219.

**Inputs → outputs:** feature vector → (Output, ok).

## Completeness review

- Complete: evaluation only; **no training** (no CART/ID3 split search, no impurity, no data API).
- Tested: `vector_decision_tree_test.go` (SimpleDecision, Reorder, Swap, AddChildAtPosition, Balance, NormThreshold, …). Run: go test -count=1 -json ./x/math/graph/... (2026-11-09) → ok except TestRRTStar_Simple2DGrid (flaky). `TestVectorDecisionTree_Balance` explicitly does not check the decision after balancing (`vector_decision_tree_test.go:193-198`).
- Correct: **no** for structural ops: `Balance()` delegates to `GenericTree.Balance`, which sorts children by subtree size (`tree.go:206-224`), silently swapping the true/false branches and changing decisions. Children beyond the second are ignored without error (:135-148) — e.g. the Balance test's 3-way root never reaches child3. `VectorDimensionThreshold` reads one element via `Slice(...).Sum()` (:199-200), an interface round-trip per comparison.
- Like [[decision-tree]], criteria are closures → not serialisable with parameters.

## TODOs

- [ ] Forbid/override `Balance` for decision trees (or store explicit true/false child slots) — vector_decision_tree.go:166, tree.go:188-230 #p1 #ER-028
- [ ] Reject >2 children or define multiway semantics — vector_decision_tree.go:135-148 #p2
- [ ] Plain-data split representation (`feature int, threshold float32`) + CART training (see [[W6-decision-trees]]) #p2

## Links

- Module: [[graph-and-planning]]; [[W6-decision-trees]]

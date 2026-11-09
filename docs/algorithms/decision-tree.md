---
type: algorithm
status: tested
tags: [easyrobot, graph, decision-tree, behaviour]
workstream: W6
module: "[[graph-and-planning]]"
summary: "Evaluation + storage round trip"
source: [x/math/graph/decision_tree.go, x/math/graph/decision_interfaces.go]
tests: [x/math/graph/decision_test.go, x/marshaller/graph/decision_expression_e2e_test.go]
test_result: pass
complete: yes
correct: yes
evidence: "decision_test 2/2 pass (threshold routing values); marshaller/graph 11/11 pass incl. TestDecisionTreeRoundTrip"
---
# Generic decision tree (hand-authored, op-per-node/edge)

**What it does:** N-ary tree (on `GenericTree`) where nodes may carry a `DecisionOp(input) (Output, ok)` and parent→child edges may carry a `DecisionEdgeOp(input) bool`. `Decide(start, inputs...)` does a depth-first first-match search: a node op that returns ok=true ends the search; otherwise children whose edge criterion passes are tried in child order, backtracking on failure.

**Where:** `x/math/graph/decision_interfaces.go:7-45` (`DecisionNode`, `DecisionEdge`, `DecisionTree`); `x/math/graph/decision_tree.go`: `GenericDecisionTree` :20-24, `NewGenericDecisionTree` :27, `AddChild` :97, `Decide` :112-132, `SetNodeOpByIndex/ID` :135-150, `SetEdgeOpByIndex/ID` :162-179, `decideFrom` :233-268, node/edge wrappers :296-379. Storage/replay: `x/marshaller/graph/stored_decision_tree.go:12-122` (ops resolved by registered function name, `ops_registry.go:116-128`).

**Inputs → outputs:** generic `Input` → `[]Output` per input; error "no decision path found" when nothing matches.

## Completeness review

- Complete: yes for evaluating hand-written trees. **No learning**: no split search (Gini/entropy/variance), no training data structures, no pruning, no probabilistic leaves.
- Tested: `TestGenericDecisionTree_Decide` (x>5 → "A", x≤5 → "B", incl. equality boundary) and interface conformance (`decision_test.go:10-72`); `TestDecisionTreeRoundTrip` marshals to files and checks `StoredDecisionTree.Decide` reproduces outputs for inputs 2,7,4,9. Run: go test -count=1 -json ./x/math/graph/... (2026-11-09) → ok except TestRRTStar_Simple2DGrid (flaky); `go test -count=1 -json ./x/marshaller/graph/...` → ok, 11 pass (same day).
- Correct: yes for tested routing. Notes: internal-node op is invoked twice when it returns ok=false (`DecisionFunction()` then `EvaluateDecision()`, :246-253) — doubles cost and side effects; wrappers are allocated on every `Edges()` iteration (:313-322, :283-294).
- Storage limitation: ops are persisted by `runtime.FuncForPC` name, so closures with captured parameters (e.g. thresholds) collide/lose their parameters — trees whose behaviour lives in data (feature index, threshold, leaf value) cannot be stored this way [INFERENCE from `ops_registry.go:116-128`].
- Embedded: maps for ops, interface wrappers, `reflect.Call` in the stored variant — heavy for ESP32/TinyGo [INFERENCE].

## TODOs

- [ ] Call the node op once in `decideFrom` — decision_tree.go:246-253 #p2
- [ ] Data-driven split node type (feature, threshold, children, leaf value) that serialises without closures #p2
- [ ] Tests for backtracking order and failing-op fallthrough #p3

## Links

- Module: [[graph-and-planning]]; workstream [[W6-decision-trees]]; storage [[marshallers]]

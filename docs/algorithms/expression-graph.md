---
type: algorithm
status: partial
tags: [easyrobot, graph, expression, dataflow]
workstream: [W6, W9]
module: "[[graph-and-planning]]"
summary: "No cycle check; AddNode handles evaluate to 0"
source: [x/math/graph/expression_graph.go, x/math/graph/expression_interfaces.go]
tests: [x/math/graph/expression_graph_test.go]
test_result: pass
complete: no
correct: no
evidence: "3 tests pass; probe exprprobe: cycle → 'start node 1 not evaluated'; Compute(handle from AddNode) returns 0 instead of 6"
---
# Expression graph (bottom-up dataflow evaluation)

**What it does:** DAG where each node has an `ExpressionOp(input, childOutputs map[id]Output) (Output, bool)`; `Compute(start, inputs...)` collects nodes reachable from `start`, orders them children-first (Kahn's algorithm on out-degree) and evaluates each node once per input.

**Where:** `x/math/graph/expression_interfaces.go:8-33` (`ExpressionNode`, `ExpressionOp`, `ExpressionGraph`); `x/math/graph/expression_graph.go`: `GenericExpressionGraph` :11-15, `AddNode` :26-38, `SetRoot` :41, `AddEdge` :51-58, `Compute` :102-131, `buildEvaluationOrder` :135-178, `collectReachableNodes` :180-197, `evaluateNode` :199-231, `computeOnce` :245-263. Persistence: `x/marshaller/graph/stored_expression_graph.go` ([[marshallers]]).

**Inputs → outputs:** generic `Input`, `Output`; edges point parent → child (operand).

## Completeness review

- Complete: no.
  - **No full cycle check:** `buildEvaluationOrder` never compares `len(result)` with `len(nodeSet)` (:162-177) and always returns `nil` error. Nodes on or above a cycle are silently dropped from the order; the failure surfaces later as `"start node %d not evaluated"` (:262). Probe (`tmp/easyrobot/filters/exprprobe`): r→p, p↔q → `err=start node 1 not evaluated`; self-loop below root → same message. If the cycle is in a branch not needed by the start node it cannot occur (all reachable nodes are ancestors of start) — so every cycle yields this misleading error, never a cycle error.
  - Child outputs are passed as `map[int64]Output` — operand order is lost (non-commutative ops must look up child IDs), and a fresh map is allocated per node per input (:206) plus a values map per input (:250).
  - `AddEdge` silently returns nil for nodes it cannot resolve (:54-56).
- Tested: `TestExpressionGraph_Compute`, `SimpleAdd`, interfaces — small trees evaluated via `Compute(nil, …)`. Run: go test -count=1 -json ./x/math/graph/... (2026-11-09) → ok except TestRRTStar_Simple2DGrid (flaky).
- Correct: **no** for the documented API: `Compute(start)` with the node handle returned by `AddNode` evaluates a detached copy with no edges (see [[generic-graph]] `AddNode` copy). Probe: diamond root=a+b, a=b=x, x=input 3 → `Compute(nil,3)=[6]`, `Compute(rootHandle,3)=[0]` with nil error; `rootHandle.Edges()` yields 0 edges.

## TODOs

- [ ] Return an error when `len(order) != len(nodeSet)` (report cycle nodes) — expression_graph.go:162-177 #p1 #ER-028
- [ ] Make `AddNode` return the stored node (or resolve handles by ID in `Compute`) — expression_graph.go:26-38, generic.go:343 #p1 #ER-028
- [ ] Ordered operands (slice in edge order) instead of map; reuse buffers across inputs — expression_graph.go:206,250 #p2
- [ ] Tests: cycle error, handle-based Compute, diamond sharing #p1 #ER-028

## Links

- Module: [[graph-and-planning]]; workstreams [[W6-decision-trees]], [[W9-go-nn-framework]] (a computation graph for NN ops lives in [[nn]]/[[tensor]], not here)
- In-tree consumer: the PEG packet parser builds its program on `GenericExpressionGraph[nodeData, float32, State, Decision]` (`x/math/protocol/peg/program.go:13-22,149-167`) and calls `Compute(nil, state)` after `SetRoot`, so it avoids the detached-handle defect; a cyclic pattern graph would surface as "start node not evaluated". See [[peg-protocol-parser]].

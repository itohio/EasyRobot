---
type: algorithm
status: tested
tags: [easyrobot, marshaller, graph, decision-tree, expression-graph]
workstream: W6
module: "[[marshallers]]"
summary: "2 e2e pass"
source: [x/marshaller/graph/ops_registry.go, x/marshaller/graph/graph_metadata.go, x/marshaller/graph/stored_decision_tree.go, x/marshaller/graph/stored_expression_graph.go, x/marshaller/graph/options.go]
tests: [x/marshaller/graph/decision_expression_e2e_test.go]
test_result: pass
complete: no
correct: unknown
evidence: "go test -count=1 -v ./x/marshaller/graph → TestDecisionTreeRoundTrip, TestExpressionGraphRoundTrip PASS (2026-11-09)"
---
# Graph marshaller: decision-tree / expression-graph op persistence

**What it does:** stores executable graphs by recording, per node/edge, the **Go symbol name** of the attached function (`runtime.FuncForPC`), plus root id and graph kind, in a protobuf `GraphMetadata` entry of the data file. On load, functions registered with `WithDecisionOp`/`WithDecisionEdge`/`WithExpressionOp` are looked up by the same name and invoked by reflection, so a `StoredDecisionTree.Decide` / `StoredExpressionGraph.Compute` runs straight from the mapped files.

**Where:** name capture `functionName` `ops_registry.go:116-128`; call adapters `callDecisionNodeOp` :130, `callDecisionEdgeOp` :145, `callExpressionOp` :160, child-output conversion :179-217; metadata capture `captureDecisionMetadata` `graph_metadata.go:177-241`, `captureExpressionMetadata` :243-279, `deriveRootID` :281; options `WithDecisionOp`/`WithDecisionEdge`/`WithExpressionOp` `options.go:158-246` (panic on non-function arguments, :177/:207/:237); runtime wrappers `stored_decision_tree.go`, `stored_expression_graph.go`.

**Inputs → outputs:** a `graph.GenericDecisionTree[N,E,In,Out]` or `GenericExpressionGraph` whose nodes expose `DecisionFunction()`/edges `CriteriaFunction()` → metadata `NodeOps map[id]name`, `EdgeOps[(parent,child)]name`; loader needs the same functions passed as options.

## Completeness review

- Complete: no — depends on [[graph-marshaller-mmap-format]], which drops node data, so trees whose leaves/thresholds live in node payloads (e.g. learned split values) lose them; only logic encoded in named functions survives. Function identity is the linker symbol (`pkg.fn`, closures become `pkg.outer.func1`), which breaks on rename/refactor and cannot carry parameters (a threshold has to be a separate named function). No versioning of op names.
- Tested: `TestDecisionTreeRoundTrip` builds a 3-node tree with two edge criteria (<5, ≥5) and two leaf ops, runs `Decide` on inputs 2,7,4,9 before and after a file round trip and requires identical outputs; `TestExpressionGraphRoundTrip` does the same for `Compute` on an input/double/sum graph. Both pass.
- Correct: unknown beyond these two fixtures; reflection conversion paths (`convertToType` :199) are only exercised for `int`/`string`/`float` inputs.

## TODOs

- [ ] Stable op identifiers (explicit names in `WithDecisionOp(name, fn)`) instead of linker symbols — x/marshaller/graph/ops_registry.go:116-128 #p2
- [ ] Parameterised ops (store per-node params as node data once node data persists) for learned trees (W6) — x/marshaller/graph/graph_metadata.go:177 #p2
- [ ] Return errors instead of panicking in option constructors — x/marshaller/graph/options.go:177,207,237 #p3

## Links

- Module: [[marshallers]]; format [[graph-marshaller-mmap-format]]; trees [[graph-and-planning]]; [[W6-decision-trees]]

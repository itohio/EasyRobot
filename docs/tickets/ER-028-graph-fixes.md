---
id: ER-028
title: "Graph: cycle checks, detached handles, RRT*, decision-tree Balance"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W6
milestone: M-d
tags: [easyrobot, ticket, w6, graph]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-028 — Graph: cycle checks, detached handles, RRT*, decision-tree Balance

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

Graph defects ER-002 expects to hit plus the decision-tree fix M-d needs ([[graph-and-planning]]):

- **Expression-graph cycle check**: a cycle yields "start node not evaluated" — `x/math/graph/expression_graph.go:162-177` ([[expression-graph]]).
- **Detached `AddNode` handles**: copies returned, `Compute(handle)` gives 0 — `expression_graph.go:26-38`, `x/math/graph/generic.go:343`.
- **Directed 2-cycle detection** skips the parent edge — `cycles.go:29` ([[graph-cycle-detection]]).
- **RRT\***: `Seed` unused, test sampler uses global `rand` (flaky `TestRRTStar_Simple2DGrid`), no steering/collision hook/choose-parent — `rrtstar.go:71-180`, `:268-293`, `rrtstar_test.go:24` ([[rrt-star]]).
- **Decision-tree `Balance`** swaps true/false branches; >2 children ignored — `vector_decision_tree.go:135-166`, `tree.go:188-230` ([[vector-decision-tree]]).

## Dependencies and scope

Dependencies: With [[ER-002-cpu-graph-semantics|ER-002]] (its fixture items 1–2 are the failing tests for the first two bullets) and [[ER-020-eager-views-newas-norm-gradients|ER-020]]. Precedes [[ER-032-flat-decision-tree-executor|ER-032]].

Not in scope: Tree/forest/kd-tree defects (`forest.go`, `tree.go:262,415,443`, `kd_tree.go:244`) stay backlog.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Failing test first: `Compute` on a cyclic expression graph returns a cycle error naming the nodes.
- [ ] Failing test first: `Compute(handleFromAddNode)` returns the node's value.
- [ ] Failing test first: `HasDirectedCycle` finds a→b→a; `HasUndirectedCycle` does not report a single undirected edge.
- [ ] RRT*: two runs with the same seed give the same path; `go test -count=30 -run TestRRTStar ./x/math/graph` passes 30/30.
- [ ] Failing test first: a decision tree decides the same before and after `Balance` (or `Balance` is rejected for decision trees).

## Effort uncertainty and revision trigger

Product's guess: part of the trusted-reference step; ≈ **M**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

---
id: ER-002
title: Establish CPU graph and tensor semantics
type: ticket
status: ready
kind: implementation
owner: easyrobot
created: 2026-11-08
workstream: W9
milestone: M-f
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-002 — Establish CPU graph and tensor semantics

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Exercise a tiny callback graph separately from a tiny tensor graph. Check repeated inputs/results, invalid dependencies/cycles, in-place versus destination writes, view aliasing and buffer lifetime.

Source: x/math/graph/expression_graph.go; x/math/tensor/types/graph.go; eager/ and gorgonia/ tensor paths. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-001-reference-environment|ER-001]] — done 2026-11-08; the packages below build CGO-free on darwin/arm64 and their existing tests pass ([[baseline-2026-11-08|baseline]]). `ready` = dependencies met and acceptance defined. Source/test edits approved 2026-11-09 by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call.

Not in scope: No new universal executor, distributed scheduling or model decoder.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Known small values produce expected scalar and tensor results over repeated computations.
- [ ] Cycles, missing inputs and unsupported operations fail explicitly rather than silently producing partial success.
- [ ] Aliasing, cloned output independence and base/view release rules are exercised; results identify the actual backend.

### First fixture: concrete acceptance (proposed 2026-11-08)

Test files only, in `x/math/graph`, `x/math/tensor/eager` and `x/math/tensor/gorgonia`; production code changes only after a failing test shows a defect, each fix with its failing-first test.

1. **Callback graph** (`NewGenericExpressionGraph`, `AddNode`, `AddEdge`, `Compute`): a diamond `d ← {b, c} ← a` with known values. Two `Compute` calls with different inputs both return the expected value, and a call counter shows each node is evaluated once per call.
2. **Explicit failures:** a cycle `a → b → a`, a node without `ExpressionNode`, and a callback returning `ok=false` each make `Compute` return an error and no partial output. The source has no full-cycle check (`buildEvaluationOrder`), so the cycle case is expected to expose a defect; record the outcome either way.
3. **Eager vs graph numerics:** the same 2×3·3×2 `MatMul` → `Add` bias → `ReLU` on eager FP32 and on the Gorgonia graph (`NewExpressionGraph`, `Compile`, `Compute`) agree within 1e-6 absolute; a second `Compute` with new input data gives the new result (VM reset); graph mutation after `Compile` is rejected. Each assertion message names the backend.
4. **Aliasing and lifetime (eager):** `FromFloat32` adopts the caller's slice (caller mutation visible); a `Slice`/`Reshape` view shares backing; `Clone` is independent; writing to `dst` versus in place is tested for `Add`; behaviour of `Release` on a base with a live view is recorded as observed.
5. **Command:** `GOTOOLCHAIN=local GOPROXY=off go test -count=3 -race -run 'Semantics' ./x/math/graph ./x/math/tensor/eager ./x/math/tensor/gorgonia` passes, or each failure is recorded here as a finding with its failing test. `-run` keeps out the pre-existing flaky `TestRRTStar_Simple2DGrid` (its sampler uses the global `rand.Float32()`, `rrtstar_test.go:24–25`; 14/30 and 16/30 failures at `-count=30`).

### Expected outcomes from ER-R2 probes (2026-11-09)

The fixture should fail on these, then drive the fixes (see [[graph-and-planning]], [[tensor]]):

- Item 2: a cycle yields "start node not evaluated" instead of a cycle error (`x/math/graph/expression_graph.go:162-177`).
- Item 1: node handles returned by `AddNode` are detached copies, so `Compute(handle)` returns 0 (`x/math/graph/generic.go:343`).
- Item 4: eager `Data()` ignores a view's offset (`x/math/tensor/eager/tensor.go:103-111`) and `x/iter/st/apply_tensor.go:53` drops the inner stride, so ops on `Slice`/transposed views are wrong; `NewAs` aliases its input (`eager/tensor.go:42-44`).

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

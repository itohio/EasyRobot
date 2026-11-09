---
id: ER-015
title: Measure approved backend costs without hidden fallback
type: ticket
status: blocked
kind: measurement
owner: unassigned-personal-profile
created: 2026-11-08
workstream: W9
milestone: M-f
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-015 — Measure approved backend costs without hidden fallback

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

For each available path, require its own completed ticket first (ER-006, ER-008, ER-010 or ER-014). Compare matched inputs, precision, thread settings and warm/cold states; retain negative results.

Source: CPU fixtures plus any individually approved target adapter or local decoder result. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-002-cpu-graph-semantics|ER-002]], [[ER-003-operator-capabilities|ER-003]]

Not in scope: No obligation to implement every optional backend just to fill a comparison table.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Results name hardware/source/model/compiler revisions and capture compile/load, compute, transfers, copies/allocations, memory and observable power limits.
- [ ] Unsupported operations and CPU fallback are explicitly counted rather than labelled accelerator execution.
- [ ] Repeated matched runs have variability and interpretation; missing hardware/model access is a blocker, never an invented benchmark.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

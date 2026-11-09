---
id: ER-008
title: Integrate one verified K230 NPU subgraph
type: ticket
status: blocked
kind: implementation
owner: unassigned-personal-profile
created: 2026-11-08
workstream: W4
milestone: M-b
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-008 — Integrate one verified K230 NPU subgraph

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Add the selected model compile/load/invoke path only after feasibility is evidenced; expose tensor conversions, unsupported-op partitioning and fallback.

Source: ER-007 board/compiler/runtime decision and CPU fixture. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-004-device-quantisation-contract|ER-004]], [[ER-007-k230-feasibility|ER-007]]

Not in scope: No generic accelerator abstraction rewrite or NPU LLM promise.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Approved board compares output/intermediates against fixed CPU inputs and declared tolerances.
- [ ] Cold compile/load, warm execution, transfers and memory are reported separately.
- [ ] Unsupported layout/operator/precision fails visibly or follows the declared measured CPU fallback.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

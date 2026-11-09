---
id: ER-010
title: Integrate one verified K510 NPU subgraph
type: ticket
status: blocked
kind: implementation
owner: unassigned-personal-profile
created: 2026-11-08
workstream: W9
milestone: none
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-010 — Integrate one verified K510 NPU subgraph

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Implement only the selected supported subgraph with explicit lifetime/layout/precision boundaries and visible CPU fallback policy.

Source: ER-009 feasibility decision and selected CPU reference. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-004-device-quantisation-contract|ER-004]], [[ER-009-k510-feasibility|ER-009]]

Not in scope: No shared ABI inferred from K230 and no whole-model port.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Approved board yields outputs within declared CPU-reference tolerance.
- [ ] Measurements separate compile/load, invoke, transfer, allocations and fallback.
- [ ] Repeat invocation, incompatible input and resource-release paths are observed on the selected target.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

---
id: ER-009
title: Investigate K510 NPU independently
type: ticket
status: backlog
kind: discovery
owner: unassigned-personal-profile
created: 2026-11-08
workstream: W9
milestone: none
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-009 — Investigate K510 NPU independently

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

UNVERIFIED hypothesis: K510 NPU support is missing or feasible. Record board, compiler/runtime/SDK, model format, operator support, RAM and transfer requirements separately from K230.

Source: Pinned source search found no named K510 compiler/runtime integration. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-001-reference-environment|ER-001]], [[ER-003-operator-capabilities|ER-003]]

Not in scope: No automatic K230-to-K510 compatibility assumption.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Independent version-pinned evidence names the real board/toolchain and access prerequisites.
- [ ] A bounded subgraph plus CPU comparator and complete copy/fallback accounting is proposed.
- [ ] Unsupported/unknown capabilities remain labelled; related vendor names are not proof of compatibility.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

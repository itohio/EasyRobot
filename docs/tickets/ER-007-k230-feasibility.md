---
id: ER-007
title: Investigate K230 NPU compiler and runtime feasibility
type: ticket
status: backlog
kind: discovery
owner: unassigned-personal-profile
created: 2026-11-08
workstream: W4
milestone: M-b
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-007 — Investigate K230 NPU compiler and runtime feasibility

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

UNVERIFIED hypothesis: K230 NPU support is missing or feasible. Identify actual board, SDK/compiler/runtime/nncase versions, model formats, supported operators, dtype/layout, memory and transfer boundaries.

Source: Pinned source search found no named K230/nncase/KPU adapter; generic tensor and TFLite paths exist. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-001-reference-environment|ER-001]], [[ER-003-operator-capabilities|ER-003]]

Not in scope: Do not infer K230 support from TFLite comments or reuse K510 conclusions.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Source/vendor evidence is version-pinned and distinguishes compilation support from board execution.
- [ ] One small supported non-LLM subgraph is proposed with CPU reference and complete transfer/fallback budget.
- [ ] No Qwen/NPU claim is made without exact-model operator coverage and board evidence.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

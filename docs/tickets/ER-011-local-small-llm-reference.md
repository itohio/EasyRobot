---
id: ER-011
title: Select and establish a small local CPU LLM reference
type: ticket
status: backlog
kind: discovery
owner: unassigned-personal-profile
created: 2026-11-08
workstream: W9
milestone: M-f
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-011 — Select and establish a small local CPU LLM reference

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Select one small Qwen-or-similar artifact/runtime that fits the actual local machine. Before any authorised download/run, record artifact revision/hash/licence, tokenizer, precision/quantisation, context and memory budget. Establish an external local comparator, not an EasyRobot integration claim.

Source: No complete Qwen/Llama decoder/tokenizer/loader path found in pinned source; source-inventory.md documents the boundary. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-001-reference-environment|ER-001]]

Not in scope: No model installation/run now, no fine-tuning, and no accelerator prerequisite.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] All model/runtime artefacts and execution settings are reproducible, local-only and licence-checked.
- [ ] After later approval, fixed prompts yield captured tokenizer IDs, logits or an explicitly accessible comparison contract, deterministic selection and bounded generation.
- [ ] Load/prefill/decode memory and latency are measured; no hosted teacher/comparison model or opaque cloud fallback is used.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

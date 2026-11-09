---
id: ER-012
title: Map selected decoder operators and state to EasyRobot
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
# ER-012 — Map selected decoder operators and state to EasyRobot

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Inspect the exact selected architecture and weight/tokenizer formats. Map embedding/gather, projections, positional encoding, masking, norms, nonlinearities, KV cache, logits and deterministic sampling to real backend coverage.

Source: Eager MatMul/normalisation/softmax paths; graph stubs; no complete model-level decoder evidenced. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-003-operator-capabilities|ER-003]], [[ER-011-local-small-llm-reference|ER-011]]

Not in scope: No requirement to reimplement every possible Qwen variant or add vectors.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Every selected-model operation has an implementation/rejection/gap and dtype/layout/state contract.
- [ ] Required variants such as positional encoding, norm, head layout and cache update derive from the selected model, not a generic transformer guess.
- [ ] A tiny fixed-weight attention/decode fixture and intermediate-comparison plan precede a full checkpoint import.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

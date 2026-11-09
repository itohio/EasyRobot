---
id: ER-013
title: Prove a tiny CPU attention and KV graph fixture
type: ticket
status: blocked
kind: implementation
owner: unassigned-personal-profile
created: 2026-11-08
workstream: W9
milestone: M-f
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-013 — Prove a tiny CPU attention and KV graph fixture

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Implement only missing primitives required by a small fixed-weight CPU block. Compare eager versus actual graph execution, including positional/mask/cache semantics and per-layer intermediates.

Source: Selected decoder map and existing eager/Gorgonia numerical paths. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-002-cpu-graph-semantics|ER-002]], [[ER-012-decoder-coverage|ER-012]]

Not in scope: No large checkpoint, quantised/NPU optimisation or training.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Hand-computable small fixtures and local reference intermediates agree within predeclared tolerance.
- [ ] Prefill versus incremental decode, causal masking, context boundaries and cache reset/isolation are exercised.
- [ ] Actual backend execution is visible; an external model call cannot masquerade as graph computation.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

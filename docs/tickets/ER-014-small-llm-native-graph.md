---
id: ER-014
title: Run a bounded small local decoder through EasyRobot graph
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
# ER-014 — Run a bounded small local decoder through EasyRobot graph

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Wire selected tokenizer/weights and required CPU graph blocks into bounded autoregressive decode. Manage KV/cache/input/output lifetimes explicitly; compare the same artifact against the external local reference.

Source: Validated tiny CPU block and selected local model manifest. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-004-device-quantisation-contract|ER-004]], [[ER-011-local-small-llm-reference|ER-011]], [[ER-013-attention-kv-cpu-fixture|ER-013]]

Not in scope: No weight learning, cloud teacher, robotics or NPU portability claim.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Fixed prompts match declared per-layer/logit tolerances and deterministic next-token selection, with documented precision differences.
- [ ] Context cap, cancellation boundary, cache reset and independent repeated sessions are exercised.
- [ ] One local end-to-end decode proves the real graph path; load/prefill/decode timings and peak memory are captured separately.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

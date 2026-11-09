---
id: ER-006
title: Implement one verified Xtensa kernel adapter
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
# ER-006 — Implement one verified Xtensa kernel adapter

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

After target selection, add only the selected kernel path with explicit unsupported/fallback and boundary ownership. Keep host CPU reference independent.

Source: ER-005 target decision; existing FP32 primitive and tensor contracts. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-004-device-quantisation-contract|ER-004]], [[ER-005-xtensa-feasibility|ER-005]]

Not in scope: No whole-model port or unmeasured SIMD rewrite.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Approved target produces matching fixtures within declared tolerance, including shape/size/alignment edge cases.
- [ ] CPU versus device end-to-end measurements include packing/transfer/setup/allocation, not only kernel time.
- [ ] Unsupported inputs take a visible declared path; no fabricated speedup or blanket target coverage.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

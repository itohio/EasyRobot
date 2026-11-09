---
id: ER-016
title: Prove graph tensor serialization and resume boundaries
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
# ER-016 — Prove graph tensor serialization and resume boundaries

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Round-trip a minimal tensor and structural callback graph in supported formats. Specify what operation registry is required; distinguish structural reload from VM/state checkpointing.

Source: x/marshaller/{graph,yaml,json,proto}/; generic callbacks; tensor view/ownership and backend graph state. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-001-reference-environment|ER-001]], [[ER-002-cpu-graph-semantics|ER-002]]

Not in scope: No promise to persist Go closures, native handles or Gorgonia VM internals.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Values/dtypes/shapes and supported metadata round-trip; unsupported executable closure/backend state is rejected or explicitly reconstructed.
- [ ] Non-contiguous views/offsets cannot silently serialize unrelated backing data or restore unsafe pool aliases.
- [ ] Unknown operations, malformed shape/dtype and unsupported graph-format dispatch fail visibly.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

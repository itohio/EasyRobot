---
id: ER-003
title: Map actual operator dtype and layout capabilities
type: ticket
status: backlog
kind: discovery
owner: easyrobot
created: 2026-11-08
workstream: W9
milestone: M-f
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-003 — Map actual operator dtype and layout capabilities

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Inventory only operations needed by the chosen CPU fixture, candidate accelerator kernel and later selected LLM. Distinguish interface declarations, implementations, explicit rejection and tested execution.

Source: x/math/tensor/types/; eager/; gorgonia/graph_tensor_stubs.go; gocv/stubs_*; marshaller/tflite/model.go. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-001-reference-environment|ER-001]]

Not in scope: No exhaustive speculative operator library.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] The matrix records dtype/rank/layout/destination/ownership limits and source symbols for each selected path.
- [ ] Future execution records numerical comparison or explicit unsupported outcome, including graph Conv2D and rank-sensitive softmax where relevant.
- [ ] Fallback/copy paths are visible; no backend is called supported because its method name exists.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

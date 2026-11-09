---
id: ER-004
title: Define device buffer transfer and quantisation ownership
type: ticket
status: blocked
kind: design
owner: unassigned-personal-profile
created: 2026-11-08
workstream: W9
milestone: M-f
tags: [easyrobot, ticket]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-004 — Define device buffer transfer and quantisation ownership

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

Specify host/device tensor ownership, lifetime, alignment, layout, scale/zero-point/per-channel and packing metadata, transfer completion, supported precision and explicit fallback. Prefer existing contracts where sufficient.

Source: x/math/tensor/types/core.go and dtype.go; eager slice views; GoCV ownership; TFLite copies. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-002-cpu-graph-semantics|ER-002]], [[ER-003-operator-capabilities|ER-003]]

Not in scope: No premature cross-vendor backend framework or claim of zero-copy.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] A small CPU fixture demonstrates declared ownership/view/copy and conversion semantics before an accelerator is attached.
- [ ] Quantisation/dequantisation has numerical fixtures, or is explicitly unsupported for a path; integer tags are not treated as a quantisation format.
- [ ] The contract exposes transfer/allocation/fallback accounting and cancellation limitations.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

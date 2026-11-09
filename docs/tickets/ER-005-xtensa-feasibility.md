---
id: ER-005
title: Investigate the exact Xtensa acceleration target
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
# ER-005 — Investigate the exact Xtensa acceleration target

[[tickets/index|Index]] · [[source-inventory|Source inventory]] · [[roadmap|Roadmap]]

## Goal and evidence

UNVERIFIED hypothesis: an Xtensa SIMD/DSP path is missing or useful. Identify the exact chip/ISA, RAM, compiler and callable library/firmware ABI. Select one bounded kernel against the CPU fixture.

Source: DESIGN.md embedded claims; x/devices/ TinyGo files; cmd/elrs-rx/; amd64-only assembly experiment. All source statements are pinned above; runtime support and accelerator/LLM feasibility remain **UNVERIFIED** until recorded below.

## Dependencies and scope

Dependencies: [[ER-001-reference-environment|ER-001]], [[ER-003-operator-capabilities|ER-003]]

Not in scope: No claim that all ESP32 devices are Xtensa, or that TinyGo bus support implies tensor acceleration.

Execution is not authorised by this ticket. No installation, model download, test, hardware or implementation has happened in this planning assignment. Resolve named missing prerequisites; do not replace local execution with cloud inference.

## Acceptance evidence required later

- [ ] Target manifest names exact hardware/toolchain/library licences and operator/layout constraints.
- [ ] Evidence separates static feasibility from any authorised later device run; absent access is a named blocker.
- [ ] A go/no-go proposal states transfer/packing/FFI costs and numerical/performance acceptance limits before implementation.

## Effort uncertainty and revision trigger

Unestimated until the named platform/operator/artifact prerequisites are evidenced. Revisit scope when a dependency, precision/layout constraint, access limit or measured resource budget changes. Do not silently widen to a complete backend/model port.

## Implementation evidence

None. Planning ticket only. Future entries must name commands/scenarios, pinned artefacts, actual outcomes and limitations; test sources or proposed procedures are not passing results.

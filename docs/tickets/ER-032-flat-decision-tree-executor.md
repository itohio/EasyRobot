---
id: ER-032
title: "Plain-data decision tree and flat executor"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W6
milestone: M-d
tags: [easyrobot, ticket, w6, decision-trees]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-032 — Plain-data decision tree and flat executor

[[tickets/index|Ticket index]] · [[roadmap]] · [[todo]]

## Goal and evidence

M-d stage 1 tree core ([[W6-decision-trees]], [[2026-11-09-on-device-learning]] §3.2). Today trees are closure-based (`vector_decision_tree.go`); there is no plain-data tree and no MCU executor.

- Plain-data tree: ≤ 1,023 nodes × 8 B (feature u8, threshold i16/f32, children u16).
- Flat-array executor: no closures, no reflection, zero allocations, TinyGo-friendly.
- Serialisation to a flash blob (versioned header) and back.

## Dependencies and scope

Dependencies: After [[ER-028-graph-fixes|ER-028]]'s `Balance` fix. Consumed by [[ER-033-cart-trainer-episode-logger|ER-033]]. No M-b dependency (O9).

Not in scope: No trainer (ER-033), no on-device learning (Hoeffding, stage 3).

Source/test edits are approved; scheduling is decided separately. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] The flat executor gives the same decisions as `VectorDecisionTree` on the same splits over a random input set.
- [ ] `testing.AllocsPerRun` = 0 for one decision.
- [ ] Blob round trip is bit-exact; a corrupted header is rejected.

## Effort uncertainty and revision trigger

Size guess: part of M-d stage 1 (**M** for the stage); this ticket ≈ **S**. Sizes are rough guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

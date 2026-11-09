---
id: ER-033
title: "CART trainer and episode logger"
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
# ER-033 — CART trainer and episode logger

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

M-d stage 1–2 learning path (accepted ER-R2 decision 10: data-driven trees learned offline from driving logs; [[2026-11-09-on-device-learning]] §3.2–3.3):

- Host CART trainer (Gini/entropy, depth ≤ 8–10, min-samples, cost-complexity pruning) emitting [[ER-032-flat-decision-tree-executor|ER-032]]'s plain-data tree.
- Episode logger format and host reader: features (16–32 per frame), operator action, outcome, at 10–20 Hz; append-only, length-prefixed.

## Dependencies and scope

Dependencies: [[ER-032-flat-decision-tree-executor|ER-032]]. The ESP32 logger run and driving sessions need hardware and owner approval (stage 2).

Not in scope: No ESP32 camera capture driver, no Hoeffding tree.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Trainer matches a small public dataset's known accuracy within 1 point (golden test, dataset vendored offline).
- [ ] A tree trained on a synthetic obstacle set classifies a held-out set above a fixed-rule baseline (M-d host checkpoint, [[roadmap#8. Next|roadmap §8]] step 5).
- [ ] Logger write → read round trip is exact; a truncated tail record is detected.

## Effort uncertainty and revision trigger

Product's guess: part of M-d stage 1 (**M**). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

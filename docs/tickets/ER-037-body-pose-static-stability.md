---
id: ER-037
title: "Body pose integration and static stability"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W3
milestone: M-a
tags: [easyrobot, ticket, w3, gait]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-037 — Body pose integration and static stability

[[tickets/index|Ticket index]] · [[roadmap]] · [[todo]]

## Goal and evidence

M-a stage 3 ([[roadmap#M-a — Hexapod / quadruped walking (W3)|roadmap M-a]]): quadruped walk, then trot, with a stability margin. From IK review §6 (hexapod/quadruped gap analysis):

- **Support polygon / stability margin / CoM**: none exist (DESIGN claim only).
- **gait/rigidbody `Advance` and `EstimateFromEndpoints` return `ErrNotImplemented`** (E20) — `x/math/control/motion/gait/rigidbody/planner.go:60-77` ([[gait-rigidbody-planner]]).
- Ideation: stability without an IMU ([[2026-11-09-gaits-w3]] §4).

## Dependencies and scope

Dependencies: [[ER-018-leg-helpers-gait-scheduler|ER-018]] (scheduler, footholds, `Body`).

Not in scope: No IMU fusion, no terrain adaptation.

Source/test edits are approved; scheduling is separate. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] **Static stability** — failing test first: CoM projection inside the stance polygon with margin m; wave gait margin > tripod margin. Fix: support polygon + margin helper.
- [ ] **gait/rigidbody** — failing test first: `Advance` integrates a constant twist into the analytic pose; `EstimateFromEndpoints` recovers a known transform. Fix: quaternion integration, Kabsch.
- [ ] A host quadruped walk keeps static margin > 0 throughout the cycle.

## Effort uncertainty and revision trigger

Size guess: M-a stage 3 **M**. Sizes are rough guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

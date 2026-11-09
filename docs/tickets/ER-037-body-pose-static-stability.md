---
id: ER-037
title: "Body pose integration and static stability"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W3
milestone: M-a
maps: [T-IK-11, T-IK-15]
tags: [easyrobot, ticket, w3, gait]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-037 — Body pose integration and static stability

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

M-a stage 3 ([[roadmap#M-a — Hexapod / quadruped walking (W3)|roadmap M-a]]): quadruped walk, then trot, with a stability margin. From IK review §6 (evidence: `~/src/docs/evidence/ER-R2-ik-review.md#6. Hexapod/quadruped gap analysis`, outside the repo):

- **Support polygon / stability margin / CoM**: none exist (DESIGN claim only).
- **gait/rigidbody `Advance` and `EstimateFromEndpoints` return `ErrNotImplemented`** (E20) — `x/math/control/motion/gait/rigidbody/planner.go:60-77` ([[gait-rigidbody-planner]]).
- Ideation: stability without an IMU ([[2026-11-09-gaits-w3]] §4).

## Dependencies and scope

Dependencies: [[ER-018-leg-helpers-gait-scheduler|ER-018]] (scheduler, footholds, `Body`).

Not in scope: No IMU fusion, no terrain adaptation.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] **T-IK-11** — failing test first: CoM projection inside the stance polygon with margin m; wave gait margin > tripod margin. Fix: support polygon + margin helper.
- [ ] **T-IK-15** — failing test first: `Advance` integrates a constant twist into the analytic pose; `EstimateFromEndpoints` recovers a known transform. Fix: quaternion integration, Kabsch.
- [ ] A host quadruped walk keeps static margin > 0 throughout the cycle.

## Effort uncertainty and revision trigger

Product's guess: M-a stage 3 **M**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

---
id: ER-039
title: "Manipulator integration: units and helpers in drivers/cmd manipulator"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W3
milestone: none
maps: [T-IK-14]
tags: [easyrobot, ticket, w3, manipulator]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-039 — Manipulator integration: units and helpers in drivers/cmd manipulator

[[tickets/index|Ticket index]] · [[roadmap]] · [[ER-R2]] · [[todo]]

## Goal and evidence

No demo needs this; it keeps the manipulator path from being silently wrong. From [[ER-R2-ik-review#4. Error list (all reproducible with the probe)|IK review §4]]:

- Host CLI sends planar limits in degrees and lengths in mm; firmware IK is a placeholder (E11) — `cmd/manipulator/main.go:346-348`; `drivers/manipulator/kinematics.planar.go:44-47`, `kinematics.dh.go:44-57`.
- Firmware maps IK radians to servo degrees with no zero offset or sign (E12) — `drivers/manipulator/main.go:333`.
- Firmware ignores `config.Motion`, always `New1D(100,100,100)` (E13) — `drivers/manipulator/main.go:351-356`.

## Dependencies and scope

Dependencies: [[ER-018-leg-helpers-gait-scheduler|ER-018]] (the `chain` helpers, T-IK-07) and [[ER-017-dh-planar-ik-robustness|ER-017]].

Not in scope: No board run; the committed ELF (E22) waits on held decision #13.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] **T-IK-14** — failing test first: host unit test of the mapping function: `cmd` config → `Chain` (rad, m) → servo degrees. Fix: use T-IK-07 in `drivers/manipulator`; honour `config.Motion`.

## Effort uncertainty and revision trigger

Product's guess: none given; **S–M** (inferred). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

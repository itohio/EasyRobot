---
id: ER-036
title: "KE-4: Abugaga find (abg.svc.find, roster_find, abu find)"
type: ticket
status: backlog
kind: implementation
owner: babyai
workstream: W10
milestone: M-g
tags: [easyrobot, ticket, w10, knowledge-engine, abugaga]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-036 — KE-4: Abugaga find (abg.svc.find, roster_find, abu find)

[[tickets/index|Ticket index]] · [[roadmap]] · [[ER-R2]] · [[todo]]

## Goal and evidence

Roster slice **KE-4**: wire the engine into Abugaga — `abg.svc.find` in `abu serve`, `roster_find`, `abu find` ([[knowledge-engine]], design [[2026-11-09-knowledge-engine]] §6.3). Abugaga code, so owned by babyai (after ABU-059); tracked here so the EasyRobot side (the `x/knowledge` API it calls) has an id.

## Dependencies and scope

Dependencies: KE-1. First after KE-1 per the roster decision, so real use drives tuning. Known unknown: whether Abugaga builds against `x/knowledge` with `GOPROXY=off`.

Not in scope: No EasyRobot-side changes beyond what the API needs; Abugaga changes follow Abugaga's own ticket rules.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] A peer finds `keys.go` from seed question 1 in one call ([[roadmap#M-g — Knowledge engine serving Abugaga (W10)|roadmap M-g]] acceptance).
- [ ] Abugaga builds offline against the nested module.

## Effort uncertainty and revision trigger

Product's guess: none given (Abugaga ticket). Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

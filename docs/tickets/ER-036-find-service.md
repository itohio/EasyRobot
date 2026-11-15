---
id: ER-036
title: "Knowledge engine: expose find to a consuming service"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W10
milestone: M-g
tags: [easyrobot, ticket, w10, knowledge-engine]
created: 2026-11-09
updated: 2026-11-15
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-036 — Knowledge engine: expose find to a consuming service

[[tickets/index|Ticket index]] · [[roadmap]] · [[todo]]

## Goal and evidence

Make the engine callable from a consuming service ([[knowledge-engine]], design [[2026-11-09-knowledge-engine]] §6.3): the service opens an index, keeps it in sync with `ingest.Sync`, and answers find requests (query text, k, filter) with `retrieve.Find` results — paths, line spans, snippets and reasons — over its request/reply transport and a CLI. The service code lives outside this repository; this ticket covers the EasyRobot side, the `x/knowledge` API it calls, so that side has an id.

## Dependencies and scope

Dependencies: the first engine slice (implemented). First of the follow-up slices, so real queries drive tuning. Known unknown: whether an external module builds against the nested `x/knowledge` module with `GOPROXY=off`.

Not in scope: No EasyRobot-side changes beyond what the API needs; the consuming service follows its own ticket rules.

Source/test edits are approved; scheduling is decided separately. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] A user of the consuming service finds the answer file of design seed question 1 in one call ([[roadmap]] milestone M-g acceptance).
- [ ] The consuming service builds offline against the nested module.

## Effort uncertainty and revision trigger

Size guess: none given (the service work is outside this repository). Sizes are rough guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

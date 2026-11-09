---
id: ER-035
title: "KE-3: py/c/ts/proto chunkers"
type: ticket
status: backlog
kind: implementation
owner: easyrobot
workstream: W10
milestone: M-g
tags: [easyrobot, ticket, w10, knowledge-engine]
created: 2026-11-09
updated: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# ER-035 — KE-3: py/c/ts/proto chunkers

[[tickets/index|Ticket index]] · [[roadmap]] · ER-R2 (evidence: `~/src/docs/evidence/ER-R2.md`, outside the repo) · [[todo]]

## Goal and evidence

Roster slice **KE-3** ([[knowledge-engine]], design [[2026-11-09-knowledge-engine]] §9): text chunkers for Python, C, TypeScript and proto, beside KE-1's Markdown and Go chunkers (symbols and imports where the language allows without CGO).

## Dependencies and scope

Dependencies: KE-1. Runs with [[ER-034-ke2-watcher-crash-test|ER-034]], after [[ER-036-ke4-abugaga-find|ER-036]] per the roster order.

Not in scope: No tree-sitter/CGO parsers.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Golden chunk tests per language on small vendored fixtures (chunk boundaries, symbol names, line spans).
- [ ] The bench harness recall@k does not regress with the new chunkers enabled.

## Effort uncertainty and revision trigger

Product's guess: **S**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

---
id: ER-035
title: "Knowledge engine: py/c/ts/proto chunkers"
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
# ER-035 — Knowledge engine: py/c/ts/proto chunkers

[[tickets/index|Ticket index]] · [[roadmap]] · [[todo]]

## Goal and evidence

Text chunkers for Python, C, TypeScript and proto ([[knowledge-engine]], design [[2026-11-09-knowledge-engine]] §9), beside the existing Markdown and Go chunkers (symbols and imports where the language allows without CGO). Today files in these languages are not indexed at all, so questions whose answer is a `.py`, `.c`, `.ts` or `.proto` file cannot be found.

## Dependencies and scope

Dependencies: the first engine slice (implemented). Runs with [[ER-034-watcher-crash-test|ER-034]], after [[ER-036-find-service|ER-036]] in that order.

Not in scope: No tree-sitter/CGO parsers.

Source/test edits are approved; scheduling is separate. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Golden chunk tests per language on small vendored fixtures (chunk boundaries, symbol names, line spans).
- [ ] The bench harness recall@k does not regress with the new chunkers enabled.

## Effort uncertainty and revision trigger

Size guess: **S**. Sizes are rough guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

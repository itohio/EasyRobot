---
id: ER-034
title: "KE-2: fsnotify watcher, crash-mid-ingest test, missing-target index"
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
# ER-034 — KE-2: fsnotify watcher, crash-mid-ingest test, missing-target index

[[tickets/index|Ticket index]] · [[roadmap]] · [[ER-R2]] · [[todo]]

## Goal and evidence

Roster slice **KE-2** of the knowledge engine ([[knowledge-engine]], design [[2026-11-09-knowledge-engine]] §9; [[roadmap#M-g — Knowledge engine serving Abugaga (W10)|roadmap M-g]]):

- pure-Go SQLite store (FTS5, edges, catalog, embedding cache) if not already in KE-1;
- incremental ingest with moves and deletions, driven by an fsnotify watcher;
- crash-mid-ingest test: kill, re-sync, store consistent;
- an index of missing link targets.

## Dependencies and scope

Dependencies: KE-1 (roster task; done/in review, [[KE-1]]). Order after KE-1 per roster decision: KE-4 ([[ER-036-ke4-abugaga-find|ER-036]]) first, then KE-2/KE-3.

Not in scope: `x/knowledge/**` is its own module; no root `go.mod` changes.

Source/test edits are approved by the owner's autonomy instruction (roster `decisions.md`); scheduling is Leader's call. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Incremental and rename timings recorded on the vault fixture.
- [ ] Crash-mid-ingest test: process killed mid-batch, restart re-syncs, every query result equals a fresh full ingest.
- [ ] A query lists links whose targets do not exist.

## Effort uncertainty and revision trigger

Product's guess: **M**. Sizes are product's guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

---
id: ER-034
title: "Knowledge engine: fsnotify watcher, crash-mid-ingest test, missing-target index"
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
# ER-034 — Knowledge engine: fsnotify watcher, crash-mid-ingest test, missing-target index

[[tickets/index|Ticket index]] · [[roadmap]] · [[todo]]

## Goal and evidence

Second slice of the knowledge engine ([[knowledge-engine]], design [[2026-11-09-knowledge-engine]] §9; [[roadmap]] milestone M-g):

- pure-Go SQLite store (FTS5, edges, catalog, embedding cache), where the first slice left gaps;
- incremental ingest with moves and deletions, driven by an fsnotify watcher;
- crash-mid-ingest test: kill, re-sync, store consistent;
- an index of missing link targets, so re-resolution after an add/move/delete touches only the documents that link to the changed path instead of every document with a missing or fuzzy link.

## Dependencies and scope

Dependencies: the first engine slice (ingest, stores, retrieval, bench; implemented). Order: [[ER-036-find-service|ER-036]] first, then this ticket and [[ER-035-chunkers|ER-035]].

Not in scope: `x/knowledge/**` is its own module; no root `go.mod` changes.

Source/test edits are approved; scheduling is separate. Every fix starts with a failing test. No hardware, flashing, TinyGo or download is authorised by this ticket.

## Acceptance evidence required later

- [ ] Incremental and rename timings recorded on the vault fixture.
- [ ] Crash-mid-ingest test: process killed mid-batch, restart re-syncs, every query result equals a fresh full ingest.
- [ ] A query lists links whose targets do not exist.

## Effort uncertainty and revision trigger

Size guess: **M**. Sizes are rough guesses ([[roadmap#1. Milestones at a glance|roadmap §1]]), not estimates from velocity. Revisit when a dependency or owner input changes the scope.

## Implementation evidence

None. Planning ticket only.

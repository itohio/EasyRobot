---
type: workstream
status: active
tags: [easyrobot, workstream, docs]
workstream: W1
---
# W1 — Knowledge vault

**Goal:** Obsidian-style docs for EasyRobot: architecture, a catalogue of modules and algorithms, each algorithm reviewed for completeness, with TODOs and objectives inferred from them.

## What exists

- Vault root `EasyRobot/docs/` with a pre-existing `.obsidian/` config (wikilinks, `templates/` folder, properties visible). Home map: [[index]], with the ranked "Next" list at the top.
- [[conventions]] (one source of truth, generated blocks, link style, TODO grammar with ticket tags, "After a fix") and `templates/{algorithm,module,workstream}.md`.
- Generator and checker: `docs/tools/vault` (Go), run as `make docs` and `make docs-check`. It writes module algorithm tables, the index status/module/ticket blocks and [[todo]], and lists stale notes.
- [[overview]], [[packages]], [[data-flow]], [[dndm-and-protobuf]], [[build-targets]], plus the earlier tensor/graph [[source-inventory|source inventory]].
- Module notes (maps of content) and one note per algorithm under `algorithms/`, each with status, test result and a completeness review.
- Workstreams W1–W10, the generated [[todo]] (by ticket), one ticket file per ER id ([[tickets/index|ticket index]], ER-001…ER-039), planning ([[roadmap]], [[known-unknown|known/unknown]]) and dated evidence (`evidence/`).

## Gaps

- Statuses describe `abb5c9c` unless a note's `source_revision` says otherwise; `make docs-check` lists notes whose cited code has changed since.
- Vision, TFLite, V4L2 and TinyGo paths are judged by reading only (no OpenCV, no TinyGo, no boards).
- Repo-side docs (README, DESIGN, per-package SPEC/PLAN files) are not reconciled with the vault; they stay untrusted claims.

## TODOs

Algorithm defects are tracked in the algorithm and module notes and rolled up by ticket in [[todo]]; this list holds workstream-level items only.

- [ ] Re-run statuses after each implementation round and update `test_result`/`evidence` fields (procedure in [[conventions#After a fix]]) #p2

## Inferred objectives (proposals)

- **Proposal:** keep one note per algorithm as the unit of review; implementation tickets link back to the note and flip its status only with a recorded run.
- **Proposal:** fold the repo's scattered SPEC/PLAN/ANALYSIS markdown into the vault over time (move, then delete in the repo), so there is one source of design truth.

## Research questions for ideation

- None for W1.

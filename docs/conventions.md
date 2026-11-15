---
type: meta
status: active
tags: [easyrobot, meta]
vault_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# Vault conventions

Obsidian vault rooted at `EasyRobot/docs/`. Open the folder in Obsidian; the home map is [[index]].

## One source of truth

- **Status and TODOs live only in algorithm notes.** Module tables, index counts, open-ticket lists and [[todo]] are generated from them. Never hand-edit a generated block.
- **Generated blocks** sit between `<!-- vault:NAME h=… -->` and `<!-- /vault:NAME -->` markers: `algorithms` in module notes; `status`, `modules` and `tickets` in [[index]]. [[todo]] is generated whole. The `h=` hash lets the tool notice text typed inside a block: `make docs-check` reports it as hand-edited and `make docs` refuses to overwrite it (move the text out of the block, or rerun with `go run ./docs/tools/vault -force gen`).
- **Regenerate:** `make docs`. **Check:** `make docs-check` (broken links, frontmatter and vocabulary rules, untagged p1 TODOs, duplicated TODOs, out-of-date generated blocks). The tool is `docs/tools/vault` (Go, run from the repo root). Generated files depend only on the notes, never on git or working-tree state, so the same notes always generate the same files.
- Module, workstream and architecture notes describe and link; they carry a TODO line only when it belongs to no algorithm note (and never repeat one that does).

## Links

- **One style: wikilinks.** `[[note]]`, `[[note|label]]`, `[[note#Heading text|label]]`. Use a path where a name is ambiguous: `[[tickets/index|ticket index]]` (the root [[index]] is the home map).
- Note names are unique across the vault, apart from the two `index` notes.
- Code that looks like a link (`m[[2,0,1]]`) goes in backticks.
- Links out of the vault (repo files, URLs) use Markdown links.

## Folders

| Folder | Holds | `type` |
|---|---|---|
| `architecture/` | layers, packages, data flow, dndm/protobuf use, build targets, source inventory | `architecture` |
| `modules/` | one note per module or package family (map of content for its algorithms) | `module` |
| `algorithms/` | one note per algorithm | `algorithm` |
| `workstreams/` | W1–W10 workstreams | `workstream` |
| `tickets/` | ER-xxx tickets, one file each, plus [[tickets/index\|the ticket index]] | `ticket` |
| `planning/` | roadmap, known/unknown map | `planning` |
| — | evidence (run records, task reports, images) is **not committed**: it lives outside the repo, and the in-repo evidence folder is gitignored. Cite it as plain text (`evidence: <file>`), never as a wikilink | — |
| `research/` | ideation research notes (owned by ideation) | `research` |
| `templates/` | Obsidian note templates | `meta` |

## Frontmatter

Every note has `type`, `status` and `tags`. Algorithm notes add:

```yaml
type: algorithm
status: tested             # vocabulary below
tags: [easyrobot, kinematics]
workstream: W3             # W1..W10, or a list
module: "[[kinematics]]"
summary: "FK positions exact; quaternion inverted"   # ≤ 100 chars, shown in the module table
source: [x/math/control/kinematics/joints/dh/denavithartenberg.go]
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d   # optional; defaults to vault_revision above
tests: [x/math/control/kinematics/joints/dh/denavithartenberg_test.go]
test_result: pass          # pass | fail | build-fail | none | not-run | flaky
complete: yes              # yes | no | unknown
correct: unknown           # yes | no | unknown (yes only with a run that checks values)
evidence: "go test -count=1 ./x/math/... → ok (2026-11-09)"
```

Module notes add `group:` (the index section). Ticket notes add `id`, `kind`, `owner`, `workstream` and `milestone`.

## Status vocabulary (algorithms)

| Status | Meaning |
|---|---|
| `tested` | implementation complete for its stated scope **and** tests that check values pass in a recorded run |
| `implemented` | code complete for its stated scope and not known wrong; tests absent, trivial or failing |
| `partial` | some paths missing, `panic("not implemented")`, TODO-gated, or known wrong (`correct: no` or `complete: no` means `partial`) |
| `stub` | placeholder or signature only |
| `unverified` | cannot be judged without hardware, native libraries or data not available offline |

A failing test never yields `tested`. Never mark anything working without a run or test.

## TODO lines

One line each, in the algorithm note's `## TODOs` section:

`- [ ] <what> — <file:line or evidence> #p1 #ER-019`

- `#p1` wrong result, crash or build break; `#p2` missing test, or a feature a workstream needs; `#p3` cleanup, docs, ergonomics.
- **Every `#p1` names its ticket** (`#ER-0xx`) or says `#backlog`. p2/p3 may name a ticket.
- The anchor (`file.go:line`) identifies the defect; the same anchor appears in one note only.

## After a fix

When a change lands in the code:

1. In the algorithm note, tick the TODO (`- [x]`, or delete it) and update `status`, `test_result`, `complete`, `correct` and `evidence` (command, date, outcome). Set `source_revision` to the commit that the note now describes.
2. If the fix closes a ticket, set the ticket's `status: done` and fill its evidence section.
3. Run `make docs` to regenerate tables, counts and [[todo]], then `make docs-check`. Commit the code, the notes and the regenerated files together.

**Stale notes** are notes whose `source` files changed since their `source_revision` (or `vault_revision`). They are listed on demand by `make docs-stale` and as warnings by `make docs-check`; they are never written into generated files. A stale note is a prompt to re-check it, not an error.

## Inferred objectives

Always labelled **proposal**; never presented as the owner's decision.

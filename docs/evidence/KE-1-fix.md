---
type: evidence
status: review
tags: [easyrobot, evidence, knowledge-engine]
task: KE-1-fix
objective: O-005
date: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# KE-1-fix — review fixes before committing the knowledge engine

**Result:** everything the brief named is fixed, each with a test written first and seen failing:
- the four major findings: S1 stemming, I1 same-second rewrites, I2 interrupted-sync relinks, I3 secrets in the index;
- Leader's two decisions: memory scores exactly like SQLite FTS5 (S2); recency uses explicit dates only (R1);
- the memory move crash (S3);
- verify's optional fix: a DocID collision between unscoped sources is now reported;
- the end-to-end floors (T1), which now fail when tuning is removed (shown by mutation);
- the cheap minors: S4, S6, I4, I6, I7, R2, R4, T2.

Gates are green: gofmt, `go vet` (also with `-tags knowledge_bench`), and `go test -race -count=1 ./...` on all 10 packages. The benchmark is regenerated. **No commits yet:** the commit split (§6) waits for review's recheck and Leader's go.

Inputs:
- review `R-KE-1` (roster folder `docs/review/R-KE-1.md`, outside the vault);
- verify `KE-1` (accepted, with one optional fix);
- brief `docs/tasks/KE-1-fix.md`.

## 1. Fixes (test first)

The fixes were written by three implementation agents (token+stores, ingest, retrieve+bench) under the lead. Raw before/after outputs are in `~/src/tmp/easyrobot/kefix/*/before*.txt` and `after.txt`.

| # | Fix | Test(s) | Before (failing) | After |
|---|---|---|---|---|
| **S1** (major) | `stem` repeats a one-step rule until nothing changes, so it is idempotent and singular/plural agree. Adds `-xes`→`-x`; keeps the ss/us/is guards | `TestStemPluralMatchesSingular`, `TestStemIdempotent` | `settings→[setting]` but `setting→[set]`; same for warnings/embeddings/mappings/readings/strings; `embedded→embed→emb` (not idempotent) | pass |
| **I1** (major) | mtime compared with `Equal` at full precision. Git's racy rule: a file whose mtime is within 2 s of the sync start gets an internal `_racy` prop and is re-hashed on later syncs until it settles | `TestSameSecondRewriteReindexed` (3 cases), `TestSameSecondSwapDetected`, `TestRacyMarkerSettles` (memory + sqlite) | `report {Unchanged:1}, want {Changed:1}`; search for the new word `[]`; a swap of two same-size files gave `Unchanged:2` | pass. The test helpers no longer move mtime by 3 s, so the old edit tests now cover same-second edits too |
| **I2** (major) | **Recovery rule:** before a sync's first add, move or delete commits, it writes an "incomplete sync" marker doc (`<Scope>/.ingest-pending`, or `.ingest-pending-<hash of root>` unscoped; Kind `ingest`, no chunks or edges). The marker is deleted only when the relink pass ends cleanly. A sync that finds the marker re-resolves every document's links and writes only docs whose links change. Documented in the package doc. No contract change | `TestRelinkSurvivesCancelAfterCommit` (add, delete, namesake add, move; memory + sqlite; a store wrapper cancels right after a commit) | `linker.md:[missing:later]` vs rebuild `[doc:later.md]`; delete case `[doc:gone.md]` vs `[missing:gone]` | pass: links equal a rebuild |
| **I3** (major, security) | Secret line spans are found on the **whole file before chunking** and blanked, keeping line numbers, hash and size. A PEM private key (RSA/EC/OPENSSH/ENCRYPTED/PGP block) is one span from BEGIN to END, or to end of file if END is missing. New formats: AWS secret key (`AWS_SECRET_ACCESS_KEY` / `aws_secret_access_key` + 40-char value), `ASIA…`, Stripe live keys, Google `AIza…`, Slack webhooks, credential URLs `scheme://user:pw@host`. Props, title and headings are still scrubbed | `TestPrivateKeyAcrossChunksNeverIndexed` (49-line key in a heredoc spanning chunks), `TestSecretFormatsNeverIndexed` (13 formats), `TestDropSecretMeta` | `private-key lines stored: 38/49`; `SecretsDropped = 3, want 13` | pass: no planted secret is in any chunk or search |
| **I7** | key=value and `sk-` rules need a value part that mixes letters and digits; `sk-` no longer matches inside hyphenated words; lines are dropped, not whole chunks | `TestSecretFalsePositivesStayIndexed` | `SecretsDropped = 4, want 0` (`token: 2026-11-09-knowledge-engine`, `password: docs/runbooks/rotation-2026`, …) | pass |
| **S2** (Leader) | Memory BM25 implements FTS5's `bm25()` exactly: idf = ln((N−n+0.5)/(n+0.5)) floored at 1e-6; tf weighted across columns (3/3/2/1); row length summed over columns | `storetest` `CommonTermFloor`; `storetest.Agree` via `sqlite.TestAgreesWithMemory` | memory `0.3567, want 1e-6`; `["alpha"] memory 1.025 vs sqlite 1.49e-06` | pass: 10 queries, 69 hits, same order, scores equal to 12 significant digits |
| **S3** | Memory `MoveDoc` computes and checks all re-keyed chunk IDs first, and errors with the store unchanged on a duplicate or a collision. SQLite does the same explicit check inside the transaction (clear error, rollback) | `storetest` `MoveDuplicateIDs` | memory: "MoveDoc with colliding re-keyed chunk ids succeeded" (then `DeleteDoc` panicked) | pass on both stores |
| **R1** (Leader) | Recency date = `updated`, then `created`, then a date in the file name. No mtime; undated docs get factor 1 and are not ranked | `TestRecencyIgnoresMTime`, `TestRecencyRanksAgainstCandidates` | `r1/a.md recency 1.3, want 1 (undated)` | pass |
| **R2** | `supersedes:` matches only targets in the same directory or top-level scope; strips `[[x\|label]]` | `TestSupersedesStaysInScope`, `TestSupersedesDemotes` | `same-scope target status 1, want 0.8`; label cases misranked | pass |
| **T1** | `bench/engine_test.go` per-type engine floors (R@1, R@8 at measured; MRR measured − 0.03), checked on both stores | mutation probes below | before: removing recency + supersede still passed `TestEngineOnFixture` | after: fails |
| verify optional | A file whose DocID belongs to another source (two unscoped roots, same relative path) is skipped and reported in `Report.Errors`; if the other file is gone, the doc is taken over | `TestScopelessDocIDCollisionReported` | `errors = []` | pass |
| **I6** | a mutex is held for the whole `Sync` | `TestConcurrentSyncsSerialised` | ≥ 30 `move …: document not found` errors | pass |
| **I4** | the linking file is excluded from base-name fallback candidates, so the link is tracked as missing | `TestMarkdownFallbackSkipsSelf` | `links = []` | pass |
| S4, S6, R4, T2 | `DeleteReAdd` and `Unicode` contract cases (they passed before the fix, so they add coverage only); `SetMaxIdleConns`; graph doc comment; behavioural instead of exact assertions in two retrieve tests | — | — | pass |

**Mutation probes** (scratch copies, `go test -run TestEngineOnFixture ./bench/`):

| Mutation | Result |
|---|---|
| Recency prior and supersede demotion off | **fails** (`recency R@1 0.250 < floor 0.750`, `MRR 0.583 < 0.845`) |
| Lexical score back to rank fusion | **fails** (overall vs grep; code, paraphrase, recency and cross-link floors) |
| Graph expansion off (β = 0) | **fails** (`cross-link R@8 0.800 < 1.000`; engine R@8 0.929 < grep 0.964) |

**Not done (minor, not in the brief):** S5 (digit→Upper camel split), I5 (alias expiry contract), I8 (symlinks bypass walk rules), I10, R3, R5–R9. They stay review findings for a later ticket.

## 2. Gates

```text
cd x/knowledge   (GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly)
gofmt -l .                                           → (nothing)
go vet ./... ; go vet -tags knowledge_bench ./bench/ → clean
go test -race -count=1 ./...                         → ok ×10 (bench, chunk/golang, chunk/md, ingest, internal/storeutil,
                                                        retrieve, store/memory, store/sqlite, token; +2 with no tests)
```

## 3. Benchmark regenerated

**Fixture (28 questions):** both stores now give identical results.

| Type | n | Engine R@1 | Engine R@8 | Engine MRR | Grep R@1 | Grep R@8 | Grep MRR |
|---|---|---|---|---|---|---|---|
| identifier | 7 | 1.000 | 1.000 | 1.000 | 0.714 | 1.000 | 0.857 |
| paraphrase | 6 | 0.667 | 0.833 | 0.750 | 0.500 | 1.000 | 0.639 |
| cross-link | 5 | 0.000 | 1.000 | 0.295 | 0.000 | 0.800 | 0.192 |
| recency | 4 | 0.750 | 1.000 | 0.875 | 0.000 | 1.000 | 0.438 |
| code | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **all** | 28 | **0.714** | **0.964** | **0.803** | 0.500 | 0.964 | 0.662 |

**Frozen copy of the workspace:**
- Built with `rsync` of all `*.md` and `*.go` under `~/src` at 03:11 EET into `~/src/tmp/easyrobot-kefix-frozen/`, excluding dot-dirs, `node_modules`, `.venv`, `_quarantine` and `_tmp`. `_tmp` is never ingested.
- 7,962 files; 7,899 are ingested after the size and benchmark exclusions.
- `.gitignore` files were not copied, so a few ignored files are included. It is not identical to a live run, but it is stable, so later runs can compare against it.
- Command: `KE_CORPUS=…/tmp/easyrobot-kefix-frozen go test -tags knowledge_bench -run Live -v ./bench/`, output saved to `tmp/easyrobot/kefix/live-frozen.txt`.

| Type | n | Grep R@1 | Grep R@8 | Grep MRR | Grep p95 | Engine (memory = sqlite) R@1 | R@8 | MRR | Memory p95 | SQLite p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| identifier | 7 | 0.000 | 0.429 | 0.104 | 111 ms | 0.714 | 0.857 | 0.786 | 17 ms | 82 ms |
| paraphrase | 11 | 0.091 | 0.455 | 0.200 | 164 ms | 0.545 | 0.818 | 0.624 | 33 ms | 186 ms |
| cross-link | 4 | 0.250 | 0.750 | 0.417 | 119 ms | 0.500 | 1.000 | 0.633 | 23 ms | 75 ms |
| recency | 2 | 0.000 | 0.000 | 0.000 | 121 ms | 0.000 | 0.500 | 0.083 | 6 ms | 48 ms |
| code | 6 | 0.167 | 0.833 | 0.347 | 142 ms | 0.833 | 1.000 | 0.917 | 20 ms | 86 ms |
| **all** | 30 | 0.100 | 0.533 | 0.223 | 156 ms | **0.600** | **0.867** | **0.686** | 29 ms | 113 ms |

**Ingest on the frozen copy:**

| Store | Full ingest | No-change resync | Secret spans dropped |
|---|---|---|---|
| Memory | 6.7 s (probe alone) | 76 ms | 103 |
| SQLite | 14.0–15.1 s | 88 ms | 103 |

- The benchmark run measured 19.7 s for memory, but that run holds the grep baseline's 70 MB file cache at the same time. The profile shows `madvise`/`memclr` pressure.
- Profile of the memory ingest: about 40 % of CPU is the secret-line regexes, a TODO (prefilter).

**What moved and why:**
- The engine's overall numbers on this corpus (0.600 / 0.867 / 0.686) are close to KE-1's live run (0.633 / 0.900 / 0.707 on memory). The corpora differ, so per-question shifts aren't attributable to one change.
- The stores now agree exactly, where KE-1 had memory 0.633 vs sqlite 0.600, which was review's S2.
- Grep scores lower here (0.533 R@8) than live (0.600), because the frozen copy has 2,300 more files.

## 4. Behaviour changes to know

- Memory scores changed completely: a term in more than half the chunks now scores ~1e-6 on both backends.
- Stemming conflates more: `string`→`str`, `embed`→`emb`, `based`→`bas`. That is the price of plural/singular agreement; a TODO is in [[knowledge-engine]].
- Freshly written files are re-hashed on each sync for 2 s after their mtime, then rewritten once to clear the marker.
- A sync that adds, moves or deletes makes two extra marker writes. During an interrupted state, the marker doc is visible to `ListDocs` (Kind `ingest`, no chunks).
- `SecretsDropped` counts secret spans (one key = 1), and only secret lines are removed now.
- mtime no longer affects ranking. A bare `supersedes` name no longer demotes files in other projects.

## 5. Docs updated

- [[KE-1]] has a post-fix section with the new numbers; the original tables are kept as history.
- [[knowledge-engine]]: numbers and two new TODOs (secret-scan cost, over-stemming).
- `make docs` and `make docs-check` were run (§7).

## 6. Commit plan (after review recheck and Leader's go; explicit pathspecs, local, personal identity)

| # | Message | Files |
|---|---|---|
| 1 | `chore(deps): mark malgo, go-audio and cobra as direct requirements [KE-1 housekeeping]` | `go.mod` |
| 2 | `chore: ignore Obsidian per-user layout` (Leader's change) | `.gitignore` |
| 3 | `feat(knowledge): data model, interfaces and tokenizer [KE-1]` | `x/knowledge/{go.mod,go.sum,knowledge.go}`, `x/knowledge/token/` |
| 4 | `feat(knowledge): memory and pure-Go SQLite (FTS5) stores with a shared contract suite [KE-1]` | `x/knowledge/store/`, `x/knowledge/internal/` |
| 5 | `feat(knowledge): Markdown and Go chunkers [KE-1]` | `x/knowledge/chunk/` |
| 6 | `feat(knowledge): incremental ingestion with crash-safe relinks and secret redaction [KE-1]` | `x/knowledge/ingest/` |
| 7 | `feat(knowledge): hybrid lexical + link-graph retrieval with spans and reasons [KE-1]` | `x/knowledge/retrieve/` |
| 8 | `test(knowledge): fixture and live benchmark against a ripgrep-style baseline [KE-1]` | `x/knowledge/bench/` |
| 9 | `feat(docs): vault tool — make docs regenerates tables and todo.md, make docs-check validates the vault [ER-docs-fix]` | `docs/tools/vault/*.go`, `Makefile` (docs targets and help lines). Gate: `go vet` and `go test -race ./docs/tools/vault/` (self-contained fixtures) |
| 10 | `docs: EasyRobot knowledge vault — notes, tickets ER-001…039, generated tables [ER-R2, ER-docs-fix]` | `docs/**` except `docs/tools/`, the KE-1 files and ideation's files; `README.md` (pointer line). The body says links into 11–12 dangle until they land |
| 11 | `docs(knowledge): KE-1 evidence and module note [KE-1]` | `docs/evidence/KE-1.md`, `docs/evidence/KE-1-fix.md`, `docs/modules/knowledge-engine.md` |
| 12 | `docs(research): ideation notes [KE-research and the W3/W4/W7/W8 research tasks]` (ideation's files) | `docs/research/ideation/**`. **Gate: `make docs-check` here, 0 errors** |

Each code commit 3–8 is re-gated cumulatively (build, vet, test) before committing.

**Docs order (Leader's question; review R-ER-docs-fix option a).** The vault (10), KE-1's docs (11) and ideation's notes (12) link to each other, so no order lets each pass `docs-check` alone. I propose landing 10–12 back to back, gating `docs-check` on 12, and saying so in commit 10's body. That keeps one commit per owner and task. The alternative (b) is one combined docs commit whose body lists each part. Run `make docs` before commit 10. Since the D1 fix, generated files no longer depend on the working tree, so a dirty tree can't leak into them.

## 6b. Added to this round: vault tool review fixes (R-ER-docs-fix)

Leader added review's vault-tool findings to this round. The VaultTool agent implemented them in `docs/tools/vault/**` plus a `docs-stale` Makefile target. Each fix has a test in `review_test.go`, seen failing first except the D8 guard. `go test -race ./docs/tools/vault/` passes; vet and gofmt are clean.

| # | Fix | Test |
|---|---|---|
| **D1** (major; decision a) | Generated files no longer read git: the stale count is gone from `index.md` `vault:status` and the stale section from [[todo]]. Staleness is reported only as `check` warnings and by `make docs-stale` (`vault stale`). A dirty cited file leaves gen output byte-identical and check passes with a warning | `TestGenIgnoresGitState` (before: `index.md changed after a cited file got dirty`) |
| D2 | Each `source_revision` is resolved on its own (`git rev-parse --verify REV^{commit}`). A bad one is an error on that note, and the others are still checked | `TestBadRevisionIsolated` |
| D3 | Opening markers carry `h=<hash>` of the generated content. Hand-edited blocks are reported by check, and gen skips them unless run with `-force` | `TestHandEditedBlock` |
| D4 | Each file is re-read before writing and skipped if it changed since load; writes go through temp file + rename | `TestWriteSkipsFileChangedSinceLoad` |
| D5 | Link problems in foreign notes (roadmap, research) are warnings | `TestForeignLinksAreWarnings` |
| D6 | gen exits 1 with `skipped <file>: no markers` | `TestGenSkipsNotesWithoutMarkers` |
| D7 | Name resolution is deterministic; an ambiguous bare name is an error (`[[index]]` still means the home map) | `TestAmbiguousLink` |
| D8 | Text outside markers survives byte for byte; unclosed marker; bad revision; dirty tree | the tests above plus `TestTextOutsideMarkersSurvives` |

One extra bug was fixed: `git status -z` records were whitespace-trimmed, with no visible effect.

On the real vault:
- `make docs` changed 39 of 39 generated files (adding the hashes); a second run changed 0.
- `make docs-check`: 0 errors, 4 stale warnings.
- `make docs-stale` lists the same 4 notes (`build-targets`, `dndm-and-protobuf`, `overview`, `packages`, caused by `Makefile`, `go.mod` and `README.md`).
- The index now reads `State: 56 tested · 29 implemented · 146 partial · 14 stub · 13 unverified (258 algorithms)`.

[[conventions]] now documents the block hashes, `-force` and `make docs-stale`.

## 6c. KE-1-fix2 (verify's required fix and docs fixes)

- **I3, AWS secret keys in JSON and CLI form.**
  - **Test:** three planted cases added to `TestSecretFormatsNeverIndexed`: `"SecretAccessKey": "…"`, `{"secret_access_key":"…"}` and `aws configure set aws_secret_access_key <key>`.
  - **Failing first:** `SecretsDropped = 13, want 16`; `awsjson.md`, `awsjsonlc.md` and `awscli.md`: secret stored and searchable.
  - **Fix:** `ingest/secret.go`. The pattern is case-insensitive, the `aws_` prefix is optional, a quote may follow the key name, and the separator may be `:=`, `:`, `=` or whitespace (CLI form).
  - **Gates:** `go test -race -count=1 ./ingest/` and `go test ./...` pass; vet and gofmt are clean.
- **Docs:**
  - [[ER-R2]] §2 now says 20 of 21 legacy notes got `type`/`tags`; `planning/roadmap.md` is product's, and product is adding its frontmatter.
  - [[W3-legged-ik]] gives the corrected warm-seed numbers: true solution ± 0.2 rad gives 100 %; the previous tick's answer gives 191/500.
  - The `R-KE-1` link in this report is now plain text.
  - Next item 5 on [[index]] already names ER-032/033.
  - `make docs` changed 0 files; `make docs-check` reports 0 errors and 4 stale warnings.

## 7. Network and boundaries

- No network contact.
- Writes: `EasyRobot/x/knowledge/**` (code and tests), `EasyRobot/docs/**` (this report, [[KE-1]], [[knowledge-engine]]), `~/src/tmp/easyrobot/kefix/`, `~/src/tmp/easyrobot-kefix-frozen/` (frozen benchmark corpus; private notes, never committed, delete when no longer needed).
- No root `go.mod`/`go.sum` changes. No commits.

## 8. Proposed mb learnings (inbox format)

```
- 2026-11-09 [code] [high] Incremental indexers: compare mtime at full precision and re-hash files whose mtime is within ~2 s of the scan (git's racy-file rule); second-rounded mtime + size misses same-second same-size edits forever — KE-1-fix (R-KE-1 I1)
- 2026-11-09 [code] [high] Redact secrets on the whole file before chunking (PEM BEGIN…END as one span), blanking lines so spans stay correct; per-chunk scrubbing let 38/49 key lines into FTS — KE-1-fix (R-KE-1 I3)
- 2026-11-09 [code] [medium] For crash-safe follow-up work after a commit (link re-resolution), write an "incomplete" marker before the first structural write and clear it only after the follow-up pass; a sync that finds it does the full pass — KE-1-fix (R-KE-1 I2)
- 2026-11-09 [ml] [medium] To make two lexical backends rank identically, port FTS5's exact bm25 (idf floor 1e-6, column-weighted tf, row length over all columns) rather than "BM25F-like"; verified identical to 12 digits — KE-1-fix (S2)
```

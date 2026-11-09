---
type: evidence
status: review
tags: [easyrobot, evidence, knowledge-engine]
task: KE-1
objective: O-005
date: 2026-11-09
source_revision: abb5c9cca7f756c632548d9c7be2436b46dc2e4d
---
# KE-1 — knowledge engine, first slice

**Result:** `x/knowledge` is a new, separate Go module (CGo-free; only dependency `modernc.org/sqlite`, taken from the module cache). It contains:
- interfaces and a native memory store;
- a pure-Go SQLite store (FTS5);
- Markdown and Go chunkers;
- incremental ingestion (moves, deletes, secret exclusion);
- hybrid lexical + link-graph retrieval returning files, line spans and reasons;
- a benchmark with a ripgrep-style baseline.

Gates are green: gofmt, `go vet`, and `go test -race` on 10 packages. Two code reviews found 13 issues; all 13 are fixed, each with a test.

On the live workspace (30 questions, 5,565 docs) the engine beats grep:

| Retriever | R@1 | R@8 | MRR | p95 |
|---|---|---|---|---|
| Engine, memory store | 0.633 | 0.900 | 0.707 | 14 ms |
| Engine, sqlite store | 0.600 | 0.900 | 0.678 | 79 ms |
| Grep baseline | 0.067 | 0.600 | 0.242 | 118 ms |

Module note: [[knowledge-engine]]. Design: [[2026-11-09-knowledge-engine]].

**Updated after [[KE-1-fix]] (2026-11-09):** review asked for four major fixes: idempotent stemming, full-precision change detection, crash-safe relinking and pre-chunk secret redaction. Leader also decided that the memory store scores exactly like SQLite FTS5 and that recency uses explicit dates only. After those changes, on a frozen copy of the workspace (7,899 docs) with the same 30 questions:

| Retriever | R@1 | R@8 | MRR | p95 |
|---|---|---|---|---|
| Engine, memory store | 0.600 | 0.867 | 0.686 | 29 ms |
| Engine, sqlite store | 0.600 | 0.867 | 0.686 | 113 ms |
| Grep baseline | 0.100 | 0.533 | 0.223 | 156 ms |

On the fixture the engine scores 0.714 / 0.964 / 0.803 on both stores; grep scores 0.500 / 0.964 / 0.662. The tables below are the original KE-1 run on the live workspace, kept as history.

## 1. Scope delivered vs brief

| Brief (KE-1.md) | Delivered |
|---|---|
| Own module `EasyRobot/x/knowledge` | yes: `github.com/itohio/EasyRobot/x/knowledge`, go 1.26.0; the root module is untouched |
| Go interfaces: vector store, graph store, embedder, chunker, ingestor, retriever | `knowledge.go`: Catalog, LexicalIndex, GraphStore, Writer, Store, VectorStore, Embedder, Chunker, Ingestor, Retriever. VectorStore and Embedder are declared but not implemented (no embeddings in this slice) |
| Native in-process store | `store/memory` (BM25F, adjacency, catalog) |
| Pure-Go SQLite store (FTS5) from the module cache | `store/sqlite`: FTS5 `bm25(3,3,2,1)` over pre-tokenized fields, an edges table, a catalog, the generation counter, WAL. Both stores pass the shared contract suite `store/storetest` |
| Markdown + Go ingestion with incremental re-read, rename by content, cascading deletes, secret exclusion | `chunk/md`, `chunk/golang`, `ingest`: stat→hash detection, move by hash (deterministic), delete cascade, re-resolution of affected links, `.gitignore` subset, secret file names, secret chunk/prop/title scrubbing |
| FTS + one-hop link-graph retrieval with line spans and reasons | `retrieve`: relative BM25 + exact id/symbol/file-name bonus, 1-hop links and 2-hop shared ids with hub damping, relative recency, status/supersedes priors, folder diversity, sections with spans/snippets/why, confidence and try-terms |
| Synthetic committed fixture + live-workspace benchmark (scores only) vs the ripgrep baseline | `bench`: 61-file fixture with 28 questions; 30 live questions (`-tags knowledge_bench`, `KE_CORPUS`); a Grep baseline modelled on peers' ripgrep use; tables print ids and numbers only |
| No embeddings | none |

## 2. Package map

| Package | Code / test lines | Author (agent) |
|---|---|---|
| `knowledge` (contract), `token` | 410 + 195 / 120 | lead |
| `chunk/md`, `chunk/golang` | 922 / 800 | KEChunkers |
| `store/memory`, `store/sqlite`, `store/storetest`, `internal/storeutil` | ~2,700 / ~600 | KEStores |
| `ingest` | ~1,350 / ~1,200 | KEIngest |
| `retrieve` | ~1,110 / ~900 | KERetrieve |
| `bench` (+ fixture) | ~650 / ~500 | KEBench; lead: `engine_test.go`, live comparison, grep exclusions and dot-dir skip, q15 |

The totals are 8.0 k lines of code and 4.1 k lines of tests.

## 3. Tests first (as recorded by each author)

- **token:** tests written first; the first run failed to build (`undefined: IsID`, `Count`). Later, the camelCase lower-case query case was caught first by `TestLowercaseQueryMatchesCamelCaseIndex` (`query "peerpermissions" term "peerpermission" not in index`), then fixed.
- **chunk/md, chunk/golang:** the first run failed to build (`undefined: Chunker/New`). The `//line` review fix was caught first by `TestLineDirectivesIgnored`: before, the chunks were `g.Parse [3 8]` and `After` was dropped.
- **store/\*:** the first run failed to build. The first real run then caught a bug in the suite itself (an idf fixture folder named `rare/`).
- **ingest:** 14 tests failed against a stub. Each review fix had its failing test recorded before the fix (§5).
- **retrieve:** 19 tests failed against a stub. The acronym fix was caught first by `TestAcronymIsNotPackageQualified`.
  - The tuning tests were written after the code, driven by the benchmark, so they were never seen failing: `TestLexicalScoresRelativeToTopHit`, the recency tests, `TestSupersedesDemotes`, `TestBareSymbolResolvesThroughPackage`, `TestNeighborsCapPerSeed` (rewritten) and `TestHubSeedCostsNeighborsLookups`.
- **bench:** the first run failed to build. The floors in `fixture_test.go` are measured values.

## 4. Commands and results (2026-11-09, go1.27.2 darwin/arm64)

All commands ran from `EasyRobot/x/knowledge` with `GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-mod=readonly`.

```text
gofmt -l .                                  → (nothing)
go vet ./... ; go vet -tags knowledge_bench ./bench/   → clean
go test -race -count=1 ./...                → ok ×10 packages (bench, chunk/golang, chunk/md, ingest, internal/storeutil,
                                               retrieve, store/memory, store/sqlite, token; knowledge and storetest have no tests)
go test -count=1 -run TestEngineOnFixture -v ./bench/
KE_CORPUS=~/src go test -tags knowledge_bench -run Live -v -timeout 30m ./bench/
```

### Fixture (28 questions, memory store; sqlite matches)

| Type | n | Engine R@1 | Engine R@8 | Engine MRR | Grep R@1 | Grep R@8 | Grep MRR |
|---|---|---|---|---|---|---|---|
| identifier | 7 | 1.000 | 1.000 | 1.000 | 0.714 | 1.000 | 0.857 |
| paraphrase | 6 | 0.667 | 0.833 | 0.750 | 0.500 | 1.000 | 0.639 |
| cross-link | 5 | 0.000 | 1.000 | 0.285 | 0.000 | 0.800 | 0.192 |
| recency | 4 | 0.750 | 1.000 | 0.875 | 0.000 | 1.000 | 0.438 |
| code | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **all** | 28 | **0.714** | **0.964** | **0.801** | 0.500 | 0.964 | 0.662 |

- **Before tuning,** the engine scored 0.500 / 0.929 / 0.631 and lost to grep. Rank fusion at chunk level flattened ranks, and the absolute recency prior never separated old notes from new ones.
- **Tuning changes,** all generic, none tied to a specific question:
  - BM25 scores relative to the top hit;
  - recency relative across candidates, stronger when the query uses recency words;
  - bare-symbol lookup through the packages of the top hits;
  - a base-name fallback for Markdown links to moved files (weight 0.5).
- **Still missed:** f12, a paraphrase sharing no vocabulary with its answer (needs vectors, KE-5). Cross-link R@1 is 0, because the asking note outranks its target.

### Live workspace (30 questions; corpus = `~/src` excluding the design note and the bench testdata, which quote the questions)

| Type | n | Grep R@1 | Grep R@8 | Grep MRR | Grep p95 | Memory R@1 | Memory R@8 | Memory MRR | Memory p95 | SQLite R@1 | SQLite R@8 | SQLite MRR | SQLite p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| identifier | 7 | 0.000 | 0.571 | 0.187 | 75 ms | 0.857 | 1.000 | 0.878 | 12 ms | 0.857 | 1.000 | 0.875 | 79 ms |
| paraphrase | 11 | 0.000 | 0.545 | 0.180 | 119 ms | 0.455 | 0.818 | 0.583 | 14 ms | 0.455 | 0.818 | 0.558 | 123 ms |
| cross-link | 4 | 0.250 | 0.750 | 0.425 | 89 ms | 0.500 | 1.000 | 0.625 | 12 ms | 0.500 | 1.000 | 0.613 | 45 ms |
| recency | 2 | 0.000 | 0.000 | 0.000 | 88 ms | 0.000 | 0.500 | 0.071 | 9 ms | 0.000 | 0.500 | 0.062 | 36 ms |
| code | 6 | 0.167 | 0.833 | 0.381 | 108 ms | 1.000 | 1.000 | 1.000 | 21 ms | 0.833 | 1.000 | 0.917 | 73 ms |
| **all** | 30 | 0.067 | 0.600 | 0.242 | 118 ms | **0.633** | **0.900** | **0.707** | 14 ms | **0.600** | **0.900** | **0.678** | 79 ms |

| Timing (live, 5,565 docs) | Memory | SQLite | Design target |
|---|---|---|---|
| Full ingest | 3.8 s | 9.7 s | < 60 s |
| Re-sync, nothing changed | 95 ms | 137 ms | — |
| Query p95 | 14 ms | 79 ms | < 150 ms |

Incremental ingest timings (probe `tmp/easyrobot/ke/lead-incr`, on a 448-file copy of EasyRobot docs + `x/knowledge`; target < 1 s):

| Change | Memory | SQLite |
|---|---|---|
| 1 file edited | 4 ms | 14 ms |
| 1 file renamed (Moved = 1) | 6 ms | 12 ms |
| 1 file deleted (22 linkers re-resolved) | 40 ms | 87 ms |

**Caveats:**
- Other agents were editing the workspace during the runs, so the live numbers drift between runs: the corpus held 6,819 docs at the first run and 5,565 at the last.
- The live question set is ideation's 20 seeds plus EasyRobot and recency additions, not real peer queries.
- q15 now also accepts the project README, because `.py` files are not indexed until KE-3.

### End-to-end smoke (EasyRobot only, sqlite, probe `tmp/easyrobot/ke/lead-live`)

Ingest of 1,616 files took 3.1 s.

| Query | Time | Top hit | Why |
|---|---|---|---|
| "why does the kalman filter freeze after the first update" | 28 ms | `EasyRobot/docs/algorithms/linear-kalman-filter.md:22-27` (Completeness review) | graph links from the EKF note, the module note and `todo.md` |
| "LD06 packet angle decoding" | 13 ms | `docs/algorithms/ld06-packet-decoder-and-scan-assembly.md:14-22` | — |

The Why lists are long on hub-linked notes; capping them is a TODO.

## 5. Code review findings and fixes

**Review 1** (reviewer agent; token, chunkers, stores, ingest): 8 issues, all confirmed by probes.

| # | Sev | Finding | Fix | Test |
|---|---|---|---|---|
| 1 | P1 | `Sync` deleted every other source's docs in a shared store | only docs the source owns (scope prefix, or Path under Root) can be missing | `TestSharedStoreScopedSources`, `TestSharedStoreScopelessSources` |
| 2 | P2 | an unreadable dir deleted the docs under it | failed paths keep their catalog docs; the error is reported | `TestUnreadableDirKeepsDocs` |
| 3 | P2 | secret filter missed `:=`, quoted JSON keys and token prefixes; frontmatter props and title were stored | broader patterns (value must contain a digit), token prefixes, props/title/heading scrubbing | `TestSecretPatternsExtended`, `TestSecretTitleDropped` |
| 4 | P2 | which file received a move depended on worker timing | move candidates are deferred and matched in path order | `TestMoveTargetDeterministic` (30 runs) |
| 5 | P2 | adding a better base-name candidate didn't re-resolve existing links | linkers of same-base-name docs are re-resolved | `TestAddedBetterCandidateMatchesRebuild`, `TestRealFileReplacesAliasForUnchangedLinker` |
| 6 | P2 | Go chunker followed `//line` directives (wrong spans, dropped decls) | `fset.PositionFor(p, false)` | `TestLineDirectivesIgnored` |
| 7 | P3 | `**/` gitignore prefix only matched at depth 0 | matches at any depth | `TestGitignoreDoubleStarAnyDepth` |
| 8 | P3 | file symlinks to outside the root were followed | the target must be inside the root; the secret-name check applies to the target | `TestFileSymlinks` |

**Review 2** (retrieve, bench): no P0/P1 issues, 5 findings.

| # | Sev | Finding | Fix |
|---|---|---|---|
| 1 | P2 | acronyms (NATS, GPT) were package-prefixed into exact hits and outranked the answer | prefix only mixed-case words; `TestAcronymIsNotPackageQualified` |
| 2 | P2 | live q15 expected a `.py` file nobody indexes | also accepts the project README |
| 3 | P3 | latency comparison was unfair to the engine (cold first query) | engines warmed before timing |
| 4 | P3 | two store calls per graph neighbour (369 GetDoc / 309 Degree) | cap per seed by edge weight before lookups: 229 / 169, sqlite 77 ms → 9 ms in the probe. Side effect: fixture f15 moved from rank 8 to 7 |
| 5 | P3 | grep baseline read hidden directories that rg skips | skips dot-dirs |

Both reviews checked and found sound:
- FTS MATCH quoting (no injection);
- filters applied before top-k;
- atomic writes and move re-keying;
- lock ordering and no goroutine leaks;
- link resolution confined to the root;
- metric formulas;
- privacy of printed tables.

## 6. Deviations and decisions taken

- **JSON question files, not YAML.** `gopkg.in/yaml.v3`'s test dependency (`gopkg.in/check.v1`) is not in the module cache, so `go mod tidy` fails offline.
- **Scope went beyond the design's KE-1 row:** the SQLite store, the Go chunker and incremental ingest were included because the brief asks for them. fsnotify, the py/c/ts chunkers and embeddings were not done.
- **The grep baseline is implemented in Go,** modelling ripgrep (distinct words, then match count, case-insensitive substring; hidden and excluded dirs skipped). This keeps it deterministic, with no subprocess. It does not honour `.gitignore`.
- **Ingest-only props** `_missing`, `_fuzzy`, `_moved_from` and `_stale` are stored in `Doc.Props`; callers should hide underscore keys.
- **`retrieve.WithNow` was removed,** because recency became relative.
- **Housekeeping:** the root `EasyRobot/go.mod` change (malgo, go-audio/audio, go-audio/wav and cobra moved to direct) is the tidy-correct marking. Per Leader it is kept, and it gets its own `chore(deps)` commit with the first gated round. `x/knowledge` does not depend on the root module.

## 7. Network and boundaries

- No network contact. All dependencies came from the module cache. The go.sum entries were seeded from verified sums in `fpv-arena/go.sum`, then checked by `go mod tidy` with `GOPROXY=off`.
- Writes went only to `EasyRobot/x/knowledge/**`, `EasyRobot/docs/**` and `~/src/tmp/easyrobot/ke/`.
- No commits.
- `git status`: ` M .gitignore` (Leader's change), ` M go.mod` (the housekeeping change), `?? docs/`, `?? x/knowledge/`.

## 8. Not verified / limitations

- Abugaga integration (KE-4) is not built.
- Benchmark numbers are from one machine under concurrent load.
- Real peer questions are not yet in the set (design "unknown knowns").
- Two scopeless sources in one store must not share relative paths, because their DocIDs would collide (documented).
- Re-resolution after an add scales with the number of linkers of same-named files (e.g. `README.md`).

## 9. Decisions for the owner

1. **Benchmark questions.** Replace half of the seed questions with 15–20 real "where is…" moments from peer transcripts.
   - **Recommend:** yes, before KE-2 tuning; Leader collects them.
2. **Next KE slice order.**
   - **Recommend:** KE-4 (Abugaga `abg.svc.find`) first, so peers start using it and real queries arrive. Then KE-3 (py/c/ts/proto chunkers) and KE-2's remaining items (fsnotify, crash test); then KE-5 embeddings.
3. **`_scratch` in the index at weight 0.5** (design decision 6, accepted). The live corpus showed scratch files from other agents' probes appearing and vanishing mid-ingest.
   - **Recommend:** keep it at 0.5, but exclude go-build temp dirs (`tmp/go-build*`), which caused transient read errors.

## 10. Proposed mb learnings (inbox format)

```
- 2026-11-09 [code] [high] Offline Go module setup: a new module can take deps purely from the module cache with GOPROXY=off if go.sum is seeded from another workspace module's verified sums; `go mod tidy` then fails only when a dependency's *test* deps aren't cached (gopkg.in/yaml.v3 → gopkg.in/check.v1) — KE-1
- 2026-11-09 [ml] [high] Hybrid lexical retrieval: reciprocal rank fusion with k=60 at chunk level plus "best + 0.3×second" favours files with several mediocre chunks; normalise BM25 by the top score instead (fixture MRR 0.63→0.78) — KE-1
- 2026-11-09 [ml] [medium] An absolute recency prior (half-life 30 days) is ~1.0 for every note older than a few months; rank dates relative to the candidate set and boost on recency words (fixture recency MRR 0.30→0.88) — KE-1
- 2026-11-09 [code] [high] go/token: use fset.PositionFor(p, false) for source spans; fset.Position follows //line directives (goyacc/cgo output) and silently remaps or drops declarations — KE-1 review
- 2026-11-09 [process] [medium] A dedicated reviewer agent after parallel implementers found 13 real bugs (incl. a P1 shared-store deletion) that green test suites missed — KE-1
```

## 11. Next action

Leader routes this report to verify (and review). After acceptance, I make two local commits: `chore(deps): mark malgo, go-audio and cobra as direct requirements` (root `go.mod`), then `feat(knowledge): …` (`x/knowledge` + `docs`). Next task per Leader's order: ER-017/018.

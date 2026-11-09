---
type: research
status: draft
owner: ideation
tags: [easyrobot, research, ideation, knowledge-engine, W10]
workstream: W10
created: 2026-11-09
updated: 2026-11-09
---

# Knowledge engine: ingestion and hybrid retrieval in EasyRobot, for Abugaga

**Brief:** `~/src/docs/tasks/KE-research.md` (objective O-005, workstream W10). easyrobot implements it as KE-1 and later tickets.

**Recommendations, not decisions.** No code was run. Commands were read-only: `find`, `grep`, `ls` of the Go module cache, and file reads. No network.

**Labels:**
- **observed:** I read the file or ran the read-only command;
- **reported:** another note or a scout's file read says so;
- **inferred:** my reasoning;
- `#verify`: a guess.

**Owner (verbatim, O-005):** "we might take some ideas for memory, but we'd have to figure out batching, graphing, retrieval stuff. probably EasyRobot can have these algorithms abstractions over vector and graph interfaces with connectors to neo4j and/or pgsql; and/or to native graph implementation. … we'd need ingestion and retrieval in easyrobot so that abugaga can more effectively find files … they'd just ask the engine and it would give me plausible starting points for them to narrow down".

**Abugaga constraints that bind this design** (reported):
- "Vectors optional and derived, never required for LF1 or authority. Queries and embeddings must preserve profile/tenant/subject permissions; no cross-profile nearest-neighbour leak." (`docs/2026-11-08-budget-runner-sandbox-storage-amendment.md` §4)
- Knowledge and lookup are an optional backend (`abugaga/docs/design/architecture.md`).

## Verdict

1. **Lexical + graph first; vectors are an optional, derived layer.** On this workspace most "where is X" questions share words, identifiers or links with the right file. BM25 with identifier-aware tokens, fused with link-graph expansion, should find most starting points with no model at all (inferred; the benchmark in §8 decides). It also satisfies Abugaga's rule that vectors are never required.
2. **The workspace is small, so there's no ANN index.**
   - About 5,100 Markdown and Go files totalling 37 MB, with about 10,100 links between notes (observed, §2).
   - That is roughly 25–40 k chunks. Exact (flat) vector search over 40 k × 384 floats is about 15 M multiply-adds per query, a few ms in Go (inferred).
   - HNSW stays behind the `VectorStore` interface until a corpus is 10× larger.
3. **One pure-Go file holds the whole persistent index.** `modernc.org/sqlite` v1.61.0 is already in the module cache, CGo-free, compiled with FTS5 (BM25) and bundling **sqlite-vec v0.1.9** (observed). One SQLite file can hold lexical, vector and graph data (edges plus recursive CTEs) with transactions. An in-memory store serves tests. PostgreSQL (pgx v5.11.0 is cached, observed) and Neo4j (driver **not** cached, so it needs network approval) come later.
4. **Abugaga uses it as a library inside `abu serve`** and exposes it to peers as one request/reply service, `abg.svc.find`. NATS pins the asking peer, so the service can filter by scope. Peers call the tool `roster_find`; the owner uses `abu find`.
5. **Embeddings without network:** none are installed on this host (observed, §7).
   - Zero-network option: vectors trained on the workspace itself (LSA).
   - One-download options: static model2vec-style vectors (pure Go at query time), or a transformer embedding model through the existing MLX setup.
   - The benchmark picks; the owner approves any download.
   - Cloud embeddings are off by default, because they send workspace text to a provider.

## 1. What to reuse from babyai

Source: `~/src/docs/research/babyai/internal/memory/**`, `pkg/embeddings/**`, `proto/memory.proto` (observed, 1,313 lines including generated code).

| babyai piece | What it does (observed) | Keep / drop | Why |
|---|---|---|---|
| `types.Store{Lookup(emb, threshold, limit); Commit(emb, text, tag)}` (`internal/memory/types/types.go`) | the whole interface: one vector per text | **Drop the shape, keep the idea of a narrow store interface** | No ids, no source path, no deletion, no update, no model id. `tag` is accepted and then discarded (`mem/store.go` `Commit` never stores it). A file index needs provenance, upsert, delete-by-source and versioning |
| Brute-force scan with `L2Similarity = 1/(1+‖a−b‖)`, full sort (`mem/store.go` `Lookup`, `pkg/embeddings/similarity.go`) | exact search | **Keep exact search**, drop the metric | Exact search is right at this scale (§2). Use cosine on normalised vectors (dot product); `fp32.Dot` / `NormalizeVec` already exist in EasyRobot (reported). Use a top-k heap, not a full sort |
| Upper cutoff "skip if score > 0.85" | drops near-identical memories | **Drop** | For retrieval, the closest match is the answer |
| **Ignore embeddings** (`tobeignored.txt`, `CalculateIgnoreScore` > 0.64 skips commit and lookup) | filters generic, content-free text | **Keep the idea**: a boilerplate filter | Generalise it into a down-weight for chunks that look like templates, licence headers or generated code, decided at ingest |
| Duplicate suppression (> 0.99 between results) | avoids repeated hits | **Keep, as diversity (MMR)** at the file level | — |
| `similarEmbeddings` / `exceptEmbeddings` in `RecallRequest` (`proto/memory.proto`) | positive and negative examples; **unused** in `recall.go` | **Keep the idea** as `near` / `not` query fields (§6) | "more like this file, not like that one" is a natural narrowing step |
| `similarity_cutoff`, `max_count` with clamps | caller-tunable limits | **Keep** as `k` and `minScore`, with server caps | — |
| SQLite through `mattn/go-sqlite3` (CGo); load everything at start, save on exit (`mem/repo.go`, `service.go` `Save`) | persistence | **Drop** | CGo, whole-RAM, and data lost on crash. Use a transactional per-document write in pure-Go SQLite (§3.3) |
| Embeddings by NATS request to an OpenAI service (`commit.go`, `recall.go`) | remote embedder | **Drop** as the default | Network, cost, privacy. The embedder becomes an optional interface (§7) |
| NATS service with proto request/response (`data-layer-sdk`) | service boundary | **Keep the pattern**: one request/reply service with proto messages | It matches Abugaga's `abg.svc.*` and EasyRobot's proto contracts (reported, `docs/architecture/dndm-and-protobuf.md`) |

**Why batching, graphing and retrieval need rethinking:**
- **Batching:** babyai embeds one request's inputs, then holds everything in RAM until exit. A file index must re-ingest incrementally (only changed files), embed in token-bounded batches, survive crashes mid-run, and never re-embed unchanged text (an embedding cache keyed by content hash).
- **Graphing:** babyai has no graph. The workspace's most reliable relevance signal is explicit structure: about 5.6 k wikilinks and about 4.5 k Markdown links (observed, §2), frontmatter ids (`ABU-061`, `O-005`), and Go imports.
- **Retrieval:** babyai returns raw fragments by vector distance. Peers need **ranked files with reasons and line spans**, small enough for a context window, and a way to narrow down.

## 2. Scale of the workspace (sizing facts)

Measured on `~/src`, excluding `.git`, `node_modules`, `.venv`, `_quarantine` and `_tmp` (observed, 2026-11-09):

| Measure | Value |
|---|---|
| Markdown files | 3,096 (largest folders: `_docs` 748, `itohio.github.io` 560, `EasyRobot` 485, `fpv-gate` 280, `_scratch` 220, `skills` 210) |
| Go files | 2,049 |
| Other code | 494 `.py`, 372 `.c`, 605 `.h`, 279 `.cpp`, 105 `.ts`, 73 `.proto` |
| Markdown + Go bytes | 37.2 MB |
| Wikilinks `[[…]]` | 5,580 |
| Markdown links to `.md` | 4,535 |
| Git repositories at top level | 13 |

**Derived estimates** (inferred):
- 25–40 k chunks at 200–800 tokens each.
- Vectors: 384-dim float32 for 40 k chunks is 61 MB (int8: 15 MB).
- Exact search costs about 15 M multiply-adds per query.
- FTS5 index: 30–60 % of the text size `#verify`.

Files also **move**. The dndm research notes cited in Abugaga docs as `dndm/docs/research/ideation/` now live in `dndm/docs/research/ideation/` (observed), so ingest must treat a rename as cheap.

## 3. Architecture in EasyRobot

### 3.1 Where it lives

- **Package root `x/knowledge`**, at L3/L4 next to `x/math/graph` and `x/marshaller/storage` (reported, `docs/architecture/overview.md`).
- **Style:** CGo-free; interface-first; backends as packages the caller imports (opt-in by import, as with the tensor backends); functional options `func(*T)`, the style the overview proposes to standardise on (reported).
- Build tags only for heavy or benchmark tests (`knowledge_bench`), as with EasyRobot's `mnist_test`.

```text
x/knowledge/                 types and interfaces (Doc, Chunk, Node, Edge, Query, Hit), Engine (compose)
x/knowledge/chunk/md         Markdown chunker: headings, frontmatter, wikilinks, md links, tags
x/knowledge/chunk/golang     Go chunker: go/parser; package, decls, doc comments, imports
x/knowledge/chunk/text       fallback: line windows with a top-level-definition regex (py, c, ts, proto)
x/knowledge/lex              pure-Go BM25 (in-memory) + identifier tokenizer
x/knowledge/graph            pure-Go typed adjacency store (in-memory)
x/knowledge/vec              flat exact vector store (in-memory, top-k heap, fp32.Dot)
x/knowledge/store/sqlite     persistent: FTS5 + sqlite-vec + edges/CTE + catalog, one file (modernc.org/sqlite)
x/knowledge/store/postgres   later: tsvector + pgvector + recursive CTE (Apache AGE optional) (pgx v5)
x/knowledge/store/neo4j      later: graph only (driver needs network approval)
x/knowledge/embed/...        optional embedders (§7)
x/knowledge/ingest           walker, change detection, batching pipeline
x/knowledge/retrieve         hybrid retriever: candidates, fusion, graph expansion, aggregation, reasons
x/knowledge/bench            recall@k harness + question sets
```

**Module boundary:** recommend **`x/knowledge` as its own Go module** (`github.com/itohio/EasyRobot/x/knowledge`, owner decision 2).
- EasyRobot's root module requires gocv, go-tflite, malgo, gorgonia and dndm through `replace ../dndm` (reported, `go.mod`).
- Abugaga today depends only on NATS, JWT, nkeys and cobra (reported).
- A nested module keeps Abugaga's module graph small. `GOPROXY=off` builds then need only the engine's own dependencies, all already cached (`modernc.org/sqlite` v1.61.0 and its `modernc.org/libc` family, observed).
- The cost: EasyRobot's first nested module, and the engine can't import root packages such as `x/math/primitive/fp32` without requiring the root module. Copy the two kernels it needs (`Dot`, normalise) or depend on the root module and accept its graph. That is the trade-off behind decision 2.

**The native graph is not `x/math/graph.GenericGraph`.**
- The vault records defects in it: `AddNode` stores a copy, `DeleteNode` is O(V·E), and it doesn't implement `Accessor` (reported, `docs/modules/graph-and-planning.md`).
- The engine needs typed edges, bulk upsert and delete-by-source.
- Recommend a small dedicated adjacency store with a per-source edge index. Fixing `GenericGraph` is easyrobot's call, but the engine should not wait on it.

### 3.2 Data model

```go
// IDs are content-derived where possible, so renames and re-ingest are cheap.
type DocID   string   // scope + "/" + relative path, e.g. "abugaga/internal/keys/keys.go"
type ChunkID [16]byte // first 16 bytes of sha256(DocID, ord, text)
type NodeID  string   // "doc:<DocID>" | "sym:<pkg>.<Name>" | "id:ABU-061" | "tag:abugaga" | "pkg:<importpath>" | "missing:<target>"

type Doc struct {
	ID       DocID
	Scope    string            // top-level project or roster ("abugaga", "dndm", "docs/roster", …)
	Path     string            // absolute, for reading
	Kind     string            // md | go | py | c | ts | proto | text
	Size     int64
	MTime    time.Time
	Hash     [32]byte          // sha256 of content
	Props    map[string]string // frontmatter: id, type, status, owner, objective, tags, created, updated; go: package
	Weight   float32           // 1.0; lower for generated code, templates, boilerplate
}

type Chunk struct {
	ID      ChunkID
	Doc     DocID
	Ord     int
	Heading []string // breadcrumb: ["Authority model v2", "8. Teams", "8.4 Team-leader restart"]
	Symbol  string   // Go: "keys.PeerPermissions"; empty for prose
	Lines   [2]int   // 1-based inclusive span in the file
	Text    string   // what is indexed (breadcrumb prepended, §4.2)
	Hash    [32]byte // sha256(Text): the embedding-cache key
	Tokens  int
}

type EdgeKind uint8 // LinksTo, Mentions (id), Tagged, Imports, Defines, Contains (doc→chunk), SameDir
type Edge struct {
	From, To NodeID
	Kind     EdgeKind
	Weight   float32
	Source   DocID // the document that asserted the edge: delete-by-source on re-ingest
}
```

### 3.3 Interfaces

```go
// Catalog tracks what is indexed: the basis of incremental ingest and deletions.
type Catalog interface {
	Get(ctx context.Context, id DocID) (Doc, bool, error)
	List(ctx context.Context, scope string) iter.Seq2[Doc, error]
	Put(ctx context.Context, d Doc, chunks []ChunkID) error // last step of a doc's ingest
	Delete(ctx context.Context, id DocID) error
	Generation(ctx context.Context) (uint64, error) // bumps on every committed change
}

type LexicalIndex interface {
	Upsert(ctx context.Context, doc DocID, chunks []Chunk) error // replaces all chunks of doc
	DeleteDoc(ctx context.Context, doc DocID) error
	Search(ctx context.Context, q LexQuery, k int, f Filter) ([]Hit, error)
}

type VectorStore interface {
	Upsert(ctx context.Context, model ModelID, recs []VecRecord) error // VecRecord{Chunk ChunkID; Doc DocID; Vec []float32}
	DeleteDoc(ctx context.Context, doc DocID) error
	Search(ctx context.Context, model ModelID, q []float32, k int, f Filter) ([]Hit, error)
	Models(ctx context.Context) ([]ModelInfo, error) // which models have vectors, with counts
}

type GraphStore interface {
	UpsertEdges(ctx context.Context, source DocID, edges []Edge) error // replaces all edges asserted by source
	DeleteSource(ctx context.Context, source DocID) error
	Neighbors(ctx context.Context, from []NodeID, kinds []EdgeKind, dir Direction, limit int) ([]Edge, error)
	Resolve(ctx context.Context, n NodeID) (DocID, bool, error) // id:ABU-061 → the doc whose frontmatter id is ABU-061
}

type Embedder interface {
	Model() ModelInfo // {ID, Dim, MaxTokens, Normalized}
	Embed(ctx context.Context, purpose Purpose, texts []string) ([][]float32, error) // Purpose: Query | Document
}

type Chunker interface {
	Accepts(path string, head []byte) bool
	Chunk(d Doc, src []byte) (chunks []Chunk, edges []Edge, props map[string]string, err error)
}

type Reranker interface { // optional; heuristic by default (§5.6)
	Rerank(ctx context.Context, q Query, cands []Candidate) ([]Candidate, error)
}

type Ingestor interface {
	Sync(ctx context.Context, src Source) (Report, error) // Report: added, changed, moved, deleted, skipped, errors, duration
}

type Retriever interface {
	Find(ctx context.Context, q Query) (Result, error)
}
```

**Composition:** `knowledge.New(opts ...Option)` with `WithCatalog`, `WithLexical`, `WithVectors`, `WithGraph`, `WithEmbedder`, `WithChunkers` and `WithReranker`.
- A backend may implement several interfaces. `store/sqlite.Open(path)` returns one value that is a `Catalog`, a `LexicalIndex`, a `VectorStore` and a `GraphStore`, so one transaction per document covers all four.
- The engine checks capabilities at build time: no `VectorStore` or no `Embedder` means the lexical and graph paths only, with no error.

### 3.4 Connectors

| Connector | Lexical | Vector | Graph | Persistence | Deps | When |
|---|---|---|---|---|---|---|
| **native memory** (`lex`, `vec`, `graph`) | pure-Go BM25 | flat exact, top-k heap | typed adjacency | none (rebuilt from files) | none | KE-1, tests |
| **native SQLite** (`store/sqlite`) | FTS5 `bm25()` with column weights | sqlite-vec `vec0` (exact), or a BLOB column scanned in Go | `edges` table + `WITH RECURSIVE` | one file, WAL, transactions | `modernc.org/sqlite` v1.61.0 + `/vec` (cached, CGo-free, observed) | KE-2: the default for Abugaga |
| **PostgreSQL** | `tsvector` / `ts_rank_cd`, or `pg_trgm` | `pgvector` (HNSW or IVFFlat) | recursive CTE; Apache AGE (openCypher) optional | server | `jackc/pgx/v5` v5.11.0 (cached, observed); the server and its extensions are owner installs | later: O-002 deployed, many writers |
| **Neo4j** | full-text index (Lucene) | vector index (5.x) `#verify` | native | server | `neo4j-go-driver` (**not cached**, so network approval) | later, only if graph queries outgrow CTEs |

**SQLite schema sketch** (KE-2):

```sql
CREATE TABLE docs   (id TEXT PRIMARY KEY, scope TEXT, path TEXT, kind TEXT, size INT, mtime INT, hash BLOB, props JSON, weight REAL);
CREATE TABLE chunks (id BLOB PRIMARY KEY, doc TEXT REFERENCES docs ON DELETE CASCADE, ord INT, heading TEXT, symbol TEXT, l0 INT, l1 INT, hash BLOB, tokens INT);
CREATE VIRTUAL TABLE chunks_fts USING fts5(heading, symbol, path, body, content='', tokenize='unicode61 remove_diacritics 2');
CREATE TABLE edges  (src TEXT, dst TEXT, kind INT, weight REAL, source TEXT, PRIMARY KEY (src, dst, kind, source));
CREATE INDEX edges_dst ON edges(dst); CREATE INDEX edges_source ON edges(source);
CREATE TABLE ids    (node TEXT PRIMARY KEY, doc TEXT);                       -- id:ABU-061 → doc
CREATE TABLE embcache (model TEXT, hash BLOB, vec BLOB, PRIMARY KEY (model, hash)); -- survives renames and re-chunking
CREATE VIRTUAL TABLE vec_<model> USING vec0(chunk BLOB PRIMARY KEY, embedding float[384]);
```

The tokenizer must keep identifiers (`ABU-061`, `PeerPermissions`, `abg.svc.find`). Feed FTS5 pre-split tokens from the identifier tokenizer (§5.1) rather than relying on `unicode61` alone `#verify`.

## 4. Ingestion

### 4.1 Pipeline and batching

```mermaid
flowchart LR
  W[walk: include/exclude, .gitignore, size cap] --> S{stat changed?<br/>size, mtime}
  S -- no --> skip[skip]
  S -- yes --> H[read + sha256]
  H --> C{hash changed?}
  C -- no --> touch[update mtime only]
  C -- yes --> K[chunk + extract edges + props]
  K --> T[one tx per doc:<br/>lexical upsert, edges replace, ids, catalog last]
  T --> E[embed queue: only chunk hashes missing from embcache]
  E --> B[batch by tokens ≤ 16k or 64 chunks]
  B --> V[vector upsert]
```

- **Change detection in two stages:** `stat` first (size, mtime). Only when those differ, read and hash. Unchanged hash → touch the mtime and do nothing else. This handles `git checkout` and touch storms (inferred).
- **Renames and moves:**
  - after the walk, docs missing from the walk are matched to new docs with the same hash, then re-keyed: catalog, chunks and edges `source`;
  - the embedding cache is keyed by `(model, chunk hash)`, so moved text never re-embeds;
  - edges pointing at the old path become `missing:` nodes until re-resolved; links are re-resolved for docs whose link targets changed.
- **Deletions:** a doc absent from the walk and not matched as a move is deleted (cascade: chunks, FTS rows, vectors, edges by `source`). Nodes that other docs still link to stay as `missing:<target>` nodes, which doubles as a broken-link report (`abu find --broken`, later).
- **Crash safety:** one transaction per document, with the catalog row written last. A crash mid-run means the next sync sees the old hash and redoes that document.
  - Embeddings run after the lexical commit, asynchronously. A doc is searchable lexically at once and gains vectors when its batch finishes.
  - `Result.index` reports `vectorsPending` so peers know (§6).
- **Batching:** bounded channels between stages (walk → read/hash → chunk → write). Worker count = `GOMAXPROCS` for read and chunk, one writer (SQLite single-writer). Embedding batches are sized by tokens, not documents.
- **Triggers:**
  - Abugaga's `abu serve` already watches the roster folder; for the workspace, use `fsnotify` (v1.9.0 cached, observed) with a 2 s debounce, plus a full stat walk every 10 min and at start (cheap: ~5 k stats);
  - `abu index rebuild` drops and re-ingests everything.
- **Model change:** vectors are stored per `ModelID`. A new embedder fills `vec_<model>` lazily from `embcache` misses. Queries use one model at a time, and old models are dropped by command.

### 4.2 What gets ingested for Abugaga

| Source | Chunks | Props | Edges |
|---|---|---|---|
| **Markdown** (roster records, objectives, tasks, ADRs, plans, research, skills, vault notes) | one per heading section, 200–800 tokens: small sections merged into the parent, long ones split at paragraph boundaries | frontmatter: `id`, `type`, `status`, `owner`, `objective`, `tags`, `aliases`, `created`, `updated` | `LinksTo` (wikilinks, Markdown links, resolved against the vault and relative paths; anchors kept); `Mentions` (ids in text that match a known `id:` such as `ABU-061`, `O-005`, `KE-1`, `D13`); `Tagged`; `SameDir` (weak) |
| **Go** | one per top-level declaration (func, method, type, const/var block) with its doc comment; one file-header chunk (package doc, imports); functions over 800 tokens split at blank lines | `package`, `generated` (`// Code generated … DO NOT EDIT.` → weight 0.2) | `Defines` (doc → `sym:`), `Imports` (`pkg:` → `pkg:`), `Contains` |
| **Other code** (py, c/h, cpp, ts, proto) | line windows (60 lines, 10 overlap) aligned to top-level definitions found by a per-language regex. `.proto`: one chunk per `message`/`service` | — | `Defines` for regex-found names |
| **Git history** (phase 2, read-only) | none | per-file `lastCommit` time and the subjects of the last N commits (`git log --name-only`), as a recency prior and extra lexical terms | — |

**Breadcrumb in every chunk:** the indexed text starts with `path › H1 › H2` (or `package.Symbol`). Keyword and vector search then see where a section sits. This is a cheap form of "contextual chunk headers" (inferred).

**Never ingested:**
- `.git`, `node_modules`, `.venv`, `_quarantine`, `_tmp`, binaries, files over 1 MB;
- `.env*`, `*.creds`, `*.nk`, keystore files, `~/.abugaga/**`;
- any chunk that matches a secret pattern (NATS seed prefixes `SU…`/`SA…`/`SO…`, JWT-shaped strings, `BEGIN … PRIVATE KEY`) is dropped and counted in the report.

The roster folder is not committed but **is** ingested locally (it is the owner's own workspace). In O-002 the scope filter becomes a tenant boundary (§6.3).

## 5. Retrieval

### 5.1 Query understanding (no model)

- **Identifier tokenizer**, used for both ingest and query:
  - splits camelCase and PascalCase, snake_case, kebab-case and dotted names (`PeerPermissions` → `peer`, `permissions`, plus `peerpermissions`; `abg.svc.find` → `abg`, `svc`, `find`, plus the whole);
  - keeps ids whole (`ABU-061`, `O-005`);
  - light English stemming for prose tokens only.
- **Exact signals:**
  - tokens that match a known `id:` node, a Go symbol, or a path segment get a direct lookup (no BM25);
  - a query like "ABU-061" or "PeerPermissions" returns that doc first.
- **Filters** come from tool arguments, never parsed from prose: `in` (path prefix), `kind`, `scope`, `status`, `since`.

### 5.2 Candidates (in parallel)

1. **BM25** over chunks with field weights: heading ×3, symbol ×3, path ×2, body ×1 (FTS5 `bm25(chunks_fts, 3, 3, 2, 1)` in SQLite). Top 100.
2. **Exact:** id, symbol and path-segment hits. Top 20.
3. **Vector kNN** (only if an embedder and vectors exist). Top 100.

### 5.3 Fusion

**Reciprocal rank fusion** (`score = Σ 1/(60 + rank)` across lists). It needs no score calibration between BM25 and cosine, and it degrades gracefully when the vector list is missing (inferred; standard practice).

### 5.4 Graph expansion

- From the top 20 fused chunks' documents, follow 1 hop of `LinksTo` (both directions), `Mentions`, `Imports` and `Defines`, capped at 10 neighbours per seed.
- A neighbour gains `β · seedScore / log2(2 + degree(neighbour))`, so hub notes such as `index.md` don't flood the results. β starts at 0.3, tuned on the benchmark.
- Each added or boosted document records its reason: `linked from authority-model-2.md`, `imports internal/keys`, `defines keys.PeerPermissions`.

### 5.5 Aggregate to files

- File score = best chunk + 0.3 × second best + graph boost, multiplied by the doc `Weight` and by priors:
  - recency: mtime or `lastCommit`, half-life 30 days, at most ×1.2;
  - status: `done` tasks and `superseded` notes ×0.8.
- **Diversity:** MMR at the file level, so no more than 3 results come from one folder unless the query was scoped there.

### 5.6 Rerank

- Default: none beyond §5.5.
- A cross-encoder needs a model (§7), and an LLM rerank through omp costs tokens and network. Both stay behind the `Reranker` interface for later.

### 5.7 Budget

- Default 8 files; each with up to 3 sections (heading, line span, a 160-character snippet, reasons).
- Total about 1–1.5 k tokens. `k` is capped at 30.

## 6. Query contract for peers

### 6.1 Messages (proto, EasyRobot style)

```proto
message FindRequest {
  string query          = 1;   // free text, ids, symbols
  uint32 k              = 2;   // default 8, max 30
  repeated string in    = 3;   // path prefixes ("abugaga/docs", "docs/tasks")
  repeated string kinds = 4;   // md | go | py | …
  repeated string near  = 5;   // paths: "more like / linked to these files" (graph + vector neighbourhood)
  repeated string not   = 6;   // paths to exclude, with their near-duplicates
  string since          = 7;   // RFC 3339; recency filter
  bool explain          = 8;   // include per-signal scores
  string cursor         = 9;   // next page of the same query
}
message Section { string heading = 1; uint32 line_start = 2; uint32 line_end = 3; string snippet = 4; repeated string why = 5; }
message Hit     { string path = 1; float score = 2; string kind = 3; string title = 4; repeated Section sections = 5; repeated string why = 6; }
message FindResponse {
  repeated Hit hits = 1;
  string confidence = 2;            // high | medium | low  (top-score margin and coverage of query terms)
  repeated string try_terms = 3;    // when low: nearby vocabulary terms and known ids
  uint64 generation = 4; uint32 docs = 5; uint32 stale_docs = 6; uint32 vectors_pending = 7;
  string cursor = 8;
}
```

**Example reply as a peer sees it** (rendered by the tool):

```text
find "how is a peer's NATS permission set built"   (8 files · confidence high · index 2 min old)
1. abugaga/internal/keys/keys.go:114-120  PeerPermissions — defines keys.PeerPermissions; terms: permission, peer, nats
2. abugaga/docs/design/adr/0024-message-kinds-in-subjects-and-leased-grants.md:11-18  Decision — terms: subjects, credentials; linked from authority-model-2.md
3. abugaga/docs/research/ideation/2026-11-09-authority-model.md:146-170  3.2 Credentials — terms: base peer credential, pub, sub
…
Next: read a span (read path:lines), or narrow: roster_find(query, in="abugaga/internal"), roster_find(near="abugaga/internal/keys/keys.go").
```

### 6.2 How a peer narrows down

1. Read the top span (`read path:105-120`).
2. If it is close but not it, run `roster_find(near=<that file>)` (linked, importing or similar files) or `in=<folder>`.
3. Use `grep` inside the returned folders for exact strings.
4. If `confidence: low`, the reply lists `try_terms`; the peer re-asks with those or falls back to `grep`.

The engine is a starting-point finder, not an answer engine. Snippets are untrusted file content: the tool labels them as data.

### 6.3 Abugaga integration

| Option | Shape | Verdict |
|---|---|---|
| **Library inside `abu serve`, exposed as `abg.svc.find`** (request/reply) | one writer (serve owns the index and watches files); peers call through their sidecar; `abu find` calls serve, or opens the index read-only if serve isn't running | **Recommended** |
| Library in every sidecar | each session opens the SQLite file read-only | rejected: N watchers or a stale index, and no single place for scope rules |
| Separate `knowledge` peer or binary | its own process and credentials | later, if the engine needs a GPU host or O-002 deployment; same proto contract |

- **Scopes:** NATS pins the asking peer (`fromPeer` in the subject, ADR 0024). The service filters results to scopes that peer may see, from roster config (default: the whole workspace for local rosters).
  - Locally every peer runs as the owner's OS user and can read every file anyway, so here the filter is about relevance, not secrecy (inferred).
  - In O-002 it becomes the "no cross-profile nearest-neighbour leak" boundary: filters apply **inside** the candidate queries (SQL `WHERE scope IN …`), never after top-k (reported constraint).
- **Index location:** `~/.abugaga/index/<roster>.db`. It is derived and rebuildable, and it stays out of the roster folder and git.
- **Surfaces:**
  - **omp tool:** `roster_find(query, k?, in?, kind?, near?, not?, since?)`, which renders as in §6.1.
  - **Console:** `/abu find <query>`.
  - **CLI:**
    ```text
    abu find "<query>" [--in <path>]... [--kind md|go] [--near <file>] [-k 8] [--json] [--why]
    abu index status                     docs · chunks · last sync · stale · vectors pending · size
    abu index rebuild [--scope <s>]
    ```
  - **Kind:** a `find` request is `abg.svc.find`, open to every peer's base credential (the same class as other `abg.svc.*` queries; authority-2 §2 unchanged).

## 7. Embeddings without host installs or network

**What exists on this host** (observed by scout, 2026-11-09):
- No embedding model weights anywhere (HF cache: an audio classifier and CLAP audio/text only; no Ollama, LM Studio or oMLX).
- No `ollama`, `llama-server` or `lms`.
- `mlx-lm` 0.32.2 only inside `experience-learning-agent/.venv`, with one **chat** model: `mlx-community/Qwen3-0.6B-4bit`, 337 MB.
- omp has embedding plumbing (an `openai-embeddings` model kind, `/v1/embeddings` on its auth-gateway, Mnemopi's local bge/e5 via fastembed + onnxruntime), but its seeded embedding models are cloud models, and Mnemopi's local runtime has never run here (`~/.omp/run/mnemopi/` absent).
- EasyRobot's NN stack can't run a transformer encoder today: no tokenizer and no weight loader, `GPTAttention.Forward` panics, and there are no residual blocks (reported, `docs/modules/nn.md`, W9).

| Option | Install / network | Quality on paraphrase (inferred) | Cost | Verdict |
|---|---|---|---|---|
| **A. Lexical + graph only** | none | weak where the question shares no words with the file; strong on ids, symbols and links | none | **v1 default** (KE-1) |
| **B. Workspace-trained LSA** (TF-IDF → randomized truncated SVD, rank 256, with gonum; v0.17.0 cached, observed) | **none** | modest: captures this workspace's co-occurrence ("lease" ≈ "grant" ≈ "expiry") but no outside knowledge | minutes to train on ~40 k chunks `#verify`; query = sparse × dense projection, ~1 ms | **v1.5 experiment**: the only zero-network semantic option |
| **C. Static model2vec-style vectors** (a distilled token-embedding table, mean-pooled; pure-Go inference: tokenizer + lookup) | one download of ~30–60 MB weights (owner approval) | better than B on general paraphrase, below transformer encoders `#verify` | microseconds per chunk; no runtime deps | **v1.5 experiment** if the owner approves one download |
| **D. Transformer embedder via MLX** (e.g. a small Qwen3 or bge embedding model behind a loopback server, like `experience-learning-agent/model_server.py`) | model download (owner approval); uses the existing venv, no host install | best local quality | a Python child process; GPU via MLX; full ingest of 40 k chunks in minutes `#verify` | **later**, if C falls short |
| E. Mean-pooled hidden states of the cached Qwen3-0.6B chat model | none | poor: decoder chat models without contrastive training make weak sentence embeddings (inferred, `#verify` on the benchmark) | heavy | **reject**, unless the benchmark surprises |
| F. Pure-Go transformer on EasyRobot NN (W9) | weights download | as D | large engineering (W9 gap list) | **after W9**; the engine's `Embedder` interface is ready for it |
| G. Cloud embeddings through omp's gateway (`text-embedding-3-small`, `qwen3-embedding-8b`) | network + provider account | high | money; **sends workspace text out** | **off by default**; owner opt-in per scope only |

**Rule:** an option is adopted only if it lifts recall@8 on the paraphrase questions by ≥ 5 points over A without hurting identifier questions (§8).

## 8. Evaluation

**Benchmark format** (`x/knowledge/bench/questions.yaml`; the live corpus is given by `KE_CORPUS=~/src`, never committed):

```yaml
- id: q01
  q: "how is a peer's NATS permission set built"
  type: paraphrase        # identifier | paraphrase | cross-link | recency | code
  expect_any: [abugaga/internal/keys/keys.go]
- id: q02
  q: "ABU-061"
  type: identifier
  expect_any: [abugaga/docs/tickets/ABU-061-message-kinds-in-subjects.md]
```

**Metrics:**
- recall@1, recall@8 and MRR, per question type;
- query latency p50/p95;
- full ingest time;
- incremental ingest time (1 file changed; 1 file renamed);
- index size;
- `stale_docs` after a burst of 50 edits.

**Baseline: what peers do today.** A ripgrep ranker: files ranked by the count of distinct query terms matched, ties by match count. The engine must beat it on every question type to justify itself.

**Starting targets** (`#verify`, set after the first run):
- lexical + graph: recall@8 ≥ 0.85 (identifier) and ≥ 0.6 (paraphrase);
- query p95 < 150 ms;
- incremental single-file ingest < 1 s;
- full lexical ingest of ~5 k files < 60 s on this Mac.

**Seed questions with known right files** (observed paths; easyrobot adds EasyRobot ones):

| # | Question | Type | Expect (any) |
|---|---|---|---|
| 1 | how is a peer's NATS permission set built | paraphrase | `abugaga/internal/keys/keys.go` |
| 2 | what fields does a roster message envelope have | paraphrase | `abugaga/internal/roster/protocol.go` |
| 3 | message kinds in subjects, leased grants | identifier | `abugaga/docs/design/adr/0024-message-kinds-in-subjects-and-leased-grants.md` |
| 4 | can the omp extension relay ask to an outside interface | paraphrase | `abugaga/docs/research/babyai-reuse/2026-11-09-abu061-ask-relay-probe.md` |
| 5 | heartbeat pulses configured by markdown files | paraphrase | `abugaga/docs/design/adr/0018-markdown-heartbeat-and-pulses.md`, `abugaga/docs/plan/iteration-1-local.md` |
| 6 | session binding selects the peer, not the profile | identifier | `abugaga/docs/design/adr/0023-session-binding-selects-peer-not-profile.md` |
| 7 | what happens when a team leader restarts | paraphrase | `abugaga/docs/research/ideation/2026-11-09-authority-model-2.md` |
| 8 | how does the owner prove a Telegram message is theirs | paraphrase (no shared terms with "delegated credential") | `abugaga/docs/research/ideation/2026-11-09-interfaces.md` |
| 9 | phone can't approve tool approvals; levers | paraphrase | `abugaga/docs/research/ideation/2026-11-09-interfaces-2.md`, `docs/ux/UX-5-interfaces.md` |
| 10 | roster rules for writing only named paths | paraphrase | `docs/README.md` |
| 11 | owner decisions about Telegram focus and persona | cross-link | `docs/objectives/O-001-abugaga-iteration-1-local.md` |
| 12 | babyai memory store lookup | code | `docs/research/babyai/internal/memory/mem/store.go` |
| 13 | pluggable keystore backends | identifier | `abugaga/docs/tickets/ABU-059-pluggable-keystore.md` |
| 14 | profile-independent session binding ticket | identifier | `abugaga/docs/tickets/ABU-053-profile-independent-session-binding.md` |
| 15 | local MLX model server for the pet | code | `experience-learning-agent/model_server.py` |
| 16 | NDN delivery and authentication research | cross-link | `dndm/docs/research/ideation/2026-11-08-ndn-delivery-auth.md`, `…-auth-2.md` |
| 17 | knowledge engine workstream | identifier | `docs/objectives/O-005-easyrobot.md` |
| 18 | verify report for the portfolio synthesis | cross-link | `docs/verification/R1-02.md` |
| 19 | kd-tree nearest neighbour bug | code | EasyRobot `x/math/graph/kd_tree.go`, `docs/algorithms/kd-tree.md` |
| 20 | what's wrong with GPT attention forward | paraphrase | EasyRobot `docs/modules/nn.md`, `x/math/nn/layers/` (easyrobot to pin the file) |

Question 8 is deliberately vocabulary-mismatched: it measures what options B–D add. Question 16 checks moved files (the old `_docs` path cited elsewhere).

**Privacy of the fixture:** roster records hold personal identifiers (the owner's email in `users/` and objectives). Commit only a **synthetic fixture** (about 50 generated notes and Go files with known links) for unit tests. The real benchmark runs against `KE_CORPUS` locally and stores only the scores.

## 9. Slices

| Slice | Content | Done when |
|---|---|---|
| **KE-1** (ticket exists) | interfaces (§3.3); native memory lexical, graph and flat vec; Markdown chunker with frontmatter, wikilinks, links, ids; identifier tokenizer; BM25 + exact + RRF + graph expansion + file aggregation + reasons; bench harness, synthetic fixture, ripgrep baseline | recall@k table for the 20 seed questions vs ripgrep; all offline |
| KE-2 | `store/sqlite` (FTS5, edges/CTE, catalog, embcache; `vec0` table created only when an embedder is set); incremental ingest with moves and deletions; fsnotify | incremental and rename timings; crash-mid-ingest test (kill, re-sync, consistent) |
| KE-3 | Go chunker (go/parser): symbols, imports graph; text chunker for py/c/ts/proto | code questions in the benchmark |
| KE-4 (Abugaga ticket) | `abg.svc.find` in `abu serve`, `roster_find`, `/abu find`, `abu find`, `abu index` | a peer finds keys.go from question 1 in one call |
| KE-5 | embeddings experiment: B (LSA, zero network) vs C (static, one approved download) | adopt or reject by the ≥ 5-point rule |
| KE-6 / KE-7 | PostgreSQL connector (O-002); Neo4j only on demonstrated need (network approval for the driver) | same contract tests as SQLite |

**Contract tests shared by every connector:**
- upsert/replace per doc;
- delete-by-source;
- move;
- filter-before-top-k (scope);
- k and score caps;
- generation increments.

## Knowledge map

| Quadrant | Item | Next step, owner |
|---|---|---|
| Known knowns | Workspace size and link counts (§2); `modernc.org/sqlite` v1.61.0 with FTS5 and sqlite-vec, `pgx` v5.11.0, `fsnotify` v1.9.0 and `gonum` v0.17.0 cached; no Neo4j driver cached; no embedding weights on the host; EasyRobot NN can't run an encoder (all observed or reported) | — |
| Known unknowns | Whether FTS5 with pre-split identifier tokens ranks code identifiers well enough, or a custom pure-Go BM25 must stay even in KE-2 | easyrobot, KE-2 benchmark |
| Known unknowns | Whether `x/knowledge` as a nested module can reuse `fp32` without requiring the root module, and whether Abugaga builds against it with `GOPROXY=off` | easyrobot, a KE-1 build probe |
| Known unknowns | LSA training time and quality on ~40 k chunks | easyrobot, KE-5 |
| Unknown knowns | Peers' real questions: the seed set is ideation's guess at what peers ask | Leader: collect 20 real "where is…" moments from peer transcripts (verbatim queries + the file they ended at) to replace half the seed set |
| Unknown knowns | Whether the owner wants git history and scratch folders (`_scratch`, `_quarantine`) searchable | owner (decision 6) |
| Unknown unknowns | How much link-graph expansion helps versus hurts on hub-heavy vaults (EasyRobot's `index.md`, ~180 algorithm notes) | benchmark with β = 0 vs tuned |

## Owner decisions

| # | Decision | Recommendation |
|---|---|---|
| 1 | v1 is lexical + link graph with no embeddings; vectors are an optional, derived layer behind the same interfaces | Yes |
| 2 | Put the engine in EasyRobot as **its own Go module** `x/knowledge` (keeps gocv, tflite and dndm out of Abugaga's module graph); connectors with server dependencies as separate packages | Yes; or the root module if you prefer one `go.mod`, and accept its dependency graph in Abugaga |
| 3 | The persistent native store is **pure-Go SQLite** (`modernc.org/sqlite` with FTS5 + sqlite-vec + recursive CTEs, cached); PostgreSQL for O-002; Neo4j only on demonstrated need (driver download needs approval) | Yes |
| 4 | Abugaga embeds the engine in `abu serve` and exposes `abg.svc.find`, with scope filtering inside the queries; tool `roster_find`, CLI `abu find`, index in `~/.abugaga/index/` | Yes |
| 5 | Embeddings: a zero-network LSA experiment first; one approved download of a static model2vec-style model as the second candidate; cloud embeddings off by default | Yes; approve the download only when KE-5 starts |
| 6 | Ingest scope: all project folders and the roster folder; exclude `_quarantine`, `_tmp` and secrets; `_scratch` included at weight 0.5; git history as metadata only (phase 2) | Yes |
| 7 | The benchmark runs on the live workspace locally (scores only are kept); the committed fixture is synthetic | Yes |

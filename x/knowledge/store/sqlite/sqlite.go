// Package sqlite is a persistent knowledge.Store in one SQLite file, using
// the pure-Go modernc.org/sqlite driver: catalog, chunks, an FTS5 index
// ranked with bm25() and per-column weights, and an edges table. File
// databases use WAL, foreign keys and a busy timeout; ":memory:" opens a
// private in-memory database on a single connection.
package sqlite

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/internal/storeutil"
	"github.com/itohio/EasyRobot/x/knowledge/token"

	_ "modernc.org/sqlite" // registers the "sqlite" driver
)

// MemoryPath opens a private in-memory database.
const MemoryPath = ":memory:"

// listPage is the ListDocs page size.
const listPage = 256

// Option configures Open.
type Option func(*config)

type config struct {
	busyTimeout time.Duration
}

// WithBusyTimeout sets how long a connection waits for a lock held by
// another connection or process before failing (default 5s).
func WithBusyTimeout(d time.Duration) Option {
	return func(c *config) { c.busyTimeout = d }
}

// Store is a SQLite-backed knowledge.Store, safe for concurrent use.
// Writers are serialised in-process and each write method is one
// transaction; readers run concurrently (WAL) on file databases.
type Store struct {
	db   *sql.DB
	path string

	wmu sync.Mutex // serialises writers

	mu     sync.RWMutex // read-held by every operation; Close takes it
	closed bool
}

var _ knowledge.Store = (*Store)(nil)

// Open opens or creates the database at path (MemoryPath for an in-memory
// database) and migrates its schema.
func Open(path string, opts ...Option) (*Store, error) {
	cfg := config{busyTimeout: 5 * time.Second}
	for _, o := range opts {
		o(&cfg)
	}
	if path == "" || strings.ContainsRune(path, '?') {
		return nil, fmt.Errorf("sqlite: invalid path %q", path)
	}
	q := fmt.Sprintf("?_txlock=immediate&_pragma=busy_timeout(%d)&_pragma=foreign_keys(1)", cfg.busyTimeout.Milliseconds())
	mem := path == MemoryPath
	if !mem {
		q += "&_pragma=journal_mode(WAL)&_pragma=synchronous(NORMAL)"
	}
	db, err := sql.Open("sqlite", path+q)
	if err != nil {
		return nil, fmt.Errorf("sqlite: open %s: %w", path, err)
	}
	n := max(4, runtime.GOMAXPROCS(0))
	if mem {
		n = 1 // every connection would be a separate database
	}
	db.SetMaxOpenConns(n)
	db.SetMaxIdleConns(n)
	if err := migrate(context.Background(), db); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("sqlite: open %s: %w", path, err)
	}
	return &Store{db: db, path: path}, nil
}

// Close closes the database. Further calls return knowledge.ErrClosed;
// closing twice is harmless.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	if err := s.db.Close(); err != nil {
		return fmt.Errorf("sqlite: close: %w", err)
	}
	return nil
}

func (s *Store) acquire() error {
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return knowledge.ErrClosed
	}
	return nil
}

func (s *Store) release() { s.mu.RUnlock() }

// write runs fn in one transaction and bumps the generation on success.
func (s *Store) write(ctx context.Context, op string, fn func(tx *sql.Tx) error) error {
	if err := s.acquire(); err != nil {
		return err
	}
	defer s.release()
	s.wmu.Lock()
	defer s.wmu.Unlock()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: %s: begin: %w", op, err)
	}
	if err := fn(tx); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("sqlite: %s: %w", op, err)
	}
	if _, err := tx.ExecContext(ctx, `UPDATE meta SET v = v + 1 WHERE k = 'generation'`); err != nil {
		_ = tx.Rollback()
		return fmt.Errorf("sqlite: %s: bump generation: %w", op, err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("sqlite: %s: commit: %w", op, err)
	}
	return nil
}

// --- Catalog ---

const docCols = `id, scope, path, kind, title, size, mtime_s, mtime_ns, hash, props, weight`

type scanner interface{ Scan(dest ...any) error }

func scanDoc(r scanner) (knowledge.Doc, error) {
	var (
		d         knowledge.Doc
		id        string
		sec, nsec int64
		hash      []byte
		props     string
		weight    float64
	)
	if err := r.Scan(&id, &d.Scope, &d.Path, &d.Kind, &d.Title, &d.Size, &sec, &nsec, &hash, &props, &weight); err != nil {
		return knowledge.Doc{}, err
	}
	d.ID = knowledge.DocID(id)
	d.MTime = time.Unix(sec, nsec).UTC()
	if err := copyHash(&d.Hash, hash); err != nil {
		return knowledge.Doc{}, fmt.Errorf("doc %q: %w", id, err)
	}
	if err := json.Unmarshal([]byte(props), &d.Props); err != nil {
		return knowledge.Doc{}, fmt.Errorf("doc %q props: %w", id, err)
	}
	d.Weight = float32(weight)
	return d, nil
}

func copyHash(dst *[32]byte, b []byte) error {
	if len(b) != len(dst) {
		return fmt.Errorf("hash has %d bytes, want %d", len(b), len(dst))
	}
	copy(dst[:], b)
	return nil
}

// GetDoc returns the catalog entry of id.
func (s *Store) GetDoc(ctx context.Context, id knowledge.DocID) (knowledge.Doc, bool, error) {
	if err := s.acquire(); err != nil {
		return knowledge.Doc{}, false, err
	}
	defer s.release()
	d, err := scanDoc(s.db.QueryRowContext(ctx, `SELECT `+docCols+` FROM docs WHERE id = ?`, string(id)))
	if errors.Is(err, sql.ErrNoRows) {
		return knowledge.Doc{}, false, nil
	}
	if err != nil {
		return knowledge.Doc{}, false, fmt.Errorf("sqlite: get doc %q: %w", id, err)
	}
	return d, true, nil
}

// ListDocs yields all documents sorted by ID, reading in pages so no
// connection or lock is held while the loop body runs.
func (s *Store) ListDocs(ctx context.Context) iter.Seq2[knowledge.Doc, error] {
	return func(yield func(knowledge.Doc, error) bool) {
		after := ""
		for {
			page, err := s.listPage(ctx, after)
			if err != nil {
				yield(knowledge.Doc{}, err)
				return
			}
			for _, d := range page {
				if !yield(d, nil) {
					return
				}
			}
			if len(page) < listPage {
				return
			}
			after = string(page[len(page)-1].ID)
		}
	}
}

func (s *Store) listPage(ctx context.Context, after string) ([]knowledge.Doc, error) {
	if err := s.acquire(); err != nil {
		return nil, err
	}
	defer s.release()
	rows, err := s.db.QueryContext(ctx, `SELECT `+docCols+` FROM docs WHERE id > ? ORDER BY id LIMIT ?`, after, listPage)
	if err != nil {
		return nil, fmt.Errorf("sqlite: list docs: %w", err)
	}
	defer rows.Close()
	var page []knowledge.Doc
	for rows.Next() {
		d, err := scanDoc(rows)
		if err != nil {
			return nil, fmt.Errorf("sqlite: list docs: %w", err)
		}
		page = append(page, d)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite: list docs: %w", err)
	}
	return page, nil
}

// Generation returns the number of committed writes since the database
// was created; it persists across reopen.
func (s *Store) Generation(ctx context.Context) (uint64, error) {
	if err := s.acquire(); err != nil {
		return 0, err
	}
	defer s.release()
	var g int64
	if err := s.db.QueryRowContext(ctx, `SELECT v FROM meta WHERE k = 'generation'`).Scan(&g); err != nil {
		return 0, fmt.Errorf("sqlite: generation: %w", err)
	}
	return uint64(g), nil
}

// --- LexicalIndex ---

const chunkCols = `id, doc, ord, heading, symbol, l0, l1, text, hash, tokens`

func scanChunk(r scanner) (knowledge.Chunk, error) {
	var (
		c            knowledge.Chunk
		id, hash     []byte
		doc, heading string
	)
	if err := r.Scan(&id, &doc, &c.Ord, &heading, &c.Symbol, &c.Lines[0], &c.Lines[1], &c.Text, &hash, &c.Tokens); err != nil {
		return knowledge.Chunk{}, err
	}
	if len(id) != len(c.ID) {
		return knowledge.Chunk{}, fmt.Errorf("chunk id has %d bytes", len(id))
	}
	copy(c.ID[:], id)
	c.Doc = knowledge.DocID(doc)
	if err := copyHash(&c.Hash, hash); err != nil {
		return knowledge.Chunk{}, fmt.Errorf("chunk %s: %w", c.ID, err)
	}
	if err := json.Unmarshal([]byte(heading), &c.Heading); err != nil {
		return knowledge.Chunk{}, fmt.Errorf("chunk %s heading: %w", c.ID, err)
	}
	return c, nil
}

// GetChunk returns the chunk with id.
func (s *Store) GetChunk(ctx context.Context, id knowledge.ChunkID) (knowledge.Chunk, bool, error) {
	if err := s.acquire(); err != nil {
		return knowledge.Chunk{}, false, err
	}
	defer s.release()
	c, err := scanChunk(s.db.QueryRowContext(ctx, `SELECT `+chunkCols+` FROM chunks WHERE id = ?`, id[:]))
	if errors.Is(err, sql.ErrNoRows) {
		return knowledge.Chunk{}, false, nil
	}
	if err != nil {
		return knowledge.Chunk{}, false, fmt.Errorf("sqlite: get chunk %s: %w", id, err)
	}
	return c, true, nil
}

// DocChunks returns the chunks of doc ordered by Ord.
func (s *Store) DocChunks(ctx context.Context, doc knowledge.DocID) ([]knowledge.Chunk, error) {
	if err := s.acquire(); err != nil {
		return nil, err
	}
	defer s.release()
	rows, err := s.db.QueryContext(ctx, `SELECT `+chunkCols+` FROM chunks WHERE doc = ? ORDER BY ord, id`, string(doc))
	if err != nil {
		return nil, fmt.Errorf("sqlite: chunks of %q: %w", doc, err)
	}
	defer rows.Close()
	var out []knowledge.Chunk
	for rows.Next() {
		c, err := scanChunk(rows)
		if err != nil {
			return nil, fmt.Errorf("sqlite: chunks of %q: %w", doc, err)
		}
		out = append(out, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite: chunks of %q: %w", doc, err)
	}
	return out, nil
}

// SearchLex ranks chunks containing any query term (OR semantics) with
// FTS5 bm25() and column weights heading 3, symbol 3, path 2, body 1;
// Score is -bm25 (higher is better). Filter f is part of the SQL WHERE
// clause, so it applies before LIMIT k. k <= 0 or a query without valid
// terms yields no hits. Ties are broken by DocID, then Ord.
func (s *Store) SearchLex(ctx context.Context, q knowledge.LexQuery, k int, f knowledge.Filter) ([]knowledge.ChunkHit, error) {
	if err := s.acquire(); err != nil {
		return nil, err
	}
	defer s.release()
	terms := storeutil.QueryTerms(q.Terms)
	if k <= 0 || len(terms) == 0 {
		return nil, nil
	}
	w := storeutil.Weights
	var sb strings.Builder
	fmt.Fprintf(&sb, `SELECT c.id, c.doc, c.ord, c.heading, c.symbol, c.text, -bm25(chunks_fts, %g, %g, %g, %g) AS score
FROM chunks_fts JOIN chunks c ON c.rid = chunks_fts.rowid JOIN docs d ON d.id = c.doc
WHERE chunks_fts MATCH ?`, w[0], w[1], w[2], w[3])
	args := []any{matchExpr(terms)}
	args = appendFilter(&sb, args, f)
	sb.WriteString(`
ORDER BY score DESC, c.doc, c.ord, c.id
LIMIT ?`)
	args = append(args, k)

	rows, err := s.db.QueryContext(ctx, sb.String(), args...)
	if err != nil {
		return nil, fmt.Errorf("sqlite: search: %w", err)
	}
	defer rows.Close()
	var hits []knowledge.ChunkHit
	for rows.Next() {
		var (
			c       knowledge.Chunk
			id      []byte
			doc     string
			heading string
			score   float64
		)
		if err := rows.Scan(&id, &doc, &c.Ord, &heading, &c.Symbol, &c.Text, &score); err != nil {
			return nil, fmt.Errorf("sqlite: search: %w", err)
		}
		copy(c.ID[:], id)
		c.Doc = knowledge.DocID(doc)
		if err := json.Unmarshal([]byte(heading), &c.Heading); err != nil {
			return nil, fmt.Errorf("sqlite: search: chunk %s heading: %w", c.ID, err)
		}
		hits = append(hits, knowledge.ChunkHit{Chunk: c.ID, Doc: c.Doc, Score: score, Matched: matched(terms, c)})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite: search: %w", err)
	}
	return hits, nil
}

// matchExpr ORs the terms as quoted FTS5 strings.
func matchExpr(terms []string) string {
	var sb strings.Builder
	for i, t := range terms {
		if i > 0 {
			sb.WriteString(" OR ")
		}
		sb.WriteByte('"')
		sb.WriteString(strings.ReplaceAll(t, `"`, `""`))
		sb.WriteByte('"')
	}
	return sb.String()
}

// appendFilter adds f's conditions on docs d to the WHERE clause in sb.
// Prefixes match like knowledge.Filter.Match: the id itself or anything
// below it; [p+"/", p+"0") is that range in byte order ('0' follows '/').
func appendFilter(sb *strings.Builder, args []any, f knowledge.Filter) []any {
	if len(f.In) > 0 {
		sb.WriteString("\n  AND (")
		for i, p := range f.In {
			p = strings.TrimSuffix(p, "/")
			if i > 0 {
				sb.WriteString(" OR ")
			}
			sb.WriteString("d.id = ? OR (d.id >= ? AND d.id < ?)")
			args = append(args, p, p+"/", p+"0")
		}
		sb.WriteString(")")
	}
	in := func(col string, vals []string, not bool) {
		if len(vals) == 0 {
			return
		}
		op := "IN"
		if not {
			op = "NOT IN"
		}
		fmt.Fprintf(sb, "\n  AND %s %s (%s)", col, op, strings.TrimSuffix(strings.Repeat("?,", len(vals)), ","))
		for _, v := range vals {
			args = append(args, v)
		}
	}
	in("d.kind", f.Kinds, false)
	in("d.scope", f.Scopes, false)
	not := make([]string, len(f.Not))
	for i, id := range f.Not {
		not[i] = string(id)
	}
	in("d.id", not, true)
	return args
}

// matched returns the query terms present in any lexical field of c.
func matched(terms []string, c knowledge.Chunk) []string {
	have := map[string]struct{}{}
	for _, text := range storeutil.FieldTexts(c) {
		for _, t := range token.Tokens(text) {
			have[t] = struct{}{}
		}
	}
	var out []string
	for _, t := range terms {
		if _, ok := have[t]; ok {
			out = append(out, t)
		}
	}
	return out
}

// --- GraphStore ---

// Neighbors returns edges touching any seed, of the given kinds (all when
// empty), at most limit per seed (no limit when <= 0). Each seed's edges
// are ordered by Weight desc, then the other endpoint, Kind, Source; seeds
// follow input order and an edge already reported for an earlier seed is
// skipped without counting towards the limit.
func (s *Store) Neighbors(ctx context.Context, from []knowledge.NodeID, kinds []knowledge.EdgeKind, dir knowledge.Direction, limit int) ([]knowledge.Edge, error) {
	if err := s.acquire(); err != nil {
		return nil, err
	}
	defer s.release()
	var where string
	switch dir {
	case knowledge.Out:
		where = `src = ?1`
	case knowledge.In:
		where = `dst = ?1`
	case knowledge.Both:
		where = `(src = ?1 OR dst = ?1)`
	default:
		return nil, fmt.Errorf("sqlite: neighbors: invalid direction %d", dir)
	}
	query := `SELECT src, dst, kind, weight, source, CASE WHEN src = ?1 THEN dst ELSE src END AS other
FROM edges WHERE ` + where
	var kindArgs []any
	if len(kinds) > 0 {
		query += ` AND kind IN (` + strings.TrimSuffix(strings.Repeat("?,", len(kinds)), ",") + `)`
		for _, k := range kinds {
			kindArgs = append(kindArgs, int64(k))
		}
	}
	query += ` ORDER BY weight DESC, other, kind, source, src, dst`

	var out []knowledge.Edge
	reported := map[knowledge.Edge]struct{}{}
	for _, seed := range storeutil.UniqueNodes(from) {
		args := append([]any{string(seed)}, kindArgs...)
		if err := s.seedEdges(ctx, query, args, limit, reported, &out); err != nil {
			return nil, fmt.Errorf("sqlite: neighbors of %q: %w", seed, err)
		}
	}
	return out, nil
}

func (s *Store) seedEdges(ctx context.Context, query string, args []any, limit int, reported map[knowledge.Edge]struct{}, out *[]knowledge.Edge) error {
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return err
	}
	defer rows.Close()
	taken := 0
	for (limit <= 0 || taken < limit) && rows.Next() {
		var (
			e                     knowledge.Edge
			src, dst, other, from string
			kind                  int64
			weight                float64
		)
		if err := rows.Scan(&src, &dst, &kind, &weight, &from, &other); err != nil {
			return err
		}
		e = knowledge.Edge{From: knowledge.NodeID(src), To: knowledge.NodeID(dst), Kind: knowledge.EdgeKind(kind), Weight: float32(weight), Source: knowledge.DocID(from)}
		key := e
		key.Weight = 0
		if _, dup := reported[key]; dup {
			continue
		}
		reported[key] = struct{}{}
		*out = append(*out, e)
		taken++
	}
	return rows.Err()
}

// Degree counts edges touching n in either direction; a self-loop counts
// once.
func (s *Store) Degree(ctx context.Context, n knowledge.NodeID) (int, error) {
	if err := s.acquire(); err != nil {
		return 0, err
	}
	defer s.release()
	var d int
	if err := s.db.QueryRowContext(ctx, `SELECT count(*) FROM edges WHERE src = ?1 OR dst = ?1`, string(n)).Scan(&d); err != nil {
		return 0, fmt.Errorf("sqlite: degree of %q: %w", n, err)
	}
	return d, nil
}

// Resolve maps a doc: node to itself when the document is indexed, and any
// other node to the smallest DocID asserting a Defines edge to it.
func (s *Store) Resolve(ctx context.Context, n knowledge.NodeID) (knowledge.DocID, bool, error) {
	if err := s.acquire(); err != nil {
		return "", false, err
	}
	defer s.release()
	var (
		doc sql.NullString
		err error
	)
	if n.Kind() == knowledge.NodeDoc {
		err = s.db.QueryRowContext(ctx, `SELECT id FROM docs WHERE id = ?`, n.Value()).Scan(&doc)
	} else {
		err = s.db.QueryRowContext(ctx, `SELECT min(source) FROM edges WHERE dst = ? AND kind = ?`,
			string(n), int64(knowledge.Defines)).Scan(&doc)
	}
	if errors.Is(err, sql.ErrNoRows) {
		return "", false, nil
	}
	if err != nil {
		return "", false, fmt.Errorf("sqlite: resolve %q: %w", n, err)
	}
	if !doc.Valid {
		return "", false, nil
	}
	return knowledge.DocID(doc.String), true, nil
}

// --- Writer ---

// PutDoc replaces the document, all its chunks and all edges it asserts,
// in one transaction.
func (s *Store) PutDoc(ctx context.Context, in knowledge.DocUpdate) error {
	u, err := storeutil.Normalize(in)
	if err != nil {
		if aerr := s.acquire(); aerr != nil {
			return aerr
		}
		s.release()
		return fmt.Errorf("sqlite: put %q: %w", in.Doc.ID, err)
	}
	return s.write(ctx, fmt.Sprintf("put %q", u.Doc.ID), func(tx *sql.Tx) error {
		if err := deleteContent(ctx, tx, u.Doc.ID); err != nil {
			return err
		}
		if err := upsertDoc(ctx, tx, u.Doc); err != nil {
			return err
		}
		if err := insertChunks(ctx, tx, u.Chunks); err != nil {
			return err
		}
		return insertEdges(ctx, tx, u.Edges)
	})
}

// DeleteDoc removes the document, its chunks and the edges it asserts;
// edges other documents assert towards it stay. Deleting an unknown
// document is a successful no-op write.
func (s *Store) DeleteDoc(ctx context.Context, id knowledge.DocID) error {
	return s.write(ctx, fmt.Sprintf("delete %q", id), func(tx *sql.Tx) error {
		return deleteDoc(ctx, tx, id)
	})
}

// MoveDoc re-keys document from as to.ID and stores the catalog entry to.
// Chunks move with it and their ids are re-derived from (to.ID, Ord, Text);
// the path field is re-indexed; edges it asserts change Source; every edge
// endpoint doc:from becomes doc:to.ID. A document already stored at to.ID
// is replaced. Edges that collide after re-keying merge, keeping the
// highest weight.
func (s *Store) MoveDoc(ctx context.Context, from knowledge.DocID, to knowledge.Doc) error {
	if to.ID == "" {
		if err := s.acquire(); err != nil {
			return err
		}
		s.release()
		return fmt.Errorf("sqlite: move %q: empty target id", from)
	}
	return s.write(ctx, fmt.Sprintf("move %q to %q", from, to.ID), func(tx *sql.Tx) error {
		var one int
		err := tx.QueryRowContext(ctx, `SELECT 1 FROM docs WHERE id = ?`, string(from)).Scan(&one)
		if errors.Is(err, sql.ErrNoRows) {
			return storeutil.ErrNotFound
		}
		if err != nil {
			return err
		}
		if to.ID == from {
			return upsertDoc(ctx, tx, to)
		}
		if err := deleteDoc(ctx, tx, to.ID); err != nil {
			return err
		}
		return moveDoc(ctx, tx, from, to)
	})
}

// TouchDoc records a new mtime for an unchanged document.
func (s *Store) TouchDoc(ctx context.Context, id knowledge.DocID, mtime time.Time) error {
	return s.write(ctx, fmt.Sprintf("touch %q", id), func(tx *sql.Tx) error {
		res, err := tx.ExecContext(ctx, `UPDATE docs SET mtime_s = ?, mtime_ns = ? WHERE id = ?`,
			mtime.Unix(), int64(mtime.Nanosecond()), string(id))
		if err != nil {
			return err
		}
		n, err := res.RowsAffected()
		if err != nil {
			return err
		}
		if n == 0 {
			return storeutil.ErrNotFound
		}
		return nil
	})
}

// --- write helpers; run inside a transaction ---

// deleteContent removes the FTS rows, chunks and asserted edges of id.
func deleteContent(ctx context.Context, tx *sql.Tx, id knowledge.DocID) error {
	if _, err := tx.ExecContext(ctx, `DELETE FROM chunks_fts WHERE rowid IN (SELECT rid FROM chunks WHERE doc = ?)`, string(id)); err != nil {
		return fmt.Errorf("delete fts rows: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM chunks WHERE doc = ?`, string(id)); err != nil {
		return fmt.Errorf("delete chunks: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM edges WHERE source = ?`, string(id)); err != nil {
		return fmt.Errorf("delete edges: %w", err)
	}
	return nil
}

func deleteDoc(ctx context.Context, tx *sql.Tx, id knowledge.DocID) error {
	if err := deleteContent(ctx, tx, id); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM docs WHERE id = ?`, string(id)); err != nil {
		return fmt.Errorf("delete doc: %w", err)
	}
	return nil
}

func docArgs(d knowledge.Doc) ([]any, error) {
	props, err := json.Marshal(d.Props)
	if err != nil {
		return nil, fmt.Errorf("encode props: %w", err)
	}
	return []any{string(d.ID), d.Scope, d.Path, d.Kind, d.Title, d.Size,
		d.MTime.Unix(), int64(d.MTime.Nanosecond()), d.Hash[:], string(props), float64(d.Weight)}, nil
}

func upsertDoc(ctx context.Context, tx *sql.Tx, d knowledge.Doc) error {
	args, err := docArgs(d)
	if err != nil {
		return err
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO docs(`+docCols+`) VALUES (?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(id) DO UPDATE SET scope = excluded.scope, path = excluded.path, kind = excluded.kind,
	title = excluded.title, size = excluded.size, mtime_s = excluded.mtime_s, mtime_ns = excluded.mtime_ns,
	hash = excluded.hash, props = excluded.props, weight = excluded.weight`, args...)
	if err != nil {
		return fmt.Errorf("upsert doc: %w", err)
	}
	return nil
}

// ftsText is the pre-tokenized form of s stored in FTS5.
func ftsText(s string) string { return strings.Join(token.Tokens(s), " ") }

func insertFTS(ctx context.Context, stmt *sql.Stmt, rid int64, c knowledge.Chunk) error {
	f := storeutil.FieldTexts(c)
	_, err := stmt.ExecContext(ctx, rid,
		ftsText(f[storeutil.FieldHeading]), ftsText(f[storeutil.FieldSymbol]),
		ftsText(f[storeutil.FieldPath]), ftsText(f[storeutil.FieldBody]))
	return err
}

const insertFTSQuery = `INSERT INTO chunks_fts(rowid, heading, symbol, path, body) VALUES (?,?,?,?,?)`

func insertChunks(ctx context.Context, tx *sql.Tx, chunks []knowledge.Chunk) error {
	if len(chunks) == 0 {
		return nil
	}
	ins, err := tx.PrepareContext(ctx, `INSERT INTO chunks(`+chunkCols+`) VALUES (?,?,?,?,?,?,?,?,?,?)`)
	if err != nil {
		return fmt.Errorf("prepare chunk insert: %w", err)
	}
	defer ins.Close()
	fts, err := tx.PrepareContext(ctx, insertFTSQuery)
	if err != nil {
		return fmt.Errorf("prepare fts insert: %w", err)
	}
	defer fts.Close()
	for _, c := range chunks {
		heading, err := json.Marshal(c.Heading)
		if err != nil {
			return fmt.Errorf("encode heading: %w", err)
		}
		res, err := ins.ExecContext(ctx, c.ID[:], string(c.Doc), c.Ord, string(heading), c.Symbol,
			c.Lines[0], c.Lines[1], c.Text, c.Hash[:], c.Tokens)
		if err != nil {
			return fmt.Errorf("insert chunk %s (ord %d): %w", c.ID, c.Ord, err)
		}
		rid, err := res.LastInsertId()
		if err != nil {
			return fmt.Errorf("insert chunk %s: %w", c.ID, err)
		}
		if err := insertFTS(ctx, fts, rid, c); err != nil {
			return fmt.Errorf("index chunk %s: %w", c.ID, err)
		}
	}
	return nil
}

func insertEdges(ctx context.Context, tx *sql.Tx, edges []knowledge.Edge) error {
	if len(edges) == 0 {
		return nil
	}
	stmt, err := tx.PrepareContext(ctx, `INSERT INTO edges(src, dst, kind, weight, source) VALUES (?,?,?,?,?)
ON CONFLICT(src, dst, kind, source) DO UPDATE SET weight = max(weight, excluded.weight)`)
	if err != nil {
		return fmt.Errorf("prepare edge insert: %w", err)
	}
	defer stmt.Close()
	for _, e := range edges {
		if _, err := stmt.ExecContext(ctx, string(e.From), string(e.To), int64(e.Kind), float64(e.Weight), string(e.Source)); err != nil {
			return fmt.Errorf("insert edge %s→%s: %w", e.From, e.To, err)
		}
	}
	return nil
}

// moveDoc re-keys from as to.ID; the target id is known to be free.
func moveDoc(ctx context.Context, tx *sql.Tx, from knowledge.DocID, to knowledge.Doc) error {
	type moved struct {
		rid   int64
		chunk knowledge.Chunk
	}
	rows, err := tx.QueryContext(ctx, `SELECT rid, ord, heading, symbol, text FROM chunks WHERE doc = ?`, string(from))
	if err != nil {
		return fmt.Errorf("read chunks: %w", err)
	}
	var chunks []moved
	for rows.Next() {
		var (
			m       moved
			heading string
		)
		if err := rows.Scan(&m.rid, &m.chunk.Ord, &heading, &m.chunk.Symbol, &m.chunk.Text); err != nil {
			rows.Close()
			return fmt.Errorf("read chunks: %w", err)
		}
		if err := json.Unmarshal([]byte(heading), &m.chunk.Heading); err != nil {
			rows.Close()
			return fmt.Errorf("read chunks: heading: %w", err)
		}
		chunks = append(chunks, m)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return fmt.Errorf("read chunks: %w", err)
	}
	// Refuse moves whose re-keyed chunk ids coincide or belong to a third
	// document; the caller's transaction rolls back, so nothing changes.
	seen := make(map[knowledge.ChunkID]bool, len(chunks))
	for _, m := range chunks {
		nid := knowledge.NewChunkID(to.ID, m.chunk.Ord, m.chunk.Text)
		if seen[nid] {
			return fmt.Errorf("chunks at ord %d re-key to the same id %s", m.chunk.Ord, nid)
		}
		seen[nid] = true
		var owner string
		err := tx.QueryRowContext(ctx, `SELECT doc FROM chunks WHERE id = ? AND doc <> ?`, nid[:], string(from)).Scan(&owner)
		if err == nil {
			return fmt.Errorf("chunk %d re-keys to id %s of %q", m.chunk.Ord, nid, owner)
		}
		if !errors.Is(err, sql.ErrNoRows) {
			return fmt.Errorf("check chunk ids: %w", err)
		}
	}
	// Park the old ids (a TEXT never equals a BLOB id), so re-keying one
	// chunk cannot hit another moved chunk's old id.
	if _, err := tx.ExecContext(ctx, `UPDATE chunks SET id = CAST(rid AS TEXT) WHERE doc = ?`, string(from)); err != nil {
		return fmt.Errorf("park chunk ids: %w", err)
	}

	if _, err := tx.ExecContext(ctx, `DELETE FROM chunks_fts WHERE rowid IN (SELECT rid FROM chunks WHERE doc = ?)`, string(from)); err != nil {
		return fmt.Errorf("delete fts rows: %w", err)
	}
	// Re-key the catalog row; ON UPDATE CASCADE moves chunks.doc along.
	args, err := docArgs(to)
	if err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `UPDATE docs SET id = ?, scope = ?, path = ?, kind = ?, title = ?, size = ?,
	mtime_s = ?, mtime_ns = ?, hash = ?, props = ?, weight = ? WHERE id = ?`, append(args, string(from))...); err != nil {
		return fmt.Errorf("re-key doc: %w", err)
	}
	if len(chunks) > 0 {
		upd, err := tx.PrepareContext(ctx, `UPDATE chunks SET id = ? WHERE rid = ?`)
		if err != nil {
			return fmt.Errorf("prepare chunk re-key: %w", err)
		}
		defer upd.Close()
		fts, err := tx.PrepareContext(ctx, insertFTSQuery)
		if err != nil {
			return fmt.Errorf("prepare fts insert: %w", err)
		}
		defer fts.Close()
		for _, m := range chunks {
			c := storeutil.RekeyChunk(m.chunk, to.ID)
			if _, err := upd.ExecContext(ctx, c.ID[:], m.rid); err != nil {
				return fmt.Errorf("re-key chunk %d: %w", c.Ord, err)
			}
			if err := insertFTS(ctx, fts, m.rid, c); err != nil {
				return fmt.Errorf("index chunk %d: %w", c.Ord, err)
			}
		}
	}

	// Re-key edges: copy every edge naming the old doc (as source or
	// endpoint) with the new names, merging collisions, then drop the
	// originals. Copies never name the old doc, so the delete only hits
	// originals.
	oldNode, newNode := string(knowledge.DocNode(from)), string(knowledge.DocNode(to.ID))
	if _, err := tx.ExecContext(ctx, `INSERT INTO edges(src, dst, kind, weight, source)
SELECT CASE WHEN src = ?1 THEN ?2 ELSE src END,
       CASE WHEN dst = ?1 THEN ?2 ELSE dst END,
       kind, weight,
       CASE WHEN source = ?3 THEN ?4 ELSE source END
FROM edges WHERE source = ?3 OR src = ?1 OR dst = ?1
ON CONFLICT(src, dst, kind, source) DO UPDATE SET weight = max(weight, excluded.weight)`,
		oldNode, newNode, string(from), string(to.ID)); err != nil {
		return fmt.Errorf("re-key edges: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM edges WHERE source = ?3 OR src = ?1 OR dst = ?1`,
		oldNode, newNode, string(from)); err != nil {
		return fmt.Errorf("drop re-keyed edges: %w", err)
	}
	return nil
}

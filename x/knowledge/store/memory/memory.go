// Package memory is an in-process knowledge.Store: an inverted index per
// lexical field scored with BM25 exactly like SQLite FTS5, typed adjacency maps with a per-source
// edge index, and a catalog map, all behind one RWMutex. Nothing persists;
// the index is rebuilt from the source files on start.
package memory

import (
	"cmp"
	"context"
	"fmt"
	"iter"
	"math"
	"slices"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/internal/storeutil"
	"github.com/itohio/EasyRobot/x/knowledge/token"
)

// BM25 parameters.
const (
	k1 = 1.2
	b  = 0.75
)

const numFields = storeutil.NumFields

// termFreq holds the frequency of one term in each lexical field.
type termFreq [numFields]int32

type docRec struct {
	doc    knowledge.Doc
	chunks []knowledge.ChunkID // ordered by Ord, then ID
}

type chunkRec struct {
	chunk knowledge.Chunk
	lens  [numFields]int32 // token count per field
	terms []string         // unique terms, to unindex
}

type edgeKey struct {
	from, to knowledge.NodeID
	kind     knowledge.EdgeKind
	source   knowledge.DocID
}

func (k edgeKey) edge(w float32) knowledge.Edge {
	return knowledge.Edge{From: k.from, To: k.to, Kind: k.kind, Weight: w, Source: k.source}
}

type keySet = map[edgeKey]struct{}

// Store is an in-memory knowledge.Store. It is safe for concurrent use;
// writers are serialised and readers never observe a partial write.
type Store struct {
	mu     sync.RWMutex
	closed bool
	gen    uint64

	docs   map[knowledge.DocID]*docRec
	chunks map[knowledge.ChunkID]*chunkRec

	postings map[string]map[knowledge.ChunkID]termFreq
	totalLen int64 // tokens over all fields of all chunks

	edges    map[edgeKey]float32
	out      map[knowledge.NodeID]keySet
	in       map[knowledge.NodeID]keySet
	bySource map[knowledge.DocID]keySet
}

var _ knowledge.Store = (*Store)(nil)

// New returns an empty store.
func New() *Store {
	return &Store{
		docs:     map[knowledge.DocID]*docRec{},
		chunks:   map[knowledge.ChunkID]*chunkRec{},
		postings: map[string]map[knowledge.ChunkID]termFreq{},
		edges:    map[edgeKey]float32{},
		out:      map[knowledge.NodeID]keySet{},
		in:       map[knowledge.NodeID]keySet{},
		bySource: map[knowledge.DocID]keySet{},
	}
}

// Close releases the index. Further calls return knowledge.ErrClosed;
// closing twice is harmless.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	s.docs, s.chunks, s.postings, s.edges = nil, nil, nil, nil
	s.out, s.in, s.bySource = nil, nil, nil
	return nil
}

func (s *Store) rlock(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.RLock()
	if s.closed {
		s.mu.RUnlock()
		return knowledge.ErrClosed
	}
	return nil
}

func (s *Store) lock(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return knowledge.ErrClosed
	}
	return nil
}

// --- Catalog ---

// GetDoc returns the catalog entry of id.
func (s *Store) GetDoc(ctx context.Context, id knowledge.DocID) (knowledge.Doc, bool, error) {
	if err := s.rlock(ctx); err != nil {
		return knowledge.Doc{}, false, err
	}
	defer s.mu.RUnlock()
	r, ok := s.docs[id]
	if !ok {
		return knowledge.Doc{}, false, nil
	}
	return storeutil.CloneDoc(r.doc), true, nil
}

// ListDocs yields a snapshot of all documents sorted by ID. The lock is
// not held while yielding, so the loop body may call the store.
func (s *Store) ListDocs(ctx context.Context) iter.Seq2[knowledge.Doc, error] {
	return func(yield func(knowledge.Doc, error) bool) {
		if err := s.rlock(ctx); err != nil {
			yield(knowledge.Doc{}, err)
			return
		}
		docs := make([]knowledge.Doc, 0, len(s.docs))
		for _, r := range s.docs {
			docs = append(docs, storeutil.CloneDoc(r.doc))
		}
		s.mu.RUnlock()
		slices.SortFunc(docs, func(a, b knowledge.Doc) int { return cmp.Compare(a.ID, b.ID) })
		for _, d := range docs {
			if !yield(d, nil) {
				return
			}
		}
	}
}

// Generation returns the number of committed writes since New.
func (s *Store) Generation(ctx context.Context) (uint64, error) {
	if err := s.rlock(ctx); err != nil {
		return 0, err
	}
	defer s.mu.RUnlock()
	return s.gen, nil
}

// --- LexicalIndex ---

// GetChunk returns the chunk with id.
func (s *Store) GetChunk(ctx context.Context, id knowledge.ChunkID) (knowledge.Chunk, bool, error) {
	if err := s.rlock(ctx); err != nil {
		return knowledge.Chunk{}, false, err
	}
	defer s.mu.RUnlock()
	r, ok := s.chunks[id]
	if !ok {
		return knowledge.Chunk{}, false, nil
	}
	return storeutil.CloneChunk(r.chunk), true, nil
}

// DocChunks returns the chunks of doc ordered by Ord.
func (s *Store) DocChunks(ctx context.Context, doc knowledge.DocID) ([]knowledge.Chunk, error) {
	if err := s.rlock(ctx); err != nil {
		return nil, err
	}
	defer s.mu.RUnlock()
	r, ok := s.docs[doc]
	if !ok {
		return nil, nil
	}
	out := make([]knowledge.Chunk, len(r.chunks))
	for i, id := range r.chunks {
		out[i] = storeutil.CloneChunk(s.chunks[id].chunk)
	}
	return out, nil
}

type candidate struct {
	chunk knowledge.ChunkID
	doc   knowledge.DocID
	ord   int
	score float64
}

// SearchLex ranks chunks containing any query term (OR semantics) by BM25
// over the fields heading, symbol, path and body, computed exactly like
// SQLite FTS5 bm25() with column weights. Filter f is applied before
// ranking; k <= 0 or a query without valid terms yields no hits. Ties are
// broken by DocID, then Ord.
//
// For the N indexed chunks, query term t and chunk d:
//
//	idf(t)   = ln((N − n(t) + 0.5) / (n(t) + 0.5)), or 1e-6 if that is ≤ 0   n: chunks containing t
//	tf(t,d)  = Σ_f w_f · tf_f(t,d)                    w = heading 3, symbol 3, path 2, body 1
//	L(d)     = tokens of d over all fields; avgL = mean L over all chunks
//	score(d) = Σ_t idf(t) · tf(t,d) · (k1 + 1) / (tf(t,d) + k1 · (1 − b + b · L(d) / avgL))
//
// with k1 = 1.2, b = 0.75.
func (s *Store) SearchLex(ctx context.Context, q knowledge.LexQuery, k int, f knowledge.Filter) ([]knowledge.ChunkHit, error) {
	if err := s.rlock(ctx); err != nil {
		return nil, err
	}
	defer s.mu.RUnlock()
	terms := storeutil.QueryTerms(q.Terms)
	if k <= 0 || len(terms) == 0 {
		return nil, nil
	}

	n := float64(len(s.chunks))
	avgL := float64(s.totalLen) / n
	pass := map[knowledge.DocID]bool{}
	at := map[knowledge.ChunkID]int{} // index into cands
	var cands []candidate
	for _, t := range terms {
		post := s.postings[t]
		if len(post) == 0 {
			continue
		}
		df := float64(len(post))
		idf := math.Log((n - df + 0.5) / (df + 0.5))
		if idf <= 0 {
			idf = 1e-6 // FTS5 floor
		}
		for id, tf := range post {
			r := s.chunks[id]
			ok, seen := pass[r.chunk.Doc]
			if !seen {
				ok = f.Match(s.docs[r.chunk.Doc].doc)
				pass[r.chunk.Doc] = ok
			}
			if !ok {
				continue
			}
			var x float64
			var rowLen int32
			for fi, c := range tf {
				x += storeutil.Weights[fi] * float64(c)
				rowLen += r.lens[fi]
			}
			i, seen := at[id]
			if !seen {
				i = len(cands)
				at[id] = i
				cands = append(cands, candidate{chunk: id, doc: r.chunk.Doc, ord: r.chunk.Ord})
			}
			cands[i].score += idf * x * (k1 + 1) / (x + k1*(1-b+b*float64(rowLen)/avgL))
		}
	}
	if len(cands) == 0 {
		return nil, nil
	}
	slices.SortFunc(cands, func(a, b candidate) int {
		if c := cmp.Compare(b.score, a.score); c != 0 {
			return c
		}
		if c := cmp.Compare(a.doc, b.doc); c != 0 {
			return c
		}
		if c := cmp.Compare(a.ord, b.ord); c != 0 {
			return c
		}
		return slices.Compare(a.chunk[:], b.chunk[:])
	})
	cands = cands[:min(k, len(cands))]
	hits := make([]knowledge.ChunkHit, len(cands))
	for i, c := range cands {
		var m []string
		for _, t := range terms {
			if _, ok := s.postings[t][c.chunk]; ok {
				m = append(m, t)
			}
		}
		hits[i] = knowledge.ChunkHit{Chunk: c.chunk, Doc: c.doc, Score: c.score, Matched: m}
	}
	return hits, nil
}

// --- GraphStore ---

// Neighbors returns edges touching any seed, of the given kinds (all when
// empty), at most limit per seed (no limit when <= 0). Each seed's edges
// are ordered by Weight desc, then the other endpoint, Kind, Source; seeds
// follow input order and an edge already reported for an earlier seed is
// skipped without counting towards the limit.
func (s *Store) Neighbors(ctx context.Context, from []knowledge.NodeID, kinds []knowledge.EdgeKind, dir knowledge.Direction, limit int) ([]knowledge.Edge, error) {
	if err := s.rlock(ctx); err != nil {
		return nil, err
	}
	defer s.mu.RUnlock()
	if dir != knowledge.Out && dir != knowledge.In && dir != knowledge.Both {
		return nil, fmt.Errorf("memory: neighbors: invalid direction %d", dir)
	}
	var out []knowledge.Edge
	reported := map[edgeKey]struct{}{}
	var buf []knowledge.Edge
	for _, seed := range storeutil.UniqueNodes(from) {
		buf = buf[:0]
		if dir == knowledge.Out || dir == knowledge.Both {
			for k := range s.out[seed] {
				if storeutil.KindIn(kinds, k.kind) {
					buf = append(buf, k.edge(s.edges[k]))
				}
			}
		}
		if dir == knowledge.In || dir == knowledge.Both {
			for k := range s.in[seed] {
				if storeutil.KindIn(kinds, k.kind) && !(dir == knowledge.Both && k.from == seed) {
					buf = append(buf, k.edge(s.edges[k]))
				}
			}
		}
		slices.SortFunc(buf, storeutil.CompareEdges(seed))
		taken := 0
		for _, e := range buf {
			if limit > 0 && taken == limit {
				break
			}
			k := edgeKey{e.From, e.To, e.Kind, e.Source}
			if _, dup := reported[k]; dup {
				continue
			}
			reported[k] = struct{}{}
			out = append(out, e)
			taken++
		}
	}
	return out, nil
}

// Degree counts edges touching n in either direction; a self-loop counts
// once.
func (s *Store) Degree(ctx context.Context, n knowledge.NodeID) (int, error) {
	if err := s.rlock(ctx); err != nil {
		return 0, err
	}
	defer s.mu.RUnlock()
	d := len(s.out[n])
	for k := range s.in[n] {
		if k.from != n {
			d++
		}
	}
	return d, nil
}

// Resolve maps a doc: node to itself when the document is indexed, and any
// other node to the smallest DocID asserting a Defines edge to it.
func (s *Store) Resolve(ctx context.Context, n knowledge.NodeID) (knowledge.DocID, bool, error) {
	if err := s.rlock(ctx); err != nil {
		return "", false, err
	}
	defer s.mu.RUnlock()
	if n.Kind() == knowledge.NodeDoc {
		id := knowledge.DocID(n.Value())
		_, ok := s.docs[id]
		if !ok {
			return "", false, nil
		}
		return id, true, nil
	}
	var best knowledge.DocID
	found := false
	for k := range s.in[n] {
		if k.kind == knowledge.Defines && (!found || k.source < best) {
			best, found = k.source, true
		}
	}
	return best, found, nil
}

// --- Writer ---

// PutDoc replaces the document, all its chunks and all edges it asserts.
func (s *Store) PutDoc(ctx context.Context, in knowledge.DocUpdate) error {
	u, err := storeutil.Normalize(in)
	if err != nil {
		return fmt.Errorf("memory: put %q: %w", in.Doc.ID, err)
	}
	if err := s.lock(ctx); err != nil {
		return err
	}
	defer s.mu.Unlock()
	id := u.Doc.ID
	for _, c := range u.Chunks {
		if r, ok := s.chunks[c.ID]; ok && r.chunk.Doc != id {
			return fmt.Errorf("memory: put %q: chunk id %s already belongs to %q", id, c.ID, r.chunk.Doc)
		}
	}
	s.removeDoc(id)
	slices.SortFunc(u.Chunks, func(a, b knowledge.Chunk) int {
		if c := cmp.Compare(a.Ord, b.Ord); c != 0 {
			return c
		}
		return slices.Compare(a.ID[:], b.ID[:])
	})
	rec := &docRec{doc: u.Doc, chunks: make([]knowledge.ChunkID, len(u.Chunks))}
	for i, c := range u.Chunks {
		s.indexChunk(c)
		rec.chunks[i] = c.ID
	}
	s.docs[id] = rec
	for _, e := range u.Edges {
		s.addEdge(edgeKey{e.From, e.To, e.Kind, e.Source}, e.Weight)
	}
	s.gen++
	return nil
}

// DeleteDoc removes the document, its chunks and the edges it asserts;
// edges other documents assert towards it stay. Deleting an unknown
// document is a successful no-op write.
func (s *Store) DeleteDoc(ctx context.Context, id knowledge.DocID) error {
	if err := s.lock(ctx); err != nil {
		return err
	}
	defer s.mu.Unlock()
	s.removeDoc(id)
	s.gen++
	return nil
}

// MoveDoc re-keys document from as to.ID and stores the catalog entry to.
// Chunks move with it and their ids are re-derived from (to.ID, Ord, Text);
// the path field is re-indexed; edges it asserts change Source; every edge
// endpoint doc:from becomes doc:to.ID. A document already stored at to.ID
// is replaced. Edges that collide after re-keying merge, keeping the
// highest weight. If two re-keyed chunk ids coincide, or one equals a chunk
// id of a third document, MoveDoc fails and the store is unchanged.
func (s *Store) MoveDoc(ctx context.Context, from knowledge.DocID, to knowledge.Doc) error {
	if to.ID == "" {
		return fmt.Errorf("memory: move %q: empty target id", from)
	}
	if err := s.lock(ctx); err != nil {
		return err
	}
	defer s.mu.Unlock()
	rec, ok := s.docs[from]
	if !ok {
		return fmt.Errorf("memory: move %q: %w", from, storeutil.ErrNotFound)
	}
	to = storeutil.CloneDoc(to)
	if to.ID == from {
		rec.doc = to
		s.gen++
		return nil
	}
	newIDs := make([]knowledge.ChunkID, len(rec.chunks))
	seen := make(map[knowledge.ChunkID]bool, len(rec.chunks))
	for i, id := range rec.chunks {
		c := s.chunks[id].chunk
		nid := knowledge.NewChunkID(to.ID, c.Ord, c.Text)
		if seen[nid] {
			return fmt.Errorf("memory: move %q to %q: chunks at ord %d re-key to the same id %s", from, to.ID, c.Ord, nid)
		}
		seen[nid] = true
		if o, ok := s.chunks[nid]; ok && o.chunk.Doc != from && o.chunk.Doc != to.ID {
			return fmt.Errorf("memory: move %q to %q: chunk %d re-keys to id %s of %q", from, to.ID, c.Ord, nid, o.chunk.Doc)
		}
		newIDs[i] = nid
	}
	if _, exists := s.docs[to.ID]; exists {
		s.removeDoc(to.ID)
	}

	// Unindex all first: a new id may equal another moved chunk's old id.
	old := make([]knowledge.Chunk, len(rec.chunks))
	for i, id := range rec.chunks {
		r := s.chunks[id]
		old[i] = r.chunk
		s.unindexChunk(r)
	}
	for i, c := range old {
		s.indexChunk(storeutil.RekeyChunk(c, to.ID))
		rec.chunks[i] = newIDs[i]
	}

	oldNode, newNode := knowledge.DocNode(from), knowledge.DocNode(to.ID)
	moved := map[edgeKey]float32{}
	for _, set := range []keySet{s.bySource[from], s.out[oldNode], s.in[oldNode]} {
		for k := range set {
			moved[k] = s.edges[k]
		}
	}
	for k := range moved {
		s.removeEdge(k)
	}
	for k, w := range moved {
		if k.source == from {
			k.source = to.ID
		}
		k.from = storeutil.RekeyNode(k.from, oldNode, newNode)
		k.to = storeutil.RekeyNode(k.to, oldNode, newNode)
		s.addEdge(k, w)
	}

	delete(s.docs, from)
	rec.doc = to
	s.docs[to.ID] = rec
	s.gen++
	return nil
}

// TouchDoc records a new mtime for an unchanged document.
func (s *Store) TouchDoc(ctx context.Context, id knowledge.DocID, mtime time.Time) error {
	if err := s.lock(ctx); err != nil {
		return err
	}
	defer s.mu.Unlock()
	rec, ok := s.docs[id]
	if !ok {
		return fmt.Errorf("memory: touch %q: %w", id, storeutil.ErrNotFound)
	}
	rec.doc.MTime = mtime
	s.gen++
	return nil
}

// --- internals; callers hold the write lock ---

func (s *Store) removeDoc(id knowledge.DocID) {
	rec, ok := s.docs[id]
	if !ok {
		return
	}
	for _, cid := range rec.chunks {
		s.unindexChunk(s.chunks[cid])
	}
	for k := range s.bySource[id] {
		s.removeEdge(k)
	}
	delete(s.docs, id)
}

func (s *Store) indexChunk(c knowledge.Chunk) {
	r := &chunkRec{chunk: c}
	tfs := map[string]termFreq{}
	for fi, text := range storeutil.FieldTexts(c) {
		toks := token.Tokens(text)
		r.lens[fi] = int32(len(toks))
		for _, t := range toks {
			tf := tfs[t]
			tf[fi]++
			tfs[t] = tf
		}
		s.totalLen += int64(len(toks))
	}
	r.terms = make([]string, 0, len(tfs))
	for t, tf := range tfs {
		post := s.postings[t]
		if post == nil {
			post = map[knowledge.ChunkID]termFreq{}
			s.postings[t] = post
		}
		post[c.ID] = tf
		r.terms = append(r.terms, t)
	}
	s.chunks[c.ID] = r
}

func (s *Store) unindexChunk(r *chunkRec) {
	id := r.chunk.ID
	for _, t := range r.terms {
		post := s.postings[t]
		delete(post, id)
		if len(post) == 0 {
			delete(s.postings, t)
		}
	}
	for _, l := range r.lens {
		s.totalLen -= int64(l)
	}
	delete(s.chunks, id)
}

func addKey[K comparable](m map[K]keySet, at K, k edgeKey) {
	set := m[at]
	if set == nil {
		set = keySet{}
		m[at] = set
	}
	set[k] = struct{}{}
}

func delKey[K comparable](m map[K]keySet, at K, k edgeKey) {
	set := m[at]
	delete(set, k)
	if len(set) == 0 {
		delete(m, at)
	}
}

func (s *Store) addEdge(k edgeKey, w float32) {
	if old, ok := s.edges[k]; ok {
		s.edges[k] = max(old, w)
		return
	}
	s.edges[k] = w
	addKey(s.out, k.from, k)
	addKey(s.in, k.to, k)
	addKey(s.bySource, k.source, k)
}

func (s *Store) removeEdge(k edgeKey) {
	if _, ok := s.edges[k]; !ok {
		return
	}
	delete(s.edges, k)
	delKey(s.out, k.from, k)
	delKey(s.in, k.to, k)
	delKey(s.bySource, k.source, k)
}

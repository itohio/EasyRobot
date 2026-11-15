// Package retrieve implements knowledge.Retriever over any knowledge.Store:
// lexical (BM25, scored relative to the best hit) and exact (id, symbol,
// file name) candidates, expanded one hop over the link graph and two hops
// through shared ids, aggregated per file with priors (weight, relative
// recency, status, supersedes), diversified per folder, and returned with
// line spans, snippets and reasons.
package retrieve

import (
	"cmp"
	"context"
	"fmt"
	"path"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/token"
)

const (
	defaultK          = 8
	maxK              = 30
	symbolPackageDocs = 20  // lexical hits whose packages qualify bare symbols
	secondChunkWeight = 0.3 // weight of a file's second-best chunk
	maxSections       = 3
	snippetRunes      = 160
	recencyGain       = 0.1 // recency prior span without recency intent
	recencyIntentGain = 0.3 // … when the query asks for the latest state
	recencyPool       = 20  // top dated candidates recency is ranked against
	statusPenalty     = 0.8
	maxTryTerms       = 5
	tryTermHits       = 3
	highMargin        = 1.5
	highCoverage      = 0.6
	lowCoverage       = 0.3
)

// Confidence levels reported in knowledge.Result.
const (
	High   = "high"
	Medium = "medium"
	Low    = "low"
)

// staleStatuses are Props["status"] values that demote a document.
var staleStatuses = []string{"done", "superseded", "archived", "deprecated"}

// recencyWords mark a query asking for the latest state of something.
var recencyWords = map[string]bool{
	"latest": true, "current": true, "currently": true, "newest": true, "recent": true, "recently": true,
	"now": true, "today": true, "supersede": true, "supersedes": true, "superseded": true,
}

// fileDate finds a YYYY-MM-DD or YYYY-MM date in a file name.
var fileDate = regexp.MustCompile(`(19|20)\d\d-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?`)

// Retriever answers queries against a knowledge.Store. It is safe for
// concurrent use.
type Retriever struct {
	store        knowledge.Store
	beta         float64
	candidates   int
	graphSeeds   int
	neighbors    int
	maxPerFolder int

	mu  sync.Mutex
	cat *catalog
}

var _ knowledge.Retriever = (*Retriever)(nil)

// Option configures a Retriever.
type Option func(*Retriever)

// WithBeta sets the graph boost factor β (default 0.3).
func WithBeta(beta float64) Option { return func(r *Retriever) { r.beta = beta } }

// WithCandidates sets how many lexical chunk hits are fused (default 100).
func WithCandidates(n int) Option { return func(r *Retriever) { r.candidates = n } }

// WithGraphSeeds sets how many top fused documents seed graph expansion
// (default 20).
func WithGraphSeeds(n int) Option { return func(r *Retriever) { r.graphSeeds = n } }

// WithNeighbors caps the graph neighbours boosted per seed (default 10).
func WithNeighbors(n int) Option { return func(r *Retriever) { r.neighbors = n } }

// WithMaxPerFolder caps hits from one parent directory (default 3).
func WithMaxPerFolder(n int) Option { return func(r *Retriever) { r.maxPerFolder = n } }

// New returns a Retriever over store.
func New(store knowledge.Store, opts ...Option) *Retriever {
	r := &Retriever{
		store:        store,
		beta:         0.3,
		candidates:   100,
		graphSeeds:   20,
		neighbors:    10,
		maxPerFolder: 3,
	}
	for _, o := range opts {
		o(r)
	}
	if r.candidates < 1 {
		r.candidates = 1
	}
	if r.maxPerFolder < 1 {
		r.maxPerFolder = 1
	}
	return r
}

// catalog is per-generation catalog data: the document count and the
// lower-cased base file name index used for exact file-name hits.
type catalog struct {
	gen    uint64
	docs   int
	byBase map[string][]knowledge.DocID
}

func (r *Retriever) catalog(ctx context.Context, gen uint64) (*catalog, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.cat != nil && r.cat.gen == gen {
		return r.cat, nil
	}
	c := &catalog{gen: gen, byBase: map[string][]knowledge.DocID{}}
	for d, err := range r.store.ListDocs(ctx) {
		if err != nil {
			return nil, fmt.Errorf("retrieve: list docs: %w", err)
		}
		c.docs++
		base := strings.ToLower(path.Base(string(d.ID)))
		c.byBase[base] = append(c.byBase[base], d.ID)
	}
	for _, ids := range c.byBase {
		slices.Sort(ids)
	}
	r.cat = c
	return c, nil
}

// Find answers q: lexical chunk hits scored relative to the best BM25 hit,
// exact candidates (id, symbol, file name) at the top chunk score, both
// filtered inside the store; graph boosts from the top documents and
// q.Near; per-file aggregation (best + 0.3 × second chunk + boost, ×
// Weight × relative recency × status/supersedes); exact hits first; folder
// diversity; then sections and confidence. An empty query without Near
// yields no hits.
func (r *Retriever) Find(ctx context.Context, q knowledge.Query) (knowledge.Result, error) {
	if err := ctx.Err(); err != nil {
		return knowledge.Result{}, err
	}
	gen, err := r.store.Generation(ctx)
	if err != nil {
		return knowledge.Result{}, fmt.Errorf("retrieve: generation: %w", err)
	}
	cat, err := r.catalog(ctx, gen)
	if err != nil {
		return knowledge.Result{}, err
	}
	res := knowledge.Result{Confidence: Low, Generation: gen, Docs: cat.docs}

	terms := token.Query(q.Text)
	if len(terms) == 0 && len(q.Near) == 0 {
		return res, nil
	}
	f := q.Filter
	f.Not = append(slices.Clone(q.Filter.Not), q.Near...)
	run := &run{
		r:       r,
		ctx:     ctx,
		q:       q,
		f:       f,
		terms:   terms,
		cat:     cat,
		cands:   map[knowledge.DocID]*cand{},
		chunks:  map[knowledge.ChunkID]*fusedChunk{},
		docs:    map[knowledge.DocID]docEntry{},
		docChs:  map[knowledge.DocID][]knowledge.Chunk{},
		degrees: map[knowledge.DocID]int{},
		near:    map[knowledge.DocID]bool{},
	}
	for _, id := range q.Near {
		run.near[id] = true
	}
	hits, err := run.find()
	if err != nil {
		return knowledge.Result{}, err
	}
	res.Hits = hits
	res.Confidence, res.TryTerms = run.confidence(hits)
	return res, nil
}

// cand is one candidate file.
type cand struct {
	id       knowledge.DocID
	doc      knowledge.Doc
	chunks   []*fusedChunk // fused chunks, best first after fuse()
	fused    float64       // best chunk + 0.3 × second
	bm25Rank int           // best lexical rank, 1-based; 0 when none
	exact    []string      // exact-match reasons
	boost    float64       // graph boost
	graphWhy []string
	weight   float64
	recency  float64
	status   float64
	// supersededBy lists candidates whose Props["supersedes"] name this one.
	supersededBy []knowledge.DocID
	score        float64
	sections     []section       // rendered sections (selected hits only)
	covered      map[string]bool // query terms matched by any of its chunks
}

// fusedChunk is one candidate chunk with its score in [0, 1].
type fusedChunk struct {
	id      knowledge.ChunkID
	doc     knowledge.DocID
	seq     int // first-seen order, for deterministic ties
	score   float64
	matched []string // nil when the chunk was not a lexical hit
}

type docEntry struct {
	doc knowledge.Doc
	ok  bool
}

// run holds the state of one Find call.
type run struct {
	r     *Retriever
	ctx   context.Context
	q     knowledge.Query
	f     knowledge.Filter
	terms []string
	cat   *catalog

	cands   map[knowledge.DocID]*cand
	chunks  map[knowledge.ChunkID]*fusedChunk
	docs    map[knowledge.DocID]docEntry
	docChs  map[knowledge.DocID][]knowledge.Chunk
	degrees map[knowledge.DocID]int
	near    map[knowledge.DocID]bool

	selected []*cand // candidates behind the returned hits, in order
}

func (x *run) find() ([]knowledge.Hit, error) {
	if err := x.lexical(); err != nil {
		return nil, err
	}
	if err := x.exact(); err != nil {
		return nil, err
	}
	x.fuse()
	if err := x.ctx.Err(); err != nil {
		return nil, err
	}
	if err := x.expand(); err != nil {
		return nil, err
	}
	ranked, err := x.aggregate()
	if err != nil {
		return nil, err
	}
	return x.hits(x.diversify(ranked))
}

func (x *run) getDoc(id knowledge.DocID) (knowledge.Doc, bool, error) {
	if e, ok := x.docs[id]; ok {
		return e.doc, e.ok, nil
	}
	d, ok, err := x.r.store.GetDoc(x.ctx, id)
	if err != nil {
		return knowledge.Doc{}, false, fmt.Errorf("retrieve: get doc %s: %w", id, err)
	}
	x.docs[id] = docEntry{d, ok}
	return d, ok, nil
}

// admit reports whether id is an indexed document passing the filter.
func (x *run) admit(id knowledge.DocID) (bool, error) {
	d, ok, err := x.getDoc(id)
	if err != nil || !ok {
		return false, err
	}
	return x.f.Match(d), nil
}

func (x *run) docChunks(id knowledge.DocID) ([]knowledge.Chunk, error) {
	if cs, ok := x.docChs[id]; ok {
		return cs, nil
	}
	cs, err := x.r.store.DocChunks(x.ctx, id)
	if err != nil {
		return nil, fmt.Errorf("retrieve: chunks of %s: %w", id, err)
	}
	cs = slices.Clone(cs)
	slices.SortFunc(cs, func(a, b knowledge.Chunk) int { return cmp.Compare(a.Ord, b.Ord) })
	x.docChs[id] = cs
	return cs, nil
}

func (x *run) cand(id knowledge.DocID) *cand {
	c, ok := x.cands[id]
	if !ok {
		c = &cand{id: id}
		x.cands[id] = c
	}
	return c
}

// addChunk records a chunk score in [0, 1], keeping the highest seen.
func (x *run) addChunk(id knowledge.ChunkID, doc knowledge.DocID, score float64, matched []string) {
	fc, ok := x.chunks[id]
	if !ok {
		fc = &fusedChunk{id: id, doc: doc, seq: len(x.chunks)}
		x.chunks[id] = fc
		c := x.cand(doc)
		c.chunks = append(c.chunks, fc)
	}
	fc.score = max(fc.score, score)
	if matched != nil && fc.matched == nil {
		fc.matched = matched
	}
}

// lexical adds the store's BM25 hits, scored relative to the best hit
// (rank 1 = 1.0) so score gaps survive, unlike rank fusion which maps
// rank 1 and rank 10 to 1/61 and 1/70.
func (x *run) lexical() error {
	if len(x.terms) == 0 {
		return nil
	}
	hits, err := x.r.store.SearchLex(x.ctx, knowledge.LexQuery{Terms: x.terms}, x.r.candidates, x.f)
	if err != nil {
		return fmt.Errorf("retrieve: lexical search: %w", err)
	}
	if len(hits) == 0 {
		return nil
	}
	top := hits[0].Score
	for i, h := range hits {
		score := 1 / float64(i+1) // stores without positive scores: rank only
		if top > 0 {
			score = max(h.Score, 0) / top
		}
		x.addChunk(h.Chunk, h.Doc, score, slices.Clone(h.Matched))
		if c := x.cand(h.Doc); c.bm25Rank == 0 {
			c.bm25Rank = i + 1
		}
	}
	return nil
}

// exactHit is a document named directly by the query.
type exactHit struct {
	doc    knowledge.DocID
	reason string
	symbol string // set for symbol hits: the chunk declaring it is preferred
}

// exact resolves ids (Resolve(IDNode)), identifier-looking words
// (Resolve(SymNode); a bare name such as "ClientPermissions" is also tried as
// "<pkg>.ClientPermissions" for the packages of the top lexical Go hits) and
// words equal to a document's base file name. Docs failing the filter or
// listed in Near are dropped.
func (x *run) exact() error {
	var found []exactHit
	resolve := func(n knowledge.NodeID, reason, sym string) (bool, error) {
		id, ok, err := x.r.store.Resolve(x.ctx, n)
		if err != nil {
			return false, fmt.Errorf("retrieve: resolve %s: %w", n, err)
		}
		if ok {
			found = append(found, exactHit{doc: id, reason: reason, symbol: sym})
		}
		return ok, nil
	}
	for _, id := range token.IDs(x.q.Text) {
		if _, err := resolve(knowledge.IDNode(id), "exact id "+id, ""); err != nil {
			return err
		}
	}
	words := queryWords(x.q.Text)
	var pkgs []string // lazily: packages of the top lexical Go hits
	for _, w := range words {
		if !identLike(w) {
			continue
		}
		ok, err := resolve(knowledge.SymNode(w), "defines "+w, w)
		if err != nil {
			return err
		}
		// Only mixed-case names (ClientPermissions) are package-qualified:
		// acronyms (NATS, LD06) are usually prose, and qualifying them with
		// whatever packages matched lexically invents exact hits.
		if ok || strings.Contains(w, ".") || !mixedCase(w) {
			continue
		}
		if pkgs == nil {
			if pkgs, err = x.lexicalPackages(); err != nil {
				return err
			}
		}
		for _, p := range pkgs {
			sym := p + "." + w
			if ok, err := resolve(knowledge.SymNode(sym), "defines "+sym, sym); err != nil {
				return err
			} else if ok {
				break
			}
		}
	}
	for _, w := range words {
		if strings.Contains(w, ".") {
			for _, id := range x.cat.byBase[strings.ToLower(w)] {
				found = append(found, exactHit{doc: id, reason: "file name " + w})
			}
		}
	}

	for _, e := range found {
		if x.near[e.doc] {
			continue
		}
		ok, err := x.admit(e.doc)
		if err != nil {
			return err
		}
		if !ok {
			continue
		}
		c := x.cand(e.doc)
		if len(c.exact) > 0 {
			if !slices.Contains(c.exact, e.reason) {
				c.exact = append(c.exact, e.reason)
			}
			continue
		}
		c.exact = append(c.exact, e.reason)
		chunk, ok, err := x.exactChunk(c, e.symbol)
		if err != nil {
			return err
		}
		if ok {
			x.addChunk(chunk, e.doc, 1, nil) // as good as the top lexical chunk
		}
	}
	return nil
}

// exactChunk picks the chunk an exact doc contributes: the chunk declaring
// the symbol, else its best lexical chunk, else its first chunk.
func (x *run) exactChunk(c *cand, symbol string) (knowledge.ChunkID, bool, error) {
	if symbol == "" && len(c.chunks) > 0 {
		return c.chunks[0].id, true, nil
	}
	cs, err := x.docChunks(c.id)
	if err != nil {
		return knowledge.ChunkID{}, false, err
	}
	for _, ch := range cs {
		if symbol != "" && ch.Symbol == symbol {
			return ch.ID, true, nil
		}
	}
	if len(c.chunks) > 0 {
		return c.chunks[0].id, true, nil
	}
	if len(cs) > 0 {
		return cs[0].ID, true, nil
	}
	return knowledge.ChunkID{}, false, nil
}

// lexicalPackages returns the distinct Go packages (Props["package"]) of
// the top symbolPackageDocs lexical candidates, in lexical rank order.
func (x *run) lexicalPackages() ([]string, error) {
	ranked := make([]*cand, 0, len(x.cands))
	for _, c := range x.cands {
		if c.bm25Rank > 0 {
			ranked = append(ranked, c)
		}
	}
	slices.SortFunc(ranked, func(a, b *cand) int { return cmp.Compare(a.bm25Rank, b.bm25Rank) })
	pkgs := []string{}
	for _, c := range ranked[:min(len(ranked), symbolPackageDocs)] {
		d, ok, err := x.getDoc(c.id)
		if err != nil {
			return nil, err
		}
		if p := d.Props["package"]; ok && p != "" && !slices.Contains(pkgs, p) {
			pkgs = append(pkgs, p)
		}
	}
	return pkgs, nil
}

// fuse orders each candidate's chunks and computes its fused doc score.
func (x *run) fuse() {
	for _, c := range x.cands {
		slices.SortFunc(c.chunks, func(a, b *fusedChunk) int {
			return cmp.Or(cmp.Compare(b.score, a.score), cmp.Compare(a.seq, b.seq))
		})
		if len(c.chunks) > 0 {
			c.fused = c.chunks[0].score
		}
		if len(c.chunks) > 1 {
			c.fused += secondChunkWeight * c.chunks[1].score
		}
	}
}

// sortedCands returns candidates by descending key, then DocID.
func (x *run) sortedCands(key func(*cand) float64) []*cand {
	out := make([]*cand, 0, len(x.cands))
	for _, c := range x.cands {
		out = append(out, c)
	}
	slices.SortFunc(out, func(a, b *cand) int {
		return cmp.Or(cmp.Compare(key(b), key(a)), cmp.Compare(a.id, b.id))
	})
	return out
}

// aggregate scores every admitted candidate and returns them best first.
func (x *run) aggregate() ([]*cand, error) {
	var out []*cand
	for _, c := range x.cands {
		if x.near[c.id] {
			continue
		}
		d, ok, err := x.getDoc(c.id)
		if err != nil {
			return nil, err
		}
		if !ok || !x.f.Match(d) {
			continue
		}
		c.doc = d
		c.weight = float64(d.Weight)
		if c.weight <= 0 {
			c.weight = 1
		}
		c.status = 1
		if slices.Contains(staleStatuses, strings.ToLower(strings.TrimSpace(d.Props["status"]))) {
			c.status = statusPenalty
		}
		out = append(out, c)
	}
	// Pre-prior order: recency is relative to the strongest candidates.
	slices.SortFunc(out, func(a, b *cand) int {
		return cmp.Or(cmp.Compare(b.fused+b.boost, a.fused+a.boost), cmp.Compare(a.id, b.id))
	})
	x.recency(out)
	supersede(out)
	maxPlain := 0.0
	for _, c := range out {
		c.score = (c.fused + c.boost) * c.weight * c.recency * c.status
		if len(c.exact) == 0 {
			maxPlain = max(maxPlain, c.score)
		}
	}
	// Exact hits outrank everything else.
	for _, c := range out {
		if len(c.exact) > 0 {
			c.score += 1 + maxPlain
		}
	}
	slices.SortFunc(out, func(a, b *cand) int {
		return cmp.Or(cmp.Compare(b.score, a.score), cmp.Compare(a.id, b.id))
	})
	return out, nil
}

// recency sets each candidate's recency prior 1 + gain·r, where r ∈ [0, 1]
// is the share of the top recencyPool dated candidates (by pre-prior score)
// that are strictly older. Ranking dates against the competing candidates,
// rather than decaying with absolute age, separates an old and a new
// version of a note however old both are. gain is recencyIntentGain when
// the query asks for the latest state, else recencyGain. Undated docs get 1.
func (x *run) recency(ranked []*cand) {
	gain := recencyGain
	for _, w := range queryWords(x.q.Text) {
		if recencyWords[strings.ToLower(w)] {
			gain = recencyIntentGain
			break
		}
	}
	dates := make([]time.Time, len(ranked))
	var pool []time.Time
	for i, c := range ranked {
		dates[i] = docDate(c.doc)
		if !dates[i].IsZero() && len(pool) < recencyPool {
			pool = append(pool, dates[i])
		}
	}
	slices.SortFunc(pool, time.Time.Compare)
	for i, c := range ranked {
		c.recency = 1
		if dates[i].IsZero() || len(pool) < 2 {
			continue
		}
		older, _ := slices.BinarySearchFunc(pool, dates[i], time.Time.Compare)
		c.recency = 1 + gain*min(float64(older)/float64(len(pool)-1), 1)
	}
}

// docDate is a document's date: Props "updated", else "created", else a
// YYYY-MM-DD or YYYY-MM in the file name, else zero (undated). MTime is
// deliberately ignored: it reflects checkout and touch times, not content.
func docDate(d knowledge.Doc) time.Time {
	for _, k := range [...]string{"updated", "created"} {
		if t, err := time.Parse(time.DateOnly, strings.TrimSpace(d.Props[k])); err == nil {
			return t
		}
	}
	if m := fileDate.FindString(path.Base(string(d.ID))); m != "" {
		if t, err := time.Parse(time.DateOnly, m); err == nil {
			return t
		}
		if t, err := time.Parse("2006-01", m[:7]); err == nil {
			return t
		}
	}
	return time.Time{}
}

// supersede demotes candidates named by another candidate's
// Props["supersedes"] (comma-separated base names, paths or [[links]] with
// optional |labels, extension optional), like a superseded status. A path
// naming a candidate's ID matches it anywhere; a bare name only matches
// candidates in the superseding doc's directory or scope (first path
// segment), so it cannot demote same-named files in other projects.
func supersede(cands []*cand) {
	byStem := map[string][]*cand{}
	for _, c := range cands {
		s := fileStem(string(c.id))
		byStem[s] = append(byStem[s], c)
	}
	for _, c := range cands {
		id := string(c.id)
		for name := range strings.SplitSeq(c.doc.Props["supersedes"], ",") {
			name = strings.TrimSpace(name)
			name = strings.TrimSuffix(strings.TrimPrefix(name, "[["), "]]")
			if i := strings.IndexByte(name, '|'); i >= 0 {
				name = name[:i]
			}
			name = strings.TrimSpace(strings.TrimSuffix(name, `\`))
			if name == "" {
				continue
			}
			ref := strings.ToLower(strings.TrimSuffix(name, path.Ext(name)))
			for _, old := range byStem[fileStem(name)] {
				oid := string(old.id)
				exact := strings.Contains(name, "/") && strings.ToLower(strings.TrimSuffix(oid, path.Ext(oid))) == ref
				near := path.Dir(oid) == path.Dir(id) || topSegment(oid) == topSegment(id)
				if old != c && (exact || near) {
					old.status = statusPenalty
					old.supersededBy = append(old.supersededBy, c.id)
				}
			}
		}
	}
}

// topSegment is the first path segment of p (its scope).
func topSegment(p string) string {
	s, _, _ := strings.Cut(p, "/")
	return s
}

// fileStem is the lower-case base name of p without its extension.
func fileStem(p string) string {
	b := path.Base(strings.ToLower(p))
	return strings.TrimSuffix(b, path.Ext(b))
}

// diversify keeps at most maxPerFolder hits per parent directory, unless
// a Filter.In prefix names that directory, and returns at most K.
func (x *run) diversify(ranked []*cand) []*cand {
	k := x.q.K
	if k <= 0 {
		k = defaultK
	}
	k = min(k, maxK)
	exempt := map[string]bool{}
	for _, p := range x.q.Filter.In {
		exempt[strings.TrimSuffix(p, "/")] = true
	}
	perDir := map[string]int{}
	var out []*cand
	for _, c := range ranked {
		if len(out) == k {
			break
		}
		dir := path.Dir(string(c.id))
		if !exempt[dir] && perDir[dir] >= x.r.maxPerFolder {
			continue
		}
		perDir[dir]++
		out = append(out, c)
	}
	return out
}

// queryWords splits s like the tokenizer does (letters, digits and inner
// connectors _ - .), keeping case; unique, in first-seen order.
func queryWords(s string) []string {
	fields := strings.FieldsFunc(s, func(r rune) bool { return !isWordRune(r) && !isConnector(r) })
	var out []string
	for _, f := range fields {
		f = strings.TrimFunc(f, isConnector)
		if f != "" && !slices.Contains(out, f) {
			out = append(out, f)
		}
	}
	return out
}

// identLike reports whether w looks like a code identifier: it contains a
// dot or underscore, or a capital after its first letter. Ids are not.
func identLike(w string) bool {
	if token.IsID(w) {
		return false
	}
	if strings.ContainsAny(w, "._") {
		return true
	}
	for i, r := range w {
		if i > 0 && unicode.IsUpper(r) {
			return true
		}
	}
	return false
}

// mixedCase reports whether w has a lower-case letter followed later by an
// upper-case one (ClientPermissions, newConn), unlike acronyms (NATS, LD06).
func mixedCase(w string) bool {
	lower := false
	for _, r := range w {
		switch {
		case unicode.IsLower(r):
			lower = true
		case lower && unicode.IsUpper(r):
			return true
		}
	}
	return false
}

// Package ingest synchronises a knowledge.Store with a folder of Markdown
// and Go files.
//
// A Sync walks the folder (stat only; link resolution needs the complete
// name index and move detection the complete set of missing documents),
// compares each accepted file with the catalog (size and full-precision
// mtime, then sha256), and runs changed files through a bounded pipeline:
// N workers read, hash, chunk and resolve links; a single writer applies
// PutDoc/TouchDoc/MoveDoc. A new file whose hash equals a missing catalog
// document is a move: MoveDoc re-keys it (and inbound edges), then it is
// re-chunked at its new location. Remaining missing documents are deleted.
// When anything was added, moved or deleted, documents with unresolved
// links (Props[PropMissing]) and documents linking into deleted documents
// are re-resolved.
//
// Racy files (as in git): a file whose mtime is no older than 2 s before
// the Sync started may be rewritten within the same mtime tick after it
// was stat'ed, so its size and mtime cannot prove it unchanged later. Such
// documents are written with Props[propRacy] ("_racy"); a later Sync
// hashes them even when size and mtime match, and rewrites them without
// the marker once their mtime has left the window (counted as Unchanged,
// or Touched when the mtime moved).
//
// Relink recovery: before the first add, move or delete of a Sync commits,
// the Sync puts an "incomplete sync" marker document (DocID
// "<Scope>/.ingest-pending", or ".ingest-pending-<hash of root>" without a
// Scope; no chunks, no edges, Kind "ingest") and deletes it once the relink
// pass has finished without error. A Sync that finds its marker (the
// previous one stopped between a structural commit and the end of its
// relink pass) re-resolves the links of every document, so links again
// match a rebuild.
//
// Concurrency: Syncs of one Ingestor are serialised; a Sync waits for a
// running one to finish (and then usually finds nothing to do).
//
// Links: wikilinks with a "/" are paths from the root or the document's
// directory (with or without ".md"), then path suffixes; others match base
// names case-insensitively, preferring the same Doc.Scope, then fewest path
// segments, then lexical order. Markdown links are relative (or "/"
// root-relative), may omit ".md", may name a directory with index.md or
// README.md, and are dropped when they escape the root. A Markdown link
// whose path does not exist but whose base name has an accepted extension
// falls back to base-name resolution among other documents (never the
// linking document itself; same preference order, as vault tools do for
// moved files); such edges weigh 0.5 instead of 1 and are counted in
// Props[PropFuzzy], not in Props[PropMissing]. Links to files with an
// extension no chunker accepts (images) and self-links are dropped; other
// unresolved links become missing: nodes. A moved document's former paths
// are kept in Props[PropMovedFrom] and resolve to it when no real file
// matches, so links written against the old name keep following it. When
// documents are added, moved or deleted, documents with missing or fallback
// links are re-resolved, so a fallback edge moves to an exact match once the
// path exists.
//
// Walk rules: the directories .git, node_modules, .venv, vendor,
// _quarantine, _tmp and every dot-directory are skipped, as are
// Source.Exclude entries, symlinks to directories, files over the size cap,
// binary files (NUL in the first 8 KiB), files no chunker accepts and
// secret-named files (.env*, *.pem, *.key, *.creds, *.nk, id_rsa*,
// id_ed25519*, *.p12, *.keystore). File symlinks are followed only when
// their resolved target lies inside the (resolved) root and its base name
// is not secret-named; they are indexed under the link's own path, so an
// in-root link duplicates its target's content. .gitignore files are
// honoured per directory for the simple subset: names, path.Match globs,
// trailing "/" for directories, leading "/" (or an inner "/") anchoring,
// leading "**/" (any depth); negation ("!") and "**" elsewhere are
// unsupported. A directory that cannot be read, or a file that cannot be
// stat'ed, is reported in Report.Errors and its catalogued documents are
// kept.
//
// Shared stores: a Sync only deletes or moves documents this source owns
// (DocID under Scope; with an empty Scope, Doc.Path under Root). Scopeless
// sources sharing a store must not have overlapping relative paths: a file
// whose DocID is catalogued by another source (whose file still exists) is
// skipped and reported in Report.Errors as a DocID collision.
//
// Moves are decided after the pipeline in a fixed order (new documents by
// path; a missing twin with the same base name first), so the outcome does
// not depend on worker timing. When documents are added or moved, the
// documents linking to a namesake (same base name, or a moved document's
// former name) are re-resolved too, so incremental links match a rebuild.
//
// Secrets: before chunking, secret lines are blanked in the whole source
// (line count kept, so spans stay correct): a PEM private key (RSA, EC,
// OpenSSH, encrypted, PGP key block, …) from BEGIN to END as one span, and
// lines with known token formats (AWS key ids and secret keys, GitHub,
// GitLab, Slack tokens and webhooks, Stripe live keys, Google API keys,
// sk- keys, NATS seeds, JWTs), URLs with a password, or key/password/token
// assignments whose value has a part mixing letters and digits. Props whose
// "key: value" matches are dropped; a matching title or chunk heading crumb
// is replaced by the file's base name. The stored hash and size are those
// of the file.
//
// Report fields: Skipped counts accepted files dropped for size or binary
// content; SecretsDropped counts secret-named files, redacted secret spans
// and dropped props, a replaced title and chunks with replaced heading
// crumbs; Changed also counts unchanged files whose links were re-resolved
// to different targets (they are then not counted as Unchanged/Touched).
package ingest

import (
	"bytes"
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/chunk/golang"
	"github.com/itohio/EasyRobot/x/knowledge/chunk/md"
)

// Props keys owned by the ingestor.
const (
	// PropMissing holds the number of unresolved links of a document
	// (absent when every link resolved).
	PropMissing = "_missing"
	// propStale marks a document moved by MoveDoc whose re-chunk at the new
	// location has not been committed yet; the next Sync re-chunks it.
	propStale = "_stale"
	// PropMovedFrom lists, newline-separated, the former root-relative
	// paths of a document detected as moved; links to an old path or name
	// keep resolving to it when no real file matches.
	PropMovedFrom = "_moved_from"
	// PropFuzzy holds the number of Markdown links resolved only by the
	// base-name fallback (absent when none); their edges weigh 0.5.
	PropFuzzy = "_fuzzy"
	// propRacy marks a document whose mtime was inside the racy window when
	// it was written; its size and mtime cannot prove it unchanged.
	propRacy = "_racy"
)

const (
	defaultMaxSize = 1 << 20
	binarySniff    = 8 << 10
	// racyWindow: a file whose mtime is no older than this before the Sync
	// started is racy.
	racyWindow = 2 * time.Second
	// pendingName is the base name of the incomplete-sync marker document.
	pendingName = ".ingest-pending"
	// kindPending is the Doc.Kind of the marker document.
	kindPending = "ingest"
)

// Source describes a folder to ingest.
type Source struct {
	// Root is the folder to walk.
	Root string
	// Scope prefixes every DocID and becomes Doc.Scope; when empty, DocIDs
	// are root-relative and Doc.Scope is the first path segment ("" for
	// files directly under Root).
	Scope string
	// Exclude lists base names (path.Match globs) or slash path prefixes
	// relative to Root that are never walked.
	Exclude []string
	// Weights multiplies Doc.Weight by the first rule whose Prefix matches
	// the DocID at a path-segment boundary.
	Weights []WeightRule
}

// WeightRule scales the weight of documents under a DocID prefix.
type WeightRule struct {
	Prefix string
	Weight float32
}

// Option configures an Ingestor.
type Option func(*Ingestor)

// WithChunkers replaces the default chunkers (Markdown and Go). The first
// chunker accepting a path handles it.
func WithChunkers(c ...knowledge.Chunker) Option {
	return func(i *Ingestor) {
		if len(c) > 0 {
			i.chunkers = slices.Clone(c)
		}
	}
}

// WithMaxSize sets the file size cap in bytes (default 1 MiB).
func WithMaxSize(n int64) Option {
	return func(i *Ingestor) {
		if n > 0 {
			i.maxSize = n
		}
	}
}

// WithWorkers sets the number of read/hash/chunk workers (default
// GOMAXPROCS).
func WithWorkers(n int) Option {
	return func(i *Ingestor) {
		if n > 0 {
			i.workers = n
		}
	}
}

// Ingestor synchronises a store with a Source. It implements
// knowledge.Ingestor. It is safe for concurrent use; Syncs are serialised.
type Ingestor struct {
	mu       sync.Mutex // held for a whole Sync
	store    knowledge.Store
	src      Source
	scope    string
	chunkers []knowledge.Chunker
	maxSize  int64
	workers  int
	now      func() time.Time // racy-window clock; tests shift it
}

var _ knowledge.Ingestor = (*Ingestor)(nil)

// New returns an Ingestor writing src into store.
func New(store knowledge.Store, src Source, opts ...Option) *Ingestor {
	i := &Ingestor{
		store:    store,
		src:      src,
		chunkers: []knowledge.Chunker{md.New(), golang.New()},
		maxSize:  defaultMaxSize,
		workers:  runtime.GOMAXPROCS(0),
		now:      time.Now,
	}
	if s := path.Clean(filepath.ToSlash(src.Scope)); s != "." {
		i.scope = strings.Trim(s, "/")
	}
	for _, o := range opts {
		o(i)
	}
	return i
}

func (i *Ingestor) chunkerFor(rel string) knowledge.Chunker {
	for _, c := range i.chunkers {
		if c.Accepts(rel) {
			return c
		}
	}
	return nil
}

func (i *Ingestor) accepts(rel string) bool { return i.chunkerFor(rel) != nil }

func (i *Ingestor) docID(rel string) knowledge.DocID {
	return knowledge.DocID(path.Join(i.scope, rel))
}

func (i *Ingestor) docScope(rel string) string {
	if i.scope != "" {
		return i.scope
	}
	if seg, _, ok := strings.Cut(rel, "/"); ok {
		return seg
	}
	return ""
}

// kindOf names the document kind of a chunker: its Kind method when it has
// one, "md"/"go" for the built-in chunkers, else the file extension.
func kindOf(c knowledge.Chunker, rel string) string {
	switch c := c.(type) {
	case interface{ Kind() string }:
		return c.Kind()
	case *md.Chunker:
		return "md"
	case *golang.Chunker:
		return "go"
	}
	if ext := strings.ToLower(strings.TrimPrefix(path.Ext(rel), ".")); ext != "" {
		return ext
	}
	return "text"
}

// weight is 1, ×0.2 for generated files, × the first matching rule.
func (i *Ingestor) weight(id knowledge.DocID, props map[string]string) float32 {
	w := float32(1)
	if props["generated"] == "true" {
		w *= 0.2
	}
	for _, r := range i.src.Weights {
		p := strings.Trim(r.Prefix, "/")
		if p != "" && (string(id) == p || strings.HasPrefix(string(id), p+"/")) {
			w *= r.Weight
			break
		}
	}
	return w
}

// Sync walks the source once and brings the store up to date. Per-file
// problems are collected in Report.Errors; an unreadable root, a failing
// catalog listing or context cancellation abort the run with an error.
// Concurrent calls are serialised.
func (i *Ingestor) Sync(ctx context.Context) (knowledge.Report, error) {
	i.mu.Lock()
	defer i.mu.Unlock()
	start := time.Now()
	racyFrom := i.now().Add(-racyWindow)
	var rep knowledge.Report
	done := func(err error) (knowledge.Report, error) {
		rep.Duration = max(time.Since(start), 1)
		return rep, err
	}
	if err := ctx.Err(); err != nil {
		return done(err)
	}
	if i.src.Root == "" {
		return done(errors.New("ingest: empty source root"))
	}
	root, err := filepath.Abs(i.src.Root)
	if err != nil {
		return done(fmt.Errorf("ingest: source root: %w", err))
	}

	catalog := make(map[knowledge.DocID]knowledge.Doc)
	for d, err := range i.store.ListDocs(ctx) {
		if err != nil {
			return done(fmt.Errorf("ingest: list catalog: %w", err))
		}
		catalog[d.ID] = d
	}
	// The marker of an interrupted Sync is not a document of the source.
	marker := i.pendingDoc(root)
	_, pending := catalog[marker.ID]
	delete(catalog, marker.ID)

	w := walker{ing: i, root: root}
	if err := w.walk(ctx); err != nil {
		return done(err)
	}
	rep.SecretsDropped += w.secrets
	rep.Skipped += w.skipped
	rep.Errors = append(rep.Errors, w.errs...)

	s := &syncState{
		ing:        i,
		rep:        &rep,
		root:       root,
		racyFrom:   racyFrom,
		marker:     marker,
		pending:    pending,
		fullRelink: pending,
		catalog:    catalog,
		movedFrom:  make(map[knowledge.DocID][]string),
		moves:      make(map[knowledge.DocID][]string),
		written:    make(map[knowledge.DocID]written),
		outcome:    make(map[knowledge.DocID]outcome, len(w.entries)),
		missing:    make(map[knowledge.DocID]bool),
		byHash:     make(map[[32]byte][]knowledge.DocID),
	}
	for id, d := range catalog {
		if v := d.Props[PropMovedFrom]; v != "" {
			s.movedFrom[id] = strings.Split(v, "\n")
		}
	}
	s.res = newResolver(w.entries, s.movedFrom, i.accepts)
	seen := make(map[knowledge.DocID]bool, len(w.entries))
	for _, e := range w.entries {
		seen[e.id] = true
	}
	for _, id := range slices.Sorted(maps.Keys(catalog)) {
		// Only this source's documents can go missing; documents under a
		// path the walk could not read are kept.
		if seen[id] || !i.owns(catalog[id], root) || w.covers(s.relOf(id)) {
			continue
		}
		s.addMissing(id)
	}

	jobs := make([]job, len(w.entries))
	for k := range w.entries {
		jobs[k] = job{e: &w.entries[k]}
	}
	if err := s.run(ctx, jobs); err != nil {
		return done(err)
	}
	if err := s.settleMoves(ctx); err != nil {
		return done(err)
	}
	return done(s.finish(ctx, w.entries))
}

// owns reports whether a catalog document belongs to this source: under
// Scope when set, else with Doc.Path inside root.
func (i *Ingestor) owns(d knowledge.Doc, root string) bool {
	if i.scope != "" {
		return string(d.ID) == i.scope || strings.HasPrefix(string(d.ID), i.scope+"/")
	}
	rel, err := filepath.Rel(root, d.Path)
	return err == nil && filepath.IsLocal(rel)
}

// pendingDoc is this source's incomplete-sync marker: under Scope, or with
// an empty Scope keyed by the root (scopeless sources share the DocID
// space) and with Doc.Path under it, so only this source owns it.
func (i *Ingestor) pendingDoc(root string) knowledge.Doc {
	id := path.Join(i.scope, pendingName)
	if i.scope == "" {
		h := sha256.Sum256([]byte(root))
		id = pendingName + "-" + hex.EncodeToString(h[:6])
	}
	return knowledge.Doc{ID: knowledge.DocID(id), Scope: i.scope, Path: filepath.Join(root, pendingName), Kind: kindPending}
}

// outcome is what the main pass did with a walked document.
type outcome uint8

const (
	outNone outcome = iota
	outUnchanged
	outTouched
	outWritten // added, changed, moved, or failed
)

// syncState is the state of one Sync. During a run, workers only read
// catalog, movedFrom and res; everything else is owned by the writer (the
// goroutine calling run).
type syncState struct {
	ing        *Ingestor
	rep        *knowledge.Report
	root       string    // absolute source root
	racyFrom   time.Time // files with mtime at or after this are racy
	marker     knowledge.Doc
	pending    bool // the marker is in the store
	marked     bool // this Sync tried to put the marker
	fullRelink bool // an earlier Sync stopped before its relink pass ended
	catalog    map[knowledge.DocID]knowledge.Doc
	movedFrom  map[knowledge.DocID][]string // former paths per doc, read-only during a run
	res        *resolver
	moves      map[knowledge.DocID][]string // former paths of docs moved in this Sync
	written    map[knowledge.DocID]written  // docs committed by PutDoc in this Sync
	outcome    map[knowledge.DocID]outcome
	missing    map[knowledge.DocID]bool // catalog docs not (validly) seen, not yet moved
	byHash     map[[32]byte][]knowledge.DocID
	deferred   []result // new documents whose hash matches a missing document
	arrived    []*entry // documents added or moved in this Sync
}

// markPending puts the incomplete-sync marker before this Sync's first add,
// move or delete commits (a marker left by an earlier Sync stays).
func (s *syncState) markPending(ctx context.Context) {
	if s.pending || s.marked {
		return
	}
	s.marked = true
	if err := s.ing.store.PutDoc(ctx, knowledge.DocUpdate{Doc: s.marker}); err != nil {
		s.rep.Errors = append(s.rep.Errors, fmt.Errorf("ingest: put sync marker: %w", err))
		return
	}
	s.pending = true
}

// clearPending deletes the marker once the relink pass has ended without
// errors (beyond the first errsBefore in the report).
func (s *syncState) clearPending(ctx context.Context, errsBefore int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !s.pending || len(s.rep.Errors) > errsBefore {
		return nil
	}
	if err := s.ing.store.DeleteDoc(ctx, s.marker.ID); err != nil {
		s.rep.Errors = append(s.rep.Errors, fmt.Errorf("ingest: delete sync marker: %w", err))
		return nil
	}
	s.pending = false
	return nil
}

// written records a document committed in this Sync.
type written struct {
	hash   [32]byte
	unsure int // missing + fallback-resolved links
}

func (s *syncState) addMissing(id knowledge.DocID) {
	if s.missing[id] {
		return
	}
	s.missing[id] = true
	h := s.catalog[id].Hash
	s.byHash[h] = append(s.byHash[h], id)
}

// takeMove returns a missing document with hash h, preferring the same
// base name as rel, then lexical order, and consumes it. With sameBase
// only missing documents with rel's base name qualify.
func (s *syncState) takeMove(h [32]byte, rel string, sameBase bool) (knowledge.DocID, bool) {
	var (
		pick  knowledge.DocID
		found bool
	)
	base := path.Base(rel)
	for _, id := range s.byHash[h] {
		if !s.missing[id] || (sameBase && path.Base(string(id)) != base) {
			continue
		}
		if !found || cmp.Or(boolCmp(path.Base(string(id)) == base, path.Base(string(pick)) == base), strings.Compare(string(id), string(pick))) < 0 {
			pick, found = id, true
		}
	}
	if found {
		delete(s.missing, pick)
	}
	return pick, found
}

// namesakeLinkers adds to linkers the sources of LinksTo edges into
// documents (or moved documents' former names) sharing a base name with a
// document added or moved in this Sync.
func (s *syncState) namesakeLinkers(ctx context.Context, linkers map[knowledge.DocID]bool) error {
	seeds := make(map[knowledge.NodeID]bool)
	for _, a := range s.arrived {
		base := strings.ToLower(path.Base(a.rel))
		for _, key := range []string{base, strings.TrimSuffix(base, path.Ext(base))} {
			for _, e := range slices.Concat(s.res.byName[key], s.res.aliasByName[key]) {
				if e.id != a.id {
					seeds[knowledge.DocNode(e.id)] = true
				}
			}
		}
	}
	if len(seeds) == 0 {
		return ctx.Err()
	}
	in, err := s.ing.store.Neighbors(ctx, slices.Sorted(maps.Keys(seeds)), []knowledge.EdgeKind{knowledge.LinksTo}, knowledge.In, 0)
	if err != nil {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		s.rep.Errors = append(s.rep.Errors, fmt.Errorf("ingest: links into namesakes: %w", err))
		return nil
	}
	for _, e := range in {
		linkers[e.Source] = true
	}
	return nil
}

// relOf returns the root-relative path of a DocID of this source.
func (s *syncState) relOf(id knowledge.DocID) string {
	if s.ing.scope == "" {
		return string(id)
	}
	return strings.TrimPrefix(string(id), s.ing.scope+"/")
}

type job struct {
	e     *entry
	force bool // re-resolution: always re-chunk, write only if links changed
}

type resultKind uint8

const (
	resCancelled resultKind = iota
	resErr
	resSkip
	resUnchanged
	resTouch
	resUpdate
)

type result struct {
	job     job
	kind    resultKind
	err     error
	doc     knowledge.Doc
	chunks  []knowledge.Chunk
	edges   []knowledge.Edge
	secrets int
	restamp bool // resUpdate of unchanged content: only the racy marker changes
}

// run feeds jobs through the worker pool and applies results in the
// calling goroutine (the single writer). It returns ctx.Err() when
// cancelled, after all workers have stopped.
func (s *syncState) run(parent context.Context, jobs []job) error {
	if len(jobs) == 0 {
		return parent.Err()
	}
	ctx, cancel := context.WithCancel(parent)
	defer cancel()
	n := min(s.ing.workers, len(jobs))
	jobCh := make(chan job, n)
	resCh := make(chan result, n)
	var wg sync.WaitGroup
	wg.Go(func() {
		defer close(jobCh)
		for _, j := range jobs {
			select {
			case jobCh <- j:
			case <-ctx.Done():
				return
			}
		}
	})
	for range n {
		wg.Go(func() {
			for j := range jobCh {
				r := s.process(ctx, j)
				select {
				case resCh <- r:
				case <-ctx.Done():
					return
				}
			}
		})
	}
	go func() {
		wg.Wait()
		close(resCh)
	}()
	for r := range resCh {
		if ctx.Err() != nil {
			break
		}
		s.apply(ctx, r)
	}
	cancel()
	for range resCh {
		// drain so every worker has exited before returning
	}
	return parent.Err()
}

// process reads, hashes and chunks one file (worker side, no writes).
func (s *syncState) process(ctx context.Context, j job) result {
	r := result{job: j}
	if ctx.Err() != nil {
		return r
	}
	e := j.e
	cat, inCat := s.catalog[e.id]
	if inCat && cat.Path != e.abs && !s.ing.owns(cat, s.root) && fileExists(cat.Path) {
		r.kind, r.err = resErr, fmt.Errorf("ingest: %s: DocID collides with %s of another source; skipped", e.id, cat.Path)
		return r
	}
	fresh := inCat && cat.Props[propStale] == "" && cat.Path == e.abs && !j.force
	racy, catRacy := !e.mtime.Before(s.racyFrom), cat.Props[propRacy] != ""
	if fresh && !catRacy && cat.Size == e.size && cat.MTime.Equal(e.mtime) {
		r.kind = resUnchanged
		return r
	}
	raw, err := readCapped(e.abs, s.ing.maxSize)
	switch {
	case errors.Is(err, errTooLarge):
		r.kind = resSkip
		return r
	case err != nil:
		r.kind, r.err = resErr, fmt.Errorf("ingest: read %s: %w", e.id, err)
		return r
	}
	if bytes.IndexByte(raw[:min(len(raw), binarySniff)], 0) >= 0 {
		r.kind = resSkip
		return r
	}
	d := knowledge.Doc{
		ID:    e.id,
		Scope: e.scope,
		Path:  e.abs,
		Kind:  e.kind,
		Size:  int64(len(raw)),
		MTime: e.mtime,
		Hash:  sha256.Sum256(raw),
	}
	if fresh && cat.Hash == d.Hash {
		if racy == catRacy {
			r.kind, r.doc = resUnchanged, d
			if !cat.MTime.Equal(d.MTime) {
				r.kind = resTouch
			}
			return r
		}
		// Same content, but the racy marker must be set or cleared.
		r.restamp = true
	}
	src, secrets := redactSecrets(raw)
	cr, err := e.chunker.Chunk(d, src)
	if err != nil {
		r.kind, r.err = resErr, fmt.Errorf("ingest: chunk %s: %w", e.id, err)
		return r
	}
	r.secrets = secrets + dropSecretMeta(e.id, &cr)
	edges, missing, fuzzy := s.res.edges(e, cr)
	props := maps.Clone(cr.Props)
	delete(props, PropMissing)
	delete(props, PropFuzzy)
	delete(props, propStale)
	delete(props, PropMovedFrom)
	delete(props, propRacy)
	if missing > 0 {
		props = setProp(props, PropMissing, strconv.Itoa(missing))
	}
	if fuzzy > 0 {
		props = setProp(props, PropFuzzy, strconv.Itoa(fuzzy))
	}
	if mf := s.movedFrom[e.id]; len(mf) > 0 {
		props = setProp(props, PropMovedFrom, strings.Join(mf, "\n"))
	}
	if racy {
		props = setProp(props, propRacy, "1")
	}
	d.Title = cr.Title
	d.Props = props
	d.Weight = s.ing.weight(e.id, props)
	r.kind, r.doc, r.chunks, r.edges = resUpdate, d, cr.Chunks, edges
	return r
}

func fileExists(name string) bool {
	_, err := os.Stat(name)
	return err == nil
}

var errTooLarge = errors.New("file over size cap")

// readCapped reads a file, failing with errTooLarge past limit bytes.
func readCapped(name string, limit int64) ([]byte, error) {
	f, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	b, err := io.ReadAll(io.LimitReader(f, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(b)) > limit {
		return nil, errTooLarge
	}
	return b, nil
}

// apply commits one result (writer side).
func (s *syncState) apply(ctx context.Context, r result) {
	if r.job.force {
		if r.kind == resUpdate {
			s.reresolve(ctx, r)
		} else if r.kind == resErr {
			s.rep.Errors = append(s.rep.Errors, r.err)
		}
		return
	}
	id := r.job.e.id
	switch r.kind {
	case resCancelled:
	case resErr:
		s.outcome[id] = outWritten
		s.rep.Errors = append(s.rep.Errors, r.err)
	case resSkip:
		s.rep.Skipped++
		if _, ok := s.catalog[id]; ok {
			s.addMissing(id) // no longer indexable: delete (or move) it
		}
	case resUnchanged:
		s.outcome[id] = outUnchanged
		s.rep.Unchanged++
	case resTouch:
		if err := s.ing.store.TouchDoc(ctx, id, r.doc.MTime); err != nil {
			s.outcome[id] = outWritten
			s.rep.Errors = append(s.rep.Errors, fmt.Errorf("ingest: touch %s: %w", id, err))
			return
		}
		s.outcome[id] = outTouched
		s.rep.Touched++
	case resUpdate:
		if r.restamp {
			s.restampDoc(ctx, r)
			return
		}
		s.outcome[id] = outWritten
		s.update(ctx, r)
	}
}

// restampDoc rewrites a document whose content is unchanged, to set or
// clear its racy marker; it counts as Unchanged, or Touched when the mtime
// moved.
func (s *syncState) restampDoc(ctx context.Context, r result) {
	id := r.doc.ID
	r.secrets = 0 // counted when the content was written
	if !s.put(ctx, r) {
		s.outcome[id] = outWritten
		return
	}
	if s.catalog[id].MTime.Equal(r.doc.MTime) {
		s.outcome[id] = outUnchanged
		s.rep.Unchanged++
		return
	}
	s.outcome[id] = outTouched
	s.rep.Touched++
}

func setProp(props map[string]string, k, v string) map[string]string {
	if props == nil {
		props = make(map[string]string, 1)
	}
	props[k] = v
	return props
}

func (s *syncState) put(ctx context.Context, r result) bool {
	err := s.ing.store.PutDoc(ctx, knowledge.DocUpdate{Doc: r.doc, Chunks: r.chunks, Edges: r.edges})
	if err != nil {
		s.rep.Errors = append(s.rep.Errors, fmt.Errorf("ingest: put %s: %w", r.doc.ID, err))
		return false
	}
	s.rep.SecretsDropped += r.secrets
	s.written[r.doc.ID] = written{hash: r.doc.Hash, unsure: unsureLinks(r.doc)}
	return true
}

func (s *syncState) update(ctx context.Context, r result) {
	id := r.doc.ID
	if _, ok := s.catalog[id]; ok {
		if s.put(ctx, r) {
			s.rep.Changed++
		}
		return
	}
	if s.hasMissing(r.doc.Hash) {
		// Possible move: decided after the pipeline, in a fixed order.
		s.deferred = append(s.deferred, r)
		return
	}
	s.markPending(ctx)
	if s.put(ctx, r) {
		s.rep.Added++
		s.arrived = append(s.arrived, r.job.e)
	}
}

func (s *syncState) hasMissing(h [32]byte) bool {
	for _, id := range s.byHash[h] {
		if s.missing[id] {
			return true
		}
	}
	return false
}

// settleMoves matches deferred new documents to missing documents with the
// same hash, independent of worker timing: first new documents (by rel)
// that have a missing twin with the same base name, then the rest (by rel)
// against the lexically first remaining twin. Unmatched ones are added.
func (s *syncState) settleMoves(ctx context.Context) error {
	slices.SortFunc(s.deferred, func(a, b result) int { return strings.Compare(a.job.e.rel, b.job.e.rel) })
	var rest []result
	for _, r := range s.deferred {
		if err := ctx.Err(); err != nil {
			return err
		}
		if old, ok := s.takeMove(r.doc.Hash, r.job.e.rel, true); ok {
			s.move(ctx, r, old)
			continue
		}
		rest = append(rest, r)
	}
	for _, r := range rest {
		if err := ctx.Err(); err != nil {
			return err
		}
		if old, ok := s.takeMove(r.doc.Hash, r.job.e.rel, false); ok {
			s.move(ctx, r, old)
			continue
		}
		s.markPending(ctx)
		if s.put(ctx, r) {
			s.rep.Added++
			s.arrived = append(s.arrived, r.job.e)
		}
	}
	s.deferred = nil
	return nil
}

// move re-keys missing document old to the new document of r, then
// re-chunks it at its new location.
func (s *syncState) move(ctx context.Context, r result, old knowledge.DocID) {
	id := r.doc.ID
	// Remember the former path so links written against it keep resolving.
	rel := r.job.e.rel
	mf := append(slices.Clone(s.movedFrom[old]), s.relOf(old))
	mf = slices.DeleteFunc(mf, func(p string) bool { return p == rel })
	slices.Sort(mf)
	mf = slices.Compact(mf)
	s.moves[id] = mf
	r.doc.Props = setProp(r.doc.Props, PropMovedFrom, strings.Join(mf, "\n"))
	// Re-key the old document (and edges pointing at it), marked stale so a
	// failed re-chunk below is redone by the next Sync.
	staged := r.doc
	staged.Props = setProp(maps.Clone(r.doc.Props), propStale, "1")
	s.markPending(ctx)
	if err := s.ing.store.MoveDoc(ctx, old, staged); err != nil {
		// old stays consumed: it remains catalogued and the next Sync retries.
		s.rep.Errors = append(s.rep.Errors, fmt.Errorf("ingest: move %s to %s: %w", old, id, err))
		return
	}
	s.rep.Moved++
	s.arrived = append(s.arrived, r.job.e)
	s.put(ctx, r) // re-chunk so relative links resolve from the new location
}

// finish deletes unmatched missing documents and re-resolves documents
// whose links may now point elsewhere: after adds, moves or deletes, or
// every document when an earlier Sync left its marker. The marker is then
// deleted unless the pass was cut short or reported errors.
func (s *syncState) finish(ctx context.Context, entries []entry) error {
	if len(s.missing) > 0 {
		s.markPending(ctx)
	}
	errs := len(s.rep.Errors)
	linkers := make(map[knowledge.DocID]bool)
	for _, id := range slices.Sorted(maps.Keys(s.missing)) {
		if err := ctx.Err(); err != nil {
			return err
		}
		in, err := s.ing.store.Neighbors(ctx, []knowledge.NodeID{knowledge.DocNode(id)}, []knowledge.EdgeKind{knowledge.LinksTo}, knowledge.In, 0)
		if err != nil {
			s.rep.Errors = append(s.rep.Errors, fmt.Errorf("ingest: inbound links of %s: %w", id, err))
		}
		for _, e := range in {
			linkers[e.Source] = true
		}
		if err := s.ing.store.DeleteDoc(ctx, id); err != nil {
			s.rep.Errors = append(s.rep.Errors, fmt.Errorf("ingest: delete %s: %w", id, err))
			continue
		}
		s.rep.Deleted++
	}
	if s.rep.Added+s.rep.Moved+s.rep.Deleted == 0 && !s.fullRelink {
		return s.clearPending(ctx, errs)
	}
	if len(s.moves) > 0 {
		maps.Copy(s.movedFrom, s.moves)
		s.res = newResolver(entries, s.movedFrom, s.ing.accepts)
	}
	// Documents linking to a namesake of an arrived document may now
	// resolve to it instead (better scope, fewer segments, a real file
	// replacing an alias).
	if err := s.namesakeLinkers(ctx, linkers); err != nil {
		return err
	}
	var jobs []job
	for k := range entries {
		e := &entries[k]
		var again bool
		switch s.outcome[e.id] {
		case outUnchanged, outTouched:
			again = s.fullRelink || linkers[e.id] || unsureLinks(s.catalog[e.id]) > 0
		case outWritten:
			// Written before this Sync's moves were known as aliases.
			again = s.rep.Moved > 0 && s.written[e.id].unsure > 0
		}
		if again {
			jobs = append(jobs, job{e: e, force: true})
		}
	}
	if err := s.run(ctx, jobs); err != nil {
		return err
	}
	return s.clearPending(ctx, errs)
}

// unsureLinks counts the links of d that a new or moved document may
// resolve better: missing ones and base-name fallbacks.
func unsureLinks(d knowledge.Doc) int {
	m, _ := strconv.Atoi(d.Props[PropMissing])
	f, _ := strconv.Atoi(d.Props[PropFuzzy])
	return m + f
}

// reresolve writes a re-chunked document only when its LinksTo targets
// differ from the stored ones (or its content changed under an unchanged
// stat). A document that was Unchanged/Touched is then counted as Changed
// instead.
func (s *syncState) reresolve(ctx context.Context, r result) {
	id := r.doc.ID
	hash := s.catalog[id].Hash
	if w, ok := s.written[id]; ok {
		hash = w.hash
	}
	if hash == r.doc.Hash {
		stored, err := s.ing.store.Neighbors(ctx, []knowledge.NodeID{knowledge.DocNode(id)}, []knowledge.EdgeKind{knowledge.LinksTo}, knowledge.Out, 0)
		if err != nil {
			s.rep.Errors = append(s.rep.Errors, fmt.Errorf("ingest: links of %s: %w", id, err))
			return
		}
		if slices.Equal(linkTargets(stored, id), linkTargets(r.edges, id)) {
			return
		}
	}
	if !s.put(ctx, r) {
		return
	}
	switch s.outcome[id] {
	case outUnchanged:
		s.rep.Unchanged--
		s.rep.Changed++
	case outTouched:
		s.rep.Touched--
		s.rep.Changed++
	}
	s.outcome[id] = outWritten
}

// linkTarget is one LinksTo edge reduced to what re-resolution compares.
type linkTarget struct {
	to     knowledge.NodeID
	weight float32
}

// linkTargets returns the sorted, unique targets (with weights) of LinksTo
// edges asserted by id.
func linkTargets(es []knowledge.Edge, id knowledge.DocID) []linkTarget {
	self := knowledge.DocNode(id)
	var out []linkTarget
	for _, e := range es {
		if e.Kind == knowledge.LinksTo && e.Source == id && e.From == self {
			out = append(out, linkTarget{e.To, e.Weight})
		}
	}
	slices.SortFunc(out, func(a, b linkTarget) int {
		return cmp.Or(strings.Compare(string(a.to), string(b.to)), cmp.Compare(a.weight, b.weight))
	})
	return slices.Compact(out)
}

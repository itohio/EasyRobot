// Package knowledge defines the data model and interfaces of a local
// knowledge engine: ingest a folder of Markdown and Go files, index it
// lexically and as a link graph, and return ranked starting points
// (files, line spans and reasons) for a question.
//
// Backends live in sub-packages and are opt-in by import: store/memory
// (in-process), store/sqlite (pure-Go SQLite with FTS5). Chunkers live in
// chunk/md and chunk/golang, ingestion in ingest, retrieval in retrieve.
// Vector search and embedders are interfaces only in this slice.
package knowledge

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"iter"
	"strconv"
	"strings"
	"time"
)

// DocID identifies a document: scope-relative slash path, e.g.
// "abugaga/internal/keys/keys.go". It is the key for catalog, chunks and
// edges asserted by the document.
type DocID string

// ChunkID is the first 16 bytes of sha256(doc, ord, text).
type ChunkID [16]byte

// NewChunkID derives a chunk id from its document, position and text.
func NewChunkID(doc DocID, ord int, text string) ChunkID {
	h := sha256.New()
	h.Write([]byte(doc))
	h.Write([]byte{0})
	h.Write([]byte(strconv.Itoa(ord)))
	h.Write([]byte{0})
	h.Write([]byte(text))
	var id ChunkID
	copy(id[:], h.Sum(nil))
	return id
}

func (c ChunkID) String() string { return hex.EncodeToString(c[:]) }

// NodeID names a graph node: "doc:<DocID>", "id:<ABU-061>", "sym:<pkg.Name>",
// "tag:<tag>", "pkg:<importpath>" or "missing:<target>".
type NodeID string

// Node kinds (the prefix before the first colon).
const (
	NodeDoc     = "doc"
	NodeIdent   = "id"
	NodeSym     = "sym"
	NodeTag     = "tag"
	NodePkg     = "pkg"
	NodeMissing = "missing"
)

func node(kind, value string) NodeID { return NodeID(kind + ":" + value) }

// DocNode is the graph node of a document.
func DocNode(id DocID) NodeID { return node(NodeDoc, string(id)) }

// IDNode is the node of a frontmatter/ticket id such as ABU-061; ids are
// case-insensitive and stored upper-case.
func IDNode(id string) NodeID { return node(NodeIdent, strings.ToUpper(id)) }

// SymNode is the node of a code symbol such as "keys.PeerPermissions".
func SymNode(sym string) NodeID { return node(NodeSym, sym) }

// TagNode is the node of a tag; tags are case-insensitive.
func TagNode(tag string) NodeID { return node(NodeTag, strings.ToLower(strings.TrimPrefix(tag, "#"))) }

// PkgNode is the node of a Go import path.
func PkgNode(path string) NodeID { return node(NodePkg, path) }

// MissingNode is the node of a link target that did not resolve.
func MissingNode(target string) NodeID { return node(NodeMissing, target) }

// Kind returns the node kind (prefix), or "" when malformed.
func (n NodeID) Kind() string {
	k, _, ok := strings.Cut(string(n), ":")
	if !ok {
		return ""
	}
	return k
}

// Value returns the part after the kind prefix.
func (n NodeID) Value() string {
	_, v, _ := strings.Cut(string(n), ":")
	return v
}

// Doc is one indexed file.
type Doc struct {
	ID     DocID
	Scope  string // first path segment or the source's scope
	Path   string // absolute path, for reading spans
	Kind   string // md | go | text
	Title  string // first H1, frontmatter title, or Go package name
	Size   int64
	MTime  time.Time
	Hash   [32]byte          // sha256 of the file content
	Props  map[string]string // frontmatter (id, type, status, owner, tags, …); Go: package
	Weight float32           // 1 by default; lower for generated or boilerplate files
}

// Chunk is one indexed section of a document.
type Chunk struct {
	ID      ChunkID
	Doc     DocID
	Ord     int
	Heading []string // breadcrumb, outermost first
	Symbol  string   // Go: "pkg.Name"; empty for prose
	Lines   [2]int   // 1-based inclusive line span in the file
	Text    string   // indexed text (body of the section)
	Hash    [32]byte // sha256(Text)
	Tokens  int      // token count of Text
}

// EdgeKind types a graph edge.
type EdgeKind uint8

const (
	LinksTo  EdgeKind = iota + 1 // doc → doc (wikilink or Markdown link)
	Mentions                     // doc → id: (an id named in text)
	Tagged                       // doc → tag:
	Imports                      // doc → pkg:
	Defines                      // doc → id: (frontmatter id) or sym: (declared symbol)
	Contains                     // reserved: doc → chunk
	SameDir                      // reserved: weak doc → doc
)

var edgeKindNames = [...]string{"", "links", "mentions", "tagged", "imports", "defines", "contains", "samedir"}

func (k EdgeKind) String() string {
	if int(k) < len(edgeKindNames) {
		return edgeKindNames[k]
	}
	return "edge(" + strconv.Itoa(int(k)) + ")"
}

// Edge is a typed, weighted relation asserted by Source. Re-ingesting
// Source replaces all of its edges.
type Edge struct {
	From, To NodeID
	Kind     EdgeKind
	Weight   float32
	Source   DocID
}

// Direction selects edge orientation for Neighbors.
type Direction uint8

const (
	Out  Direction = iota + 1 // edges whose From is the seed
	In                        // edges whose To is the seed
	Both                      // either
)

// Filter restricts results before ranking (never after top-k).
type Filter struct {
	In     []string // DocID prefixes ("abugaga/docs", "docs/tasks")
	Kinds  []string // md | go | text
	Scopes []string
	Not    []DocID // exact documents to exclude
}

// Match reports whether d passes the filter.
func (f Filter) Match(d Doc) bool {
	if len(f.In) > 0 && !hasPrefixAny(string(d.ID), f.In) {
		return false
	}
	if len(f.Kinds) > 0 && !contains(f.Kinds, d.Kind) {
		return false
	}
	if len(f.Scopes) > 0 && !contains(f.Scopes, d.Scope) {
		return false
	}
	for _, n := range f.Not {
		if n == d.ID {
			return false
		}
	}
	return true
}

func hasPrefixAny(s string, prefixes []string) bool {
	for _, p := range prefixes {
		p = strings.TrimSuffix(p, "/")
		if s == p || strings.HasPrefix(s, p+"/") {
			return true
		}
	}
	return false
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}

// LexQuery is a pre-tokenized lexical query (see package token).
type LexQuery struct {
	Terms []string
}

// ChunkHit is a lexical match. Score is higher-is-better and comparable
// only within one result list.
type ChunkHit struct {
	Chunk   ChunkID
	Doc     DocID
	Score   float64
	Matched []string // query terms present in the chunk
}

// DocUpdate is everything one document contributes to the index.
type DocUpdate struct {
	Doc    Doc
	Chunks []Chunk
	Edges  []Edge // all edges asserted by Doc (Source must equal Doc.ID)
}

// Catalog tracks indexed documents.
type Catalog interface {
	GetDoc(ctx context.Context, id DocID) (Doc, bool, error)
	ListDocs(ctx context.Context) iter.Seq2[Doc, error]
	// Generation increases on every committed change.
	Generation(ctx context.Context) (uint64, error)
}

// LexicalIndex searches chunks by terms.
type LexicalIndex interface {
	// SearchLex returns up to k chunks ranked by BM25 with field weights
	// heading 3, symbol 3, path 2, body 1, restricted by f before ranking.
	SearchLex(ctx context.Context, q LexQuery, k int, f Filter) ([]ChunkHit, error)
	GetChunk(ctx context.Context, id ChunkID) (Chunk, bool, error)
	DocChunks(ctx context.Context, doc DocID) ([]Chunk, error)
}

// GraphStore holds typed edges.
type GraphStore interface {
	// Neighbors returns edges touching any seed, of the given kinds (all
	// kinds when empty), at most limit per seed (no limit when <= 0).
	Neighbors(ctx context.Context, from []NodeID, kinds []EdgeKind, dir Direction, limit int) ([]Edge, error)
	// Degree counts edges touching n in either direction.
	Degree(ctx context.Context, n NodeID) (int, error)
	// Resolve maps a node to the document it stands for: doc: nodes to
	// themselves when indexed; id: and sym: nodes to the source of a
	// Defines edge pointing at them.
	Resolve(ctx context.Context, n NodeID) (DocID, bool, error)
}

// Writer mutates the index; each call is atomic.
type Writer interface {
	// PutDoc replaces the document, all its chunks and all edges it asserts.
	PutDoc(ctx context.Context, u DocUpdate) error
	// DeleteDoc removes the document, its chunks and edges it asserts.
	// Edges other documents assert towards it stay (they dangle).
	DeleteDoc(ctx context.Context, id DocID) error
	// MoveDoc re-keys a document whose content is unchanged: catalog,
	// chunks, edges it asserts, and edges pointing at doc:from.
	MoveDoc(ctx context.Context, from DocID, to Doc) error
	// TouchDoc records a new mtime for an unchanged document.
	TouchDoc(ctx context.Context, id DocID, mtime time.Time) error
}

// Store is a complete index backend.
type Store interface {
	Catalog
	LexicalIndex
	GraphStore
	Writer
	Close() error
}

// ModelID names an embedding model.
type ModelID string

// ModelInfo describes an embedding model.
type ModelInfo struct {
	ID         ModelID
	Dim        int
	MaxTokens  int
	Normalized bool
}

// VecRecord is one chunk vector.
type VecRecord struct {
	Chunk ChunkID
	Doc   DocID
	Vec   []float32
}

// Purpose distinguishes query from document embeddings.
type Purpose uint8

const (
	PurposeQuery Purpose = iota + 1
	PurposeDocument
)

// VectorStore is the optional vector layer (not implemented in this slice).
type VectorStore interface {
	UpsertVectors(ctx context.Context, model ModelID, recs []VecRecord) error
	DeleteDocVectors(ctx context.Context, doc DocID) error
	SearchVectors(ctx context.Context, model ModelID, q []float32, k int, f Filter) ([]ChunkHit, error)
	Models(ctx context.Context) ([]ModelInfo, error)
}

// Embedder turns text into vectors (not implemented in this slice).
type Embedder interface {
	Model() ModelInfo
	Embed(ctx context.Context, purpose Purpose, texts []string) ([][]float32, error)
}

// LinkKind types a raw link found by a chunker.
type LinkKind uint8

const (
	WikiLink     LinkKind = iota + 1 // [[target]] or [[target|label]]
	MarkdownLink                     // [label](relative/path.md#anchor)
)

// Link is an unresolved link; the ingestor resolves it to a DocID.
type Link struct {
	Target string // as written, without anchor or label
	Kind   LinkKind
	Line   int // 1-based
}

// ChunkResult is what a chunker extracts from one file.
type ChunkResult struct {
	Title  string
	Props  map[string]string
	Chunks []Chunk
	Edges  []Edge // edges needing no resolution (ids, tags, symbols, imports)
	Links  []Link // edges needing resolution against the corpus
}

// Chunker splits one kind of file.
type Chunker interface {
	Accepts(path string) bool
	Chunk(d Doc, src []byte) (ChunkResult, error)
}

// Query is a retrieval request.
type Query struct {
	Text    string
	K       int     // default 8, capped at 30
	Filter  Filter  // applied before ranking
	Near    []DocID // "more like / linked to these files"
	Explain bool    // include per-signal scores
}

// Section is a ranked span inside a hit.
type Section struct {
	Heading string
	Lines   [2]int
	Snippet string
	Why     []string
}

// Hit is one ranked file.
type Hit struct {
	Doc      DocID
	Path     string
	Kind     string
	Title    string
	Score    float64
	Sections []Section
	Why      []string
	Signals  map[string]float64 // with Query.Explain
}

// Result is a retrieval answer.
type Result struct {
	Hits       []Hit
	Confidence string   // high | medium | low
	TryTerms   []string // suggestions when confidence is low
	Generation uint64
	Docs       int
}

// Retriever answers queries.
type Retriever interface {
	Find(ctx context.Context, q Query) (Result, error)
}

// Report summarises one ingest run.
type Report struct {
	Added, Changed, Moved, Deleted, Unchanged, Touched, Skipped int
	SecretsDropped                                              int // chunks or files dropped by secret rules
	Errors                                                      []error
	Duration                                                    time.Duration
}

// Ingestor synchronises an index with a source folder.
type Ingestor interface {
	Sync(ctx context.Context) (Report, error)
}

// ErrClosed is returned by a closed store.
var ErrClosed = errors.New("knowledge: store closed")

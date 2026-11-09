// Package storetest is the contract suite every knowledge.Store backend
// must pass. A backend's test calls Run (or RunWith) with a function that
// opens a fresh, empty store and registers its Close with t.Cleanup.
package storetest

import (
	"crypto/sha256"
	"errors"
	"fmt"
	"maps"
	"slices"
	"testing"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/token"
)

// Opener returns a fresh, empty store and registers its Close with
// t.Cleanup. Closing an already closed store must be harmless.
type Opener = func(t *testing.T) knowledge.Store

// RunOpts configures optional parts of the suite.
type RunOpts struct {
	// Reopen closes s and returns a store opened on the same data. When
	// nil the persistence subtest is skipped.
	Reopen func(t *testing.T, s knowledge.Store) knowledge.Store
}

// Run runs the contract suite without the persistence subtest.
func Run(t *testing.T, open Opener) { RunWith(t, open, RunOpts{}) }

// RunWith runs the contract suite with optional hooks.
func RunWith(t *testing.T, open Opener, o RunOpts) {
	t.Run("RoundTrip", func(t *testing.T) { testRoundTrip(t, open(t)) })
	t.Run("PutReplaces", func(t *testing.T) { testPutReplaces(t, open(t)) })
	t.Run("PutRejectsInvalid", func(t *testing.T) { testPutRejectsInvalid(t, open(t)) })
	t.Run("SearchLex", func(t *testing.T) { testSearchLex(t, open) })
	t.Run("DeleteCascades", func(t *testing.T) { testDelete(t, open(t)) })
	t.Run("MoveDoc", func(t *testing.T) { testMove(t, open(t)) })
	t.Run("MoveOntoExisting", func(t *testing.T) { testMoveOntoExisting(t, open(t)) })
	t.Run("TouchDoc", func(t *testing.T) { testTouch(t, open(t)) })
	t.Run("MoveDuplicateIDs", func(t *testing.T) { testMoveDuplicateIDs(t, open(t)) })
	t.Run("DeleteReAdd", func(t *testing.T) { testDeleteReAdd(t, open(t)) })
	t.Run("Unicode", func(t *testing.T) { testUnicode(t, open(t)) })
	t.Run("CommonTermFloor", func(t *testing.T) { testCommonTermFloor(t, open(t)) })
	t.Run("Generation", func(t *testing.T) { testGeneration(t, open(t)) })
	t.Run("Reopen", func(t *testing.T) {
		if o.Reopen == nil {
			t.Skip("backend has no Reopen hook")
		}
		testReopen(t, open(t), o.Reopen)
	})
	t.Run("Neighbors", func(t *testing.T) { testNeighbors(t, open(t)) })
	t.Run("Degree", func(t *testing.T) { testDegree(t, open(t)) })
	t.Run("Resolve", func(t *testing.T) { testResolve(t, open(t)) })
	t.Run("ListDocs", func(t *testing.T) { testListDocs(t, open(t)) })
	t.Run("ConcurrentReaders", func(t *testing.T) { testConcurrent(t, open(t)) })
	t.Run("Closed", func(t *testing.T) { testClosed(t, open(t)) })
}

var baseTime = time.Date(2026, 11, 9, 10, 11, 12, 345, time.UTC)

func mkDoc(id, kind, scope string) knowledge.Doc {
	return knowledge.Doc{
		ID:     knowledge.DocID(id),
		Scope:  scope,
		Path:   "/ws/" + id,
		Kind:   kind,
		Title:  "title of " + id,
		Size:   int64(100 + len(id)),
		MTime:  baseTime,
		Hash:   sha256.Sum256([]byte(id)),
		Weight: 1,
	}
}

func mkChunk(doc knowledge.DocID, ord int, heading []string, symbol, text string) knowledge.Chunk {
	return knowledge.Chunk{
		ID:      knowledge.NewChunkID(doc, ord, text),
		Doc:     doc,
		Ord:     ord,
		Heading: heading,
		Symbol:  symbol,
		Lines:   [2]int{ord*10 + 1, ord*10 + 9},
		Text:    text,
		Hash:    sha256.Sum256([]byte(text)),
		Tokens:  token.Count(text),
	}
}

func mkEdge(from, to knowledge.NodeID, kind knowledge.EdgeKind, w float32, src string) knowledge.Edge {
	return knowledge.Edge{From: from, To: to, Kind: kind, Weight: w, Source: knowledge.DocID(src)}
}

func dn(id string) knowledge.NodeID { return knowledge.DocNode(knowledge.DocID(id)) }

func put(t *testing.T, s knowledge.Store, d knowledge.Doc, chunks []knowledge.Chunk, edges ...knowledge.Edge) {
	t.Helper()
	if err := s.PutDoc(t.Context(), knowledge.DocUpdate{Doc: d, Chunks: chunks, Edges: edges}); err != nil {
		t.Fatalf("PutDoc(%s): %v", d.ID, err)
	}
}

// textDoc stores an md document in scope "ws" with one chunk per text.
func textDoc(t *testing.T, s knowledge.Store, id string, texts ...string) []knowledge.Chunk {
	t.Helper()
	d := mkDoc(id, "md", "ws")
	var cs []knowledge.Chunk
	for i, txt := range texts {
		cs = append(cs, mkChunk(d.ID, i, nil, "", txt))
	}
	put(t, s, d, cs)
	return cs
}

// fillers stores n documents whose chunks have ten tokens of unrelated
// prose, so corpus statistics (idf, average length) are realistic.
func fillers(t *testing.T, s knowledge.Store, n int) {
	t.Helper()
	for i := range n {
		textDoc(t, s, fmt.Sprintf("filler/f%02d.md", i),
			"lorem ipsum dolor sit amet consectetur adipiscing elit sed tempor")
	}
}

func search(t *testing.T, s knowledge.Store, terms []string, k int, f knowledge.Filter) []knowledge.ChunkHit {
	t.Helper()
	hits, err := s.SearchLex(t.Context(), knowledge.LexQuery{Terms: terms}, k, f)
	if err != nil {
		t.Fatalf("SearchLex(%q, %d, %+v): %v", terms, k, f, err)
	}
	return hits
}

func hitDocs(hits []knowledge.ChunkHit) []knowledge.DocID {
	out := make([]knowledge.DocID, len(hits))
	for i, h := range hits {
		out[i] = h.Doc
	}
	return out
}

func wantDocs(t *testing.T, hits []knowledge.ChunkHit, want ...knowledge.DocID) {
	t.Helper()
	if got := hitDocs(hits); !slices.Equal(got, want) {
		t.Fatalf("hit docs = %q, want %q (hits %+v)", got, want, hits)
	}
}

func hitFor(t *testing.T, hits []knowledge.ChunkHit, doc knowledge.DocID) knowledge.ChunkHit {
	t.Helper()
	for _, h := range hits {
		if h.Doc == doc {
			return h
		}
	}
	t.Fatalf("no hit for %s in %+v", doc, hits)
	return knowledge.ChunkHit{}
}

func getDoc(t *testing.T, s knowledge.Store, id knowledge.DocID) (knowledge.Doc, bool) {
	t.Helper()
	d, ok, err := s.GetDoc(t.Context(), id)
	if err != nil {
		t.Fatalf("GetDoc(%s): %v", id, err)
	}
	return d, ok
}

func docChunks(t *testing.T, s knowledge.Store, id knowledge.DocID) []knowledge.Chunk {
	t.Helper()
	cs, err := s.DocChunks(t.Context(), id)
	if err != nil {
		t.Fatalf("DocChunks(%s): %v", id, err)
	}
	return cs
}

func getChunk(t *testing.T, s knowledge.Store, id knowledge.ChunkID) (knowledge.Chunk, bool) {
	t.Helper()
	c, ok, err := s.GetChunk(t.Context(), id)
	if err != nil {
		t.Fatalf("GetChunk(%s): %v", id, err)
	}
	return c, ok
}

func neighbors(t *testing.T, s knowledge.Store, from []knowledge.NodeID, kinds []knowledge.EdgeKind, dir knowledge.Direction, limit int) []knowledge.Edge {
	t.Helper()
	es, err := s.Neighbors(t.Context(), from, kinds, dir, limit)
	if err != nil {
		t.Fatalf("Neighbors(%q, %v, %d, %d): %v", from, kinds, dir, limit, err)
	}
	return es
}

func wantEdges(t *testing.T, got []knowledge.Edge, want ...knowledge.Edge) {
	t.Helper()
	if len(got) == 0 && len(want) == 0 {
		return
	}
	if !slices.Equal(got, want) {
		t.Fatalf("edges:\n got  %+v\n want %+v", got, want)
	}
}

func generation(t *testing.T, s knowledge.Store) uint64 {
	t.Helper()
	g, err := s.Generation(t.Context())
	if err != nil {
		t.Fatalf("Generation: %v", err)
	}
	return g
}

func equalDoc(t *testing.T, got, want knowledge.Doc) {
	t.Helper()
	same := got.ID == want.ID && got.Scope == want.Scope && got.Path == want.Path &&
		got.Kind == want.Kind && got.Title == want.Title && got.Size == want.Size &&
		got.MTime.Truncate(time.Second).Equal(want.MTime.Truncate(time.Second)) &&
		got.Hash == want.Hash && maps.Equal(got.Props, want.Props) && got.Weight == want.Weight
	if !same {
		t.Fatalf("doc:\n got  %+v\n want %+v", got, want)
	}
}

func equalChunk(t *testing.T, got, want knowledge.Chunk) {
	t.Helper()
	same := got.ID == want.ID && got.Doc == want.Doc && got.Ord == want.Ord &&
		slices.Equal(got.Heading, want.Heading) && got.Symbol == want.Symbol &&
		got.Lines == want.Lines && got.Text == want.Text && got.Hash == want.Hash &&
		got.Tokens == want.Tokens
	if !same {
		t.Fatalf("chunk:\n got  %+v\n want %+v", got, want)
	}
}

func equalChunks(t *testing.T, got, want []knowledge.Chunk) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("got %d chunks, want %d: %+v", len(got), len(want), got)
	}
	for i := range got {
		equalChunk(t, got[i], want[i])
	}
}

func isClosed(t *testing.T, what string, err error) {
	t.Helper()
	if !errors.Is(err, knowledge.ErrClosed) {
		t.Errorf("%s after Close: err = %v, want ErrClosed", what, err)
	}
}

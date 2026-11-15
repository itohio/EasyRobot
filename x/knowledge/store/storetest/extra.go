package storetest

import (
	"fmt"
	"math"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// testCommonTermFloor: a term in more than half the chunks has a
// non-positive BM25 idf, which FTS5 floors at 1e-6. With equal row lengths
// and tf = 1 in the body (weight 1) every hit then scores
// 1e-6 · 1 · (k1+1) / (1 + k1) = 1e-6.
func testCommonTermFloor(t *testing.T, s knowledge.Store) {
	for _, id := range []string{"c/a.md", "c/b.md", "c/d.md"} {
		textDoc(t, s, id, "common xx yy")
	}
	textDoc(t, s, "c/e.md", "rare xx yy")
	hits := search(t, s, []string{"common"}, 10, knowledge.Filter{})
	wantDocs(t, hits, "c/a.md", "c/b.md", "c/d.md")
	for _, h := range hits {
		if math.Abs(h.Score-1e-6) > 1e-12 {
			t.Fatalf("%s score = %g, want 1e-6 (idf floor)", h.Doc, h.Score)
		}
	}
	hits = search(t, s, []string{"common", "rare"}, 10, knowledge.Filter{})
	if len(hits) != 4 || hits[0].Doc != "c/e.md" {
		t.Fatalf("rare term did not win over floored common term: %+v", hits)
	}
}

// testDeleteReAdd: a deleted document can be stored again under the same
// id with the same chunks and is fully searchable and gettable.
func testDeleteReAdd(t *testing.T, s knowledge.Store) {
	fillers(t, s, 3)
	d := mkDoc("re/a.md", "md", "ws")
	cs := []knowledge.Chunk{mkChunk(d.ID, 0, []string{"Readd"}, "", "phoenix rises"), mkChunk(d.ID, 1, nil, "", "again")}
	e := mkEdge(dn("re/a.md"), knowledge.TagNode("bird"), knowledge.Tagged, 1, "re/a.md")
	put(t, s, d, cs, e)
	before := search(t, s, []string{"phoenix", "again"}, 10, knowledge.Filter{})
	if err := s.DeleteDoc(t.Context(), d.ID); err != nil {
		t.Fatalf("DeleteDoc: %v", err)
	}
	if hits := search(t, s, []string{"phoenix"}, 10, knowledge.Filter{}); len(hits) != 0 {
		t.Fatalf("deleted doc searchable: %+v", hits)
	}
	put(t, s, d, cs, e)
	got, ok := getDoc(t, s, d.ID)
	if !ok {
		t.Fatal("re-added doc missing")
	}
	equalDoc(t, got, d)
	equalChunks(t, docChunks(t, s, d.ID), cs)
	after := search(t, s, []string{"phoenix", "again"}, 10, knowledge.Filter{})
	if len(after) != len(before) {
		t.Fatalf("hits after re-add = %+v, before %+v", after, before)
	}
	for i := range after {
		if after[i].Chunk != before[i].Chunk || !close6(after[i].Score, before[i].Score) {
			t.Fatalf("hit %d after re-add = %+v, before %+v", i, after[i], before[i])
		}
	}
	wantEdges(t, neighbors(t, s, []knowledge.NodeID{knowledge.TagNode("bird")}, nil, knowledge.Both, 0), e)
}

// testUnicode: non-ASCII text round-trips and its words are searchable.
func testUnicode(t *testing.T, s knowledge.Store) {
	fillers(t, s, 3)
	d := mkDoc("u/žodis.md", "md", "ws")
	c := mkChunk(d.ID, 0, []string{"Šaknis", "Ąžuolas"}, "", "Gražus žodynas ąžuolų giraitėje: 日本語 テキスト")
	put(t, s, d, []knowledge.Chunk{c})
	equalChunks(t, docChunks(t, s, d.ID), []knowledge.Chunk{c})
	got, ok := getDoc(t, s, d.ID)
	if !ok {
		t.Fatal("unicode doc missing")
	}
	equalDoc(t, got, d)
	for _, term := range []string{"gražus", "žodynas", "ąžuolų", "šaknis", "ąžuolas", "日本語", "テキスト", "žodis.md"} {
		hits := search(t, s, []string{term}, 10, knowledge.Filter{})
		wantDocs(t, hits, d.ID)
		if hits[0].Score <= 0 {
			t.Fatalf("%s: score %v", term, hits[0].Score)
		}
	}
	if hits := search(t, s, []string{"GRAŽUS"}, 10, knowledge.Filter{}); len(hits) != 1 {
		t.Fatalf("upper-case unicode query hits = %+v", hits)
	}
}

// testMoveDuplicateIDs: two chunks that differ only in their caller ids
// re-key to the same id; the move must fail without changing the store,
// and later deletes must work.
func testMoveDuplicateIDs(t *testing.T, s knowledge.Store) {
	a := mkDoc("dup/a.md", "md", "ws")
	c1 := mkChunk(a.ID, 0, nil, "", "same text")
	c2 := c1
	c1.ID = knowledge.ChunkID{1}
	c2.ID = knowledge.ChunkID{2}
	put(t, s, a, []knowledge.Chunk{c1, c2})
	other := mkDoc("dup/b.md", "md", "ws")
	put(t, s, other, []knowledge.Chunk{mkChunk(other.ID, 0, nil, "", "other text")})
	g := generation(t, s)

	to := mkDoc("dup/moved.md", "md", "ws")
	if err := s.MoveDoc(t.Context(), a.ID, to); err == nil {
		t.Fatal("MoveDoc with colliding re-keyed chunk ids succeeded")
	}
	if _, ok := getDoc(t, s, to.ID); ok {
		t.Fatal("failed move created the target doc")
	}
	if _, ok := getDoc(t, s, a.ID); !ok {
		t.Fatal("failed move removed the source doc")
	}
	equalChunks(t, docChunks(t, s, a.ID), []knowledge.Chunk{c1, c2})
	if hits := search(t, s, []string{"same"}, 10, knowledge.Filter{}); len(hits) != 2 {
		t.Fatalf("index changed by failed move: %+v", hits)
	}
	if g2 := generation(t, s); g2 != g {
		t.Fatalf("generation %d → %d after failed move", g, g2)
	}

	// Moving onto a doc whose chunk already holds a re-keyed id also fails.
	b := mkDoc("dup/c.md", "md", "ws")
	cb := mkChunk(b.ID, 0, nil, "", "clash")
	put(t, s, b, []knowledge.Chunk{cb})
	thief := mkDoc("dup/thief.md", "md", "ws")
	stolen := mkChunk(thief.ID, 0, nil, "", "clash")
	stolen.ID = knowledge.NewChunkID("dup/target.md", 0, "clash")
	put(t, s, thief, []knowledge.Chunk{stolen})
	if err := s.MoveDoc(t.Context(), b.ID, mkDoc("dup/target.md", "md", "ws")); err == nil {
		t.Fatal("MoveDoc colliding with another doc's chunk id succeeded")
	}
	equalChunks(t, docChunks(t, s, b.ID), []knowledge.Chunk{cb})
	equalChunks(t, docChunks(t, s, thief.ID), []knowledge.Chunk{stolen})

	for _, id := range []knowledge.DocID{a.ID, other.ID, b.ID, thief.ID} {
		if err := s.DeleteDoc(t.Context(), id); err != nil {
			t.Fatalf("DeleteDoc(%s): %v", id, err)
		}
	}
	if hits := search(t, s, []string{"same", "other", "clash"}, 10, knowledge.Filter{}); len(hits) != 0 {
		t.Fatalf("hits after deleting all: %+v", hits)
	}
}

func close6(a, b float64) bool {
	return math.Abs(a-b) <= 1e-6*math.Max(math.Abs(a), math.Abs(b))
}

// Agree stores the same corpus in two backends and checks that SearchLex
// returns the same chunks in the same order with scores equal within 1e-6
// relative on several queries, including a floored common term.
func Agree(t *testing.T, a, b Opener) {
	sa, sb := a(t), b(t)
	for _, s := range []knowledge.Store{sa, sb} {
		agreeCorpus(t, s)
	}
	queries := [][]string{
		{"alpha"}, {"alpha", "beta"}, {"common"}, {"common", "zeta"}, {"server"},
		{"clientpermissions", "client", "permission"}, {"lorem", "alpha"}, {"guide"},
		{"ąžuolas"}, {"handler", "request", "common"},
	}
	for _, q := range queries {
		ha := search(t, sa, q, 50, knowledge.Filter{})
		hb := search(t, sb, q, 50, knowledge.Filter{})
		if len(ha) == 0 {
			t.Fatalf("%q: no hits", q)
		}
		if len(ha) != len(hb) {
			t.Fatalf("%q: %d vs %d hits\n a %+v\n b %+v", q, len(ha), len(hb), ha, hb)
		}
		for i := range ha {
			if ha[i].Chunk != hb[i].Chunk || !close6(ha[i].Score, hb[i].Score) {
				t.Fatalf("%q hit %d: a %s %.12g vs b %s %.12g", q, i, ha[i].Doc, ha[i].Score, hb[i].Doc, hb[i].Score)
			}
		}
		if testing.Verbose() {
			for i := range ha {
				t.Logf("%q #%d %s a=%.12g b=%.12g", q, i, ha[i].Doc, ha[i].Score, hb[i].Score)
			}
		}
	}
}

func agreeCorpus(t *testing.T, s knowledge.Store) {
	fillers(t, s, 4)
	for i := range 6 {
		textDoc(t, s, fmt.Sprintf("com/c%d.md", i), fmt.Sprintf("common filler text number %d alpha", i))
	}
	textDoc(t, s, "docs/guide.md", "alpha beta gamma", "beta beta delta common", "zeta")
	h := mkDoc("docs/server.md", "md", "ws")
	put(t, s, h, []knowledge.Chunk{
		mkChunk(h.ID, 0, []string{"Server", "Guide"}, "", "the HTTPServer handles request routing"),
		mkChunk(h.ID, 1, []string{"Server", "Clients"}, "", "ClientPermissions gate every request common"),
	})
	g := mkDoc("code/handler.go", "go", "code")
	put(t, s, g, []knowledge.Chunk{
		mkChunk(g.ID, 0, nil, "Handler", "func Handler(w ResponseWriter, r *Request) { alpha() }"),
		mkChunk(g.ID, 1, nil, "ClientPermissions", "type ClientPermissions struct{ client string }"),
	})
	textDoc(t, s, "u/ąžuolas.md", "Ąžuolas auga giraitėje alpha alpha")
	// Delete and replace to exercise length bookkeeping.
	textDoc(t, s, "tmp/x.md", "alpha common beta beta")
	if err := s.DeleteDoc(t.Context(), "tmp/x.md"); err != nil {
		t.Fatal(err)
	}
	textDoc(t, s, "docs/guide.md", "alpha beta gamma guide", "beta beta delta common", "zeta zeta")
}

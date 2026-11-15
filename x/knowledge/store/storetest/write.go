package storetest

import (
	"crypto/sha256"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
)

func testRoundTrip(t *testing.T, s knowledge.Store) {
	d := mkDoc("relay/docs/auth.md", "md", "relay")
	d.Props = map[string]string{"id": "DEV-061", "tags": "auth,keys", "status": "done"}
	d.Weight = 0.5
	d.Size = 4321
	c0 := mkChunk(d.ID, 0, []string{"Authority model", "8. Teams"}, "", "Team leads restart clients.")
	c1 := mkChunk(d.ID, 1, []string{"Authority model", "9. Keys"}, "keys.ClientPermissions", "ClientPermissions grants rights.")
	put(t, s, d, []knowledge.Chunk{c1, c0}) // out of order on purpose

	got, ok := getDoc(t, s, d.ID)
	if !ok {
		t.Fatalf("GetDoc(%s) not found", d.ID)
	}
	equalDoc(t, got, d)
	equalChunks(t, docChunks(t, s, d.ID), []knowledge.Chunk{c0, c1})
	for _, c := range []knowledge.Chunk{c0, c1} {
		gc, ok := getChunk(t, s, c.ID)
		if !ok {
			t.Fatalf("GetChunk(%s) not found", c.ID)
		}
		equalChunk(t, gc, c)
	}

	// A bare document: nil props, zero time, no chunks.
	bare := knowledge.Doc{ID: "bare.txt", Kind: "text"}
	put(t, s, bare, nil)
	gotBare, ok := getDoc(t, s, bare.ID)
	if !ok {
		t.Fatal("bare doc not found")
	}
	equalDoc(t, gotBare, bare)
	if cs := docChunks(t, s, bare.ID); len(cs) != 0 {
		t.Fatalf("bare doc chunks = %+v, want none", cs)
	}

	// Missing entries.
	if _, ok := getDoc(t, s, "nope.md"); ok {
		t.Fatal("GetDoc(nope.md) found")
	}
	if _, ok := getChunk(t, s, knowledge.NewChunkID("nope.md", 0, "x")); ok {
		t.Fatal("GetChunk(missing) found")
	}
	if cs := docChunks(t, s, "nope.md"); len(cs) != 0 {
		t.Fatalf("DocChunks(nope.md) = %+v", cs)
	}

	// The store owns its copies: caller mutations never leak in or out.
	d.Props["status"] = "mutated"
	c0.Heading[0] = "mutated"
	got.Props["extra"] = "x"
	again, _ := getDoc(t, s, d.ID)
	if again.Props["status"] != "done" || again.Props["extra"] != "" {
		t.Fatalf("props aliased caller maps: %v", again.Props)
	}
	gc, _ := getChunk(t, s, c0.ID)
	if gc.Heading[0] != "Authority model" {
		t.Fatalf("heading aliased caller slice: %q", gc.Heading)
	}
}

func testPutReplaces(t *testing.T, s knowledge.Store) {
	a := mkDoc("r/a.md", "md", "r")
	b := mkDoc("r/b.md", "md", "r")
	put(t, s, b, nil, mkEdge(dn("r/b.md"), dn("r/a.md"), knowledge.LinksTo, 1, "r/b.md"))
	old := mkChunk(a.ID, 0, nil, "", "oldterm appears here")
	put(t, s, a, []knowledge.Chunk{old},
		mkEdge(dn("r/a.md"), dn("r/b.md"), knowledge.LinksTo, 1, "r/a.md"),
		mkEdge(dn("r/a.md"), knowledge.TagNode("old"), knowledge.Tagged, 1, "r/a.md"))

	a2 := a
	a2.Title = "v2"
	a2.Hash = sha256.Sum256([]byte("v2"))
	fresh := mkChunk(a.ID, 0, nil, "", "newterm appears here")
	put(t, s, a2, []knowledge.Chunk{fresh},
		mkEdge(dn("r/a.md"), knowledge.TagNode("new"), knowledge.Tagged, 1, "r/a.md"))

	if hits := search(t, s, []string{"oldterm"}, 10, knowledge.Filter{}); len(hits) != 0 {
		t.Fatalf("old chunk still searchable: %+v", hits)
	}
	if _, ok := getChunk(t, s, old.ID); ok {
		t.Fatal("old chunk still gettable")
	}
	wantDocs(t, search(t, s, []string{"newterm"}, 10, knowledge.Filter{}), a.ID)
	equalChunks(t, docChunks(t, s, a.ID), []knowledge.Chunk{fresh})
	got, _ := getDoc(t, s, a.ID)
	equalDoc(t, got, a2)

	wantEdges(t, neighbors(t, s, []knowledge.NodeID{dn("r/a.md")}, nil, knowledge.Out, 0),
		mkEdge(dn("r/a.md"), knowledge.TagNode("new"), knowledge.Tagged, 1, "r/a.md"))
	wantEdges(t, neighbors(t, s, []knowledge.NodeID{knowledge.TagNode("old")}, nil, knowledge.Both, 0))
	wantEdges(t, neighbors(t, s, []knowledge.NodeID{dn("r/a.md")}, nil, knowledge.In, 0),
		mkEdge(dn("r/b.md"), dn("r/a.md"), knowledge.LinksTo, 1, "r/b.md"))

	// Empty Chunk.Doc, zero Chunk.ID and empty Edge.Source are filled
	// from the document.
	c := knowledge.Chunk{Ord: 0, Text: "implicit owner"}
	put(t, s, mkDoc("r/c.md", "md", "r"), []knowledge.Chunk{c},
		knowledge.Edge{From: dn("r/c.md"), To: knowledge.TagNode("implicit"), Kind: knowledge.Tagged, Weight: 1})
	cs := docChunks(t, s, "r/c.md")
	if len(cs) != 1 || cs[0].Doc != "r/c.md" || cs[0].ID != knowledge.NewChunkID("r/c.md", 0, "implicit owner") {
		t.Fatalf("implicit chunk = %+v", cs)
	}
	wantEdges(t, neighbors(t, s, []knowledge.NodeID{knowledge.TagNode("implicit")}, nil, knowledge.In, 0),
		mkEdge(dn("r/c.md"), knowledge.TagNode("implicit"), knowledge.Tagged, 1, "r/c.md"))
}

func testPutRejectsInvalid(t *testing.T, s knowledge.Store) {
	a := mkDoc("v/a.md", "md", "v")
	ca := mkChunk(a.ID, 0, nil, "", "keepme")
	put(t, s, a, []knowledge.Chunk{ca})
	g := generation(t, s)

	other := mkDoc("v/b.md", "md", "v")
	bad := map[string]knowledge.DocUpdate{
		"empty id": {Doc: knowledge.Doc{Kind: "md"}},
		"edge source mismatch": {Doc: other, Edges: []knowledge.Edge{
			mkEdge(dn("v/b.md"), knowledge.TagNode("x"), knowledge.Tagged, 1, "v/a.md")}},
		"chunk doc mismatch": {Doc: other, Chunks: []knowledge.Chunk{mkChunk("v/a.md", 0, nil, "", "x")}},
		"duplicate chunk id": {Doc: other, Chunks: []knowledge.Chunk{
			mkChunk(other.ID, 0, nil, "", "dup"), mkChunk(other.ID, 0, nil, "", "dup")}},
		"chunk id owned by another doc": {Doc: other, Chunks: []knowledge.Chunk{
			{ID: ca.ID, Doc: other.ID, Text: "steal"}}},
	}
	for name, u := range bad {
		if err := s.PutDoc(t.Context(), u); err == nil {
			t.Errorf("PutDoc(%s) succeeded, want error", name)
		}
	}
	if got := generation(t, s); got != g {
		t.Fatalf("generation moved on failed writes: %d → %d", g, got)
	}
	if _, ok := getDoc(t, s, other.ID); ok {
		t.Fatal("rejected doc was stored")
	}
	equalChunks(t, docChunks(t, s, a.ID), []knowledge.Chunk{ca})
	wantDocs(t, search(t, s, []string{"keepme"}, 10, knowledge.Filter{}), a.ID)
}

func testDelete(t *testing.T, s knowledge.Store) {
	a := mkDoc("d/a.md", "md", "d")
	b := mkDoc("d/b.md", "md", "d")
	ca := mkChunk(a.ID, 0, nil, "", "deleteterm alpha")
	put(t, s, a, []knowledge.Chunk{ca},
		mkEdge(dn("d/a.md"), dn("d/b.md"), knowledge.LinksTo, 1, "d/a.md"),
		mkEdge(dn("d/a.md"), knowledge.TagNode("x"), knowledge.Tagged, 1, "d/a.md"),
		mkEdge(dn("d/a.md"), knowledge.IDNode("DEL-1"), knowledge.Defines, 1, "d/a.md"))
	put(t, s, b, []knowledge.Chunk{mkChunk(b.ID, 0, nil, "", "alpha")},
		mkEdge(dn("d/b.md"), dn("d/a.md"), knowledge.LinksTo, 1, "d/b.md"))

	if err := s.DeleteDoc(t.Context(), a.ID); err != nil {
		t.Fatalf("DeleteDoc: %v", err)
	}
	if _, ok := getDoc(t, s, a.ID); ok {
		t.Fatal("deleted doc still in catalog")
	}
	if cs := docChunks(t, s, a.ID); len(cs) != 0 {
		t.Fatalf("deleted doc chunks = %+v", cs)
	}
	if _, ok := getChunk(t, s, ca.ID); ok {
		t.Fatal("deleted chunk still gettable")
	}
	if hits := search(t, s, []string{"deleteterm"}, 10, knowledge.Filter{}); len(hits) != 0 {
		t.Fatalf("deleted chunk searchable: %+v", hits)
	}
	wantDocs(t, search(t, s, []string{"alpha"}, 10, knowledge.Filter{}), b.ID)
	// Edges b asserts towards a dangle but stay.
	wantEdges(t, neighbors(t, s, []knowledge.NodeID{dn("d/a.md")}, nil, knowledge.Both, 0),
		mkEdge(dn("d/b.md"), dn("d/a.md"), knowledge.LinksTo, 1, "d/b.md"))
	wantEdges(t, neighbors(t, s, []knowledge.NodeID{knowledge.TagNode("x")}, nil, knowledge.Both, 0))
	if n, err := s.Degree(t.Context(), knowledge.TagNode("x")); err != nil || n != 0 {
		t.Fatalf("Degree(tag:x) = %d, %v; want 0", n, err)
	}
	if _, ok, err := s.Resolve(t.Context(), knowledge.IDNode("DEL-1")); err != nil || ok {
		t.Fatalf("Resolve(id:DEL-1) = %v, %v; want false", ok, err)
	}
	if err := s.DeleteDoc(t.Context(), "d/missing.md"); err != nil {
		t.Fatalf("DeleteDoc(missing) = %v, want nil", err)
	}
}

func testMove(t *testing.T, s knowledge.Store) {
	const oldID, newID = "old/dir/a.md", "new/dir/a.md"
	a := mkDoc(oldID, "md", "old")
	c0 := mkChunk(a.ID, 0, nil, "", "moveterm alpha")
	c1 := mkChunk(a.ID, 1, []string{"Sec"}, "", "beta")
	put(t, s, a, []knowledge.Chunk{c0, c1},
		mkEdge(dn(oldID), knowledge.TagNode("t"), knowledge.Tagged, 1, oldID),
		mkEdge(dn(oldID), dn("other/c.md"), knowledge.LinksTo, 1, oldID),
		mkEdge(dn(oldID), dn(oldID), knowledge.LinksTo, 0.5, oldID))
	put(t, s, mkDoc("other/b.md", "md", "other"), nil,
		mkEdge(dn("other/b.md"), dn(oldID), knowledge.LinksTo, 1, "other/b.md"))
	put(t, s, mkDoc("other/c.md", "md", "other"), nil)

	to := mkDoc(newID, "md", "new")
	to.Title = "moved"
	to.Hash = a.Hash
	if err := s.MoveDoc(t.Context(), oldID, to); err != nil {
		t.Fatalf("MoveDoc: %v", err)
	}

	if _, ok := getDoc(t, s, oldID); ok {
		t.Fatal("old id still in catalog")
	}
	got, ok := getDoc(t, s, newID)
	if !ok {
		t.Fatal("new id not in catalog")
	}
	equalDoc(t, got, to)

	var want []knowledge.Chunk
	for _, c := range []knowledge.Chunk{c0, c1} {
		c.Doc = newID
		c.ID = knowledge.NewChunkID(newID, c.Ord, c.Text)
		want = append(want, c)
	}
	equalChunks(t, docChunks(t, s, newID), want)
	if cs := docChunks(t, s, oldID); len(cs) != 0 {
		t.Fatalf("old id still has chunks: %+v", cs)
	}
	if _, ok := getChunk(t, s, c0.ID); ok {
		t.Fatal("chunk still gettable under its old id")
	}
	if gc, ok := getChunk(t, s, want[0].ID); !ok || gc.Doc != newID {
		t.Fatalf("GetChunk(new id) = %+v, %v", gc, ok)
	}

	hits := search(t, s, []string{"moveterm"}, 10, knowledge.Filter{})
	wantDocs(t, hits, newID)
	if hits[0].Chunk != want[0].ID {
		t.Fatalf("hit chunk = %s, want %s", hits[0].Chunk, want[0].ID)
	}
	if hits := search(t, s, []string{"new"}, 10, knowledge.Filter{}); len(hits) != 2 || hits[0].Doc != newID {
		t.Fatalf("path field not re-keyed: %+v", hits)
	}
	if hits := search(t, s, []string{"old"}, 10, knowledge.Filter{}); len(hits) != 0 {
		t.Fatalf("old path still indexed: %+v", hits)
	}
	wantDocs(t, search(t, s, []string{"moveterm"}, 10, knowledge.Filter{In: []string{"new"}}), newID)

	wantEdges(t, neighbors(t, s, []knowledge.NodeID{dn(newID)}, nil, knowledge.Out, 0),
		mkEdge(dn(newID), dn("other/c.md"), knowledge.LinksTo, 1, newID),
		mkEdge(dn(newID), knowledge.TagNode("t"), knowledge.Tagged, 1, newID),
		mkEdge(dn(newID), dn(newID), knowledge.LinksTo, 0.5, newID))
	wantEdges(t, neighbors(t, s, []knowledge.NodeID{dn(newID)}, nil, knowledge.In, 0),
		mkEdge(dn("other/b.md"), dn(newID), knowledge.LinksTo, 1, "other/b.md"),
		mkEdge(dn(newID), dn(newID), knowledge.LinksTo, 0.5, newID))
	wantEdges(t, neighbors(t, s, []knowledge.NodeID{dn(oldID)}, nil, knowledge.Both, 0))

	// Moving onto itself updates the catalog only.
	to.Path = "/elsewhere/a.md"
	if err := s.MoveDoc(t.Context(), newID, to); err != nil {
		t.Fatalf("MoveDoc(same id): %v", err)
	}
	got, _ = getDoc(t, s, newID)
	equalDoc(t, got, to)
	equalChunks(t, docChunks(t, s, newID), want)

	if err := s.MoveDoc(t.Context(), "missing.md", mkDoc("x.md", "md", "")); err == nil {
		t.Fatal("MoveDoc(missing) succeeded")
	}
	if err := s.MoveDoc(t.Context(), newID, knowledge.Doc{}); err == nil {
		t.Fatal("MoveDoc to empty id succeeded")
	}
}

func testMoveOntoExisting(t *testing.T, s knowledge.Store) {
	a := mkDoc("m/a.md", "md", "m")
	put(t, s, a, []knowledge.Chunk{mkChunk(a.ID, 0, nil, "", "sourceterm")})
	target := mkDoc("m/t.md", "md", "m")
	put(t, s, target, []knowledge.Chunk{mkChunk(target.ID, 0, nil, "", "targetterm")},
		mkEdge(dn("m/t.md"), knowledge.TagNode("z"), knowledge.Tagged, 1, "m/t.md"))

	if err := s.MoveDoc(t.Context(), a.ID, target); err != nil {
		t.Fatalf("MoveDoc onto existing: %v", err)
	}
	if hits := search(t, s, []string{"targetterm"}, 10, knowledge.Filter{}); len(hits) != 0 {
		t.Fatalf("replaced target still searchable: %+v", hits)
	}
	wantDocs(t, search(t, s, []string{"sourceterm"}, 10, knowledge.Filter{}), target.ID)
	wantEdges(t, neighbors(t, s, []knowledge.NodeID{knowledge.TagNode("z")}, nil, knowledge.Both, 0))
	if _, ok := getDoc(t, s, a.ID); ok {
		t.Fatal("source id still in catalog")
	}
}

func testTouch(t *testing.T, s knowledge.Store) {
	d := mkDoc("t/a.md", "md", "t")
	d.Props = map[string]string{"status": "todo"}
	c := mkChunk(d.ID, 0, []string{"H"}, "", "touchterm")
	put(t, s, d, []knowledge.Chunk{c})
	t2 := baseTime.Add(time.Hour + 5*time.Second)
	if err := s.TouchDoc(t.Context(), d.ID, t2); err != nil {
		t.Fatalf("TouchDoc: %v", err)
	}
	want := d
	want.MTime = t2
	got, _ := getDoc(t, s, d.ID)
	equalDoc(t, got, want)
	equalChunks(t, docChunks(t, s, d.ID), []knowledge.Chunk{c})
	wantDocs(t, search(t, s, []string{"touchterm"}, 10, knowledge.Filter{}), d.ID)
	if err := s.TouchDoc(t.Context(), "t/missing.md", t2); err == nil {
		t.Fatal("TouchDoc(missing) succeeded")
	}
}

func testGeneration(t *testing.T, s knowledge.Store) {
	g := generation(t, s)
	step := func(what string, err error) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", what, err)
		}
		n := generation(t, s)
		if n <= g {
			t.Fatalf("generation after %s = %d, want > %d", what, n, g)
		}
		g = n
	}
	ctx := t.Context()
	a, b := mkDoc("g/a.md", "md", "g"), mkDoc("g/b.md", "md", "g")
	step("put a", s.PutDoc(ctx, knowledge.DocUpdate{Doc: a, Chunks: []knowledge.Chunk{mkChunk(a.ID, 0, nil, "", "x")}}))
	step("put b", s.PutDoc(ctx, knowledge.DocUpdate{Doc: b, Edges: []knowledge.Edge{
		mkEdge(dn("g/b.md"), dn("g/a.md"), knowledge.LinksTo, 1, "g/b.md")}}))
	step("touch a", s.TouchDoc(ctx, a.ID, baseTime.Add(time.Minute)))
	step("move a", s.MoveDoc(ctx, a.ID, mkDoc("g/a2.md", "md", "g")))
	step("delete b", s.DeleteDoc(ctx, b.ID))

	if err := s.TouchDoc(ctx, "g/missing.md", baseTime); err == nil {
		t.Fatal("TouchDoc(missing) succeeded")
	}
	if err := s.MoveDoc(ctx, "g/missing.md", mkDoc("g/z.md", "md", "g")); err == nil {
		t.Fatal("MoveDoc(missing) succeeded")
	}
	if err := s.PutDoc(ctx, knowledge.DocUpdate{}); err == nil {
		t.Fatal("PutDoc(empty) succeeded")
	}
	if n := generation(t, s); n != g {
		t.Fatalf("generation after failed writes = %d, want %d", n, g)
	}
}

func testReopen(t *testing.T, s knowledge.Store, reopen func(*testing.T, knowledge.Store) knowledge.Store) {
	a := mkDoc("p/a.md", "md", "p")
	a.Props = map[string]string{"id": "RE-1"}
	ca := mkChunk(a.ID, 0, []string{"Persist", "Me"}, "p.Sym", "persistterm survives restart")
	put(t, s, a, []knowledge.Chunk{ca},
		mkEdge(dn("p/a.md"), knowledge.IDNode("RE-1"), knowledge.Defines, 1, "p/a.md"))
	put(t, s, mkDoc("p/b.md", "md", "p"), nil,
		mkEdge(dn("p/b.md"), dn("p/a.md"), knowledge.LinksTo, 1, "p/b.md"))
	if err := s.TouchDoc(t.Context(), "p/b.md", baseTime.Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	g := generation(t, s)

	s = reopen(t, s)
	got, ok := getDoc(t, s, a.ID)
	if !ok {
		t.Fatal("doc lost on reopen")
	}
	equalDoc(t, got, a)
	equalChunks(t, docChunks(t, s, a.ID), []knowledge.Chunk{ca})
	wantDocs(t, search(t, s, []string{"persistterm"}, 10, knowledge.Filter{}), a.ID)
	wantEdges(t, neighbors(t, s, []knowledge.NodeID{dn("p/a.md")}, nil, knowledge.In, 0),
		mkEdge(dn("p/b.md"), dn("p/a.md"), knowledge.LinksTo, 1, "p/b.md"))
	if d, ok, err := s.Resolve(t.Context(), knowledge.IDNode("re-1")); err != nil || !ok || d != a.ID {
		t.Fatalf("Resolve after reopen = %q, %v, %v", d, ok, err)
	}
	if n := generation(t, s); n != g {
		t.Fatalf("generation after reopen = %d, want %d", n, g)
	}
	put(t, s, mkDoc("p/c.md", "md", "p"), nil)
	if n := generation(t, s); n <= g {
		t.Fatalf("generation after reopen+put = %d, want > %d", n, g)
	}
}

func testListDocs(t *testing.T, s knowledge.Store) {
	for _, err := range s.ListDocs(t.Context()) {
		if err != nil {
			t.Fatalf("ListDocs(empty): %v", err)
		}
		t.Fatal("ListDocs on an empty store yielded a doc")
	}
	want := map[knowledge.DocID]knowledge.Doc{}
	for _, id := range []string{"l/c.md", "l/a.md", "l/b/z.go", "l/b.md"} {
		d := mkDoc(id, "md", "l")
		want[d.ID] = d
		put(t, s, d, nil)
	}
	var ids []knowledge.DocID
	for d, err := range s.ListDocs(t.Context()) {
		if err != nil {
			t.Fatalf("ListDocs: %v", err)
		}
		equalDoc(t, d, want[d.ID])
		ids = append(ids, d.ID)
	}
	wantIDs := []knowledge.DocID{"l/a.md", "l/b.md", "l/b/z.go", "l/c.md"}
	if fmt.Sprint(ids) != fmt.Sprint(wantIDs) {
		t.Fatalf("ListDocs order = %q, want %q", ids, wantIDs)
	}
	n := 0
	for range s.ListDocs(t.Context()) {
		n++
		break
	}
	if n != 1 {
		t.Fatalf("early break yielded %d", n)
	}
}

func testConcurrent(t *testing.T, s knowledge.Store) {
	ctx := t.Context()
	const docs = 20
	id := func(prefix string, i int) knowledge.DocID {
		return knowledge.DocID(fmt.Sprintf("c/%s%02d.md", prefix, i))
	}
	for i := range docs {
		d := mkDoc(string(id("d", i)), "md", "c")
		put(t, s, d, []knowledge.Chunk{mkChunk(d.ID, 0, nil, "", fmt.Sprintf("shared term%d", i))},
			mkEdge(knowledge.DocNode(d.ID), knowledge.DocNode(id("d", (i+1)%docs)), knowledge.LinksTo, 1, string(d.ID)))
	}

	stop := make(chan struct{})
	errs := make(chan error, 16)
	read := func() error {
		if _, err := s.SearchLex(ctx, knowledge.LexQuery{Terms: []string{"shared", "term3"}}, 5, knowledge.Filter{In: []string{"c"}}); err != nil {
			return err
		}
		if _, err := s.Neighbors(ctx, []knowledge.NodeID{knowledge.DocNode(id("d", 0))}, nil, knowledge.Both, 3); err != nil {
			return err
		}
		if _, _, err := s.GetDoc(ctx, id("d", 1)); err != nil {
			return err
		}
		cs, err := s.DocChunks(ctx, id("d", 2))
		if err != nil {
			return err
		}
		for _, c := range cs {
			if _, _, err := s.GetChunk(ctx, c.ID); err != nil {
				return err
			}
		}
		if _, err := s.Generation(ctx); err != nil {
			return err
		}
		if _, err := s.Degree(ctx, knowledge.DocNode(id("d", 3))); err != nil {
			return err
		}
		if _, _, err := s.Resolve(ctx, knowledge.DocNode(id("d", 4))); err != nil {
			return err
		}
		for _, err := range s.ListDocs(ctx) {
			if err != nil {
				return err
			}
		}
		return nil
	}
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			for n := 0; ; n++ {
				select {
				case <-stop:
					if n > 0 {
						return
					}
				default:
				}
				if err := read(); err != nil {
					errs <- err
					return
				}
			}
		})
	}

	for i := range 30 {
		w := mkDoc(string(id("w", i)), "md", "c")
		if err := s.PutDoc(ctx, knowledge.DocUpdate{Doc: w, Chunks: []knowledge.Chunk{mkChunk(w.ID, 0, nil, "", "shared writer")}}); err != nil {
			t.Error(err)
		}
		if err := s.TouchDoc(ctx, id("d", i%docs), baseTime.Add(time.Duration(i)*time.Second)); err != nil {
			t.Error(err)
		}
		if i%3 == 0 {
			if err := s.MoveDoc(ctx, w.ID, mkDoc(string(id("m", i)), "md", "c")); err != nil {
				t.Error(err)
			}
		}
		if i%6 == 0 {
			if err := s.DeleteDoc(ctx, id("m", i)); err != nil {
				t.Error(err)
			}
		}
	}
	close(stop)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("reader: %v", err)
	}

	n := 0
	for _, err := range s.ListDocs(ctx) {
		if err != nil {
			t.Fatal(err)
		}
		n++
	}
	if want := docs + 30 - 5; n != want { // 5 of the 10 moved docs deleted
		t.Fatalf("docs after concurrent writes = %d, want %d", n, want)
	}
}

func testClosed(t *testing.T, s knowledge.Store) {
	ctx := t.Context()
	a := mkDoc("x/a.md", "md", "x")
	put(t, s, a, nil)
	if err := s.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	_, _, err := s.GetDoc(ctx, a.ID)
	isClosed(t, "GetDoc", err)
	yielded := false
	for _, err := range s.ListDocs(ctx) {
		yielded = true
		isClosed(t, "ListDocs", err)
		break
	}
	if !yielded {
		t.Error("ListDocs after Close yielded nothing, want ErrClosed")
	}
	_, err = s.Generation(ctx)
	isClosed(t, "Generation", err)
	_, err = s.SearchLex(ctx, knowledge.LexQuery{Terms: []string{"x"}}, 5, knowledge.Filter{})
	isClosed(t, "SearchLex", err)
	_, _, err = s.GetChunk(ctx, knowledge.ChunkID{})
	isClosed(t, "GetChunk", err)
	_, err = s.DocChunks(ctx, a.ID)
	isClosed(t, "DocChunks", err)
	_, err = s.Neighbors(ctx, []knowledge.NodeID{dn("x/a.md")}, nil, knowledge.Both, 0)
	isClosed(t, "Neighbors", err)
	_, err = s.Degree(ctx, dn("x/a.md"))
	isClosed(t, "Degree", err)
	_, _, err = s.Resolve(ctx, dn("x/a.md"))
	isClosed(t, "Resolve", err)
	isClosed(t, "PutDoc", s.PutDoc(ctx, knowledge.DocUpdate{Doc: a}))
	isClosed(t, "DeleteDoc", s.DeleteDoc(ctx, a.ID))
	isClosed(t, "MoveDoc", s.MoveDoc(ctx, a.ID, mkDoc("x/b.md", "md", "x")))
	isClosed(t, "TouchDoc", s.TouchDoc(ctx, a.ID, baseTime))
	if err := s.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
}

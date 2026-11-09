package storetest

import (
	"slices"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/token"
)

func testSearchLex(t *testing.T, open Opener) {
	none := knowledge.Filter{}

	t.Run("MoreTermsRankHigher", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 10)
		textDoc(t, s, "rank/one.md", "alpha gamma delta omega")
		textDoc(t, s, "rank/both.md", "alpha beta gamma delta")
		wantDocs(t, search(t, s, []string{"alpha", "beta"}, 10, none), "rank/both.md", "rank/one.md")
	})

	t.Run("RarerTermRanksHigher", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 10)
		textDoc(t, s, "idf/x.md", "common apple banana cherry")
		textDoc(t, s, "idf/y.md", "rare apple banana cherry")
		for _, id := range []string{"idf/c1.md", "idf/c2.md", "idf/c3.md", "idf/c4.md"} {
			textDoc(t, s, id, "common melon grape lemon")
		}
		hits := search(t, s, []string{"rare", "common"}, 10, none)
		if len(hits) != 6 || hits[0].Doc != "idf/y.md" {
			t.Fatalf("rare term did not win: %+v", hits)
		}
	})

	t.Run("TermFrequencySaturates", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 10)
		textDoc(t, s, "sat/many.md", "foo foo foo foo foo foo foo foo foo foo")
		textDoc(t, s, "sat/once.md", "foo one two three four five six seven eight nine")
		hits := search(t, s, []string{"foo"}, 10, none)
		wantDocs(t, hits, "sat/many.md", "sat/once.md")
		if ratio := hits[0].Score / hits[1].Score; ratio <= 1 || ratio >= 3 {
			t.Fatalf("score(tf=10)/score(tf=1) = %.3f, want in (1, 3): tf must saturate", ratio)
		}
	})

	t.Run("HeadingOutranksBody", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 10)
		body := mkDoc("fw/body.md", "md", "ws")
		put(t, s, body, []knowledge.Chunk{mkChunk(body.ID, 0, []string{"other"}, "", "zeta bb cc")})
		head := mkDoc("fw/head.md", "md", "ws")
		put(t, s, head, []knowledge.Chunk{mkChunk(head.ID, 0, []string{"zeta"}, "", "aa bb cc")})
		wantDocs(t, search(t, s, []string{"zeta"}, 10, none), "fw/head.md", "fw/body.md")
	})

	t.Run("SymbolOutranksBody", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 10)
		body := mkDoc("fw/body.go", "go", "ws")
		put(t, s, body, []knowledge.Chunk{mkChunk(body.ID, 0, nil, "Other", "zeta bb cc")})
		sym := mkDoc("fw/sym.go", "go", "ws")
		put(t, s, sym, []knowledge.Chunk{mkChunk(sym.ID, 0, nil, "Zeta", "aa bb cc")})
		wantDocs(t, search(t, s, []string{"zeta"}, 10, none), "fw/sym.go", "fw/body.go")
	})

	t.Run("PathCounts", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 10)
		textDoc(t, s, "other/x.md", "zeta bb cc")
		textDoc(t, s, "zeta/x.md", "aa bb cc")
		hits := search(t, s, []string{"zeta"}, 10, none)
		wantDocs(t, hits, "zeta/x.md", "other/x.md")
		if !slices.Equal(hits[0].Matched, []string{"zeta"}) {
			t.Fatalf("path-only hit Matched = %q", hits[0].Matched)
		}
	})

	t.Run("Identifiers", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 10)
		textDoc(t, s, "ids/ticket.md", "Ticket ABU-061 calls abg.svc.find to check PeerPermissions.")
		textDoc(t, s, "ids/split.md", "abu 061 abg svc find peer permissions")
		for _, term := range []string{"abu-061", "abg.svc.find", "peerpermissions"} {
			hits := search(t, s, []string{term}, 10, none)
			wantDocs(t, hits, "ids/ticket.md")
			if !slices.Equal(hits[0].Matched, []string{term}) {
				t.Fatalf("%s: Matched = %q", term, hits[0].Matched)
			}
		}
		q := token.Query("abg.svc.find")
		hits := search(t, s, q, 10, none)
		wantDocs(t, hits, "ids/ticket.md", "ids/split.md")
		if !slices.Equal(hits[0].Matched, q) || !slices.Equal(hits[1].Matched, []string{"abg", "svc", "find"}) {
			t.Fatalf("Matched = %q / %q", hits[0].Matched, hits[1].Matched)
		}
	})

	t.Run("ORSemanticsAndMatched", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 10)
		textDoc(t, s, "or/a.md", "alpha aa bb")
		textDoc(t, s, "or/b.md", "beta aa bb")
		textDoc(t, s, "or/ab.md", "alpha beta aa")
		hits := search(t, s, []string{"beta", "alpha", "nomatchterm"}, 10, none)
		if len(hits) != 3 || hits[0].Doc != "or/ab.md" {
			t.Fatalf("OR query hits = %+v", hits)
		}
		want := map[knowledge.DocID][]string{"or/a.md": {"alpha"}, "or/b.md": {"beta"}, "or/ab.md": {"beta", "alpha"}}
		for _, h := range hits {
			if !slices.Equal(h.Matched, want[h.Doc]) {
				t.Errorf("%s Matched = %q, want %q", h.Doc, h.Matched, want[h.Doc])
			}
			if h.Score <= 0 {
				t.Errorf("%s Score = %v, want > 0", h.Doc, h.Score)
			}
		}
		// Upper-case and duplicated terms behave like their normalised form.
		again := search(t, s, []string{"BETA", "alpha", "alpha"}, 10, none)
		if !slices.Equal(hitDocs(again), hitDocs(hits)) || !slices.Equal(again[0].Matched, []string{"beta", "alpha"}) {
			t.Fatalf("normalised query = %+v, want %+v", again, hits)
		}
	})

	t.Run("KRespected", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 5)
		textDoc(t, s, "k/1.md", "kterm aa bb cc dd")
		textDoc(t, s, "k/2.md", "kterm kterm bb cc dd")
		textDoc(t, s, "k/3.md", "kterm kterm kterm cc dd")
		textDoc(t, s, "k/4.md", "kterm kterm kterm kterm dd")
		all := search(t, s, []string{"kterm"}, 10, none)
		wantDocs(t, all, "k/4.md", "k/3.md", "k/2.md", "k/1.md")
		top := search(t, s, []string{"kterm"}, 2, none)
		wantDocs(t, top, "k/4.md", "k/3.md")
		if top[0].Score != all[0].Score || top[1].Score != all[1].Score {
			t.Fatalf("scores depend on k: %+v vs %+v", top, all)
		}
		if hits := search(t, s, []string{"kterm"}, 0, none); len(hits) != 0 {
			t.Fatalf("k=0 returned %+v", hits)
		}
	})

	t.Run("FilterBeforeTopK", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 10)
		strong := mkDoc("notes/strong.md", "md", "notes")
		put(t, s, strong, []knowledge.Chunk{mkChunk(strong.ID, 0, nil, "", "fterm fterm fterm aa")})
		code := mkDoc("code/x.go", "go", "code")
		put(t, s, code, []knowledge.Chunk{mkChunk(code.ID, 0, nil, "", "fterm fterm aa bb")})
		weak := mkDoc("keep/weak.md", "md", "keep")
		put(t, s, weak, []knowledge.Chunk{mkChunk(weak.ID, 0, nil, "", "fterm aa bb cc")})
		q := []string{"fterm"}

		wantDocs(t, search(t, s, q, 1, none), "notes/strong.md")
		wantDocs(t, search(t, s, q, 1, knowledge.Filter{In: []string{"keep"}}), "keep/weak.md")
		wantDocs(t, search(t, s, q, 1, knowledge.Filter{In: []string{"keep/"}}), "keep/weak.md")
		wantDocs(t, search(t, s, q, 1, knowledge.Filter{In: []string{"keep/weak.md"}}), "keep/weak.md")
		wantDocs(t, search(t, s, q, 10, knowledge.Filter{In: []string{"kee"}}))
		wantDocs(t, search(t, s, q, 10, knowledge.Filter{In: []string{"keep", "code"}}), "code/x.go", "keep/weak.md")
		wantDocs(t, search(t, s, q, 1, knowledge.Filter{Kinds: []string{"go"}}), "code/x.go")
		wantDocs(t, search(t, s, q, 1, knowledge.Filter{Scopes: []string{"keep"}}), "keep/weak.md")
		wantDocs(t, search(t, s, q, 1, knowledge.Filter{Not: []knowledge.DocID{"notes/strong.md"}}), "code/x.go")
		wantDocs(t, search(t, s, q, 10, knowledge.Filter{Kinds: []string{"md"}, Not: []knowledge.DocID{"notes/strong.md"}}), "keep/weak.md")
	})

	t.Run("EmptyQuery", func(t *testing.T) {
		s := open(t)
		textDoc(t, s, "e/a.md", "alpha")
		for _, terms := range [][]string{nil, {}, {""}, {" "}, {"!!"}} {
			if hits := search(t, s, terms, 10, none); len(hits) != 0 {
				t.Fatalf("query %q returned %+v", terms, hits)
			}
		}
	})

	t.Run("TiesByDocThenOrd", func(t *testing.T) {
		s := open(t)
		fillers(t, s, 10)
		b := mkDoc("tie/b.md", "md", "ws")
		cb := mkChunk(b.ID, 0, nil, "", "tieterm aa")
		put(t, s, b, []knowledge.Chunk{cb})
		a := mkDoc("tie/a.md", "md", "ws")
		ca1 := mkChunk(a.ID, 1, nil, "", "tieterm aa")
		ca0 := mkChunk(a.ID, 0, nil, "", "tieterm aa")
		put(t, s, a, []knowledge.Chunk{ca1, ca0})
		hits := search(t, s, []string{"tieterm"}, 10, none)
		got := []knowledge.ChunkID{}
		for _, h := range hits {
			got = append(got, h.Chunk)
		}
		if want := []knowledge.ChunkID{ca0.ID, ca1.ID, cb.ID}; !slices.Equal(got, want) {
			t.Fatalf("tie order = %v, want %v (hits %+v)", got, want, hits)
		}
		if hits[0].Score != hits[2].Score {
			t.Fatalf("tie scores differ: %+v", hits)
		}
		wantDocs(t, search(t, s, []string{"tieterm"}, 2, none), "tie/a.md", "tie/a.md")
	})
}

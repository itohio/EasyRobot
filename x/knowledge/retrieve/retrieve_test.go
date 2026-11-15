package retrieve

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/itohio/EasyRobot/x/knowledge"
)

func TestIDQueryReturnsDefiningDocFirst(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("notes/authority.md", sec{text: "Client authority model decision record. " + filler(30)}).definesID("DEV-061"),
			doc("misc/chatter.md", sec{text: repeat("DEV-061", 10)}).mentionsID("DEV-061"),
			doc("misc/other.md", sec{text: filler(30)}),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "DEV-061"})
		wantFirst(t, res, "notes/authority.md")
		wantWhy(t, res.Hits[0], "exact id DEV-061")
		if res.Confidence != "high" {
			t.Fatalf("confidence %q, want high", res.Confidence)
		}
		// The chatter doc is found lexically and also via the id graph.
		wantWhy(t, hitOf(t, res, "misc/chatter.md"), "mentions DEV-061 defined in notes/authority.md")
		// Lower-case ids resolve too.
		wantFirst(t, find(t, newRetriever(s), knowledge.Query{Text: "what is dev-061 about"}), "notes/authority.md")
	})
}

func TestSymbolQuery(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("code/keys/keys.go",
				sec{heading: "package keys", text: "package keys " + filler(20)},
				sec{heading: "ClientPermissions", symbol: "keys.ClientPermissions", text: "func ClientPermissions(p Client) Permissions { return build(p) }"},
			).definesSym("keys.ClientPermissions"),
			doc("docs/perm.md", sec{heading: "ClientPermissions", text: repeat("keys.ClientPermissions ClientPermissions client permissions", 8)}),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "where is keys.ClientPermissions built"})
		wantFirst(t, res, "code/keys/keys.go")
		wantWhy(t, res.Hits[0], "defines keys.ClientPermissions")
		if res.Confidence != "high" {
			t.Fatalf("confidence %q, want high", res.Confidence)
		}
		// The exact doc contributes the chunk declaring the symbol.
		if sec := res.Hits[0].Sections; len(sec) == 0 || sec[0].Lines != [2]int{11, 19} {
			t.Fatalf("sections %+v, want first span 11-19", sec)
		}
	})
}

func TestFileNameQuery(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("code/keys/keys.go", sec{text: "package keys " + filler(20)}),
			doc("docs/notes.md", sec{text: repeat("keys.go", 10)}),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "keys.go"})
		wantFirst(t, res, "code/keys/keys.go")
		wantWhy(t, res.Hits[0], "file name keys.go")
		if res.Confidence != "high" {
			t.Fatalf("confidence %q, want high", res.Confidence)
		}
	})
}

func TestBM25Ordering(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("a/one.md", sec{text: "raft consensus " + filler(20)}),
			doc("b/two.md", sec{text: "raft " + filler(21)}),
			doc("c/three.md", sec{text: filler(22)}),
			doc("d/four.md", sec{text: "consensus raft raft log replication raft " + filler(16)}),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "raft consensus"})
		if got, want := hitIDs(res), []string{"d/four.md", "a/one.md", "b/two.md"}; !slices.Equal(got, want) {
			t.Fatalf("order %v, want %v", got, want)
		}
		for i := 1; i < len(res.Hits); i++ {
			if res.Hits[i].Score > res.Hits[i-1].Score {
				t.Fatalf("scores not descending: %v", res.Hits)
			}
		}
	})
}

func TestFieldWeighting(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("a/b.md", sec{heading: "Overview", text: "telemetry " + filler(29)}),
			doc("b/h.md", sec{heading: "Telemetry", text: filler(30)}),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "telemetry"})
		if got, want := hitIDs(res), []string{"b/h.md", "a/b.md"}; !slices.Equal(got, want) {
			t.Fatalf("order %v, want heading match first %v", got, want)
		}
	})
}

func TestGraphExpansionLinkedDoc(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("a/seed.md", sec{text: "quantum entanglement " + filler(20)}).links("b/linked.md"),
			doc("b/linked.md", sec{text: filler(25)}),
			doc("c/linker.md", sec{text: filler(26)}).links("a/seed.md"),
			doc("d/unrelated.md", sec{text: filler(27)}),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "quantum", Explain: true})
		wantFirst(t, res, "a/seed.md")
		linked := hitOf(t, res, "b/linked.md")
		wantWhy(t, linked, "linked from a/seed.md")
		if linked.Signals["graph"] <= 0 {
			t.Fatalf("linked graph signal %v", linked.Signals)
		}
		wantWhy(t, hitOf(t, res, "c/linker.md"), "links to a/seed.md")
		if rankOf(res, "d/unrelated.md") >= 0 {
			t.Fatalf("unrelated doc returned: %v", hitIDs(res))
		}
		// A graph-only hit still points at a span to read.
		if len(linked.Sections) == 0 || linked.Sections[0].Lines != [2]int{1, 9} {
			t.Fatalf("linked sections %+v", linked.Sections)
		}
	})
}

func TestHubDamping(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		hub := doc("b/hub.md", sec{text: filler(20)})
		var fill []*tdoc
		for _, d := range folderDocs("f", 20, filler(10)) {
			hub.links(string(d.u.Doc.ID))
			fill = append(fill, d)
		}
		load(t, s, fill...)
		load(t, s,
			doc("a/seed.md", sec{text: "quantum " + filler(20)}).links("b/hub.md", "c/leaf.md"),
			hub,
			doc("c/leaf.md", sec{text: filler(20)}),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "quantum", Explain: true})
		seed, h, l := hitOf(t, res, "a/seed.md"), hitOf(t, res, "b/hub.md"), hitOf(t, res, "c/leaf.md")
		wantBefore(t, res, "c/leaf.md", "b/hub.md")
		fused := seed.Signals["fused"]
		if fused <= 0 {
			t.Fatalf("seed signals %v", seed.Signals)
		}
		wantLeaf := 0.3 * fused / math.Log2(2+1)
		wantHub := 0.3 * fused / math.Log2(2+21)
		if !near(l.Signals["graph"], wantLeaf) || !near(h.Signals["graph"], wantHub) {
			t.Fatalf("graph boosts leaf %v hub %v, want %v %v", l.Signals["graph"], h.Signals["graph"], wantLeaf, wantHub)
		}
	})
}

func TestTwoHopThroughIDs(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("a/a.md", sec{text: "zebra stripes " + filler(20)}).mentionsID("DOC-7"),
			doc("b/b.md", sec{text: filler(21)}).definesID("DOC-7"),
			doc("c/c.md", sec{text: "zebra herd " + filler(20)}).definesID("DOC-9"),
			doc("d/d.md", sec{text: filler(22)}).mentionsID("DOC-9"),
			doc("e/e.md", sec{text: filler(23)}).mentionsID("DOC-7"),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "zebra"})
		wantWhy(t, hitOf(t, res, "b/b.md"), "defines DOC-7 mentioned in a/a.md")
		wantWhy(t, hitOf(t, res, "d/d.md"), "mentions DOC-9 defined in c/c.md")
		if rankOf(res, "e/e.md") >= 0 {
			t.Fatalf("co-mentioning doc returned: %v", hitIDs(res))
		}
	})
}

func TestFilterBeforeTopK(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		strong := []string{"x1/s.md", "x2/s.md", "x3/s.md", "x4/s.md"}
		for _, id := range strong {
			load(t, s, doc(id, sec{heading: "Orbit", text: repeat("orbit", 6) + " " + filler(10)}))
		}
		load(t, s,
			doc("y/weak.md", sec{text: "orbit " + filler(30)}).links("z/n.md"),
			doc("z/n.md", sec{text: filler(12)}),
			doc("g/code.go", sec{text: "orbit " + filler(31)}),
			doc("x1/def.md", sec{text: filler(12)}).definesID("OR-1"),
		)
		r := newRetriever(s)
		if got := hitIDs(find(t, r, knowledge.Query{Text: "orbit", K: 1, Filter: knowledge.Filter{In: []string{"y"}}})); !slices.Equal(got, []string{"y/weak.md"}) {
			t.Fatalf("In: got %v", got)
		}
		if got := hitIDs(find(t, r, knowledge.Query{Text: "orbit", K: 1, Filter: knowledge.Filter{Kinds: []string{"go"}}})); !slices.Equal(got, []string{"g/code.go"}) {
			t.Fatalf("Kinds: got %v", got)
		}
		not := make([]knowledge.DocID, len(strong))
		for i, id := range strong {
			not[i] = knowledge.DocID(id)
		}
		got := hitIDs(find(t, r, knowledge.Query{Text: "orbit", K: 1, Filter: knowledge.Filter{Not: not}}))
		if len(got) != 1 || slices.Contains(strong, got[0]) {
			t.Fatalf("Not: got %v", got)
		}
		// Exact and graph candidates obey the filter too.
		if got := hitIDs(find(t, r, knowledge.Query{Text: "OR-1 orbit", Filter: knowledge.Filter{In: []string{"y"}}})); slices.Contains(got, "x1/def.md") || slices.Contains(got, "z/n.md") {
			t.Fatalf("filtered exact/graph docs returned: %v", got)
		}
		if got := hitIDs(find(t, r, knowledge.Query{Text: "OR-1", Filter: knowledge.Filter{Not: []knowledge.DocID{"x1/def.md"}}})); slices.Contains(got, "x1/def.md") {
			t.Fatalf("Not exact doc returned: %v", got)
		}
		if got := hitIDs(find(t, r, knowledge.Query{Text: "orbit"})); !slices.Contains(got, "z/n.md") {
			t.Fatalf("unfiltered graph neighbour missing: %v", got)
		}
	})
}

func TestDiversityCap(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s, folderDocs("same", 5, repeat("nebula", 5)+" "+filler(10))...)
		load(t, s, folderDocs("other", 2, "nebula "+filler(30))...)
		count := func(res knowledge.Result, dir string) int {
			n := 0
			for _, id := range hitIDs(res) {
				if strings.HasPrefix(id, dir+"/") {
					n++
				}
			}
			return n
		}
		r := newRetriever(s)
		res := find(t, r, knowledge.Query{Text: "nebula"})
		if count(res, "same") != 3 || count(res, "other") != 2 {
			t.Fatalf("diversity: got %v", hitIDs(res))
		}
		res = find(t, r, knowledge.Query{Text: "nebula", Filter: knowledge.Filter{In: []string{"same/"}}})
		if count(res, "same") != 5 || len(res.Hits) != 5 {
			t.Fatalf("In exemption: got %v", hitIDs(res))
		}
		res = find(t, newRetriever(s, WithMaxPerFolder(1)), knowledge.Query{Text: "nebula"})
		if count(res, "same") != 1 || count(res, "other") != 1 {
			t.Fatalf("WithMaxPerFolder(1): got %v", hitIDs(res))
		}
	})
}

// twin returns two documents with identical text, so only priors differ.
func twin(a, b string) (*tdoc, *tdoc) {
	return doc(a, sec{text: "comet tail " + filler(10)}), doc(b, sec{text: "comet tail " + filler(10)})
}

func TestRecencyPriorIsRelative(t *testing.T) {
	for _, flip := range []bool{false, true} {
		forEachStore(t, func(t *testing.T, s knowledge.Store) {
			a, b := twin("r1/a.md", "r2/b.md")
			fresh, stale := a, b
			if flip {
				fresh, stale = b, a
			}
			// Both notes are years old: an absolute decay would make them
			// equal; ranking against the candidates still separates them.
			fresh.prop("updated", "2020-02-01")
			stale.prop("updated", "2019-01-01").mtime(now) // Props win over MTime.
			load(t, s, a, b)
			r := newRetriever(s)
			res := find(t, r, knowledge.Query{Text: "comet", Explain: true})
			wantFirst(t, res, string(fresh.u.Doc.ID))
			wantWhy(t, res.Hits[0], "recency ×1.1")
			if got := res.Hits[0].Signals["recency"]; !near(got, 1.1) {
				t.Fatalf("newer recency %v, want 1.1", got)
			}
			if got := hitOf(t, res, string(stale.u.Doc.ID)).Signals["recency"]; got != 1 {
				t.Fatalf("older recency %v, want 1", got)
			}
			// Recency intent strengthens the prior.
			for _, q := range []string{"latest comet", "the current comet", "comet superseded"} {
				res := find(t, r, knowledge.Query{Text: q, Explain: true})
				if got := hitOf(t, res, string(fresh.u.Doc.ID)).Signals["recency"]; !near(got, 1.3) {
					t.Fatalf("%q: recency %v, want 1.3", q, got)
				}
			}
		})
	}
}

func TestRecencyRanksAgainstCandidates(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		a, b := twin("r1/a.md", "r2/b.md")
		c, d := twin("r3/c.md", "r4/d.md")
		a.prop("updated", "2026-03-01")
		b.prop("updated", "2025-01-01")
		c.prop("updated", "2025-06-01")
		d.mtime(now) // only an mtime: undated, not in the pool
		load(t, s, a, b, c, d)
		res := find(t, newRetriever(s), knowledge.Query{Text: "comet", Explain: true})
		want := map[string]float64{"r2/b.md": 1, "r3/c.md": 1.05, "r1/a.md": 1.1, "r4/d.md": 1}
		for id, w := range want {
			if got := hitOf(t, res, id).Signals["recency"]; !near(got, w) {
				t.Fatalf("%s recency %v, want %v", id, got, w)
			}
		}
		if got := hitIDs(res); got[0] != "r1/a.md" || got[1] != "r3/c.md" {
			t.Fatalf("order %v", got)
		}
	})
}

func TestRecencyDateSources(t *testing.T) {
	cases := []struct {
		name       string
		old, young *tdoc // young must win; its ID sorts last so ties cannot help
	}{
		{"created", doc("r1/a.md", sec{text: "comet tail"}).prop("created", "2026-01-01"), doc("r2/b.md", sec{text: "comet tail"}).prop("created", "2026-05-01")},
		{"file date", doc("r1/n-2026-03-15.md", sec{text: "comet tail"}), doc("r2/n-2026-10-01.md", sec{text: "comet tail"})},
		{"file month", doc("r1/c-2026-03.md", sec{text: "comet tail"}), doc("r2/c-2026-10.md", sec{text: "comet tail"})},
	}
	for _, tc := range cases {
		forEachStore(t, func(t *testing.T, s knowledge.Store) {
			young, old := tc.young, tc.old
			load(t, s, old, young)
			res := find(t, newRetriever(s), knowledge.Query{Text: "comet", Explain: true})
			wantFirst(t, res, string(young.u.Doc.ID))
			if got := res.Hits[0].Signals["recency"]; !near(got, 1.1) {
				t.Fatalf("%s: recency %v, want 1.1", tc.name, got)
			}
		})
	}
}

// TestRecencyIgnoresMTime: a document with only an mtime is undated, so
// checkout or touch times cannot reorder results.
func TestRecencyIgnoresMTime(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		a, b := twin("r1/a.md", "r2/b.md")
		a.mtime(now)
		b.mtime(oldTime)
		load(t, s, a, b)
		res := find(t, newRetriever(s), knowledge.Query{Text: "latest comet", Explain: true})
		for _, id := range []string{"r1/a.md", "r2/b.md"} {
			if got := hitOf(t, res, id).Signals["recency"]; got != 1 {
				t.Fatalf("%s recency %v, want 1 (undated)", id, got)
			}
		}
	})
}

func TestSupersedesDemotes(t *testing.T) {
	for _, ref := range []string{"old", "old.md", "[[old]]", "[[old|Old plan]]", `[[old\|Old plan]]`, "s/a/old.md", "other, old"} {
		forEachStore(t, func(t *testing.T, s knowledge.Store) {
			old, young := twin("s/a/old.md", "s/new.md")
			young.prop("supersedes", ref)
			load(t, s, old, young)
			res := find(t, newRetriever(s), knowledge.Query{Text: "comet", Explain: true})
			wantFirst(t, res, "s/new.md")
			h := hitOf(t, res, "s/a/old.md")
			wantWhy(t, h, "superseded by s/new.md ×0.8")
			if h.Signals["status"] != 0.8 || !near(h.Score, 0.8*res.Hits[0].Score) {
				t.Fatalf("%q: old %+v vs new %v", ref, h.Signals, res.Hits[0].Score)
			}
		})
	}
}

// TestSupersedesStaysInScope: a bare name only demotes same-named files in
// the superseding doc's directory or scope, not other projects' files.
func TestSupersedesStaysInScope(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		mine, theirs := twin("p1/docs/architecture.md", "p2/architecture.md")
		young := doc("p1/new.md", sec{text: "comet tail " + filler(10)}).prop("supersedes", "[[architecture|Arch]]")
		load(t, s, mine, theirs, young)
		res := find(t, newRetriever(s), knowledge.Query{Text: "comet", Explain: true})
		if got := hitOf(t, res, "p1/docs/architecture.md").Signals["status"]; got != 0.8 {
			t.Fatalf("same-scope target status %v, want 0.8", got)
		}
		if got, ok := hitOf(t, res, "p2/architecture.md").Signals["status"]; ok && got != 1 {
			t.Fatalf("other project's architecture.md demoted: status %v", got)
		}
	})
}

func TestLexicalScoresRelativeToTopHit(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("a/strong.md", sec{text: "raft raft raft consensus " + filler(6)}),
			doc("b/multi.md",
				sec{heading: "One", text: "raft " + filler(9)},
				sec{heading: "Two", text: "raft " + filler(10)},
				sec{heading: "Three", text: "raft " + filler(11)},
			),
			doc("c/weak.md", sec{text: "consensus " + filler(9)}),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "raft consensus", Explain: true})
		// One strong chunk beats several mediocre ones (rank fusion with
		// k=60 put multi.md first: 1/62 + 0.3/63 > 1/61).
		wantFirst(t, res, "a/strong.md")
		for _, h := range res.Hits[1:] {
			if f := h.Signals["fused"]; f <= 0 || f >= 1 {
				t.Fatalf("%s: fused %v, want in (0,1) relative to the top hit", h.Doc, f)
			}
		}
		if res.Hits[0].Signals["fused"] != 1 {
			t.Fatalf("top fused %v, want 1", res.Hits[0].Signals["fused"])
		}
	})
}

func TestBareSymbolResolvesThroughPackage(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		keys := doc("code/keys/keys.go",
			sec{heading: "package keys", text: "package keys " + filler(20)},
			sec{heading: "ClientPermissions", symbol: "keys.ClientPermissions", text: "func ClientPermissions(p Client) Permissions { return build(p) }"},
		).definesSym("keys.ClientPermissions").prop("package", "keys")
		load(t, s, keys,
			doc("docs/perm.md", sec{heading: "ClientPermissions", text: repeat("ClientPermissions client permissions", 8)}),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "ClientPermissions"})
		wantFirst(t, res, "code/keys/keys.go")
		wantWhy(t, res.Hits[0], "defines keys.ClientPermissions")
		if res.Confidence != "high" || res.Hits[0].Sections[0].Lines != [2]int{11, 19} {
			t.Fatalf("confidence %q sections %+v", res.Confidence, res.Hits[0].Sections)
		}
	})
}

func TestStatusPrior(t *testing.T) {
	for _, status := range []string{"done", "superseded", "archived", "Deprecated"} {
		for _, flip := range []bool{false, true} {
			forEachStore(t, func(t *testing.T, s knowledge.Store) {
				a := doc("s1/a.md", sec{text: "comet tail " + filler(10)}).prop("status", "active")
				b := doc("s2/b.md", sec{text: "comet tail " + filler(10)})
				dead, live := a, b
				if flip {
					dead, live = b, a
				}
				dead.prop("status", status)
				load(t, s, a, b)
				res := find(t, newRetriever(s), knowledge.Query{Text: "comet", Explain: true})
				wantFirst(t, res, string(live.u.Doc.ID))
				h := hitOf(t, res, string(dead.u.Doc.ID))
				wantWhy(t, h, "status "+strings.ToLower(status))
				if h.Signals["status"] != 0.8 {
					t.Fatalf("status signal %v", h.Signals["status"])
				}
			})
		}
	}
}

func TestWeightMultiplier(t *testing.T) {
	for _, flip := range []bool{false, true} {
		forEachStore(t, func(t *testing.T, s knowledge.Store) {
			a := doc("w1/a.md", sec{text: "comet tail " + filler(10)})
			b := doc("w2/b.md", sec{text: "comet tail " + filler(10)})
			light := a
			if flip {
				light = b
			}
			light.weight(0.5)
			load(t, s, a, b)
			res := find(t, newRetriever(s), knowledge.Query{Text: "comet", Explain: true})
			if len(res.Hits) != 2 || string(res.Hits[1].Doc) != string(light.u.Doc.ID) {
				t.Fatalf("got %v, want %s last", hitIDs(res), light.u.Doc.ID)
			}
			if res.Hits[1].Signals["weight"] != 0.5 || res.Hits[1].Score >= 0.51*res.Hits[0].Score {
				t.Fatalf("weight not applied: %+v", res.Hits)
			}
			wantWhy(t, res.Hits[1], "weight ×0.5")
		})
	}
}

func TestSections(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("s/doc.md",
				sec{heading: "Intro", text: filler(20)},
				sec{heading: "Two", text: filler(15) + "\n\n  alpha   beta\tgamma " + filler(15)},
				sec{heading: "Three", text: "alpha " + filler(25)},
				sec{heading: "Four", text: "beta " + filler(25)},
				sec{heading: "Five", text: "alpha beta " + filler(5)},
			),
		)
		res := find(t, newRetriever(s), knowledge.Query{Text: "alpha beta"})
		h := hitOf(t, res, "s/doc.md")
		if h.Path != "/ws/s/doc.md" || h.Kind != "md" || h.Title != "doc" {
			t.Fatalf("hit metadata %+v", h)
		}
		if len(h.Sections) != 3 {
			t.Fatalf("got %d sections, want 3: %+v", len(h.Sections), h.Sections)
		}
		var alpha *knowledge.Section
		for i, sc := range h.Sections {
			if sc.Heading == "doc › Intro" {
				t.Fatalf("non-matching section returned: %+v", sc)
			}
			if !strings.HasPrefix(sc.Heading, "doc › ") {
				t.Fatalf("heading %q", sc.Heading)
			}
			if n := utf8.RuneCountInString(sc.Snippet); n == 0 || n > 160 {
				t.Fatalf("snippet length %d: %q", n, sc.Snippet)
			}
			if strings.ContainsAny(sc.Snippet, "\n\t") || strings.Contains(sc.Snippet, "  ") {
				t.Fatalf("whitespace not collapsed: %q", sc.Snippet)
			}
			if sc.Heading == "doc › Two" {
				alpha = &h.Sections[i]
			}
		}
		// Five matches both terms in the shortest text: best section.
		if h.Sections[0].Heading != "doc › Five" || h.Sections[0].Lines != [2]int{41, 49} {
			t.Fatalf("best section %+v", h.Sections[0])
		}
		if !slices.Equal(h.Sections[0].Why, []string{"terms: alpha, beta"}) {
			t.Fatalf("section why %q", h.Sections[0].Why)
		}
		if alpha == nil {
			t.Fatalf("Alpha section missing: %+v", h.Sections)
		}
		if alpha.Lines != [2]int{11, 19} || !strings.Contains(alpha.Snippet, "alpha beta gamma") || !strings.HasPrefix(alpha.Snippet, "…") {
			t.Fatalf("alpha section %+v", *alpha)
		}
	})
}

func TestConfidence(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("a/one.md", sec{text: "kestrel falcon " + filler(10)}),
			doc("b/two.md", sec{text: "talon raptor raptor raptor wing wing " + filler(10)}),
			doc("c/three.md", sec{text: "harrier " + filler(10)}),
			doc("d/four.md", sec{text: "harrier buzzard " + filler(10)}),
			doc("e/five.md", sec{text: "harrier buzzard " + filler(11)}),
		)
		r := newRetriever(s)
		if res := find(t, r, knowledge.Query{Text: "kestrel falcon"}); res.Confidence != "high" || len(res.TryTerms) != 0 {
			t.Fatalf("dominant full-coverage top: %q %v %v", res.Confidence, hitIDs(res), res.TryTerms)
		}
		if res := find(t, r, knowledge.Query{Text: "harrier buzzard"}); res.Confidence != "medium" {
			t.Fatalf("close race: %q %v", res.Confidence, hitIDs(res))
		}
		res := find(t, r, knowledge.Query{Text: "talon osprey heron egret ibis"})
		if res.Confidence != "low" {
			t.Fatalf("low coverage: %q %v", res.Confidence, hitIDs(res))
		}
		if len(res.TryTerms) == 0 || len(res.TryTerms) > 5 || res.TryTerms[0] != "raptor" {
			t.Fatalf("try terms %v", res.TryTerms)
		}
		for _, term := range res.TryTerms {
			if slices.Contains([]string{"talon", "osprey", "heron", "egret", "ibis"}, term) {
				t.Fatalf("try terms repeat the query: %v", res.TryTerms)
			}
		}
		res = find(t, r, knowledge.Query{Text: "nonexistentword"})
		if res.Confidence != "low" || len(res.Hits) != 0 {
			t.Fatalf("no hits: %q %v", res.Confidence, hitIDs(res))
		}
	})
}

func TestKDefaultAndCap(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		for i := range 40 {
			load(t, s, doc("k"+string(rune('a'+i/26))+string(rune('a'+i%26))+"/doc.md", sec{text: "pulsar " + filler(5+i)}))
		}
		r := newRetriever(s)
		for _, tc := range []struct{ k, want int }{{0, 8}, {-3, 8}, {3, 3}, {100, 30}} {
			if got := len(find(t, r, knowledge.Query{Text: "pulsar", K: tc.k}).Hits); got != tc.want {
				t.Fatalf("K=%d: got %d hits, want %d", tc.k, got, tc.want)
			}
		}
	})
}

func TestNearSeeds(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("n/near.md", sec{text: "pulsar " + filler(10)}).links("m/m.md"),
			doc("m/m.md", sec{text: filler(11)}),
			doc("o/o.md", sec{text: filler(12)}).links("n/near.md"),
			doc("p/p.md", sec{text: "pulsar " + filler(13)}),
		)
		r := newRetriever(s)
		res := find(t, r, knowledge.Query{Near: []knowledge.DocID{"n/near.md"}})
		if rankOf(res, "n/near.md") >= 0 {
			t.Fatalf("near doc returned: %v", hitIDs(res))
		}
		wantWhy(t, hitOf(t, res, "m/m.md"), "linked from n/near.md")
		wantWhy(t, hitOf(t, res, "o/o.md"), "links to n/near.md")
		res = find(t, r, knowledge.Query{Text: "pulsar", Near: []knowledge.DocID{"n/near.md"}})
		if rankOf(res, "n/near.md") >= 0 || rankOf(res, "p/p.md") < 0 || rankOf(res, "m/m.md") < 0 {
			t.Fatalf("near with text: %v", hitIDs(res))
		}
	})
}

func TestEmptyQuery(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s, doc("a/a.md", sec{text: "the of and " + filler(5)}))
		r := newRetriever(s)
		for _, text := range []string{"", "   ", "the of and"} {
			res, err := r.Find(context.Background(), knowledge.Query{Text: text})
			if err != nil || len(res.Hits) != 0 || res.Confidence != "low" {
				t.Fatalf("%q: %v %v %q", text, err, hitIDs(res), res.Confidence)
			}
			if res.Docs != 1 {
				t.Fatalf("%q: docs %d", text, res.Docs)
			}
		}
	})
}

func TestGenerationAndDocs(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s, folderDocs("g", 3, "pulsar")...)
		r := newRetriever(s)
		res := find(t, r, knowledge.Query{Text: "pulsar"})
		gen, err := s.Generation(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		if res.Docs != 3 || res.Generation != gen {
			t.Fatalf("docs %d gen %d, want 3 %d", res.Docs, res.Generation, gen)
		}
		load(t, s, doc("h/new.md", sec{text: "pulsar"}))
		res = find(t, r, knowledge.Query{Text: "pulsar"})
		if res.Docs != 4 || res.Generation <= gen {
			t.Fatalf("after put: docs %d gen %d", res.Docs, res.Generation)
		}
	})
}

func TestExplainSignals(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("a/a.md", sec{text: "pulsar " + filler(5)}).definesID("PU-1"),
			doc("b/b.md", sec{text: "pulsar pulsar " + filler(5)}),
		)
		r := newRetriever(s)
		if res := find(t, r, knowledge.Query{Text: "pulsar"}); res.Hits[0].Signals != nil {
			t.Fatalf("signals without Explain: %v", res.Hits[0].Signals)
		}
		res := find(t, r, knowledge.Query{Text: "PU-1 pulsar", Explain: true})
		wantFirst(t, res, "a/a.md")
		sig := res.Hits[0].Signals
		for _, k := range []string{"bm25_rank", "exact", "fused", "graph", "weight", "recency", "status"} {
			if _, ok := sig[k]; !ok {
				t.Fatalf("signal %q missing: %v", k, sig)
			}
		}
		if sig["exact"] != 1 || hitOf(t, res, "b/b.md").Signals["exact"] != 0 {
			t.Fatalf("exact signals %v", sig)
		}
		if hitOf(t, res, "b/b.md").Signals["bm25_rank"] != 1 {
			t.Fatalf("bm25 rank %v", hitOf(t, res, "b/b.md").Signals)
		}
	})
}

func TestCancelledContext(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s, doc("a/a.md", sec{text: "pulsar"}))
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := newRetriever(s).Find(ctx, knowledge.Query{Text: "pulsar"}); !errors.Is(err, context.Canceled) {
			t.Fatalf("err %v, want context.Canceled", err)
		}
	})
}

func near(a, b float64) bool { return math.Abs(a-b) <= 1e-9*math.Max(1, math.Abs(b)) }

func TestConcurrentFind(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s, folderDocs("c", 6, "pulsar quasar")...)
		load(t, s, doc("d/def.md", sec{text: "quasar"}).definesID("QU-1").links("c/d00.md"))
		r := newRetriever(s)
		queries := []knowledge.Query{{Text: "pulsar"}, {Text: "QU-1 quasar"}, {Near: []knowledge.DocID{"d/def.md"}}}
		want := make([]knowledge.Result, len(queries))
		for i, q := range queries {
			want[i] = find(t, r, q)
		}
		errs := make(chan error, 8*len(queries))
		for range 8 {
			for i, q := range queries {
				go func() {
					got, err := r.Find(context.Background(), q)
					if err == nil && !slices.Equal(hitIDs(got), hitIDs(want[i])) {
						err = errors.New("result differs: " + strings.Join(hitIDs(got), ","))
					}
					errs <- err
				}()
			}
		}
		for range 8 * len(queries) {
			if err := <-errs; err != nil {
				t.Fatal(err)
			}
		}
	})
}

func TestNeighborsCapPerSeed(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		seed := doc("a/seed.md", sec{text: "quantum " + filler(10)})
		// n0 has degree 1 (the seed link), n1 degree 2, … n4 degree 5; the
		// seed's link to n_i weighs 1+i. The cap keeps the heaviest edges.
		for i := range 5 {
			id := "n" + string(rune('0'+i)) + "/n.md"
			seed.edge(knowledge.DocNode(knowledge.DocID(id)), knowledge.LinksTo)
			seed.u.Edges[len(seed.u.Edges)-1].Weight = float32(1 + i)
			d := doc(id, sec{text: filler(10)})
			for j := range i {
				d.links("f/f" + string(rune('0'+j)) + ".md")
			}
			load(t, s, d)
		}
		load(t, s, seed)
		res := find(t, newRetriever(s, WithNeighbors(2)), knowledge.Query{Text: "quantum"})
		// n3 (degree 4) is boosted more than n4 (degree 5).
		if got, want := hitIDs(res), []string{"a/seed.md", "n3/n.md", "n4/n.md"}; !slices.Equal(got, want) {
			t.Fatalf("got %v, want %v", got, want)
		}
	})
}

// countingStore counts per-document lookups.
type countingStore struct {
	knowledge.Store
	degree, getDoc int
}

func (c *countingStore) Degree(ctx context.Context, n knowledge.NodeID) (int, error) {
	c.degree++
	return c.Store.Degree(ctx, n)
}

func (c *countingStore) GetDoc(ctx context.Context, id knowledge.DocID) (knowledge.Doc, bool, error) {
	c.getDoc++
	return c.Store.GetDoc(ctx, id)
}

func TestHubSeedCostsNeighborsLookups(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		hub := doc("a/hub.md", sec{text: "quantum " + filler(10)})
		for _, d := range folderDocs("n", 40, filler(10)) {
			hub.links(string(d.u.Doc.ID))
			load(t, s, d)
		}
		load(t, s, hub)
		c := &countingStore{Store: s}
		res := find(t, New(c, WithNeighbors(3)), knowledge.Query{Text: "quantum"})
		// One seed: Degree for its 3 kept neighbours, GetDoc for the seed
		// and those 3; not one per link.
		if c.degree > 3 || c.getDoc > 3+1 {
			t.Fatalf("Degree %d GetDoc %d, want ≤ Neighbors (3) and ≤ Neighbors+1 (hits %v)", c.degree, c.getDoc, hitIDs(res))
		}
	})
}

func TestAcronymIsNotPackageQualified(t *testing.T) {
	forEachStore(t, func(t *testing.T, s knowledge.Store) {
		load(t, s,
			doc("relay/internal/keys/keys.go", sec{heading: "ClientPermissions", symbol: "keys.ClientPermissions",
				text: "ClientPermissions builds the NATS permission set for a client: publish and subscribe allow lists are built from the registry"}).
				definesSym("keys.ClientPermissions").prop("package", "keys"),
			doc("pkg/config/config.go", sec{heading: "NATS", symbol: "config.NATS", text: "NATS holds the NATS server url and credentials file"}).
				definesSym("config.NATS").prop("package", "config"),
		)
		r := newRetriever(s)
		res := find(t, r, knowledge.Query{Text: "how is a client's NATS permission set built"})
		wantFirst(t, res, "relay/internal/keys/keys.go")
		if h := hitOf(t, res, "pkg/config/config.go"); hasWhy(h.Why, "defines config.NATS") {
			t.Fatalf("acronym NATS qualified as config.NATS: %q", h.Why)
		}
		// Written in full, the symbol is still an exact hit.
		res = find(t, r, knowledge.Query{Text: "config.NATS"})
		wantFirst(t, res, "pkg/config/config.go")
		wantWhy(t, res.Hits[0], "defines config.NATS")
	})
}

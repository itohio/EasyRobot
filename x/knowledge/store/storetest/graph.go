package storetest

import (
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
)

const (
	ga, gb, gc = "g/a.md", "g/b.md", "g/c.md"
)

// graphFixture stores three documents asserting:
//
//	a: a→b links 1, a→c links 2, a→tag:x tagged 0.5, a→id:ABU-1 defines 1
//	b: b→a links 1, b→c links 1, b→sym:pkg.F defines 1
//	c: c→a mentions 0.7, c→sym:pkg.F defines 1, c→tag:x tagged 0.5
func graphFixture(t *testing.T, s knowledge.Store) {
	t.Helper()
	put(t, s, mkDoc(ga, "md", "g"), nil,
		mkEdge(dn(ga), dn(gb), knowledge.LinksTo, 1, ga),
		mkEdge(dn(ga), dn(gc), knowledge.LinksTo, 2, ga),
		mkEdge(dn(ga), knowledge.TagNode("x"), knowledge.Tagged, 0.5, ga),
		mkEdge(dn(ga), knowledge.IDNode("ABU-1"), knowledge.Defines, 1, ga))
	put(t, s, mkDoc(gb, "md", "g"), nil,
		mkEdge(dn(gb), dn(ga), knowledge.LinksTo, 1, gb),
		mkEdge(dn(gb), dn(gc), knowledge.LinksTo, 1, gb),
		mkEdge(dn(gb), knowledge.SymNode("pkg.F"), knowledge.Defines, 1, gb))
	put(t, s, mkDoc(gc, "go", "g"), nil,
		mkEdge(dn(gc), dn(ga), knowledge.Mentions, 0.7, gc),
		mkEdge(dn(gc), knowledge.SymNode("pkg.F"), knowledge.Defines, 1, gc),
		mkEdge(dn(gc), knowledge.TagNode("x"), knowledge.Tagged, 0.5, gc))
}

func testNeighbors(t *testing.T, s knowledge.Store) {
	graphFixture(t, s)
	var (
		ab   = mkEdge(dn(ga), dn(gb), knowledge.LinksTo, 1, ga)
		ac   = mkEdge(dn(ga), dn(gc), knowledge.LinksTo, 2, ga)
		ax   = mkEdge(dn(ga), knowledge.TagNode("x"), knowledge.Tagged, 0.5, ga)
		aid  = mkEdge(dn(ga), knowledge.IDNode("ABU-1"), knowledge.Defines, 1, ga)
		ba   = mkEdge(dn(gb), dn(ga), knowledge.LinksTo, 1, gb)
		bc   = mkEdge(dn(gb), dn(gc), knowledge.LinksTo, 1, gb)
		ca   = mkEdge(dn(gc), dn(ga), knowledge.Mentions, 0.7, gc)
		cx   = mkEdge(dn(gc), knowledge.TagNode("x"), knowledge.Tagged, 0.5, gc)
		seed = func(n ...knowledge.NodeID) []knowledge.NodeID { return n }
		link = []knowledge.EdgeKind{knowledge.LinksTo}
	)

	// Weight desc, then the other endpoint asc ("doc:g/b.md" < "id:ABU-1").
	wantEdges(t, neighbors(t, s, seed(dn(ga)), nil, knowledge.Out, 0), ac, ab, aid, ax)
	wantEdges(t, neighbors(t, s, seed(dn(ga)), link, knowledge.Out, 0), ac, ab)
	wantEdges(t, neighbors(t, s, seed(dn(ga)), nil, knowledge.Out, 2), ac, ab)
	wantEdges(t, neighbors(t, s, seed(dn(ga)), nil, knowledge.In, 0), ba, ca)
	// ab and ba tie on weight, other endpoint and kind: Source breaks it.
	wantEdges(t, neighbors(t, s, seed(dn(ga)), nil, knowledge.Both, 0), ac, ab, ba, aid, ca, ax)
	wantEdges(t, neighbors(t, s, seed(dn(ga)), []knowledge.EdgeKind{knowledge.Mentions, knowledge.Tagged}, knowledge.Both, 0), ca, ax)
	// Limit applies per seed; results follow seed order.
	wantEdges(t, neighbors(t, s, seed(dn(ga), dn(gb)), link, knowledge.Out, 1), ac, ba)
	// An edge reached from two seeds is reported once.
	wantEdges(t, neighbors(t, s, seed(dn(gb), dn(ga)), link, knowledge.Both, 0), ab, ba, bc, ac)
	wantEdges(t, neighbors(t, s, seed(dn(ga), dn(ga)), nil, knowledge.Out, 1), ac)
	wantEdges(t, neighbors(t, s, seed(knowledge.TagNode("x")), nil, knowledge.In, 0), ax, cx)
	wantEdges(t, neighbors(t, s, seed(knowledge.TagNode("x")), nil, knowledge.Out, 0))
	wantEdges(t, neighbors(t, s, nil, nil, knowledge.Both, 0))
	wantEdges(t, neighbors(t, s, seed(dn("g/unknown.md")), nil, knowledge.Both, 0))

	if _, err := s.Neighbors(t.Context(), seed(dn(ga)), nil, knowledge.Direction(0), 0); err == nil {
		t.Fatal("Neighbors with an invalid direction succeeded")
	}

	// Duplicate edges in one update merge, keeping the highest weight.
	put(t, s, mkDoc("g/dup.md", "md", "g"), nil,
		mkEdge(dn("g/dup.md"), knowledge.TagNode("y"), knowledge.Tagged, 0.3, "g/dup.md"),
		mkEdge(dn("g/dup.md"), knowledge.TagNode("y"), knowledge.Tagged, 0.9, "g/dup.md"))
	wantEdges(t, neighbors(t, s, seed(knowledge.TagNode("y")), nil, knowledge.Both, 0),
		mkEdge(dn("g/dup.md"), knowledge.TagNode("y"), knowledge.Tagged, 0.9, "g/dup.md"))
	// The same relation asserted by two sources is two edges.
	put(t, s, mkDoc("g/dup2.md", "md", "g"), nil,
		mkEdge(dn("g/dup.md"), knowledge.TagNode("y"), knowledge.Tagged, 0.9, "g/dup2.md"))
	wantEdges(t, neighbors(t, s, seed(knowledge.TagNode("y")), nil, knowledge.In, 0),
		mkEdge(dn("g/dup.md"), knowledge.TagNode("y"), knowledge.Tagged, 0.9, "g/dup.md"),
		mkEdge(dn("g/dup.md"), knowledge.TagNode("y"), knowledge.Tagged, 0.9, "g/dup2.md"))
}

func testDegree(t *testing.T, s knowledge.Store) {
	graphFixture(t, s)
	put(t, s, mkDoc("g/self.md", "md", "g"), nil,
		mkEdge(dn("g/self.md"), dn("g/self.md"), knowledge.LinksTo, 1, "g/self.md"))
	for n, want := range map[knowledge.NodeID]int{
		dn(ga):                     6, // 4 out + b→a + c→a
		dn(gb):                     4, // 3 out + a→b
		dn(gc):                     5, // 3 out + a→c + b→c
		knowledge.TagNode("x"):     2,
		knowledge.SymNode("pkg.F"): 2,
		dn("g/self.md"):            1, // a self-loop counts once
		dn("g/unknown.md"):         0,
	} {
		got, err := s.Degree(t.Context(), n)
		if err != nil {
			t.Fatalf("Degree(%s): %v", n, err)
		}
		if got != want {
			t.Errorf("Degree(%s) = %d, want %d", n, got, want)
		}
	}
}

func testResolve(t *testing.T, s knowledge.Store) {
	graphFixture(t, s)
	check := func(n knowledge.NodeID, want knowledge.DocID, wantOK bool) {
		t.Helper()
		got, ok, err := s.Resolve(t.Context(), n)
		if err != nil {
			t.Fatalf("Resolve(%s): %v", n, err)
		}
		if ok != wantOK || got != want {
			t.Errorf("Resolve(%s) = %q, %v; want %q, %v", n, got, ok, want, wantOK)
		}
	}
	check(dn(ga), ga, true)
	check(dn("g/unknown.md"), "", false) // linked to by nobody, not indexed
	check(knowledge.IDNode("abu-1"), ga, true)
	check(knowledge.SymNode("pkg.F"), gb, true) // b and c define it: smallest DocID
	check(knowledge.IDNode("NONE-1"), "", false)
	check(knowledge.TagNode("x"), "", false)
	check("garbage", "", false)

	if err := s.DeleteDoc(t.Context(), gb); err != nil {
		t.Fatal(err)
	}
	check(knowledge.SymNode("pkg.F"), gc, true)
	check(dn(gb), "", false) // a still links to b, but b is no longer indexed
}

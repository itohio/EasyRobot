package storeutil

import (
	"slices"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
)

func TestNormalize(t *testing.T) {
	d := knowledge.Doc{ID: "a.md", Props: map[string]string{"k": "v"}}
	heading := []string{"H"}
	u := knowledge.DocUpdate{
		Doc:    d,
		Chunks: []knowledge.Chunk{{Ord: 1, Text: "two", Heading: heading}, {Doc: "a.md", Ord: 0, Text: "one"}},
		Edges: []knowledge.Edge{
			{From: "doc:a.md", To: "tag:x", Kind: knowledge.Tagged, Weight: 0.2},
			{From: "doc:a.md", To: "tag:y", Kind: knowledge.Tagged, Weight: 1, Source: "a.md"},
			{From: "doc:a.md", To: "tag:x", Kind: knowledge.Tagged, Weight: 0.8, Source: "a.md"},
		},
	}
	got, err := Normalize(u)
	if err != nil {
		t.Fatal(err)
	}
	if got.Chunks[0].Doc != "a.md" || got.Chunks[0].ID != knowledge.NewChunkID("a.md", 1, "two") {
		t.Fatalf("chunk not filled: %+v", got.Chunks[0])
	}
	want := []knowledge.Edge{
		{From: "doc:a.md", To: "tag:x", Kind: knowledge.Tagged, Weight: 0.8, Source: "a.md"},
		{From: "doc:a.md", To: "tag:y", Kind: knowledge.Tagged, Weight: 1, Source: "a.md"},
	}
	if !slices.Equal(got.Edges, want) {
		t.Fatalf("edges = %+v, want %+v", got.Edges, want)
	}
	heading[0] = "mutated"
	d.Props["k"] = "mutated"
	if got.Chunks[0].Heading[0] != "H" || got.Doc.Props["k"] != "v" {
		t.Fatal("Normalize aliases caller data")
	}

	for name, bad := range map[string]knowledge.DocUpdate{
		"empty id":        {},
		"chunk doc":       {Doc: d, Chunks: []knowledge.Chunk{{Doc: "b.md"}}},
		"edge source":     {Doc: d, Edges: []knowledge.Edge{{From: "doc:a.md", To: "tag:x", Source: "b.md"}}},
		"duplicate chunk": {Doc: d, Chunks: []knowledge.Chunk{{Text: "x"}, {Text: "x"}}},
	} {
		if _, err := Normalize(bad); err == nil {
			t.Errorf("%s: Normalize succeeded", name)
		}
	}
}

func TestQueryTerms(t *testing.T) {
	got := QueryTerms([]string{"BETA", "alpha", "alpha", "", " ", "a b", "abu-061", "-x", "Ünïcode"})
	want := []string{"beta", "alpha", "abu-061", "-x", "ünïcode"}
	if !slices.Equal(got, want) {
		t.Fatalf("QueryTerms = %q, want %q", got, want)
	}
}

func TestFieldTexts(t *testing.T) {
	c := knowledge.Chunk{Doc: "dir/a.md", Heading: []string{"A", "B"}, Symbol: "pkg.F", Text: "body"}
	got := FieldTexts(c)
	want := [NumFields]string{"A › B", "pkg.F", "dir/a.md", "body"}
	if got != want {
		t.Fatalf("FieldTexts = %q, want %q", got, want)
	}
}

func TestCompareEdges(t *testing.T) {
	seed := knowledge.NodeID("doc:s")
	es := []knowledge.Edge{
		{From: "doc:s", To: "tag:b", Kind: knowledge.Tagged, Weight: 1, Source: "s"},
		{From: "doc:z", To: "doc:s", Kind: knowledge.LinksTo, Weight: 1, Source: "z"},
		{From: "doc:s", To: "doc:z", Kind: knowledge.LinksTo, Weight: 1, Source: "s"},
		{From: "doc:s", To: "tag:a", Kind: knowledge.Tagged, Weight: 2, Source: "s"},
	}
	slices.SortFunc(es, CompareEdges(seed))
	want := []knowledge.Edge{
		{From: "doc:s", To: "tag:a", Kind: knowledge.Tagged, Weight: 2, Source: "s"},
		{From: "doc:s", To: "doc:z", Kind: knowledge.LinksTo, Weight: 1, Source: "s"},
		{From: "doc:z", To: "doc:s", Kind: knowledge.LinksTo, Weight: 1, Source: "z"},
		{From: "doc:s", To: "tag:b", Kind: knowledge.Tagged, Weight: 1, Source: "s"},
	}
	if !slices.Equal(es, want) {
		t.Fatalf("sorted = %+v\nwant %+v", es, want)
	}
}

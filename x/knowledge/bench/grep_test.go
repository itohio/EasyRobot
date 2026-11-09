package bench

import (
	"context"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/itohio/EasyRobot/x/knowledge"
)

func TestQueryWords(t *testing.T) {
	cases := map[string][]string{
		"the alpha, beta and gamma?":                   {"alpha", "beta", "gamma"},
		"Where's the RV-007 ticket, about kd_tree.go?": {"rv-007", "ticket", "kd_tree.go"},
		"files. Files! ab x":                           {"files"},
		"--flag-- ..dots.. __under__":                  {"flag", "dots", "under"},
		"":                                             nil,
		"how is it":                                    nil,
		"Kalman P aliasing: copy(k.P, k.tempM) aliasing!": {"kalman", "aliasing", "copy", "k.p", "k.tempm"},
	}
	for in, want := range cases {
		if got := queryWords(in); !reflect.DeepEqual(got, want) {
			t.Errorf("queryWords(%q) = %q, want %q", in, got, want)
		}
	}
}

func grepCorpus(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"a.md":                "alpha beta gamma\n",
		"b.md":                "intro\nalpha beta\nalpha beta alpha\n",
		"c.md":                "Alpha\nBETA\n",
		"d.go":                "package d\n// gamma beta\n",
		"e.txt":               "alpha beta gamma\n",
		"f.md":                "nothing here\n",
		"sub/g.md":            "intro\n\nthe gamma ray\n",
		"h.md":                strings.Repeat("x", 200) + " zeta " + strings.Repeat("é", 200) + "\n",
		".git/x.md":           "alpha beta gamma\n",
		"node_modules/x.md":   "alpha beta gamma\n",
		".venv/x.md":          "alpha beta gamma\n",
		"_quarantine/x.md":    "alpha beta gamma\n",
		"_tmp/x.md":           "alpha beta gamma\n",
		"sub/_tmp/y.md":       "alpha beta gamma\n",
		"sub/.git/objects.md": "alpha beta gamma\n",
		"big.md":              "alpha beta gamma\n" + strings.Repeat("y", 1<<20),
	}
	for p, c := range files {
		writeFile(t, filepath.Join(root, filepath.FromSlash(p)), c)
	}
	return root
}

func docs(res knowledge.Result) []knowledge.DocID {
	var out []knowledge.DocID
	for _, h := range res.Hits {
		out = append(out, h.Doc)
	}
	return out
}

func TestGrepRanking(t *testing.T) {
	root := grepCorpus(t)
	g := NewGrep(root)
	res, err := g.Find(context.Background(), knowledge.Query{Text: "the Alpha, beta and GAMMA?"})
	if err != nil {
		t.Fatal(err)
	}
	// a: 3 distinct; b: 2 distinct, 5 matches; c and d: 2 distinct, 2
	// matches, path order; sub/g: 1 distinct. e.txt (extension), f.md (no
	// match), big.md (size) and skipped folders never appear.
	want := []knowledge.DocID{"a.md", "b.md", "c.md", "d.go", "sub/g.md"}
	if got := docs(res); !reflect.DeepEqual(got, want) {
		t.Fatalf("ranking = %q, want %q", got, want)
	}
	for i := 1; i < len(res.Hits); i++ {
		if res.Hits[i].Score > res.Hits[i-1].Score {
			t.Errorf("score increases at %d: %v > %v", i, res.Hits[i].Score, res.Hits[i-1].Score)
		}
	}
	if !(res.Hits[0].Score > res.Hits[1].Score && res.Hits[1].Score > res.Hits[2].Score && res.Hits[2].Score == res.Hits[3].Score && res.Hits[3].Score > res.Hits[4].Score) {
		t.Errorf("scores do not reflect (distinct, total): %v %v %v %v %v",
			res.Hits[0].Score, res.Hits[1].Score, res.Hits[2].Score, res.Hits[3].Score, res.Hits[4].Score)
	}

	byDoc := map[knowledge.DocID]knowledge.Hit{}
	for _, h := range res.Hits {
		byDoc[h.Doc] = h
	}
	for doc, want := range map[knowledge.DocID]struct {
		line    int
		snippet string
		kind    string
	}{
		"a.md":     {1, "alpha beta gamma", "md"},
		"b.md":     {2, "alpha beta", "md"},
		"c.md":     {1, "Alpha", "md"},
		"d.go":     {2, "// gamma beta", "go"},
		"sub/g.md": {3, "the gamma ray", "md"},
	} {
		h := byDoc[doc]
		if len(h.Sections) != 1 {
			t.Fatalf("%s: %d sections", doc, len(h.Sections))
		}
		s := h.Sections[0]
		if s.Lines != [2]int{want.line, want.line} || s.Snippet != want.snippet {
			t.Errorf("%s: section %v %q, want line %d %q", doc, s.Lines, s.Snippet, want.line, want.snippet)
		}
		if h.Kind != want.kind {
			t.Errorf("%s: kind %q, want %q", doc, h.Kind, want.kind)
		}
		if h.Path != filepath.Join(root, filepath.FromSlash(string(doc))) {
			t.Errorf("%s: path %q", doc, h.Path)
		}
	}
}

func TestGrepLimitsAndFilters(t *testing.T) {
	g := NewGrep(grepCorpus(t))
	ctx := context.Background()

	res, err := g.Find(ctx, knowledge.Query{Text: "alpha beta gamma", K: 2})
	if err != nil {
		t.Fatal(err)
	}
	if got := docs(res); !reflect.DeepEqual(got, []knowledge.DocID{"a.md", "b.md"}) {
		t.Errorf("K=2 → %q", got)
	}

	res, err = g.Find(ctx, knowledge.Query{Text: "alpha beta gamma", Filter: knowledge.Filter{In: []string{"sub/"}}})
	if err != nil {
		t.Fatal(err)
	}
	if got := docs(res); !reflect.DeepEqual(got, []knowledge.DocID{"sub/g.md"}) {
		t.Errorf("In=sub/ → %q", got)
	}

	res, err = g.Find(ctx, knowledge.Query{Text: "alpha beta gamma", Filter: knowledge.Filter{Kinds: []string{"go"}}})
	if err != nil {
		t.Fatal(err)
	}
	if got := docs(res); !reflect.DeepEqual(got, []knowledge.DocID{"d.go"}) {
		t.Errorf("Kinds=go → %q", got)
	}

	res, err = g.Find(ctx, knowledge.Query{Text: "how is it a"})
	if err != nil || len(res.Hits) != 0 {
		t.Errorf("stopword-only query → %d hits, err %v", len(res.Hits), err)
	}

	res, err = g.Find(ctx, knowledge.Query{Text: "ab zeta"})
	if err != nil {
		t.Fatal(err)
	}
	if got := docs(res); !reflect.DeepEqual(got, []knowledge.DocID{"h.md"}) {
		t.Fatalf("zeta → %q", got)
	}
	snip := res.Hits[0].Sections[0].Snippet
	if n := utf8.RuneCountInString(snip); n > 160 || !utf8.ValidString(snip) {
		t.Errorf("snippet %d runes, valid=%v", n, utf8.ValidString(snip))
	}
	if !strings.HasPrefix(snip, "xxxx") {
		t.Errorf("snippet should start at line start: %q", snip[:10])
	}

	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := NewGrep(t.TempDir()).Find(cctx, knowledge.Query{Text: "alpha"}); err == nil {
		t.Error("cancelled context: expected error")
	}
	if _, err := NewGrep(filepath.Join(t.TempDir(), "absent")).Find(ctx, knowledge.Query{Text: "alpha"}); err == nil {
		t.Error("missing root: expected error")
	}
}

func TestGrepConcurrent(t *testing.T) {
	g := NewGrep(grepCorpus(t))
	errs := make(chan error, 8)
	for range 8 {
		go func() {
			res, err := g.Find(context.Background(), knowledge.Query{Text: "alpha beta gamma"})
			if err == nil && (len(res.Hits) == 0 || res.Hits[0].Doc != "a.md") {
				err = context.Canceled
			}
			errs <- err
		}()
	}
	for range 8 {
		if err := <-errs; err != nil {
			t.Error(err)
		}
	}
}

func TestGrepLoad(t *testing.T) {
	ctx := context.Background()
	if err := NewGrep(filepath.Join(t.TempDir(), "absent")).Load(ctx); err == nil {
		t.Fatal("missing root: expected error")
	}
	root := grepCorpus(t)
	g := NewGrep(root)
	if err := g.Load(ctx); err != nil {
		t.Fatal(err)
	}
	// Contents are cached: a file added after Load is not seen.
	writeFile(t, filepath.Join(root, "late.md"), "alpha beta gamma\n")
	res, err := g.Find(ctx, knowledge.Query{Text: "alpha beta gamma"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Docs != 7 {
		t.Errorf("Docs = %d, want 7 (a b c d f sub/g h)", res.Docs)
	}
	for _, h := range res.Hits {
		if h.Doc == "late.md" {
			t.Error("late.md seen after Load")
		}
	}
}

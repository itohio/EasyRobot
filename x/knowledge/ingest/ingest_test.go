package ingest

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/token"
)

// ---- helpers ---------------------------------------------------------------

func writeFile(t *testing.T, root, rel, content string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// bumpMTime moves a file's mtime forward by d (second precision matters).
func bumpMTime(t *testing.T, root, rel string, d time.Duration) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	fi, err := os.Stat(p)
	if err != nil {
		t.Fatal(err)
	}
	mt := fi.ModTime().Add(d)
	if err := os.Chtimes(p, mt, mt); err != nil {
		t.Fatal(err)
	}
}

// editFile rewrites a file without moving its mtime away: same-second edits
// must be detected (full-precision mtime plus the racy rule).
func editFile(t *testing.T, root, rel, content string) {
	t.Helper()
	writeFile(t, root, rel, content)
}

func sync1(t *testing.T, ing *Ingestor) knowledge.Report {
	t.Helper()
	rep, err := ing.Sync(context.Background())
	if err != nil {
		t.Fatalf("Sync: %v", err)
	}
	if len(rep.Errors) > 0 {
		t.Fatalf("Sync errors: %v", rep.Errors)
	}
	if rep.Duration <= 0 {
		t.Errorf("Duration = %v, want > 0", rep.Duration)
	}
	return rep
}

type counts struct{ Added, Changed, Moved, Deleted, Unchanged, Touched, Skipped, Secrets int }

func countsOf(r knowledge.Report) counts {
	return counts{r.Added, r.Changed, r.Moved, r.Deleted, r.Unchanged, r.Touched, r.Skipped, r.SecretsDropped}
}

func wantCounts(t *testing.T, r knowledge.Report, want counts) {
	t.Helper()
	if got := countsOf(r); got != want {
		t.Errorf("report = %+v, want %+v", got, want)
	}
}

func docIDs(t *testing.T, st knowledge.Store) []knowledge.DocID {
	t.Helper()
	var ids []knowledge.DocID
	for d, err := range st.ListDocs(context.Background()) {
		if err != nil {
			t.Fatal(err)
		}
		ids = append(ids, d.ID)
	}
	slices.Sort(ids)
	return ids
}

func getDoc(t *testing.T, st knowledge.Store, id knowledge.DocID) (knowledge.Doc, bool) {
	t.Helper()
	d, ok, err := st.GetDoc(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	return d, ok
}

func search(t *testing.T, st knowledge.Store, q string) []knowledge.DocID {
	t.Helper()
	hits, err := st.SearchLex(context.Background(), knowledge.LexQuery{Terms: token.Query(q)}, 20, knowledge.Filter{})
	if err != nil {
		t.Fatal(err)
	}
	var out []knowledge.DocID
	for _, h := range hits {
		if !slices.Contains(out, h.Doc) {
			out = append(out, h.Doc)
		}
	}
	slices.Sort(out)
	return out
}

// links returns the sorted targets of LinksTo edges asserted by id.
func links(t *testing.T, st knowledge.Store, id knowledge.DocID) []knowledge.NodeID {
	t.Helper()
	es, err := st.Neighbors(context.Background(), []knowledge.NodeID{knowledge.DocNode(id)}, []knowledge.EdgeKind{knowledge.LinksTo}, knowledge.Out, 0)
	if err != nil {
		t.Fatal(err)
	}
	var out []knowledge.NodeID
	for _, e := range es {
		if e.Source == id && e.From == knowledge.DocNode(id) {
			out = append(out, e.To)
		}
	}
	slices.Sort(out)
	return out
}

// inbound returns the sorted sources of LinksTo edges pointing at id.
func inbound(t *testing.T, st knowledge.Store, id knowledge.DocID) []knowledge.DocID {
	t.Helper()
	es, err := st.Neighbors(context.Background(), []knowledge.NodeID{knowledge.DocNode(id)}, []knowledge.EdgeKind{knowledge.LinksTo}, knowledge.In, 0)
	if err != nil {
		t.Fatal(err)
	}
	var out []knowledge.DocID
	for _, e := range es {
		out = append(out, e.Source)
	}
	slices.Sort(out)
	return out
}

func dn(id string) knowledge.NodeID { return knowledge.DocNode(knowledge.DocID(id)) }

func ids(s ...string) []knowledge.DocID {
	out := make([]knowledge.DocID, len(s))
	for i, v := range s {
		out[i] = knowledge.DocID(v)
	}
	return out
}

func nodes(n ...knowledge.NodeID) []knowledge.NodeID {
	slices.Sort(n)
	return n
}

// ---- lifecycle -------------------------------------------------------------

func TestFirstSyncAndDocFields(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "notes/alpha.md", "# Alpha\n\nalpha body text\n")
	writeFile(t, root, "notes/beta.md", "# Beta\n\nbeta body links [[alpha]]\n")
	writeFile(t, root, "code/keys/keys.go", "// Package keys manages keys.\npackage keys\n\n// Rotate rotates.\nfunc Rotate() {}\n")
	writeFile(t, root, "top.md", "# Top\n\ntop level\n")
	writeFile(t, root, "img/logo.png", "\x89PNG")
	writeFile(t, root, "notes/readme.txt", "plain text no chunker")

	st := newStore(t)
	rep := sync1(t, New(st, Source{Root: root}, testOptions()...))
	wantCounts(t, rep, counts{Added: 4})

	if got, want := docIDs(t, st), ids("code/keys/keys.go", "notes/alpha.md", "notes/beta.md", "top.md"); !slices.Equal(got, want) {
		t.Fatalf("docs = %v, want %v", got, want)
	}
	d, _ := getDoc(t, st, "notes/beta.md")
	if d.Scope != "notes" || d.Kind != "md" || d.Title != "Beta" || d.Weight != 1 {
		t.Errorf("beta doc = scope %q kind %q title %q weight %v", d.Scope, d.Kind, d.Title, d.Weight)
	}
	if want := filepath.Join(root, "notes", "beta.md"); d.Path != want {
		t.Errorf("Path = %q, want %q", d.Path, want)
	}
	if !filepath.IsAbs(d.Path) {
		t.Errorf("Path %q not absolute", d.Path)
	}
	data, _ := os.ReadFile(d.Path)
	if d.Size != int64(len(data)) || d.Hash == ([32]byte{}) || d.MTime.IsZero() {
		t.Errorf("stat fields: size %d hash %x mtime %v", d.Size, d.Hash, d.MTime)
	}
	if _, ok := d.Props["_missing"]; ok {
		t.Errorf("_missing present on fully resolved doc: %v", d.Props)
	}
	g, _ := getDoc(t, st, "code/keys/keys.go")
	if g.Kind != "go" || g.Scope != "code" || g.Props["package"] != "keys" {
		t.Errorf("go doc = kind %q scope %q props %v", g.Kind, g.Scope, g.Props)
	}
	if got := links(t, st, "notes/beta.md"); !slices.Equal(got, nodes(dn("notes/alpha.md"))) {
		t.Errorf("beta links = %v", got)
	}
	if got := search(t, st, "alpha body"); !slices.Contains(got, "notes/alpha.md") {
		t.Errorf("search alpha = %v", got)
	}

	// An explicit scope prefixes ids and becomes Doc.Scope.
	st2 := newStore(t)
	sync1(t, New(st2, Source{Root: root, Scope: "ws"}, testOptions()...))
	if got, want := docIDs(t, st2), ids("ws/code/keys/keys.go", "ws/notes/alpha.md", "ws/notes/beta.md", "ws/top.md"); !slices.Equal(got, want) {
		t.Fatalf("scoped docs = %v, want %v", got, want)
	}
	d2, _ := getDoc(t, st2, "ws/notes/beta.md")
	if d2.Scope != "ws" {
		t.Errorf("scoped Scope = %q", d2.Scope)
	}
	if got := links(t, st2, "ws/notes/beta.md"); !slices.Equal(got, nodes(dn("ws/notes/alpha.md"))) {
		t.Errorf("scoped beta links = %v", got)
	}
}

func TestUnchangedTouchEdit(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a.md", "# A\n\nalphaword here\n")
	writeFile(t, root, "b.md", "# B\n\nbetaword here\n")
	writeFile(t, root, "c.go", "package c\n\nfunc C() {}\n")
	st := newStore(t)
	ing := New(st, Source{Root: root}, testOptions()...)
	wantCounts(t, sync1(t, ing), counts{Added: 3})

	gen := func() uint64 {
		g, err := st.Generation(context.Background())
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	g0 := gen()
	wantCounts(t, sync1(t, ing), counts{Unchanged: 3})
	if g := gen(); g != g0 {
		t.Errorf("generation moved on unchanged sync: %d → %d", g0, g)
	}

	// Touch: new mtime, same content.
	bumpMTime(t, root, "b.md", 5*time.Second)
	wantCounts(t, sync1(t, ing), counts{Touched: 1, Unchanged: 2})
	fi, _ := os.Stat(filepath.Join(root, "b.md"))
	d, _ := getDoc(t, st, "b.md")
	if d.MTime.Unix() != fi.ModTime().Unix() {
		t.Errorf("touched MTime = %v, want %v", d.MTime, fi.ModTime())
	}
	wantCounts(t, sync1(t, ing), counts{Unchanged: 3})

	// Edit: new content is searchable, old content is gone.
	editFile(t, root, "a.md", "# A\n\nomegaword now\n")
	wantCounts(t, sync1(t, ing), counts{Changed: 1, Unchanged: 2})
	if got := search(t, st, "omegaword"); !slices.Equal(got, ids("a.md")) {
		t.Errorf("search new = %v", got)
	}
	if got := search(t, st, "alphaword"); len(got) != 0 {
		t.Errorf("search old = %v, want none", got)
	}
}

func TestRenameKeepsInboundLinks(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "notes/target.md", "# Target\n\nthe zanzibarword note\n")
	writeFile(t, root, "linker.md", "# Linker\n\nsee [[target]]\n")
	st := newStore(t)
	ing := New(st, Source{Root: root}, testOptions()...)
	wantCounts(t, sync1(t, ing), counts{Added: 2})

	if err := os.MkdirAll(filepath.Join(root, "archive"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(filepath.Join(root, "notes", "target.md"), filepath.Join(root, "archive", "renamed.md")); err != nil {
		t.Fatal(err)
	}
	rep := sync1(t, ing)
	wantCounts(t, rep, counts{Moved: 1, Unchanged: 1})
	if _, ok := getDoc(t, st, "notes/target.md"); ok {
		t.Error("old doc still in catalog")
	}
	nd, ok := getDoc(t, st, "archive/renamed.md")
	if !ok || nd.Scope != "archive" || nd.Path != filepath.Join(root, "archive", "renamed.md") {
		t.Fatalf("new doc = %+v ok=%v", nd, ok)
	}
	if got := inbound(t, st, "archive/renamed.md"); !slices.Equal(got, ids("linker.md")) {
		t.Errorf("inbound to renamed = %v, want [linker.md]", got)
	}
	if got := search(t, st, "zanzibarword"); !slices.Equal(got, ids("archive/renamed.md")) {
		t.Errorf("search after move = %v", got)
	}
	if got := docIDs(t, st); !slices.Equal(got, ids("archive/renamed.md", "linker.md")) {
		t.Errorf("docs = %v", got)
	}
	if nd.Props[PropMovedFrom] != "notes/target.md" {
		t.Errorf("%s = %q", PropMovedFrom, nd.Props[PropMovedFrom])
	}

	// The former name stays an alias: re-chunking the linker (edit) keeps
	// the link on the moved document, as does a second move.
	editFile(t, root, "linker.md", "# Linker\n\nstill see [[target]] and [old](notes/target.md)\n")
	wantCounts(t, sync1(t, ing), counts{Changed: 1, Unchanged: 1})
	if got := links(t, st, "linker.md"); !slices.Equal(got, nodes(dn("archive/renamed.md"))) {
		t.Errorf("linker links after edit = %v", got)
	}
	if err := os.Rename(filepath.Join(root, "archive", "renamed.md"), filepath.Join(root, "final.md")); err != nil {
		t.Fatal(err)
	}
	wantCounts(t, sync1(t, ing), counts{Moved: 1, Unchanged: 1})
	if got := inbound(t, st, "final.md"); !slices.Equal(got, ids("linker.md")) {
		t.Errorf("inbound to final = %v", got)
	}
	if fd, _ := getDoc(t, st, "final.md"); fd.Props[PropMovedFrom] != "archive/renamed.md\nnotes/target.md" {
		t.Errorf("%s = %q", PropMovedFrom, fd.Props[PropMovedFrom])
	}
	// A real file at the old name wins over the alias for newly resolved links.
	writeFile(t, root, "notes/target.md", "# New target\n\nfresh\n")
	editFile(t, root, "linker.md", "# Linker\n\nnow [[target]]\n")
	sync1(t, ing)
	if got := links(t, st, "linker.md"); !slices.Equal(got, nodes(dn("notes/target.md"))) {
		t.Errorf("linker links with real file = %v", got)
	}
}

func TestMoveReresolvesRelativeLinks(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a/doc.md", "# Doc\n\nsee [sib](sib.md)\n")
	writeFile(t, root, "a/sib.md", "# Sib A\n\nsib in a\n")
	writeFile(t, root, "b/sib.md", "# Sib B\n\nsib in b\n")
	st := newStore(t)
	ing := New(st, Source{Root: root}, testOptions()...)
	sync1(t, ing)
	if got := links(t, st, "a/doc.md"); !slices.Equal(got, nodes(dn("a/sib.md"))) {
		t.Fatalf("links before = %v", got)
	}
	if err := os.Rename(filepath.Join(root, "a", "doc.md"), filepath.Join(root, "b", "doc.md")); err != nil {
		t.Fatal(err)
	}
	wantCounts(t, sync1(t, ing), counts{Moved: 1, Unchanged: 2})
	if got := links(t, st, "b/doc.md"); !slices.Equal(got, nodes(dn("b/sib.md"))) {
		t.Errorf("links after move = %v, want [doc:b/sib.md]", got)
	}
}

func TestDeleteCascadesAndReresolves(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "target.md", "# Target\n\nuniqueword target\n")
	writeFile(t, root, "linker.md", "# Linker\n\nsee [[target]]\n")
	writeFile(t, root, "other.md", "# Other\n\nunrelated\n")
	st := newStore(t)
	ing := New(st, Source{Root: root}, testOptions()...)
	sync1(t, ing)

	if err := os.Remove(filepath.Join(root, "target.md")); err != nil {
		t.Fatal(err)
	}
	rep := sync1(t, ing)
	// linker.md is re-resolved: its edge changed, so it counts as Changed.
	wantCounts(t, rep, counts{Deleted: 1, Changed: 1, Unchanged: 1})
	if _, ok := getDoc(t, st, "target.md"); ok {
		t.Error("deleted doc still in catalog")
	}
	cs, err := st.DocChunks(context.Background(), "target.md")
	if err != nil || len(cs) != 0 {
		t.Errorf("chunks of deleted doc = %d (err %v)", len(cs), err)
	}
	if got := search(t, st, "uniqueword"); len(got) != 0 {
		t.Errorf("search deleted = %v", got)
	}
	if got := links(t, st, "linker.md"); !slices.Equal(got, nodes(knowledge.MissingNode("target"))) {
		t.Errorf("linker links = %v, want [missing:target]", got)
	}
	ld, _ := getDoc(t, st, "linker.md")
	if ld.Props["_missing"] != "1" {
		t.Errorf("linker _missing = %q", ld.Props["_missing"])
	}
	wantCounts(t, sync1(t, ing), counts{Unchanged: 2})
}

func TestMissingLinkResolvesLater(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "linker.md", "# Linker\n\nsee [[Later]] and [l](sub/doc.md)\n")
	writeFile(t, root, "plain.md", "# Plain\n\nnothing\n")
	st := newStore(t)
	ing := New(st, Source{Root: root}, testOptions()...)
	sync1(t, ing)
	if got := links(t, st, "linker.md"); !slices.Equal(got, nodes(knowledge.MissingNode("Later"), knowledge.MissingNode("sub/doc.md"))) {
		t.Fatalf("links before = %v", got)
	}
	if d, _ := getDoc(t, st, "linker.md"); d.Props["_missing"] != "2" {
		t.Errorf("_missing = %q, want 2", d.Props["_missing"])
	}

	writeFile(t, root, "later.md", "# Later\n\nnow exists\n")
	wantCounts(t, sync1(t, ing), counts{Added: 1, Changed: 1, Unchanged: 1})
	if got := links(t, st, "linker.md"); !slices.Equal(got, nodes(dn("later.md"), knowledge.MissingNode("sub/doc.md"))) {
		t.Errorf("links after first add = %v", got)
	}
	if d, _ := getDoc(t, st, "linker.md"); d.Props["_missing"] != "1" {
		t.Errorf("_missing = %q, want 1", d.Props["_missing"])
	}

	writeFile(t, root, "sub/doc.md", "# Sub\n\nsub doc\n")
	wantCounts(t, sync1(t, ing), counts{Added: 1, Changed: 1, Unchanged: 2})
	if got := links(t, st, "linker.md"); !slices.Equal(got, nodes(dn("later.md"), dn("sub/doc.md"))) {
		t.Errorf("links after second add = %v", got)
	}
	if d, _ := getDoc(t, st, "linker.md"); d.Props["_missing"] != "" {
		t.Errorf("_missing = %q, want absent", d.Props["_missing"])
	}
}

// ---- link resolution ---------------------------------------------------------

func TestWikiLinkCandidates(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "alpha/deep/notes/Topic.md", "# Topic alpha\n")
	writeFile(t, root, "beta/Topic.md", "# Topic beta\n")
	writeFile(t, root, "gamma/x/topic.md", "# Topic gamma\n")
	writeFile(t, root, "alpha/a.md", "# A\n\n[[topic]]\n") // same scope wins over fewer segments
	writeFile(t, root, "beta/b.md", "# B\n\n[[TOPIC]]\n")  // same scope, case-insensitive
	writeFile(t, root, "delta/d.md", "# D\n\n[[Topic]]\n") // no same scope: fewest segments
	writeFile(t, root, "delta/e.md", "# E\n\n[[gamma/x/topic]] [[x/topic]] [[beta/Topic.md]]\n")
	writeFile(t, root, "delta/f.md", "# F\n\n[[f]] [[keys.go]] [[diagram.png]]\n") // self-link, file with ext, non-doc ext
	writeFile(t, root, "delta/keys.go", "package delta\n")
	writeFile(t, root, "eps/one/z.md", "# Z1\n")
	writeFile(t, root, "eps/two/z.md", "# Z2\n")
	writeFile(t, root, "zeta/q.md", "# Q\n\n[[z]]\n") // tie on segments: lexical

	st := newStore(t)
	sync1(t, New(st, Source{Root: root}, testOptions()...))
	tests := []struct {
		doc  string
		want []knowledge.NodeID
	}{
		{"alpha/a.md", nodes(dn("alpha/deep/notes/Topic.md"))},
		{"beta/b.md", nodes(dn("beta/Topic.md"))},
		{"delta/d.md", nodes(dn("beta/Topic.md"))},
		{"delta/e.md", nodes(dn("gamma/x/topic.md"), dn("beta/Topic.md"))},
		{"delta/f.md", nodes(dn("delta/keys.go"))},
		{"zeta/q.md", nodes(dn("eps/one/z.md"))},
	}
	for _, tt := range tests {
		if got := links(t, st, knowledge.DocID(tt.doc)); !slices.Equal(got, tt.want) {
			t.Errorf("%s links = %v, want %v", tt.doc, got, tt.want)
		}
	}
}

func TestMarkdownLinks(t *testing.T) {
	parent := t.TempDir()
	root := filepath.Join(parent, "root")
	writeFile(t, parent, "outside.md", "# Outside\n")
	writeFile(t, root, "docs/b/other.md", "# Other\n")
	writeFile(t, root, "docs/a/sib.md", "# Sib\n")
	writeFile(t, root, "docs/c/index.md", "# C index\n")
	writeFile(t, root, "docs/d/README.md", "# D readme\n")
	writeFile(t, root, "top.md", "# Top\n")
	writeFile(t, root, "docs/a/page.md", strings.Join([]string{
		"# Page",
		"",
		"[up](../b/other.md#section) [same](sib.md) [again](./sib.md)",
		"[noext](sib) [dir](../c/) [readme](../d)",
		"[esc](../../../outside.md) [esc2](../../../root/top.md)",
		"[gone](nothere.md) [self](page.md#x) [img](pic.png)",
		"[root](/top.md) [web](https://example.com/x.md)",
		"",
	}, "\n"))

	st := newStore(t)
	sync1(t, New(st, Source{Root: root}, testOptions()...))
	want := nodes(dn("docs/b/other.md"), dn("docs/a/sib.md"), dn("docs/c/index.md"), dn("docs/d/README.md"), dn("top.md"), knowledge.MissingNode("nothere.md"))
	if got := links(t, st, "docs/a/page.md"); !slices.Equal(got, want) {
		t.Errorf("links =\n%v\nwant\n%v", got, want)
	}
	if d, _ := getDoc(t, st, "docs/a/page.md"); d.Props["_missing"] != "1" {
		t.Errorf("_missing = %q, want 1", d.Props["_missing"])
	}
}

// ---- walk filters ------------------------------------------------------------

func TestWalkFilters(t *testing.T) {
	root := t.TempDir()
	keep := []string{
		"keep.md", "sub/top.md", "other/local.md", "proj/keep/k.md", "proj/skipper.md", "notbuild/x.md",
	}
	for _, rel := range keep {
		writeFile(t, root, rel, "# K\n\nkeep "+rel+"\n")
	}
	drop := []string{
		".obsidian/w.md", ".hidden/x.md", ".git/y.md", "node_modules/m.md", ".venv/v.md", "vendor/v.go",
		"_quarantine/q.md", "_tmp/t.md", "drafts/d.md", "a/drafts/d2.md", "proj/skip/s.md",
		"build/b.md", "top.md", "gen_a.md", "sub/local.md", "sub/deeper/local.md", "sub/bdir/x.md",
	}
	for _, rel := range drop {
		writeFile(t, root, rel, "# D\n\ndrop "+rel+"\n")
	}
	writeFile(t, root, ".gitignore", "# comment\n\nbuild/\n/top.md\ngen_*.md\n!keep.md\n")
	writeFile(t, root, "sub/.gitignore", "local.md\nbdir/\n")
	writeFile(t, root, "notbuild/.gitignore", "") // empty .gitignore is harmless
	// Symlinked directory is skipped.
	writeFile(t, root, "real/r.md", "# R\n\nreal\n")
	keep = append(keep, "real/r.md")
	if err := os.Symlink(filepath.Join(root, "real"), filepath.Join(root, "link")); err != nil {
		t.Fatal(err)
	}
	// Size cap and binary.
	writeFile(t, root, "big.md", "# Big\n\n"+strings.Repeat("x", 300)+"\n")
	writeFile(t, root, "bin.md", "# Bin\n\x00\x01\x02")
	// Secret-named files.
	for _, rel := range []string{".env", ".env.local", "certs/key.pem", "k/server.key", "id_rsa", "id_rsa.pub", "id_ed25519", "x.creds", "s.nk", "a.p12", "app.keystore"} {
		writeFile(t, root, rel, "secret")
	}

	st := newStore(t)
	rep := sync1(t, New(st, Source{Root: root, Exclude: []string{"drafts", "proj/skip"}}, append(testOptions(), WithMaxSize(200))...))
	wantCounts(t, rep, counts{Added: len(keep), Skipped: 2, Secrets: 11})
	var want []knowledge.DocID
	for _, rel := range keep {
		want = append(want, knowledge.DocID(rel))
	}
	slices.Sort(want)
	if got := docIDs(t, st); !slices.Equal(got, want) {
		t.Errorf("docs =\n%v\nwant\n%v", got, want)
	}
}

func TestSecretLinesDropped(t *testing.T) {
	root := t.TempDir()
	seed := "SUA" + strings.Repeat("ABCDEFGH234567", 4)
	files := map[string]string{
		"pk.md":    "# PK\n\nprivatekeymarker [[clean]]\n\n-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu\n-----END RSA PRIVATE KEY-----\n",
		"seed.md":  "# Seed\n\nseedmarker " + seed + "\n",
		"jwt.md":   "# JWT\n\njwtmarker eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U\n",
		"aws.md":   "# AWS\n\nawsmarker AKIAIOSFODNN7EXAMPLE\n",
		"pass.md":  "# Pass\n\npassmarker password = \"hunter2hunter2hunter2\"\n",
		"clean.md": "# Clean\n\ncleanmarker token = short\n",
	}
	for rel, content := range files {
		writeFile(t, root, rel, content)
	}
	st := newStore(t)
	rep := sync1(t, New(st, Source{Root: root}, testOptions()...))
	wantCounts(t, rep, counts{Added: 6, Secrets: 5})
	// Markers on a secret line go with it.
	for _, m := range []string{"seedmarker", "jwtmarker", "awsmarker", "passmarker", "MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu"} {
		if got := search(t, st, m); len(got) != 0 {
			t.Errorf("search %s = %v, want none", m, got)
		}
	}
	// Only secret lines are dropped: the rest of the chunk, and its link,
	// stay.
	for q, want := range map[string]string{"cleanmarker": "clean.md", "privatekeymarker": "pk.md"} {
		if got := search(t, st, q); !slices.Equal(got, ids(want)) {
			t.Errorf("search %s = %v, want [%s]", q, got, want)
		}
	}
	if got := links(t, st, "pk.md"); !slices.Equal(got, nodes(dn("clean.md"))) {
		t.Errorf("pk.md links = %v, want [doc:clean.md]", got)
	}
}

// ---- weights -------------------------------------------------------------------

func TestWeights(t *testing.T) {
	root := t.TempDir()
	gen := "// Code generated by stringer; DO NOT EDIT.\n\npackage g\n\nfunc G() {}\n"
	writeFile(t, root, "pkg/gen.go", gen)
	writeFile(t, root, "pkg/plain.go", "package g\n\nfunc P() {}\n")
	writeFile(t, root, "_scratch/note.md", "# Note\n\nscratch\n")
	writeFile(t, root, "_scratch/gen.go", gen)
	writeFile(t, root, "_scratchpad/x.md", "# X\n\nnot scratch\n")
	writeFile(t, root, "docs/a.md", "# A\n")
	st := newStore(t)
	src := Source{Root: root, Weights: []WeightRule{{Prefix: "_scratch", Weight: 0.5}, {Prefix: "_scratch/note.md", Weight: 0.1}, {Prefix: "docs/", Weight: 2}}}
	sync1(t, New(st, src, testOptions()...))
	want := map[string]float32{
		"pkg/gen.go": 0.2, "pkg/plain.go": 1, "_scratch/note.md": 0.5, "_scratch/gen.go": 0.1, "_scratchpad/x.md": 1, "docs/a.md": 2,
	}
	for id, w := range want {
		d, ok := getDoc(t, st, knowledge.DocID(id))
		if !ok {
			t.Fatalf("%s missing", id)
		}
		if diff := d.Weight - w; diff > 1e-6 || diff < -1e-6 {
			t.Errorf("%s weight = %v, want %v", id, d.Weight, w)
		}
	}
}

// ---- failures and cancellation ---------------------------------------------------

// flakyStore fails PutDoc once per armed document.
type flakyStore struct {
	knowledge.Store
	mu   sync.Mutex
	fail map[knowledge.DocID]bool
}

var errFlaky = errors.New("flaky put")

func (f *flakyStore) arm(id knowledge.DocID) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.fail[id] = true
}

func (f *flakyStore) PutDoc(ctx context.Context, u knowledge.DocUpdate) error {
	f.mu.Lock()
	fail := f.fail[u.Doc.ID]
	delete(f.fail, u.Doc.ID)
	f.mu.Unlock()
	if fail {
		return errFlaky
	}
	return f.Store.PutDoc(ctx, u)
}

func TestFailedPutIsRetried(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a.md", "# A\n\noldword\n")
	writeFile(t, root, "b.md", "# B\n\nbword\n")
	st := &flakyStore{Store: newStore(t), fail: map[knowledge.DocID]bool{}}
	ing := New(st, Source{Root: root}, testOptions()...)

	// New doc fails: not catalogued, next sync adds it.
	st.arm("b.md")
	rep, err := ing.Sync(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Errors) != 1 || !errors.Is(rep.Errors[0], errFlaky) || !strings.Contains(rep.Errors[0].Error(), "b.md") {
		t.Fatalf("errors = %v", rep.Errors)
	}
	wantCounts(t, rep, counts{Added: 1})
	if got := docIDs(t, st); !slices.Equal(got, ids("a.md")) {
		t.Fatalf("docs = %v", got)
	}
	wantCounts(t, sync1(t, ing), counts{Added: 1, Unchanged: 1})

	// Changed doc fails: old row and old text stay, next sync redoes it.
	editFile(t, root, "a.md", "# A\n\nnewword\n")
	st.arm("a.md")
	rep, err = ing.Sync(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Errors) != 1 || !errors.Is(rep.Errors[0], errFlaky) {
		t.Fatalf("errors = %v", rep.Errors)
	}
	wantCounts(t, rep, counts{Unchanged: 1})
	if got := search(t, st, "oldword"); !slices.Equal(got, ids("a.md")) {
		t.Errorf("old text after failure = %v", got)
	}
	wantCounts(t, sync1(t, ing), counts{Changed: 1, Unchanged: 1})
	if got := search(t, st, "newword"); !slices.Equal(got, ids("a.md")) {
		t.Errorf("new text after retry = %v", got)
	}
	if got := search(t, st, "oldword"); len(got) != 0 {
		t.Errorf("old text after retry = %v", got)
	}

	// Moved doc whose re-chunk fails: the next sync still re-chunks it.
	if err := os.Rename(filepath.Join(root, "b.md"), filepath.Join(root, "c.md")); err != nil {
		t.Fatal(err)
	}
	st.arm("c.md")
	rep, err = ing.Sync(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Errors) != 1 {
		t.Fatalf("errors = %v", rep.Errors)
	}
	rep = sync1(t, ing)
	if rep.Changed != 1 {
		t.Errorf("retry after failed move re-chunk: %+v", countsOf(rep))
	}
	if got := docIDs(t, st); !slices.Equal(got, ids("a.md", "c.md")) {
		t.Errorf("docs = %v", got)
	}
	if got := search(t, st, "bword"); !slices.Equal(got, ids("c.md")) {
		t.Errorf("search moved = %v", got)
	}
}

// cancelChunker cancels the sync's context on its first call.
type cancelChunker struct {
	knowledge.Chunker
	once   sync.Once
	cancel context.CancelFunc
}

func (c *cancelChunker) Chunk(d knowledge.Doc, src []byte) (knowledge.ChunkResult, error) {
	c.once.Do(c.cancel)
	return c.Chunker.Chunk(d, src)
}

func TestContextCancel(t *testing.T) {
	root := t.TempDir()
	for i := range 200 {
		writeFile(t, root, filepath.ToSlash(filepath.Join("d", string(rune('a'+i%26)), strings.Repeat("n", 1+i/26)+".md")), "# N\n\nbody\n")
	}
	st := newStore(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := New(st, Source{Root: root}, testOptions()...).Sync(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("pre-cancelled Sync err = %v", err)
	}
	if got := docIDs(t, st); len(got) != 0 {
		t.Errorf("pre-cancelled sync wrote %d docs", len(got))
	}

	ctx, cancel = context.WithCancel(context.Background())
	defer cancel()
	cc := &cancelChunker{Chunker: testChunkers()[0], cancel: cancel}
	start := time.Now()
	_, err := New(st, Source{Root: root}, WithChunkers(cc), WithWorkers(2)).Sync(ctx)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("mid-run Sync err = %v", err)
	}
	if d := time.Since(start); d > 5*time.Second {
		t.Errorf("cancel took %v", d)
	}
	if n := len(docIDs(t, st)); n >= 200 {
		t.Errorf("cancelled sync still wrote all %d docs", n)
	}
	// A later sync completes the job.
	rep := sync1(t, New(st, Source{Root: root}, testOptions()...))
	if got := len(docIDs(t, st)); got != 200 || rep.Added+rep.Unchanged != 200 {
		t.Errorf("after resume: %d docs, report %+v", got, countsOf(rep))
	}
}

func TestBadSource(t *testing.T) {
	st := newStore(t)
	if _, err := New(st, Source{Root: filepath.Join(t.TempDir(), "nope")}, testOptions()...).Sync(context.Background()); err == nil {
		t.Error("missing root: want error")
	}
	if _, err := New(st, Source{}, testOptions()...).Sync(context.Background()); err == nil {
		t.Error("empty root: want error")
	}
}

// ---- backend-independent scenario --------------------------------------------

// runLifecycle exercises add, unchanged, touch, edit, rename with inbound
// links, delete with re-resolution and late link resolution on st.
func runLifecycle(t *testing.T, st knowledge.Store) {
	root := t.TempDir()
	writeFile(t, root, "notes/target.md", "# Target\n\nquokkaword target body\n")
	writeFile(t, root, "notes/linker.md", "# Linker\n\nsee [[target]] and [[Pending]] and [up](../code/k.go)\n")
	writeFile(t, root, "code/k.go", "// Package k does k.\npackage k\n\n// Rotate rotates wombatword keys.\nfunc Rotate() {}\n")
	ing := New(st, Source{Root: root, Scope: "ws"}, testOptions()...)

	wantCounts(t, sync1(t, ing), counts{Added: 3})
	if got := links(t, st, "ws/notes/linker.md"); !slices.Equal(got, nodes(dn("ws/notes/target.md"), dn("ws/code/k.go"), knowledge.MissingNode("Pending"))) {
		t.Fatalf("links = %v", got)
	}
	wantCounts(t, sync1(t, ing), counts{Unchanged: 3})

	bumpMTime(t, root, "code/k.go", 4*time.Second)
	wantCounts(t, sync1(t, ing), counts{Touched: 1, Unchanged: 2})

	editFile(t, root, "code/k.go", "// Package k does k.\npackage k\n\n// Rotate rotates numbatword keys.\nfunc Rotate() {}\n")
	wantCounts(t, sync1(t, ing), counts{Changed: 1, Unchanged: 2})
	if got := search(t, st, "numbatword"); !slices.Equal(got, ids("ws/code/k.go")) {
		t.Errorf("search new = %v", got)
	}
	if got := search(t, st, "wombatword"); len(got) != 0 {
		t.Errorf("search old = %v", got)
	}

	if err := os.Rename(filepath.Join(root, "notes", "target.md"), filepath.Join(root, "code", "moved.md")); err != nil {
		t.Fatal(err)
	}
	// linker.md has an unresolved link, so it is re-resolved after the move,
	// but its targets are unchanged (the inbound edge was re-keyed).
	wantCounts(t, sync1(t, ing), counts{Moved: 1, Unchanged: 2})
	if got := inbound(t, st, "ws/code/moved.md"); !slices.Equal(got, ids("ws/notes/linker.md")) {
		t.Errorf("inbound to moved = %v", got)
	}
	if got := search(t, st, "quokkaword"); !slices.Equal(got, ids("ws/code/moved.md")) {
		t.Errorf("search moved = %v", got)
	}

	writeFile(t, root, "notes/pending.md", "# Pending\n\nnow here\n")
	wantCounts(t, sync1(t, ing), counts{Added: 1, Changed: 1, Unchanged: 2})
	if got := links(t, st, "ws/notes/linker.md"); !slices.Equal(got, nodes(dn("ws/code/moved.md"), dn("ws/code/k.go"), dn("ws/notes/pending.md"))) {
		t.Errorf("links after add = %v", got)
	}

	if err := os.Remove(filepath.Join(root, "code", "k.go")); err != nil {
		t.Fatal(err)
	}
	wantCounts(t, sync1(t, ing), counts{Deleted: 1, Changed: 1, Unchanged: 2})
	if got := links(t, st, "ws/notes/linker.md"); !slices.Equal(got, nodes(dn("ws/code/moved.md"), dn("ws/notes/pending.md"), knowledge.MissingNode("../code/k.go"))) {
		t.Errorf("links after delete = %v", got)
	}
	if cs, err := st.DocChunks(context.Background(), "ws/code/k.go"); err != nil || len(cs) != 0 {
		t.Errorf("chunks of deleted = %d, %v", len(cs), err)
	}
	if got := docIDs(t, st); !slices.Equal(got, ids("ws/code/moved.md", "ws/notes/linker.md", "ws/notes/pending.md")) {
		t.Errorf("docs = %v", got)
	}
	wantCounts(t, sync1(t, ing), counts{Unchanged: 3})
}

func TestLifecycleMemory(t *testing.T) { runLifecycle(t, newStore(t)) }

// linkWeights maps LinksTo targets asserted by id to their edge weight.
func linkWeights(t *testing.T, st knowledge.Store, id knowledge.DocID) map[knowledge.NodeID]float32 {
	t.Helper()
	es, err := st.Neighbors(context.Background(), []knowledge.NodeID{knowledge.DocNode(id)}, []knowledge.EdgeKind{knowledge.LinksTo}, knowledge.Out, 0)
	if err != nil {
		t.Fatal(err)
	}
	out := make(map[knowledge.NodeID]float32)
	for _, e := range es {
		if e.Source == id {
			out[e.To] = e.Weight
		}
	}
	return out
}

func TestMarkdownLinkBaseNameFallback(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "rover/docs/imu-calibration.md", "# IMU calibration\n")
	writeFile(t, root, "atlas/imu-calibration.md", "# Other calibration\n") // fewer segments, other scope
	writeFile(t, root, "rover/docs/motor.md", "# Motor\n")
	writeFile(t, root, "rover/sib.md", "# Sib\n")
	writeFile(t, root, "rover/docs/imu-fusion.md", strings.Join([]string{
		"# IMU fusion",
		"",
		"see the [calibration procedure](old/imu-calibration.md#steps)",
		"and [motor](motor.md) exactly, [again](../docs/motor.md)",
		"[noext](old/sib) [ghost](old/ghost.md) [esc](../../../x/motor.md) [pic](old/motor.png)",
		"",
	}, "\n"))
	st := newStore(t)
	ing := New(st, Source{Root: root}, testOptions()...)
	sync1(t, ing)

	want := map[knowledge.NodeID]float32{
		dn("rover/docs/imu-calibration.md"):   0.5, // base-name fallback, same scope wins
		dn("rover/docs/motor.md"):             1,   // exact path
		knowledge.MissingNode("old/sib"):      1,   // no extension: no fallback
		knowledge.MissingNode("old/ghost.md"): 1,   // no file with that base name
	}
	got := linkWeights(t, st, "rover/docs/imu-fusion.md")
	if len(got) != len(want) {
		t.Errorf("links = %v, want %v", got, want)
	}
	for to, w := range want {
		if gw, ok := got[to]; !ok || gw != w {
			t.Errorf("link %s weight = %v (present %v), want %v", to, gw, ok, w)
		}
	}
	d, _ := getDoc(t, st, "rover/docs/imu-fusion.md")
	if d.Props[PropMissing] != "2" || d.Props[PropFuzzy] != "1" {
		t.Errorf("props %s=%q %s=%q, want 2 and 1", PropMissing, d.Props[PropMissing], PropFuzzy, d.Props[PropFuzzy])
	}

	// The exact path appears: the fallback edge is re-resolved to it.
	writeFile(t, root, "rover/docs/old/imu-calibration.md", "# Moved back\n")
	wantCounts(t, sync1(t, ing), counts{Added: 1, Changed: 1, Unchanged: 4})
	got = linkWeights(t, st, "rover/docs/imu-fusion.md")
	if w, ok := got[dn("rover/docs/old/imu-calibration.md")]; !ok || w != 1 {
		t.Errorf("exact link after add = %v", got)
	}
	if _, ok := got[dn("rover/docs/imu-calibration.md")]; ok {
		t.Errorf("fallback edge kept after exact match: %v", got)
	}
	if d, _ := getDoc(t, st, "rover/docs/imu-fusion.md"); d.Props[PropFuzzy] != "" {
		t.Errorf("%s = %q, want absent", PropFuzzy, d.Props[PropFuzzy])
	}
}

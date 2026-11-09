package ingest

import (
	"context"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// ---- shared stores -------------------------------------------------------------

func TestSharedStoreScopedSources(t *testing.T) {
	st := newStore(t)
	r1, r2 := t.TempDir(), t.TempDir()
	writeFile(t, r1, "a.md", "# A\n\nalpha\n")
	writeFile(t, r2, "b.md", "# B\n\nbeta\n")
	one := New(st, Source{Root: r1, Scope: "one"}, testOptions()...)
	two := New(st, Source{Root: r2, Scope: "two"}, testOptions()...)
	wantCounts(t, sync1(t, one), counts{Added: 1})
	wantCounts(t, sync1(t, two), counts{Added: 1})
	wantCounts(t, sync1(t, one), counts{Unchanged: 1})
	wantCounts(t, sync1(t, two), counts{Unchanged: 1})
	if got := docIDs(t, st); !slices.Equal(got, ids("one/a.md", "two/b.md")) {
		t.Errorf("docs = %v", got)
	}
	// A deletion in one source removes only its own doc.
	if err := os.Remove(filepath.Join(r1, "a.md")); err != nil {
		t.Fatal(err)
	}
	wantCounts(t, sync1(t, one), counts{Deleted: 1})
	if got := docIDs(t, st); !slices.Equal(got, ids("two/b.md")) {
		t.Errorf("docs after delete = %v", got)
	}
}

func TestSharedStoreScopelessSources(t *testing.T) {
	st := newStore(t)
	r1, r2 := t.TempDir(), t.TempDir()
	writeFile(t, r1, "a/x.md", "# X\n\nxword\n")
	writeFile(t, r2, "b/y.md", "# Y\n\nyword\n")
	one := New(st, Source{Root: r1}, testOptions()...)
	two := New(st, Source{Root: r2}, testOptions()...)
	sync1(t, one)
	sync1(t, two)
	wantCounts(t, sync1(t, one), counts{Unchanged: 1})
	wantCounts(t, sync1(t, two), counts{Unchanged: 1})
	if got := docIDs(t, st); !slices.Equal(got, ids("a/x.md", "b/y.md")) {
		t.Errorf("docs = %v", got)
	}
}

// ---- unreadable paths ------------------------------------------------------------

func TestUnreadableDirKeepsDocs(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions")
	}
	st := newStore(t)
	root := t.TempDir()
	writeFile(t, root, "keep.md", "# K\n\nkeep\n")
	writeFile(t, root, "locked/x.md", "# X\n\nxword\n")
	writeFile(t, root, "locked/deeper/y.md", "# Y\n\nyword\n")
	ing := New(st, Source{Root: root}, testOptions()...)
	wantCounts(t, sync1(t, ing), counts{Added: 3})

	locked := filepath.Join(root, "locked")
	if err := os.Chmod(locked, 0); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(locked, 0o755) })
	rep, err := ing.Sync(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Errors) != 1 || !strings.Contains(rep.Errors[0].Error(), "locked") {
		t.Errorf("errors = %v, want one about locked", rep.Errors)
	}
	wantCounts(t, rep, counts{Unchanged: 1})
	if got := docIDs(t, st); !slices.Equal(got, ids("keep.md", "locked/deeper/y.md", "locked/x.md")) {
		t.Errorf("docs = %v", got)
	}
}

// ---- secrets ----------------------------------------------------------------------

func TestSecretPatternsExtended(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		"k.go":      "package k\n\nfunc F() {\n\tapiKey := \"zq9Xv2Lm8Rt4Wy6Pk3Hn7Bc5\"\n\t_ = apiKey\n}\n",
		"j.md":      "# Config\n\njsonmarker\n\n```json\n{\"password\": \"hunter2hunter2hunter2xx\"}\n```\n",
		"ghp.md":    "# G\n\nghpmarker ghp_abcdefghijklmnopqrstuvwxyz0123456789\n",
		"oauth.md":  "# G\n\ngho_marker gho_abcdefghijklmnopqrstuvwxyz0123456789\n",
		"pat.md":    "# G\n\npatmarker github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuv\n",
		"slack.md":  "# S\n\nslackmarker xoxb-123456789012-abcdefghijkl\n",
		"openai.md": "# O\n\nopenaimarker sk-abcdefghijklmnopqrstuvwx0123\n",
		"gitlab.md": "# L\n\ngitlabmarker glpat-abcdefghij0123456789\n",
		"fm.md":     "---\ntitle: FM\napi_key: abcdefghijklmnopqrstuvwxyz012345\nowner: me\n---\n# FM\n\nfmmarker body\n",
		"clean.md":  "# Clean\n\ncleanmarker token = short\n\ntoken: tokenSourceProviderFactory\n\npassword := os.Getenv(\"DB_PASSWORD\")\n\nsk-learn is a library; the github_pat_ prefix is documented.\n",
		"clean.go":  "package c\n\n// Refresh renews cleangomarker credentials.\nfunc Refresh(token string) { tok := token; _ = tok }\n",
	}
	for rel, content := range files {
		writeFile(t, root, rel, content)
	}
	st := newStore(t)
	rep := sync1(t, New(st, Source{Root: root}, testOptions()...))
	// 8 secret lines (k.go, j.md, six token files) + 1 frontmatter line.
	if rep.SecretsDropped != 9 {
		t.Errorf("SecretsDropped = %d, want 9", rep.SecretsDropped)
	}
	for _, q := range []string{"zq9Xv2Lm8Rt4Wy6Pk3Hn7Bc5", "hunter2hunter2hunter2xx", "ghpmarker", "gho_marker", "patmarker", "slackmarker", "openaimarker", "gitlabmarker"} {
		if got := search(t, st, q); len(got) != 0 {
			t.Errorf("search %q = %v, want none", q, got)
		}
	}
	// Lines, not chunks: jsonmarker shares j.md's chunk, not its line.
	for q, want := range map[string]string{"cleanmarker": "clean.md", "cleangomarker": "clean.go", "fmmarker": "fm.md", "jsonmarker": "j.md"} {
		if got := search(t, st, q); !slices.Equal(got, ids(want)) {
			t.Errorf("search %q = %v, want [%s]", q, got, want)
		}
	}
	fm, _ := getDoc(t, st, "fm.md")
	if _, ok := fm.Props["api_key"]; ok || fm.Props["owner"] != "me" || fm.Title != "FM" {
		t.Errorf("fm.md title %q props %v", fm.Title, fm.Props)
	}
}

func TestSecretTitleDropped(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "t.md", "---\ntitle: \"secret: zq9Xv2Lm8Rt4Wy6Pk3Hn7Bc5\"\n---\n\nplain words\n")
	st := newStore(t)
	rep := sync1(t, New(st, Source{Root: root}, testOptions()...))
	d, ok := getDoc(t, st, "t.md")
	if !ok {
		t.Fatal("t.md not indexed")
	}
	if strings.Contains(d.Title, "zq9Xv2") || strings.Contains(d.Props["title"], "zq9Xv2") {
		t.Errorf("secret kept: title %q props %v", d.Title, d.Props)
	}
	if rep.SecretsDropped != 1 { // the frontmatter title line
		t.Errorf("SecretsDropped = %d, want 1", rep.SecretsDropped)
	}
	if got := search(t, st, "zq9Xv2Lm8Rt4Wy6Pk3Hn7Bc5"); len(got) != 0 {
		t.Errorf("search secret = %v", got)
	}
}

// Props, title and heading crumbs a chunker derives are scrubbed too.
func TestDropSecretMeta(t *testing.T) {
	cr := knowledge.ChunkResult{
		Title:  "secret: zq9Xv2Lm8Rt4Wy6Pk3Hn7Bc5",
		Props:  map[string]string{"api_key": "abcdefghijklmnopqrstuvwxyz012345", "owner": "me"},
		Chunks: []knowledge.Chunk{{Heading: []string{"Top", "key AKIAIOSFODNN7EXAMPLE"}}, {Heading: []string{"Top"}}},
	}
	if n := dropSecretMeta("dir/note.md", &cr); n != 3 {
		t.Errorf("dropSecretMeta = %d, want 3", n)
	}
	if cr.Title != "note" || !slices.Equal(slices.Sorted(maps.Keys(cr.Props)), []string{"owner"}) || !slices.Equal(cr.Chunks[0].Heading, []string{"Top", "note"}) {
		t.Errorf("after scrub: title %q props %v heading %v", cr.Title, cr.Props, cr.Chunks[0].Heading)
	}
}

// ---- deterministic moves -------------------------------------------------------------

func moverOf(t *testing.T, st knowledge.Store) []knowledge.DocID {
	t.Helper()
	var out []knowledge.DocID
	for d, err := range st.ListDocs(context.Background()) {
		if err != nil {
			t.Fatal(err)
		}
		if d.Props[PropMovedFrom] != "" {
			out = append(out, d.ID)
		}
	}
	return out
}

func TestMoveTargetDeterministic(t *testing.T) {
	content := "# Same\n\nidentical content body words here for move\n"
	for _, tc := range []struct {
		name    string
		targets []string
		want    knowledge.DocID
	}{
		{"same base name wins", []string{"z/b.md", "y/a.md"}, "y/a.md"},
		{"lexical rel wins", []string{"r/s.md", "p/q.md"}, "p/q.md"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for run := range 30 {
				st := newStore(t)
				root := t.TempDir()
				writeFile(t, root, "x/a.md", content)
				writeFile(t, root, "linker.md", "# L\n\nsee [l](x/a.md)\n")
				ing := New(st, Source{Root: root}, WithChunkers(testChunkers()...), WithWorkers(8))
				sync1(t, ing)
				if err := os.Remove(filepath.Join(root, "x", "a.md")); err != nil {
					t.Fatal(err)
				}
				for _, rel := range tc.targets {
					writeFile(t, root, rel, content)
				}
				for j := range 30 {
					writeFile(t, root, "pad/"+strings.Repeat("p", j+1)+".md", "# P\n\npad\n")
				}
				rep := sync1(t, ing)
				if rep.Moved != 1 || rep.Added != 31 || rep.Deleted != 0 {
					t.Fatalf("run %d: report %+v", run, countsOf(rep))
				}
				if got := moverOf(t, st); !slices.Equal(got, []knowledge.DocID{tc.want}) {
					t.Fatalf("run %d: moved to %v, want %s", run, got, tc.want)
				}
				if got := inbound(t, st, tc.want); !slices.Equal(got, ids("linker.md")) {
					t.Fatalf("run %d: inbound = %v", run, got)
				}
			}
		})
	}
}

// ---- better candidates ----------------------------------------------------------------

func TestAddedBetterCandidateMatchesRebuild(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a/deep/Topic.md", "# Deep topic\n")
	writeFile(t, root, "linker.md", "# L\n\nsee [[Topic]]\n")
	writeFile(t, root, "other.md", "# O\n\nnothing\n")
	st := newStore(t)
	ing := New(st, Source{Root: root}, testOptions()...)
	sync1(t, ing)
	if got := links(t, st, "linker.md"); !slices.Equal(got, nodes(dn("a/deep/Topic.md"))) {
		t.Fatalf("links before = %v", got)
	}
	writeFile(t, root, "Topic.md", "# Root topic\n")
	wantCounts(t, sync1(t, ing), counts{Added: 1, Changed: 1, Unchanged: 2})

	rebuilt := newStore(t)
	sync1(t, New(rebuilt, Source{Root: root}, testOptions()...))
	for _, id := range docIDs(t, rebuilt) {
		if got, want := links(t, st, id), links(t, rebuilt, id); !slices.Equal(got, want) {
			t.Errorf("%s: incremental links %v, rebuild %v", id, got, want)
		}
	}
	if got := links(t, st, "linker.md"); !slices.Equal(got, nodes(dn("Topic.md"))) {
		t.Errorf("links after add = %v", got)
	}
}

func TestRealFileReplacesAliasForUnchangedLinker(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "notes/target.md", "# Target\n\nbody\n")
	writeFile(t, root, "linker.md", "# L\n\nsee [[target]]\n")
	st := newStore(t)
	ing := New(st, Source{Root: root}, testOptions()...)
	sync1(t, ing)
	if err := os.Rename(filepath.Join(root, "notes", "target.md"), filepath.Join(root, "renamed.md")); err != nil {
		t.Fatal(err)
	}
	sync1(t, ing)
	writeFile(t, root, "notes/target.md", "# New target\n\nfresh\n")
	wantCounts(t, sync1(t, ing), counts{Added: 1, Changed: 1, Unchanged: 1})
	if got := links(t, st, "linker.md"); !slices.Equal(got, nodes(dn("notes/target.md"))) {
		t.Errorf("linker links = %v", got)
	}
}

// ---- gitignore and symlinks -------------------------------------------------------------

func TestGitignoreDoubleStarAnyDepth(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, ".gitignore", "**/gen/out\n")
	writeFile(t, root, "sub/.gitignore", "**/tmp.md\n")
	for _, rel := range []string{"gen/out/b.md", "x/y/gen/out/a.md", "sub/tmp.md", "sub/q/r/tmp.md"} {
		writeFile(t, root, rel, "# D\n")
	}
	keep := []string{"gen/other.md", "x/gen/outx/c.md", "tmp.md", "out/gen/d.md"}
	for _, rel := range keep {
		writeFile(t, root, rel, "# K\n")
	}
	st := newStore(t)
	sync1(t, New(st, Source{Root: root}, testOptions()...))
	want := ids(keep...)
	slices.Sort(want)
	if got := docIDs(t, st); !slices.Equal(got, want) {
		t.Errorf("docs = %v, want %v", got, want)
	}
}

func TestFileSymlinks(t *testing.T) {
	parent := t.TempDir()
	root := filepath.Join(parent, "root")
	writeFile(t, parent, "outside.md", "# Outside\n\noutsideword\n")
	writeFile(t, root, "in.md", "# In\n\ninword\n")
	writeFile(t, root, "creds/server.pem", "-----BEGIN PRIVATE KEY-----\n")
	for link, target := range map[string]string{
		"link.md":   filepath.Join(parent, "outside.md"),
		"rel.md":    filepath.Join("..", "outside.md"),
		"inlink.md": "in.md",
		"notes.md":  filepath.Join("creds", "server.pem"),
	} {
		if err := os.Symlink(target, filepath.Join(root, link)); err != nil {
			t.Fatal(err)
		}
	}
	st := newStore(t)
	rep := sync1(t, New(st, Source{Root: root}, testOptions()...))
	// The in-root link is indexed under its own path (content duplicated).
	if got := docIDs(t, st); !slices.Equal(got, ids("in.md", "inlink.md")) {
		t.Errorf("docs = %v", got)
	}
	if got := search(t, st, "outsideword"); len(got) != 0 {
		t.Errorf("outside content indexed: %v", got)
	}
	if rep.SecretsDropped != 2 { // creds/server.pem and the link to it
		t.Errorf("SecretsDropped = %d, want 2", rep.SecretsDropped)
	}
}

// ---- DocID collisions, concurrent syncs, self fallback -----------------------------

func TestScopelessDocIDCollisionReported(t *testing.T) {
	st := newStore(t)
	r1, r2 := t.TempDir(), t.TempDir()
	writeFile(t, r1, "notes/a.md", "# A\n\nfirstroot words\n")
	writeFile(t, r2, "notes/a.md", "# A\n\nsecondroot words here\n")
	one := New(st, Source{Root: r1}, testOptions()...)
	two := New(st, Source{Root: r2}, testOptions()...)
	wantCounts(t, sync1(t, one), counts{Added: 1})

	rep, err := two.Sync(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Errors) != 1 || !strings.Contains(rep.Errors[0].Error(), "notes/a.md") || !strings.Contains(rep.Errors[0].Error(), "collides") {
		t.Fatalf("errors = %v, want one DocID collision for notes/a.md", rep.Errors)
	}
	// The first source's document is untouched.
	if got := search(t, st, "firstroot"); !slices.Equal(got, ids("notes/a.md")) {
		t.Errorf("search firstroot = %v", got)
	}
	if got := search(t, st, "secondroot"); len(got) != 0 {
		t.Errorf("search secondroot = %v, want none", got)
	}
	if d, _ := getDoc(t, st, "notes/a.md"); !strings.HasPrefix(d.Path, r1) {
		t.Errorf("doc path = %s, want under %s", d.Path, r1)
	}
	wantCounts(t, sync1(t, one), counts{Unchanged: 1})
}

// slowStore widens the window in which two Syncs could overlap.
type slowStore struct{ knowledge.Store }

func (s slowStore) MoveDoc(ctx context.Context, from knowledge.DocID, to knowledge.Doc) error {
	time.Sleep(time.Millisecond)
	return s.Store.MoveDoc(ctx, from, to)
}

func (s slowStore) DeleteDoc(ctx context.Context, id knowledge.DocID) error {
	time.Sleep(time.Millisecond)
	return s.Store.DeleteDoc(ctx, id)
}

func TestConcurrentSyncsSerialised(t *testing.T) {
	root := t.TempDir()
	for i := range 40 {
		writeFile(t, root, "d/n"+strconv.Itoa(i)+".md", "# N\n\nbody number "+strconv.Itoa(i)+"\n")
	}
	st := slowStore{newStore(t)}
	ing := New(st, Source{Root: root}, testOptions()...)
	sync1(t, ing)
	for i := range 20 {
		if err := os.Rename(filepath.Join(root, "d", "n"+strconv.Itoa(i)+".md"), filepath.Join(root, "d", "m"+strconv.Itoa(i)+".md")); err != nil {
			t.Fatal(err)
		}
		if err := os.Remove(filepath.Join(root, "d", "n"+strconv.Itoa(20+i)+".md")); err != nil {
			t.Fatal(err)
		}
	}
	var (
		wg   sync.WaitGroup
		reps [2]knowledge.Report
		errs [2]error
	)
	for k := range reps {
		wg.Go(func() { reps[k], errs[k] = ing.Sync(context.Background()) })
	}
	wg.Wait()
	moved, deleted := 0, 0
	for k := range reps {
		if errs[k] != nil || len(reps[k].Errors) > 0 {
			t.Errorf("sync %d: err %v, report errors %v", k, errs[k], reps[k].Errors)
		}
		moved += reps[k].Moved
		deleted += reps[k].Deleted
	}
	if moved != 20 || deleted != 20 {
		t.Errorf("moved %d deleted %d across both syncs, want 20 and 20", moved, deleted)
	}
	if n := len(docIDs(t, st)); n != 20 {
		t.Errorf("%d docs, want 20", n)
	}
}

func TestMarkdownFallbackSkipsSelf(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "topic.md", "# Topic\n\nsee [old](gone/topic.md)\n")
	st := newStore(t)
	ing := New(st, Source{Root: root}, testOptions()...)
	sync1(t, ing)
	if got := links(t, st, "topic.md"); !slices.Equal(got, nodes(knowledge.MissingNode("gone/topic.md"))) {
		t.Errorf("links = %v, want the missing target", got)
	}
	writeFile(t, root, "a/deep/topic.md", "# Deep topic\n")
	sync1(t, ing)
	if got := links(t, st, "topic.md"); !slices.Equal(got, nodes(dn("a/deep/topic.md"))) {
		t.Errorf("links after the namesake appears = %v", got)
	}
	fresh := newStore(t)
	sync1(t, New(fresh, Source{Root: root}, testOptions()...))
	if got, want := links(t, st, "topic.md"), links(t, fresh, "topic.md"); !slices.Equal(got, want) {
		t.Errorf("incremental links %v, rebuild %v", got, want)
	}
}

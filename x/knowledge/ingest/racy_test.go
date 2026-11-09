package ingest

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// setMTime sets a file's mtime exactly (no rounding).
func setMTime(t *testing.T, root, rel string, mt time.Time) {
	t.Helper()
	if err := os.Chtimes(filepath.Join(root, filepath.FromSlash(rel)), mt, mt); err != nil {
		t.Fatal(err)
	}
}

func mtimeOf(t *testing.T, root, rel string) time.Time {
	t.Helper()
	fi, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel)))
	if err != nil {
		t.Fatal(err)
	}
	return fi.ModTime()
}

// rewriteSameTick rewrites a file with same-size content and restores mtime
// mt, as a coarse-mtime filesystem does for an edit within the same tick.
func rewriteSameTick(t *testing.T, root, rel, content string, mt time.Time) {
	t.Helper()
	writeFile(t, root, rel, content)
	setMTime(t, root, rel, mt)
}

func TestSameSecondRewriteReindexed(t *testing.T) {
	const before, after = "# A\n\noldword here\n", "# A\n\nnewword here\n"
	cases := []struct {
		name string
		// prepare runs after the first write and returns the mtime the
		// rewrite carries.
		prepare func(t *testing.T, root string) time.Time
	}{
		{
			// Old file (outside the racy window): only the sub-second part of
			// the mtime changes.
			name: "sub-second mtime change",
			prepare: func(t *testing.T, root string) time.Time {
				base := time.Now().Add(-time.Hour).Truncate(time.Second).Add(100 * time.Millisecond)
				setMTime(t, root, "a.md", base)
				return base.Add(500 * time.Millisecond)
			},
		},
		{
			// Freshly written file, rewritten with the identical mtime.
			name: "identical mtime just after the write",
			prepare: func(t *testing.T, root string) time.Time {
				return mtimeOf(t, root, "a.md")
			},
		},
		{
			// Old file touched to now, then rewritten within the same tick.
			name: "touch then same-tick rewrite",
			prepare: func(t *testing.T, root string) time.Time {
				setMTime(t, root, "a.md", time.Now().Add(-time.Hour))
				return time.Now()
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			root := t.TempDir()
			writeFile(t, root, "a.md", before)
			mt := tc.prepare(t, root)
			st := newStore(t)
			ing := New(st, Source{Root: root}, testOptions()...)
			wantCounts(t, sync1(t, ing), counts{Added: 1})
			if tc.name == "touch then same-tick rewrite" {
				setMTime(t, root, "a.md", mt)
				wantCounts(t, sync1(t, ing), counts{Touched: 1})
			}

			rewriteSameTick(t, root, "a.md", after, mt)
			wantCounts(t, sync1(t, ing), counts{Changed: 1})
			if got := search(t, st, "newword"); !slices.Equal(got, ids("a.md")) {
				t.Errorf("search new word = %v, want [a.md]", got)
			}
			if got := search(t, st, "oldword"); len(got) != 0 {
				t.Errorf("search old word = %v, want none", got)
			}
			wantCounts(t, sync1(t, ing), counts{Unchanged: 1})
		})
	}
}

func TestSameSecondSwapDetected(t *testing.T) {
	root := t.TempDir()
	const a, b = "# S\n\nappleword\n", "# S\n\ncherryxyz\n"
	writeFile(t, root, "a.md", a)
	writeFile(t, root, "b.md", b)
	st := newStore(t)
	ing := New(st, Source{Root: root}, testOptions()...)
	wantCounts(t, sync1(t, ing), counts{Added: 2})

	ma, mb := mtimeOf(t, root, "a.md"), mtimeOf(t, root, "b.md")
	rewriteSameTick(t, root, "a.md", b, ma)
	rewriteSameTick(t, root, "b.md", a, mb)
	wantCounts(t, sync1(t, ing), counts{Changed: 2})
	if got := search(t, st, "appleword"); !slices.Equal(got, ids("b.md")) {
		t.Errorf("search appleword = %v, want [b.md]", got)
	}
	if got := search(t, st, "cherryxyz"); !slices.Equal(got, ids("a.md")) {
		t.Errorf("search cherryxyz = %v, want [a.md]", got)
	}
}

// Once a file's mtime has left the racy window, the catalog stat is trusted
// again: one bookkeeping write, then syncs write nothing (which also needs
// the store to keep the mtime at full precision).
func TestRacyMarkerSettles(t *testing.T) {
	for name, open := range map[string]func(*testing.T) knowledge.Store{"memory": newStore, "sqlite": newSQLiteStore} {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			writeFile(t, root, "a.md", "# A\n\nalphaword\n")
			st := open(t)
			ing := New(st, Source{Root: root}, testOptions()...)
			wantCounts(t, sync1(t, ing), counts{Added: 1})
			gen := func() uint64 {
				g, err := st.Generation(context.Background())
				if err != nil {
					t.Fatal(err)
				}
				return g
			}
			// Still racy: re-hashed, nothing written.
			g0 := gen()
			wantCounts(t, sync1(t, ing), counts{Unchanged: 1})
			if g := gen(); g != g0 {
				t.Errorf("racy unchanged sync wrote: generation %d → %d", g0, g)
			}

			ing.now = func() time.Time { return time.Now().Add(time.Minute) }
			wantCounts(t, sync1(t, ing), counts{Unchanged: 1})
			g1 := gen()
			if g1 == g0 {
				t.Error("leaving the racy window wrote nothing")
			}
			wantCounts(t, sync1(t, ing), counts{Unchanged: 1})
			if g := gen(); g != g1 {
				t.Errorf("settled sync wrote: generation %d → %d", g1, g)
			}
			if got := search(t, st, "alphaword"); !slices.Equal(got, ids("a.md")) {
				t.Errorf("search = %v", got)
			}
		})
	}
}

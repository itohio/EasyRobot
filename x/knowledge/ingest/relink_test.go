package ingest

import (
	"context"
	"errors"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// commitCancelStore cancels a context right after one successful commit
// (PutDoc, MoveDoc or DeleteDoc of one DocID), as a crash or shutdown
// between the structural commit and the relink pass would.
type commitCancelStore struct {
	knowledge.Store
	mu     sync.Mutex
	op     string
	id     knowledge.DocID
	cancel context.CancelFunc
}

func (s *commitCancelStore) arm(op string, id knowledge.DocID, cancel context.CancelFunc) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.op, s.id, s.cancel = op, id, cancel
}

func (s *commitCancelStore) committed(op string, id knowledge.DocID) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.cancel != nil && op == s.op && id == s.id {
		s.cancel()
		s.cancel = nil
	}
}

func (s *commitCancelStore) PutDoc(ctx context.Context, u knowledge.DocUpdate) error {
	if err := s.Store.PutDoc(ctx, u); err != nil {
		return err
	}
	s.committed("put", u.Doc.ID)
	return nil
}

func (s *commitCancelStore) MoveDoc(ctx context.Context, from knowledge.DocID, to knowledge.Doc) error {
	if err := s.Store.MoveDoc(ctx, from, to); err != nil {
		return err
	}
	s.committed("move", to.ID)
	return nil
}

func (s *commitCancelStore) DeleteDoc(ctx context.Context, id knowledge.DocID) error {
	if err := s.Store.DeleteDoc(ctx, id); err != nil {
		return err
	}
	s.committed("delete", id)
	return nil
}

// allLinks maps every catalogued document to its sorted LinksTo targets.
func allLinks(t *testing.T, st knowledge.Store) map[knowledge.DocID][]knowledge.NodeID {
	t.Helper()
	out := make(map[knowledge.DocID][]knowledge.NodeID)
	for _, id := range docIDs(t, st) {
		out[id] = links(t, st, id)
	}
	return out
}

func TestRelinkSurvivesCancelAfterCommit(t *testing.T) {
	cases := []struct {
		name  string
		files map[string]string
		// change edits the tree after the first sync.
		change func(t *testing.T, root string)
		op     string
		id     knowledge.DocID
	}{
		{
			name:   "add resolves a missing link",
			files:  map[string]string{"linker.md": "# L\n\nsee [[later]]\n", "plain.md": "# P\n\nplain\n"},
			change: func(t *testing.T, root string) { writeFile(t, root, "later.md", "# Later\n\nlater body\n") },
			op:     "put", id: "later.md",
		},
		{
			name:  "delete leaves a dangling link",
			files: map[string]string{"linker.md": "# L\n\nsee [[gone]]\n", "gone.md": "# Gone\n\ngone body\n", "plain.md": "# P\n\nplain\n"},
			change: func(t *testing.T, root string) {
				if err := os.Remove(filepath.Join(root, "gone.md")); err != nil {
					t.Fatal(err)
				}
			},
			op: "delete", id: "gone.md",
		},
		{
			name:   "added namesake is a better match",
			files:  map[string]string{"linker.md": "# L\n\nsee [[topic]]\n", "deep/x/topic.md": "# Deep topic\n\ndeep\n"},
			change: func(t *testing.T, root string) { writeFile(t, root, "topic.md", "# Topic\n\nroot topic\n") },
			op:     "put", id: "topic.md",
		},
		{
			name: "moved namesake is a better match",
			files: map[string]string{
				"linker.md":         "# L\n\nsee [[target]]\n",
				"deep/q/target.md":  "# Target q\n\nq body\n",
				"x/y/z/target.md":   "# Target z\n\nz body\n",
				"unrelated/note.md": "# N\n\nnote\n",
			},
			change: func(t *testing.T, root string) {
				if err := os.Rename(filepath.Join(root, "x/y/z/target.md"), filepath.Join(root, "target.md")); err != nil {
					t.Fatal(err)
				}
			},
			op: "move", id: "target.md",
		},
	}
	stores := []struct {
		name string
		open func(*testing.T) knowledge.Store
	}{{"memory", newStore}, {"sqlite", newSQLiteStore}}
	for _, sc := range stores {
		for _, tc := range cases {
			t.Run(sc.name+"/"+tc.name, func(t *testing.T) {
				root := t.TempDir()
				for rel, content := range tc.files {
					writeFile(t, root, rel, content)
				}
				st := &commitCancelStore{Store: sc.open(t)}
				ing := New(st, Source{Root: root}, testOptions()...)
				sync1(t, ing)

				tc.change(t, root)
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				st.arm(tc.op, tc.id, cancel)
				if _, err := ing.Sync(ctx); !errors.Is(err, context.Canceled) {
					t.Fatalf("interrupted Sync err = %v, want context.Canceled", err)
				}

				sync1(t, ing)
				fresh := sc.open(t)
				sync1(t, New(fresh, Source{Root: root}, testOptions()...))
				got, want := allLinks(t, st), allLinks(t, fresh)
				if !maps.EqualFunc(got, want, slices.Equal) {
					t.Errorf("links after recovery = %v\nrebuild = %v", got, want)
				}
				// The recovery leaves nothing behind.
				if g, w := docIDs(t, st), docIDs(t, fresh); !slices.Equal(g, w) {
					t.Errorf("docs after recovery = %v, rebuild = %v", g, w)
				}
				wantCounts(t, sync1(t, ing), counts{Unchanged: len(docIDs(t, st))})
			})
		}
	}
}

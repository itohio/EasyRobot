package sqlite

import (
	"database/sql"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/store/memory"
	"github.com/itohio/EasyRobot/x/knowledge/store/storetest"
)

func openAt(t *testing.T, path string) *Store {
	t.Helper()
	s, err := Open(path)
	if err != nil {
		t.Fatalf("Open(%s): %v", path, err)
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestStoreFile(t *testing.T) {
	storetest.RunWith(t,
		func(t *testing.T) knowledge.Store { return openAt(t, filepath.Join(t.TempDir(), "k.db")) },
		storetest.RunOpts{Reopen: func(t *testing.T, s knowledge.Store) knowledge.Store {
			st := s.(*Store)
			if err := st.Close(); err != nil {
				t.Fatalf("Close: %v", err)
			}
			return openAt(t, st.path)
		}})
}

func TestStoreInMemory(t *testing.T) {
	storetest.Run(t, func(t *testing.T) knowledge.Store { return openAt(t, ":memory:") })
}

func TestRejectsUnknownSchemaVersion(t *testing.T) {
	path := filepath.Join(t.TempDir(), "k.db")
	s := openAt(t, path)
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`PRAGMA user_version = 99`); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	if s, err := Open(path); err == nil {
		_ = s.Close()
		t.Fatal("Open accepted schema version 99")
	}
}

func TestOpenBadPath(t *testing.T) {
	if s, err := Open(filepath.Join(t.TempDir(), "missing", "dir", "k.db")); err == nil {
		_ = s.Close()
		t.Fatal("Open in a missing directory succeeded")
	}
}

func BenchmarkSearchLex(b *testing.B) {
	for _, tc := range []struct{ name, path string }{
		{"file", filepath.Join(b.TempDir(), "k.db")},
		{"memory", ":memory:"},
	} {
		b.Run(tc.name, func(b *testing.B) {
			s, err := Open(tc.path)
			if err != nil {
				b.Fatal(err)
			}
			b.Cleanup(func() { _ = s.Close() })
			storetest.BenchSearchLex(b, s, 1000, 5)
		})
	}
}

// TestAgreesWithMemory: the memory store's BM25 mirrors FTS5 bm25().
func TestAgreesWithMemory(t *testing.T) {
	storetest.Agree(t,
		func(t *testing.T) knowledge.Store {
			s := memory.New()
			t.Cleanup(func() { _ = s.Close() })
			return s
		},
		func(t *testing.T) knowledge.Store { return openAt(t, filepath.Join(t.TempDir(), "k.db")) })
}

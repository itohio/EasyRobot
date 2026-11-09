package ingest

import (
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/store/sqlite"
)

func newSQLiteStore(t *testing.T) knowledge.Store {
	t.Helper()
	st, err := sqlite.Open(filepath.Join(t.TempDir(), "k.db"))
	if err != nil {
		t.Fatalf("sqlite.Open: %v", err)
	}
	t.Cleanup(func() { _ = st.Close() })
	return st
}

func TestLifecycleSQLite(t *testing.T) { runLifecycle(t, newSQLiteStore(t)) }

package retrieve

import (
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/store/memory"
	"github.com/itohio/EasyRobot/x/knowledge/store/sqlite"
)

// opener names a store backend the suite runs against.
type opener struct {
	name string
	open func(t testing.TB) knowledge.Store
}

var openers = []opener{
	{"memory", func(testing.TB) knowledge.Store { return memory.New() }},
	{"sqlite", func(t testing.TB) knowledge.Store {
		s, err := sqlite.Open(filepath.Join(t.TempDir(), "k.db"))
		if err != nil {
			t.Fatalf("sqlite.Open: %v", err)
		}
		return s
	}},
}

// forEachStore runs fn as a subtest on a fresh store of every backend.
func forEachStore(t *testing.T, fn func(t *testing.T, s knowledge.Store)) {
	t.Helper()
	for _, o := range openers {
		t.Run(o.name, func(t *testing.T) {
			s := o.open(t)
			t.Cleanup(func() {
				if err := s.Close(); err != nil {
					t.Errorf("close: %v", err)
				}
			})
			fn(t, s)
		})
	}
}

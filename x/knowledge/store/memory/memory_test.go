package memory_test

import (
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/store/memory"
	"github.com/itohio/EasyRobot/x/knowledge/store/storetest"
)

func open(t *testing.T) knowledge.Store {
	s := memory.New()
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func TestStore(t *testing.T) { storetest.Run(t, open) }

func BenchmarkSearchLex(b *testing.B) {
	s := memory.New()
	b.Cleanup(func() { _ = s.Close() })
	storetest.BenchSearchLex(b, s, 1000, 5)
}

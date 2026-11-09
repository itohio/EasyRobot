package ingest

import (
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/chunk/golang"
	"github.com/itohio/EasyRobot/x/knowledge/chunk/md"
	"github.com/itohio/EasyRobot/x/knowledge/store/memory"
)

func newStore(t *testing.T) knowledge.Store {
	t.Helper()
	s := memory.New()
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func testChunkers() []knowledge.Chunker { return []knowledge.Chunker{md.New(), golang.New()} }

func testOptions() []Option { return []Option{WithChunkers(testChunkers()...), WithWorkers(4)} }

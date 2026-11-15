package bench

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/ingest"
	"github.com/itohio/EasyRobot/x/knowledge/retrieve"
	"github.com/itohio/EasyRobot/x/knowledge/store/memory"
	"github.com/itohio/EasyRobot/x/knowledge/store/sqlite"
)

// newEngine ingests root into st and returns a retriever over it.
func newEngine(t testing.TB, st knowledge.Store, src ingest.Source) (knowledge.Retriever, knowledge.Report) {
	t.Helper()
	rep, err := ingest.New(st, src).Sync(context.Background())
	if err != nil {
		t.Fatalf("ingest %s: %v", src.Root, err)
	}
	for _, e := range rep.Errors {
		t.Logf("ingest error: %v", e)
	}
	return retrieve.New(st), rep
}

// engineFloor is the minimum the engine must score on one question type of
// the committed fixture (both stores).
type engineFloor struct {
	recall1, recallK, mrr float64
}

// Engine floors measured on the committed fixture (memory and sqlite
// identical). R@1 and R@k sit at the measured fraction (one lost
// question is a regression); MRR sits 0.03 below the measured value, less
// than one rank step for any of these questions. They make the tuning
// (recency prior, supersede demotion, score-relative fusion, graph
// expansion) fail this test when removed; each was checked by mutation.
var engineFloors = map[Type]engineFloor{
	TypeIdentifier: {recall1: 7.0 / 7, recallK: 7.0 / 7, mrr: 0.97},  // measured 1.000 / 1.000 / 1.000
	TypeParaphrase: {recall1: 4.0 / 6, recallK: 5.0 / 6, mrr: 0.72},  // measured 0.667 / 0.833 / 0.750
	TypeCrossLink:  {recall1: 0.0 / 5, recallK: 5.0 / 5, mrr: 0.278}, // measured 0.000 / 1.000 / 0.308
	TypeRecency:    {recall1: 3.0 / 4, recallK: 4.0 / 4, mrr: 0.845}, // measured 0.750 / 1.000 / 0.875
	TypeCode:       {recall1: 6.0 / 6, recallK: 6.0 / 6, mrr: 0.97},  // measured 1.000 / 1.000 / 1.000
}

// TestEngineOnFixture runs ingest + hybrid retrieval end to end on the
// synthetic fixture, against both stores, and compares with the grep
// baseline. The engine must not lose to grep on recall@k or MRR, and must
// answer the vocabulary-mismatch question (f14) that grep misses.
func TestEngineOnFixture(t *testing.T) {
	qs, err := LoadQuestions("testdata/fixture-questions.json")
	if err != nil {
		t.Fatal(err)
	}
	root, err := filepath.Abs("testdata/corpus")
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	grep := NewGrep(root)
	gs := Run(ctx, grep, qs, DefaultK)

	db, err := sqlite.Open(filepath.Join(t.TempDir(), "index.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	stores := map[string]knowledge.Store{"memory": memory.New(), "sqlite": db}
	for name, st := range stores {
		t.Run(name, func(t *testing.T) {
			eng, rep := newEngine(t, st, ingest.Source{Root: root})
			if len(rep.Errors) > 0 || rep.Added == 0 {
				t.Fatalf("ingest report: %+v", rep)
			}
			es := Run(ctx, eng, qs, DefaultK)
			t.Logf("ingest: added %d, skipped %d, %v\n%s", rep.Added, rep.Skipped, rep.Duration,
				Compare(ctx, qs, DefaultK, map[string]knowledge.Retriever{"grep": grep, "engine": eng}))
			if es.Overall.RecallK < gs.Overall.RecallK {
				t.Errorf("engine recall@%d %.3f < grep %.3f", DefaultK, es.Overall.RecallK, gs.Overall.RecallK)
			}
			if es.Overall.MRR < gs.Overall.MRR {
				t.Errorf("engine MRR %.3f < grep %.3f", es.Overall.MRR, gs.Overall.MRR)
			}
			for typ, f := range engineFloors {
				m, ok := es.ByType[typ]
				if !ok {
					t.Errorf("%s: no questions", typ)
					continue
				}
				if m.Recall1 < f.recall1-1e-9 {
					t.Errorf("%s R@1 %.3f < floor %.3f", typ, m.Recall1, f.recall1)
				}
				if m.RecallK < f.recallK-1e-9 {
					t.Errorf("%s R@%d %.3f < floor %.3f", typ, DefaultK, m.RecallK, f.recallK)
				}
				if m.MRR < f.mrr {
					t.Errorf("%s MRR %.3f < floor %.3f", typ, m.MRR, f.mrr)
				}
			}
			for _, o := range es.Outcomes {
				if o.ID == "f14" && (o.Rank == 0 || o.Rank > DefaultK) {
					t.Errorf("vocabulary-mismatch question f14: rank %d, want within top %d", o.Rank, DefaultK)
				}
			}
		})
	}
}

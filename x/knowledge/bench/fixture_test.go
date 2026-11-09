package bench

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const fixtureCorpus = "testdata/corpus"

func TestFixtureAnswersExistAndAreDocumented(t *testing.T) {
	qs, err := LoadQuestions("testdata/fixture-questions.json")
	if err != nil {
		t.Fatal(err)
	}
	readme, err := os.ReadFile("testdata/README.md")
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range qs {
		if !strings.Contains(string(readme), "| "+q.ID+" |") {
			t.Errorf("README does not document %s", q.ID)
		}
		for _, p := range q.ExpectAny {
			if _, err := os.Stat(filepath.Join(fixtureCorpus, filepath.FromSlash(p))); err != nil {
				t.Errorf("%s: expected file missing: %v", q.ID, err)
			}
			if !strings.Contains(string(readme), p) {
				t.Errorf("README does not name %s's answer %s", q.ID, p)
			}
		}
	}
}

// Floors measured on the committed fixture (see README); they guard the
// baseline against silent regressions, not set targets.
const (
	floorGrepIdentifierRecallK = 7.0 / 7   // measured 2026-11-09: 1.000
	floorGrepIdentifierMRR     = 6.0 / 7   // measured: 0.857 (f02, f03 at rank 2)
	floorGrepOverallRecallK    = 27.0 / 28 // measured: 0.964 (f14 missed by design)
)

func TestGrepOnFixture(t *testing.T) {
	qs, err := LoadQuestions("testdata/fixture-questions.json")
	if err != nil {
		t.Fatal(err)
	}
	g := NewGrep(fixtureCorpus)
	if err := g.Load(context.Background()); err != nil {
		t.Fatal(err)
	}
	s := Run(context.Background(), g, qs, DefaultK)
	t.Logf("grep baseline on the fixture:\n%s", s.Table())
	for _, o := range s.Outcomes {
		if o.Err != "" {
			t.Errorf("%s: %s", o.ID, o.Err)
		}
	}
	if got := s.ByType[TypeIdentifier].RecallK; got < floorGrepIdentifierRecallK {
		t.Errorf("identifier recall@%d = %.3f, floor %.3f", s.K, got, floorGrepIdentifierRecallK)
	}
	if got := s.ByType[TypeIdentifier].MRR; got < floorGrepIdentifierMRR-1e-9 {
		t.Errorf("identifier MRR = %.3f, floor %.3f", got, floorGrepIdentifierMRR)
	}
	if got := s.Overall.RecallK; got < floorGrepOverallRecallK-1e-9 {
		t.Errorf("overall recall@%d = %.3f, floor %.3f", s.K, got, floorGrepOverallRecallK)
	}
}

package bench

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// fakeClock advances only when the fake retriever says so, making
// latencies exact.
type fakeClock struct{ now time.Time }

func (c *fakeClock) Now() time.Time { return c.now }

type fakeAnswer struct {
	docs    []knowledge.DocID
	latency time.Duration
	err     error
}

type fakeRetriever struct {
	clock   *fakeClock
	answers map[string]fakeAnswer
	gotK    []int
}

func (f *fakeRetriever) Find(_ context.Context, q knowledge.Query) (knowledge.Result, error) {
	f.gotK = append(f.gotK, q.K)
	a := f.answers[q.Text]
	if f.clock != nil {
		f.clock.now = f.clock.now.Add(a.latency)
	}
	if a.err != nil {
		return knowledge.Result{}, a.err
	}
	var res knowledge.Result
	for i, d := range a.docs {
		res.Hits = append(res.Hits, knowledge.Hit{Doc: d, Score: float64(100 - i)})
	}
	return res, nil
}

func ms(n int) time.Duration { return time.Duration(n) * time.Millisecond }

func fixtureRun() ([]Question, *fakeRetriever, *fakeClock) {
	qs := []Question{
		{ID: "a", Q: "qa", Type: TypeIdentifier, ExpectAny: []string{"a.md"}},
		{ID: "b", Q: "qb", Type: TypeIdentifier, ExpectAny: []string{"b.md"}},
		{ID: "c", Q: "qc", Type: TypeParaphrase, ExpectAny: []string{"c.md"}},
		{ID: "d", Q: "qd", Type: TypeParaphrase, ExpectAny: []string{"d.md"}},
		{ID: "e", Q: "qe", Type: TypeCode, ExpectAny: []string{"e.go"}},
		{ID: "f", Q: "qf", Type: TypeCode, ExpectAny: []string{"f1.md", "f2.md"}},
	}
	clock := &fakeClock{now: time.Unix(1000, 0)}
	r := &fakeRetriever{clock: clock, answers: map[string]fakeAnswer{
		"qa": {docs: []knowledge.DocID{"a.md", "x.md"}, latency: ms(10)},
		"qb": {docs: []knowledge.DocID{"x.md", "b.md", "y.md"}, latency: ms(20)},
		"qc": {docs: []knowledge.DocID{"x.md", "y.md", "c.md"}, latency: ms(30)},
		// expected doc at position 4: beyond k=3, must count as a miss.
		"qd": {docs: []knowledge.DocID{"x.md", "y.md", "z.md", "d.md", "w.md"}, latency: ms(40)},
		"qe": {err: errors.New("boom"), latency: ms(50)},
		// either expected doc counts; the first one found sets the rank.
		"qf": {docs: []knowledge.DocID{"x.md", "f2.md", "f1.md"}, latency: ms(60)},
	}}
	return qs, r, clock
}

func approx(t *testing.T, name string, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Errorf("%s = %.6f, want %.6f", name, got, want)
	}
}

func TestRunMetrics(t *testing.T) {
	qs, r, clock := fixtureRun()
	s := run(context.Background(), r, qs, 3, clock.Now)

	if s.K != 3 {
		t.Fatalf("K = %d, want 3", s.K)
	}
	for i, k := range r.gotK {
		if k != 3 {
			t.Fatalf("query %d asked with K=%d, want 3", i, k)
		}
	}
	wantRanks := map[string]int{"a": 1, "b": 2, "c": 3, "d": 0, "e": 0, "f": 2}
	if len(s.Outcomes) != len(qs) {
		t.Fatalf("%d outcomes, want %d", len(s.Outcomes), len(qs))
	}
	for i, o := range s.Outcomes {
		if o.ID != qs[i].ID || o.Type != qs[i].Type {
			t.Errorf("outcome %d = %s/%s, want %s/%s", i, o.ID, o.Type, qs[i].ID, qs[i].Type)
		}
		if o.Rank != wantRanks[o.ID] {
			t.Errorf("rank(%s) = %d, want %d", o.ID, o.Rank, wantRanks[o.ID])
		}
		if want := ms(10 * (i + 1)); o.Latency != want {
			t.Errorf("latency(%s) = %v, want %v", o.ID, o.Latency, want)
		}
	}
	if s.Outcomes[4].Err == "" {
		t.Error("error of question e not recorded")
	}

	all := s.Overall
	if all.N != 6 {
		t.Errorf("overall N = %d", all.N)
	}
	approx(t, "overall recall@1", all.Recall1, 1.0/6)
	approx(t, "overall recall@k", all.RecallK, 4.0/6)
	approx(t, "overall MRR", all.MRR, (1+0.5+1.0/3+0.5)/6)
	if all.P50 != ms(30) || all.P95 != ms(60) {
		t.Errorf("overall p50/p95 = %v/%v, want 30ms/60ms", all.P50, all.P95)
	}

	id := s.ByType[TypeIdentifier]
	approx(t, "identifier recall@1", id.Recall1, 0.5)
	approx(t, "identifier recall@k", id.RecallK, 1)
	approx(t, "identifier MRR", id.MRR, 0.75)
	if id.N != 2 || id.P50 != ms(10) || id.P95 != ms(20) {
		t.Errorf("identifier N/p50/p95 = %d/%v/%v", id.N, id.P50, id.P95)
	}
	para := s.ByType[TypeParaphrase]
	approx(t, "paraphrase recall@1", para.Recall1, 0)
	approx(t, "paraphrase recall@k", para.RecallK, 0.5)
	approx(t, "paraphrase MRR", para.MRR, 1.0/6)
	code := s.ByType[TypeCode]
	approx(t, "code recall@k", code.RecallK, 0.5)
	approx(t, "code MRR", code.MRR, 0.25)
	if _, ok := s.ByType[TypeRecency]; ok {
		t.Error("recency present without questions")
	}
}

func TestRunDefaultK(t *testing.T) {
	r := &fakeRetriever{answers: map[string]fakeAnswer{
		"q": {docs: []knowledge.DocID{"1", "2", "3", "4", "5", "6", "7", "t.md"}},
	}}
	qs := []Question{{ID: "q", Q: "q", Type: TypeCode, ExpectAny: []string{"t.md"}}}
	s := Run(context.Background(), r, qs, 0)
	if s.K != DefaultK || r.gotK[0] != DefaultK {
		t.Fatalf("K = %d (asked %d), want %d", s.K, r.gotK[0], DefaultK)
	}
	if s.Outcomes[0].Rank != 8 {
		t.Fatalf("rank = %d, want 8", s.Outcomes[0].Rank)
	}
	approx(t, "MRR", s.Overall.MRR, 1.0/8)
}

func TestTableHasScoresOnly(t *testing.T) {
	qs, r, clock := fixtureRun()
	s := run(context.Background(), r, qs, 3, clock.Now)
	tab := s.Table()
	for _, want := range []string{"| type ", "R@1", "R@3", "MRR", "p50", "p95", "| identifier |", "| paraphrase |", "| code |", "| all |", "0.389", "| a |", "| d |"} {
		if !strings.Contains(tab, want) {
			t.Errorf("table lacks %q:\n%s", want, tab)
		}
	}
	for _, q := range qs {
		if strings.Contains(tab, "| "+q.Q+" ") {
			t.Errorf("table leaks question text %q", q.Q)
		}
	}
	for _, leak := range []string{"a.md", "x.md", "boom"} {
		if strings.Contains(tab, leak) {
			t.Errorf("table leaks %q:\n%s", leak, tab)
		}
	}
	if strings.Contains(tab, "| recency |") {
		t.Errorf("table lists a type with no questions:\n%s", tab)
	}
}

func TestCompare(t *testing.T) {
	qs, good, _ := fixtureRun()
	bad := &fakeRetriever{answers: map[string]fakeAnswer{}}
	out := Compare(context.Background(), qs, 3, map[string]knowledge.Retriever{"zeta": bad, "alpha": good})
	header := ""
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "| type") {
			header = line
			break
		}
	}
	if header == "" {
		t.Fatalf("no header:\n%s", out)
	}
	ia, iz := strings.Index(header, "alpha R@1"), strings.Index(header, "zeta R@1")
	if ia < 0 || iz < 0 || ia > iz {
		t.Fatalf("column groups missing or unsorted: %s", header)
	}
	for _, want := range []string{"alpha R@3", "alpha MRR", "zeta MRR", "| all |", "| f |"} {
		if !strings.Contains(out, want) {
			t.Errorf("comparison lacks %q:\n%s", want, out)
		}
	}
	// The all row carries alpha's MRR then zeta's zero MRR.
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "| all |") {
			if !strings.Contains(line, "0.389") || !strings.Contains(line, "0.000") {
				t.Errorf("all row = %s", line)
			}
		}
	}
}

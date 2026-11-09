package bench

import (
	"context"
	"fmt"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// DefaultK is the result depth used when Run is given k <= 0.
const DefaultK = 8

// Outcome is the result of one question.
type Outcome struct {
	ID      string
	Type    Type
	Rank    int // 1-based rank of the first expected doc within k; 0 = miss
	Latency time.Duration
	Err     string // retriever error, if any (counts as a miss)
}

// Metrics aggregates outcomes.
type Metrics struct {
	N       int
	Recall1 float64 // share of questions answered at rank 1
	RecallK float64 // share answered within k
	MRR     float64 // mean reciprocal rank (misses count 0)
	P50     time.Duration
	P95     time.Duration
}

// Scores is the result of a benchmark run.
type Scores struct {
	K        int
	Outcomes []Outcome        // in question order
	ByType   map[Type]Metrics // only types that have questions
	Overall  Metrics
}

// Run asks r every question with depth k (DefaultK when k <= 0) and scores
// the rank of the first expected document. Questions run sequentially so
// latencies are not distorted by each other.
func Run(ctx context.Context, r knowledge.Retriever, qs []Question, k int) Scores {
	return run(ctx, r, qs, k, time.Now)
}

func run(ctx context.Context, r knowledge.Retriever, qs []Question, k int, now func() time.Time) Scores {
	if k <= 0 {
		k = DefaultK
	}
	s := Scores{K: k, Outcomes: make([]Outcome, 0, len(qs))}
	for _, q := range qs {
		o := Outcome{ID: q.ID, Type: q.Type}
		start := now()
		res, err := r.Find(ctx, knowledge.Query{Text: q.Q, K: k})
		o.Latency = now().Sub(start)
		if err != nil {
			o.Err = err.Error()
		} else {
			o.Rank = rankOf(res.Hits, q.ExpectAny, k)
		}
		s.Outcomes = append(s.Outcomes, o)
	}
	s.Overall = aggregate(s.Outcomes, k)
	s.ByType = make(map[Type]Metrics)
	for _, t := range Types() {
		var sub []Outcome
		for _, o := range s.Outcomes {
			if o.Type == t {
				sub = append(sub, o)
			}
		}
		if len(sub) > 0 {
			s.ByType[t] = aggregate(sub, k)
		}
	}
	return s
}

// rankOf returns the 1-based position of the first hit whose doc is
// expected, looking at the first k hits only; 0 when none matches.
func rankOf(hits []knowledge.Hit, expect []string, k int) int {
	for i, h := range hits {
		if i >= k {
			break
		}
		if slices.Contains(expect, string(h.Doc)) {
			return i + 1
		}
	}
	return 0
}

func aggregate(outs []Outcome, k int) Metrics {
	m := Metrics{N: len(outs)}
	if m.N == 0 {
		return m
	}
	lat := make([]time.Duration, 0, len(outs))
	for _, o := range outs {
		lat = append(lat, o.Latency)
		if o.Rank == 0 {
			continue
		}
		if o.Rank == 1 {
			m.Recall1++
		}
		if o.Rank <= k {
			m.RecallK++
			m.MRR += 1 / float64(o.Rank)
		}
	}
	n := float64(m.N)
	m.Recall1 /= n
	m.RecallK /= n
	m.MRR /= n
	slices.Sort(lat)
	m.P50 = percentile(lat, 50)
	m.P95 = percentile(lat, 95)
	return m
}

// percentile uses the nearest-rank method on sorted values.
func percentile(sorted []time.Duration, p int) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	idx := (p*len(sorted) + 99) / 100 // ceil(p/100 * n)
	return sorted[max(idx, 1)-1]
}

// Table renders the scores as Markdown: one row per question type plus
// "all", then each question's rank. Only ids and numbers are printed.
func (s Scores) Table() string {
	var b strings.Builder
	fmt.Fprintf(&b, "| type | n | R@1 | R@%d | MRR | p50 | p95 |\n", s.K)
	b.WriteString("|---|---|---|---|---|---|---|\n")
	row := func(name string, m Metrics) {
		fmt.Fprintf(&b, "| %s | %d | %.3f | %.3f | %.3f | %s | %s |\n",
			name, m.N, m.Recall1, m.RecallK, m.MRR, fmtDur(m.P50), fmtDur(m.P95))
	}
	for _, t := range Types() {
		if m, ok := s.ByType[t]; ok {
			row(string(t), m)
		}
	}
	row("all", s.Overall)

	b.WriteString("\n| id | type | rank | latency |\n|---|---|---|---|\n")
	for _, o := range s.Outcomes {
		fmt.Fprintf(&b, "| %s | %s | %s | %s |\n", o.ID, o.Type, fmtRank(o), fmtDur(o.Latency))
	}
	return b.String()
}

// Compare runs every named retriever over qs and renders one Markdown table
// with a column group (R@1, R@k, MRR, p95) per retriever, in name order,
// followed by per-question ranks. Only ids and numbers are printed.
func Compare(ctx context.Context, qs []Question, k int, named map[string]knowledge.Retriever) string {
	names := make([]string, 0, len(named))
	for n := range named {
		names = append(names, n)
	}
	sort.Strings(names)
	runs := make([]Scores, len(names))
	for i, n := range names {
		runs[i] = Run(ctx, named[n], qs, k)
	}
	if k <= 0 {
		k = DefaultK
	}

	var b strings.Builder
	b.WriteString("| type | n |")
	sep := "|---|---|"
	for _, n := range names {
		fmt.Fprintf(&b, " %s R@1 | %s R@%d | %s MRR | %s p95 |", n, n, k, n, n)
		sep += "---|---|---|---|"
	}
	b.WriteString("\n" + sep + "\n")
	row := func(name string, pick func(Scores) (Metrics, bool)) {
		n := 0
		cells := make([]string, 0, len(runs))
		for _, s := range runs {
			m, ok := pick(s)
			if !ok {
				return
			}
			n = m.N
			cells = append(cells, fmt.Sprintf(" %.3f | %.3f | %.3f | %s |", m.Recall1, m.RecallK, m.MRR, fmtDur(m.P95)))
		}
		fmt.Fprintf(&b, "| %s | %d |%s\n", name, n, strings.Join(cells, ""))
	}
	for _, t := range Types() {
		row(string(t), func(s Scores) (Metrics, bool) { m, ok := s.ByType[t]; return m, ok })
	}
	row("all", func(s Scores) (Metrics, bool) { return s.Overall, true })

	b.WriteString("\n| id | type |")
	sep = "|---|---|"
	for _, n := range names {
		fmt.Fprintf(&b, " %s rank |", n)
		sep += "---|"
	}
	b.WriteString("\n" + sep + "\n")
	for i, q := range qs {
		fmt.Fprintf(&b, "| %s | %s |", q.ID, q.Type)
		for _, s := range runs {
			fmt.Fprintf(&b, " %s |", fmtRank(s.Outcomes[i]))
		}
		b.WriteString("\n")
	}
	return b.String()
}

func fmtRank(o Outcome) string {
	switch {
	case o.Err != "":
		return "err"
	case o.Rank == 0:
		return "—"
	default:
		return fmt.Sprint(o.Rank)
	}
}

func fmtDur(d time.Duration) string {
	return fmt.Sprintf("%.2fms", float64(d)/float64(time.Millisecond))
}

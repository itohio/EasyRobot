package storetest

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
)

// BenchSearchLex fills s with a synthetic corpus of docs×chunksPerDoc
// chunks (Zipf-distributed vocabulary, deterministic) and benchmarks
// SearchLex with k=10 for a common, a selective and a filtered query.
func BenchSearchLex(b *testing.B, s knowledge.Store, docs, chunksPerDoc int) {
	ctx := b.Context()
	rng := rand.New(rand.NewPCG(1, 2))
	vocab := make([]string, 4000)
	for i := range vocab {
		vocab[i] = fmt.Sprintf("w%04d", i)
	}
	zipf := rand.NewZipf(rng, 1.1, 1, uint64(len(vocab)-1))
	text := func(n int) string {
		var sb strings.Builder
		for i := range n {
			if i > 0 {
				sb.WriteByte(' ')
			}
			sb.WriteString(vocab[zipf.Uint64()])
		}
		return sb.String()
	}
	for d := range docs {
		doc := mkDoc(fmt.Sprintf("bench/d%03d/file%04d.md", d%50, d), "md", "bench")
		cs := make([]knowledge.Chunk, chunksPerDoc)
		for c := range cs {
			cs[c] = mkChunk(doc.ID, c, []string{text(4)}, "", text(120))
		}
		if err := s.PutDoc(ctx, knowledge.DocUpdate{Doc: doc, Chunks: cs}); err != nil {
			b.Fatal(err)
		}
	}
	cases := []struct {
		name  string
		terms []string
		f     knowledge.Filter
	}{
		{"common", []string{vocab[3], vocab[40], vocab[700]}, knowledge.Filter{}},        // ~99% of chunks match
		{"selective", []string{vocab[100], vocab[700], vocab[2000]}, knowledge.Filter{}}, // ~14% match
		{"filtered", []string{vocab[100], vocab[700], vocab[2000]}, knowledge.Filter{In: []string{"bench/d007"}}},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			q := knowledge.LexQuery{Terms: tc.terms}
			b.ReportAllocs()
			for b.Loop() {
				hits, err := s.SearchLex(ctx, q, 10, tc.f)
				if err != nil {
					b.Fatal(err)
				}
				if len(hits) == 0 {
					b.Fatal("no hits")
				}
			}
		})
	}
}

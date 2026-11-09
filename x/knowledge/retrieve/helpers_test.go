package retrieve

import (
	"context"
	"crypto/sha256"
	"fmt"
	"path"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/token"
)

// now is a reference date for test documents.
var now = time.Date(2026, 11, 9, 12, 0, 0, 0, time.UTC)

// oldTime is the default MTime of test documents.
var oldTime = time.Date(2015, 1, 1, 0, 0, 0, 0, time.UTC)

// sec is one chunk of a test document.
type sec struct {
	heading string
	symbol  string
	text    string
}

// tdoc builds a DocUpdate for a test corpus.
type tdoc struct {
	u knowledge.DocUpdate
}

func doc(id string, secs ...sec) *tdoc {
	kind := "md"
	if strings.HasSuffix(id, ".go") {
		kind = "go"
	}
	scope, _, _ := strings.Cut(id, "/")
	title := strings.TrimSuffix(path.Base(id), path.Ext(id))
	d := knowledge.Doc{
		ID:     knowledge.DocID(id),
		Scope:  scope,
		Path:   "/ws/" + id,
		Kind:   kind,
		Title:  title,
		MTime:  oldTime,
		Weight: 1,
		Props:  map[string]string{},
	}
	t := &tdoc{u: knowledge.DocUpdate{Doc: d}}
	for i, s := range secs {
		heading := []string{title}
		if s.heading != "" {
			heading = append(heading, s.heading)
		}
		t.u.Chunks = append(t.u.Chunks, knowledge.Chunk{
			ID:      knowledge.NewChunkID(d.ID, i, s.text),
			Doc:     d.ID,
			Ord:     i,
			Heading: heading,
			Symbol:  s.symbol,
			Lines:   [2]int{10*i + 1, 10*i + 9},
			Text:    s.text,
			Hash:    sha256.Sum256([]byte(s.text)),
			Tokens:  token.Count(s.text),
		})
	}
	return t
}

func (t *tdoc) edge(to knowledge.NodeID, kind knowledge.EdgeKind) *tdoc {
	t.u.Edges = append(t.u.Edges, knowledge.Edge{
		From: knowledge.DocNode(t.u.Doc.ID), To: to, Kind: kind, Weight: 1, Source: t.u.Doc.ID,
	})
	return t
}

func (t *tdoc) links(to ...string) *tdoc {
	for _, id := range to {
		t.edge(knowledge.DocNode(knowledge.DocID(id)), knowledge.LinksTo)
	}
	return t
}

func (t *tdoc) definesID(id string) *tdoc { return t.edge(knowledge.IDNode(id), knowledge.Defines) }

func (t *tdoc) definesSym(sym string) *tdoc { return t.edge(knowledge.SymNode(sym), knowledge.Defines) }

func (t *tdoc) mentionsID(id string) *tdoc { return t.edge(knowledge.IDNode(id), knowledge.Mentions) }

func (t *tdoc) prop(k, v string) *tdoc { t.u.Doc.Props[k] = v; return t }

func (t *tdoc) mtime(m time.Time) *tdoc { t.u.Doc.MTime = m; return t }

func (t *tdoc) weight(w float32) *tdoc { t.u.Doc.Weight = w; return t }

func load(t testing.TB, s knowledge.Store, docs ...*tdoc) {
	t.Helper()
	for _, d := range docs {
		if err := s.PutDoc(context.Background(), d.u); err != nil {
			t.Fatalf("PutDoc %s: %v", d.u.Doc.ID, err)
		}
	}
}

func newRetriever(s knowledge.Store, opts ...Option) *Retriever {
	return New(s, opts...)
}

func find(t testing.TB, r *Retriever, q knowledge.Query) knowledge.Result {
	t.Helper()
	res, err := r.Find(context.Background(), q)
	if err != nil {
		t.Fatalf("Find(%q): %v", q.Text, err)
	}
	return res
}

func hitIDs(res knowledge.Result) []string {
	out := make([]string, len(res.Hits))
	for i, h := range res.Hits {
		out[i] = string(h.Doc)
	}
	return out
}

func rankOf(res knowledge.Result, id string) int {
	return slices.Index(hitIDs(res), id)
}

func hitOf(t testing.TB, res knowledge.Result, id string) knowledge.Hit {
	t.Helper()
	i := rankOf(res, id)
	if i < 0 {
		t.Fatalf("%s not in hits %v", id, hitIDs(res))
	}
	return res.Hits[i]
}

func wantFirst(t testing.TB, res knowledge.Result, id string) {
	t.Helper()
	if len(res.Hits) == 0 || string(res.Hits[0].Doc) != id {
		t.Fatalf("first hit: got %v, want %s first", hitIDs(res), id)
	}
}

func wantBefore(t testing.TB, res knowledge.Result, a, b string) {
	t.Helper()
	ra, rb := rankOf(res, a), rankOf(res, b)
	if ra < 0 || (rb >= 0 && ra > rb) {
		t.Fatalf("want %s before %s, got %v", a, b, hitIDs(res))
	}
}

func hasWhy(why []string, want string) bool {
	for _, w := range why {
		if strings.Contains(w, want) {
			return true
		}
	}
	return false
}

func wantWhy(t testing.TB, h knowledge.Hit, want string) {
	t.Helper()
	if !hasWhy(h.Why, want) {
		t.Fatalf("%s: Why %q lacks %q", h.Doc, h.Why, want)
	}
}

// filler returns n words of neutral text that match no test query.
func filler(n int) string {
	words := []string{"lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit"}
	var b strings.Builder
	for i := range n {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(words[i%len(words)])
	}
	return b.String()
}

func repeat(word string, n int) string {
	return strings.TrimSpace(strings.Repeat(word+" ", n))
}

func folderDocs(prefix string, n int, text string) []*tdoc {
	out := make([]*tdoc, n)
	for i := range n {
		out[i] = doc(fmt.Sprintf("%s/d%02d.md", prefix, i), sec{text: text})
	}
	return out
}

package md

import (
	"crypto/sha256"
	"fmt"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/token"
)

// lineOf returns the 1-based line of the first line containing sub.
func lineOf(t *testing.T, src, sub string) int {
	t.Helper()
	for i, l := range strings.Split(src, "\n") {
		if strings.Contains(l, sub) {
			return i + 1
		}
	}
	t.Fatalf("substring %q not in source", sub)
	return 0
}

// span returns the source lines l0..l1 (1-based inclusive) joined by "\n".
func span(src string, l0, l1 int) string {
	lines := strings.Split(src, "\n")
	return strings.Join(lines[l0-1:l1], "\n")
}

func chunkIt(t *testing.T, c *Chunker, id, src string) knowledge.ChunkResult {
	t.Helper()
	res, err := c.Chunk(knowledge.Doc{ID: knowledge.DocID(id), Kind: "md"}, []byte(src))
	if err != nil {
		t.Fatalf("Chunk: %v", err)
	}
	checkInvariants(t, knowledge.DocID(id), src, res.Chunks)
	return res
}

// checkInvariants verifies ids, hashes, ords, token counts and that every
// chunk's Text is exactly its source span.
func checkInvariants(t *testing.T, doc knowledge.DocID, src string, chunks []knowledge.Chunk) {
	t.Helper()
	n := len(strings.Split(src, "\n"))
	for i, c := range chunks {
		if c.Ord != i {
			t.Errorf("chunk %d: Ord = %d", i, c.Ord)
		}
		if c.Doc != doc {
			t.Errorf("chunk %d: Doc = %q", i, c.Doc)
		}
		if c.Lines[0] < 1 || c.Lines[1] < c.Lines[0] || c.Lines[1] > n {
			t.Fatalf("chunk %d: bad span %v (file has %d lines)", i, c.Lines, n)
		}
		if want := span(src, c.Lines[0], c.Lines[1]); c.Text != want {
			t.Errorf("chunk %d: Text does not match span %v:\n got %q\nwant %q", i, c.Lines, c.Text, want)
		}
		if c.ID != knowledge.NewChunkID(doc, i, c.Text) {
			t.Errorf("chunk %d: ID mismatch", i)
		}
		if c.Hash != sha256.Sum256([]byte(c.Text)) {
			t.Errorf("chunk %d: Hash mismatch", i)
		}
		if c.Tokens != token.Count(c.Text) {
			t.Errorf("chunk %d: Tokens = %d, want %d", i, c.Tokens, token.Count(c.Text))
		}
		if c.Symbol != "" {
			t.Errorf("chunk %d: Symbol = %q, want empty", i, c.Symbol)
		}
	}
}

func headings(chunks []knowledge.Chunk) [][]string {
	out := make([][]string, len(chunks))
	for i, c := range chunks {
		out[i] = c.Heading
	}
	return out
}

func spans(chunks []knowledge.Chunk) [][2]int {
	out := make([][2]int, len(chunks))
	for i, c := range chunks {
		out[i] = c.Lines
	}
	return out
}

func edgesOf(res knowledge.ChunkResult, kind knowledge.EdgeKind) []knowledge.NodeID {
	var out []knowledge.NodeID
	for _, e := range res.Edges {
		if e.Kind == kind {
			out = append(out, e.To)
		}
	}
	return out
}

const body = "with enough plain words in it to stay a chunk of its own here"

func TestAccepts(t *testing.T) {
	c := New()
	for path, want := range map[string]bool{
		"a/b.md": true, "x.markdown": true, "X.MD": true,
		"a.go": false, "md": false, "a.md.txt": false, "README": false,
	} {
		if got := c.Accepts(path); got != want {
			t.Errorf("Accepts(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestFrontmatterAndTitle(t *testing.T) {
	tests := []struct {
		name      string
		id        string
		src       string
		title     string
		props     map[string]string
		defines   []knowledge.NodeID
		tags      []knowledge.NodeID
		firstLine int
	}{
		{
			name:  "frontmatter title wins",
			id:    "docs/tasks/ABU-061.md",
			src:   "---\nid: ABU-061\ntitle: \"Key Leases\"\ntype: task\ntags: [security, 'nats', \"#Lease\"]\nowner: 'mb'\n---\n# Heading One\n\nBody " + body + ".\n",
			title: "Key Leases",
			props: map[string]string{
				"id": "ABU-061", "title": "Key Leases", "type": "task",
				"tags": "security,nats,#Lease", "owner": "mb",
			},
			defines:   []knowledge.NodeID{"id:ABU-061"},
			tags:      []knowledge.NodeID{"tag:security", "tag:nats", "tag:lease"},
			firstLine: 8,
		},
		{
			name:      "first H1 when no title",
			id:        "notes/x.md",
			src:       "---\nstatus: open\nid: not an id\n---\n\nIntro " + body + ".\n\n# Real Title\n\nBody " + body + ".\n",
			title:     "Real Title",
			props:     map[string]string{"status": "open", "id": "not an id"},
			firstLine: 6,
		},
		{
			name:      "base name fallback",
			id:        "notes/my-note.md",
			src:       "## Sub\n\nBody " + body + ".\n",
			title:     "my-note",
			props:     map[string]string{},
			firstLine: 1,
		},
		{
			name:      "block list tags and lowercase id",
			id:        "a.md",
			src:       "---\nid: ke-1\ntags:\n  - alpha\n  - \"beta\"\n---\n# T\n\nBody " + body + ".\n",
			title:     "T",
			props:     map[string]string{"id": "ke-1", "tags": "alpha,beta"},
			defines:   []knowledge.NodeID{"id:KE-1"},
			tags:      []knowledge.NodeID{"tag:alpha", "tag:beta"},
			firstLine: 7,
		},
		{
			name:      "unterminated frontmatter is content",
			id:        "u.md",
			src:       "---\ntitle: nope\n\nText " + body + ".\n",
			title:     "u",
			props:     map[string]string{},
			firstLine: 1,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := chunkIt(t, New(), tt.id, tt.src)
			if res.Title != tt.title {
				t.Errorf("Title = %q, want %q", res.Title, tt.title)
			}
			if res.Props == nil {
				t.Fatal("Props is nil")
			}
			if !reflect.DeepEqual(res.Props, tt.props) {
				t.Errorf("Props = %v, want %v", res.Props, tt.props)
			}
			if got := edgesOf(res, knowledge.Defines); !slices.Equal(got, tt.defines) {
				t.Errorf("Defines = %v, want %v", got, tt.defines)
			}
			if got := edgesOf(res, knowledge.Tagged); !slices.Equal(got, tt.tags) {
				t.Errorf("Tagged = %v, want %v", got, tt.tags)
			}
			if len(res.Chunks) == 0 || res.Chunks[0].Lines[0] != tt.firstLine {
				t.Errorf("first chunk spans %v, want start %d", spans(res.Chunks), tt.firstLine)
			}
			for _, e := range res.Edges {
				if e.From != knowledge.DocNode(knowledge.DocID(tt.id)) || e.Source != knowledge.DocID(tt.id) || e.Weight != 1 {
					t.Errorf("edge %+v: wrong From/Source/Weight", e)
				}
			}
		})
	}
}

const fixture = `---
id: ABU-061
title: Key Leases
tags: [security]
---

Intro mentions ABU-061 and KE-1 and #intro-tag ` + body + `.

# Leases

Leases are granted by the broker to peers for a bounded time; see [[ABU-060|parent]] and [design](../docs/design%20notes.md#leases).

## Expiry #wip

When a lease expires the peer must re-request it. Not tags: issue #123, http://x.io/a#frag, a#b, (#x).

` + "```go" + `
# not a heading
[[NotALink]] and [x](not/a/link.md) #notatag O-999
` + "```" + `

### Renewal

Renewal happens automatically ![[diagram.png]] and ![alt](img/a.png) and [[Note#Section]] and [[ Spaced ]].
External [ext](https://example.com/x.md), [mail](mailto:a@b.c), [self](#local), ` + "`[[InCode]]` and `[c](code.md)` and `#codetag`" + ` mention o-5 and ABU-060 again.

~~~
## fenced tilde heading
~~~

# Second

Back at top level with [[Leases]] ` + body + ` #wip again.
`

func TestSectionsAndBreadcrumbs(t *testing.T) {
	src := fixture
	res := chunkIt(t, New(), "docs/leases.md", src)
	wantH := [][]string{
		{"Key Leases"},
		{"Leases"},
		{"Leases", "Expiry #wip"},
		{"Leases", "Expiry #wip", "Renewal"},
		{"Second"},
	}
	if got := headings(res.Chunks); !reflect.DeepEqual(got, wantH) {
		t.Fatalf("headings = %q, want %q", got, wantH)
	}
	wantS := [][2]int{
		{lineOf(t, src, "Intro mentions"), lineOf(t, src, "Intro mentions")},
		{lineOf(t, src, "# Leases"), lineOf(t, src, "Leases are granted")},
		{lineOf(t, src, "## Expiry"), lineOf(t, src, "```go") + 3},
		{lineOf(t, src, "### Renewal"), lineOf(t, src, "~~~") + 2},
		{lineOf(t, src, "# Second"), lineOf(t, src, "Back at top level")},
	}
	if got := spans(res.Chunks); !reflect.DeepEqual(got, wantS) {
		t.Fatalf("spans = %v, want %v", got, wantS)
	}
	if !strings.HasPrefix(res.Chunks[1].Text, "# Leases\n") {
		t.Errorf("section text must start with its heading line: %q", res.Chunks[1].Text)
	}
	if !strings.Contains(res.Chunks[2].Text, "# not a heading") {
		t.Errorf("fenced content must stay in the enclosing section")
	}
}

func TestLinks(t *testing.T) {
	src := fixture
	res := chunkIt(t, New(), "docs/leases.md", src)
	l1 := lineOf(t, src, "Leases are granted")
	l2 := lineOf(t, src, "Renewal happens")
	l3 := lineOf(t, src, "Back at top level")
	want := []knowledge.Link{
		{Target: "ABU-060", Kind: knowledge.WikiLink, Line: l1},
		{Target: "../docs/design notes.md", Kind: knowledge.MarkdownLink, Line: l1},
		{Target: "diagram.png", Kind: knowledge.WikiLink, Line: l2},
		{Target: "img/a.png", Kind: knowledge.MarkdownLink, Line: l2},
		{Target: "Note", Kind: knowledge.WikiLink, Line: l2},
		{Target: "Spaced", Kind: knowledge.WikiLink, Line: l2},
		{Target: "Leases", Kind: knowledge.WikiLink, Line: l3},
	}
	if !reflect.DeepEqual(res.Links, want) {
		t.Errorf("Links =\n%+v\nwant\n%+v", res.Links, want)
	}
}

func TestLinkForms(t *testing.T) {
	wiki := func(target string) []knowledge.Link {
		return []knowledge.Link{{Target: target, Kind: knowledge.WikiLink, Line: 1}}
	}
	mdl := func(target string) []knowledge.Link {
		return []knowledge.Link{{Target: target, Kind: knowledge.MarkdownLink, Line: 1}}
	}
	tests := []struct {
		line string
		want []knowledge.Link
	}{
		{"[[a]]", wiki("a")},
		{"[[a|label]]", wiki("a")},
		{"[[a#anchor|label]]", wiki("a")},
		{"![[img.png]]", wiki("img.png")},
		{"[[#local]]", nil},
		{"[l](b.md)", mdl("b.md")},
		{"[l](./dir/My%20File.md#sec)", mdl("./dir/My File.md")},
		{`[l](b.md "title")`, mdl("b.md")},
		{"[l](<my file.md>)", mdl("my file.md")},
		{"![alt](p.png)", mdl("p.png")},
		{"[l](http://x/y.md) [l](https://x) [l](mailto:a@b) [l](#a)", nil},
		{"`[[x]]` and `` [y](y.md) ``", nil},
	}
	for _, tt := range tests {
		t.Run(tt.line, func(t *testing.T) {
			res := chunkIt(t, New(), "d.md", tt.line+"\n")
			if !reflect.DeepEqual(res.Links, tt.want) {
				t.Errorf("Links = %+v, want %+v", res.Links, tt.want)
			}
		})
	}
}

func TestTagsAndMentions(t *testing.T) {
	res := chunkIt(t, New(), "docs/leases.md", fixture)
	wantTags := []knowledge.NodeID{"tag:security", "tag:intro-tag", "tag:wip"}
	if got := edgesOf(res, knowledge.Tagged); !slices.Equal(got, wantTags) {
		t.Errorf("Tagged = %v, want %v", got, wantTags)
	}
	wantIDs := []knowledge.NodeID{"id:KE-1", "id:ABU-060", "id:O-5"}
	if got := edgesOf(res, knowledge.Mentions); !slices.Equal(got, wantIDs) {
		t.Errorf("Mentions = %v, want %v", got, wantIDs)
	}
	if got := edgesOf(res, knowledge.Defines); !slices.Equal(got, []knowledge.NodeID{"id:ABU-061"}) {
		t.Errorf("Defines = %v", got)
	}
}

func TestInlineTagForms(t *testing.T) {
	tests := []struct {
		text string
		want []knowledge.NodeID
	}{
		{"plain #tag here", []knowledge.NodeID{"tag:tag"}},
		{"#start of line", []knowledge.NodeID{"tag:start"}},
		{"nested #area/sub-topic.", []knowledge.NodeID{"tag:area/sub-topic"}},
		{"mixed #Case and #case", []knowledge.NodeID{"tag:case"}},
		{"number #123 and #2026", nil},
		{"id #v2 ok", []knowledge.NodeID{"tag:v2"}},
		{"url http://a.b/c#frag and a#b", nil},
		{"code `#inline` only", nil},
		{"anchor [x](#sec) and [[n#h]]", nil},
		{"# Heading only", nil},
		{"## Heading #h2tag", []knowledge.NodeID{"tag:h2tag"}},
	}
	for _, tt := range tests {
		t.Run(tt.text, func(t *testing.T) {
			res := chunkIt(t, New(), "d.md", tt.text+"\n")
			if got := edgesOf(res, knowledge.Tagged); !slices.Equal(got, tt.want) {
				t.Errorf("Tagged = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMergeTinySections(t *testing.T) {
	long := "C body " + body + "."
	tests := []struct {
		name  string
		src   string
		heads [][]string
		spans [][2]int
	}{
		{
			name:  "heading stub into child, tiny into sibling",
			src:   "# A\n## B\nshort.\n## C\n" + long + "\n",
			heads: [][]string{{"A", "C"}},
			spans: [][2]int{{1, 5}},
		},
		{
			name:  "tiny last section stays",
			src:   "# A\n" + long + "\n## B\ntiny\n",
			heads: [][]string{{"A"}, {"A", "B"}},
			spans: [][2]int{{1, 2}, {3, 4}},
		},
		{
			name:  "tiny before shallower heading stays",
			src:   "# A\n" + long + "\n## B\ntiny\n# C\n" + long + "\n",
			heads: [][]string{{"A"}, {"A", "B"}, {"C"}},
			spans: [][2]int{{1, 2}, {3, 4}, {5, 6}},
		},
		{
			name:  "tiny preamble merges into first section",
			src:   "Hi.\n\n# A\n" + long + "\n",
			heads: [][]string{{"A"}},
			spans: [][2]int{{1, 4}},
		},
		{
			name:  "blank preamble skipped, trailing blanks trimmed",
			src:   "---\ntitle: T\n---\n\n\n# A\n" + long + "\n\n\n# B\n" + long + "\n\n",
			heads: [][]string{{"A"}, {"B"}},
			spans: [][2]int{{6, 7}, {10, 11}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := chunkIt(t, New(), "m.md", tt.src)
			if got := headings(res.Chunks); !reflect.DeepEqual(got, tt.heads) {
				t.Errorf("headings = %q, want %q", got, tt.heads)
			}
			if got := spans(res.Chunks); !reflect.DeepEqual(got, tt.spans) {
				t.Errorf("spans = %v, want %v", got, tt.spans)
			}
		})
	}
}

func TestSplitLongSections(t *testing.T) {
	para := func(i int) string { return fmt.Sprintf("p%d one two three four five six seven", i) } // 8 tokens
	var b strings.Builder
	b.WriteString("# Top\n\n## Long\n")
	for i := range 5 {
		b.WriteString(para(i) + "\n\n")
	}
	b.WriteString("```\nfence a one two three four five six\n\nfence b one two three four five six\n```\n")
	src := b.String()
	res := chunkIt(t, New(WithMaxTokens(20)), "s.md", src)
	// Top's empty section merges into its child. "# Top" (2) + "## Long" (2)
	// + p0 (8) + p1 (8) = 20; + p2 would exceed 20. The fence (18 tokens) is
	// atomic: its inner blank line is not a split point.
	wantS := [][2]int{
		{1, lineOf(t, src, "p1 one")},
		{lineOf(t, src, "p2 one"), lineOf(t, src, "p3 one")},
		{lineOf(t, src, "p4 one"), lineOf(t, src, "p4 one")},
		{lineOf(t, src, "```"), lineOf(t, src, "fence b") + 1},
	}
	if got := spans(res.Chunks); !reflect.DeepEqual(got, wantS) {
		t.Fatalf("spans = %v, want %v", got, wantS)
	}
	for i, c := range res.Chunks {
		if !slices.Equal(c.Heading, []string{"Top", "Long"}) {
			t.Errorf("chunk %d heading = %q", i, c.Heading)
		}
	}
}

func TestSplitDefaultLimit(t *testing.T) {
	var b strings.Builder
	b.WriteString("# Big\n\n")
	for i := range 100 {
		fmt.Fprintf(&b, "paragraph %d %s %s.\n\n", i, body, body) // 28 tokens
	}
	src := b.String()
	res := chunkIt(t, New(), "big.md", src)
	if len(res.Chunks) < 3 {
		t.Fatalf("got %d chunks, want >= 3", len(res.Chunks))
	}
	prev := 0
	for i, c := range res.Chunks {
		if c.Tokens > 800 {
			t.Errorf("chunk %d has %d tokens > 800", i, c.Tokens)
		}
		if c.Lines[0] <= prev {
			t.Errorf("chunk %d overlaps previous: %v", i, c.Lines)
		}
		prev = c.Lines[1]
	}
	if last := res.Chunks[len(res.Chunks)-1]; last.Lines[1] != lineOf(t, src, "paragraph 99 ") {
		t.Errorf("last chunk ends at %d", last.Lines[1])
	}
}

func TestEmptyAndCRLF(t *testing.T) {
	res := chunkIt(t, New(), "e.md", "")
	if len(res.Chunks) != 0 || res.Title != "e" {
		t.Errorf("empty: chunks=%d title=%q", len(res.Chunks), res.Title)
	}
	res, err := New().Chunk(knowledge.Doc{ID: "c.md"}, []byte("---\r\ntitle: Win\r\n---\r\n# H\r\n"+body+"\r\n"))
	if err != nil {
		t.Fatal(err)
	}
	if res.Title != "Win" || len(res.Chunks) != 1 || res.Chunks[0].Lines != [2]int{4, 5} || !slices.Equal(res.Chunks[0].Heading, []string{"H"}) {
		t.Errorf("crlf: title=%q chunks=%+v", res.Title, res.Chunks)
	}
}

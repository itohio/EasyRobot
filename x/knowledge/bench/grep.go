package bench

import (
	"cmp"
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"unicode"
	"unicode/utf8"

	"github.com/itohio/EasyRobot/x/knowledge"
)

const (
	grepMaxFileSize = 1 << 20 // larger files are skipped
	grepMaxK        = 30
	grepSnippetLen  = 160 // runes
	grepMinWordLen  = 3   // runes
)

// grepSkipDirs are folder names never descended into.
var grepSkipDirs = map[string]bool{
	".git": true, "node_modules": true, ".venv": true, "_quarantine": true, "_tmp": true,
}

// grepStopwords are dropped from queries, as a person typing rg would.
var grepStopwords = map[string]bool{
	"about": true, "after": true, "all": true, "also": true, "and": true, "any": true, "are": true,
	"been": true, "before": true, "being": true, "but": true, "can": true, "could": true,
	"did": true, "does": true, "each": true, "for": true, "from": true, "had": true, "has": true,
	"have": true, "here": true, "how": true, "into": true, "its": true, "just": true, "not": true,
	"now": true, "only": true, "our": true, "out": true, "over": true, "should": true, "some": true,
	"such": true, "than": true, "that": true, "the": true, "their": true, "them": true, "then": true,
	"there": true, "these": true, "they": true, "this": true, "those": true, "very": true,
	"was": true, "were": true, "what": true, "when": true, "where": true, "which": true, "who": true,
	"why": true, "will": true, "with": true, "would": true, "you": true, "your": true,
}

// Grep is the ripgrep-style baseline retriever: it ranks Markdown and Go
// files under a root by the number of distinct query words they contain
// (case-insensitive substring), then by total match count, then by path.
// File contents are read once and cached; Grep is safe for concurrent use.
type Grep struct {
	root    string
	exclude []string // slash paths relative to root, skipped with everything below

	mu    sync.Mutex
	files []grepFile // nil until loaded
}

type grepFile struct {
	doc   knowledge.DocID
	path  string
	kind  string
	text  string
	lower string
}

// NewGrep returns a baseline retriever over root, skipping the given
// root-relative slash paths (files or folders). Files are loaded on the
// first Find or by Load.
func NewGrep(root string, exclude ...string) *Grep {
	return &Grep{root: root, exclude: exclude}
}

func (g *Grep) excluded(rel string) bool {
	for _, e := range g.exclude {
		e = strings.TrimSuffix(e, "/")
		if rel == e || strings.HasPrefix(rel, e+"/") {
			return true
		}
	}
	return false
}

// Find implements knowledge.Retriever.
func (g *Grep) Find(ctx context.Context, q knowledge.Query) (knowledge.Result, error) {
	if err := ctx.Err(); err != nil {
		return knowledge.Result{}, err
	}
	files, err := g.load(ctx)
	if err != nil {
		return knowledge.Result{}, err
	}
	words := queryWords(q.Text)
	res := knowledge.Result{Docs: len(files)}
	if len(words) == 0 {
		return res, nil
	}
	k := q.K
	if k <= 0 {
		k = DefaultK
	}
	k = min(k, grepMaxK)

	type cand struct {
		f        *grepFile
		distinct int
		total    int
	}
	var cands []cand
	for i := range files {
		if i%256 == 0 {
			if err := ctx.Err(); err != nil {
				return knowledge.Result{}, err
			}
		}
		f := &files[i]
		if !q.Filter.Match(knowledge.Doc{ID: f.doc, Scope: scopeOf(f.doc), Path: f.path, Kind: f.kind}) {
			continue
		}
		c := cand{f: f}
		for _, w := range words {
			if n := strings.Count(f.lower, w); n > 0 {
				c.distinct++
				c.total += n
			}
		}
		if c.distinct > 0 {
			cands = append(cands, c)
		}
	}
	slices.SortFunc(cands, func(a, b cand) int {
		return cmp.Or(
			cmp.Compare(b.distinct, a.distinct),
			cmp.Compare(b.total, a.total),
			cmp.Compare(a.f.doc, b.f.doc),
		)
	})
	if len(cands) > k {
		cands = cands[:k]
	}
	res.Hits = make([]knowledge.Hit, 0, len(cands))
	for _, c := range cands {
		res.Hits = append(res.Hits, knowledge.Hit{
			Doc:      c.f.doc,
			Path:     c.f.path,
			Kind:     c.f.kind,
			Score:    float64(c.distinct) + float64(c.total)/float64(c.total+1),
			Sections: []knowledge.Section{firstMatch(c.f.text, words)},
			Why:      []string{fmt.Sprintf("grep: %d/%d words, %d matches", c.distinct, len(words), c.total)},
		})
	}
	return res, nil
}

// Load walks the root and caches file contents now, so that the first
// Find's latency does not include the walk. It is a no-op once loaded.
func (g *Grep) Load(ctx context.Context) error {
	_, err := g.load(ctx)
	return err
}

// load walks the root once; a failed walk is not cached so a later call
// can retry.
func (g *Grep) load(ctx context.Context) ([]grepFile, error) {
	g.mu.Lock()
	defer g.mu.Unlock()
	if g.files != nil {
		return g.files, nil
	}
	files := []grepFile{}
	err := filepath.WalkDir(g.root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == g.root {
				return err
			}
			if d != nil && d.IsDir() {
				return fs.SkipDir
			}
			return nil // unreadable entry: skip, as rg does
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		rel, err := filepath.Rel(g.root, p)
		if err != nil {
			return fmt.Errorf("relative path of %s: %w", p, err)
		}
		rel = filepath.ToSlash(rel)
		if d.IsDir() {
			// Like rg's defaults (and ingest), skip hidden directories.
			if p != g.root && (grepSkipDirs[d.Name()] || strings.HasPrefix(d.Name(), ".") || g.excluded(rel)) {
				return fs.SkipDir
			}
			return nil
		}
		kind := grepKind(d.Name())
		if kind == "" || !d.Type().IsRegular() || g.excluded(rel) {
			return nil
		}
		info, err := d.Info()
		if err != nil || info.Size() > grepMaxFileSize {
			return nil
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return nil
		}
		text := string(data)
		files = append(files, grepFile{
			doc:   knowledge.DocID(rel),
			path:  p,
			kind:  kind,
			text:  text,
			lower: strings.ToLower(text),
		})
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("bench: grep walk %s: %w", g.root, err)
	}
	slices.SortFunc(files, func(a, b grepFile) int { return cmp.Compare(a.doc, b.doc) })
	g.files = files
	return files, nil
}

func grepKind(name string) string {
	switch strings.ToLower(filepath.Ext(name)) {
	case ".md":
		return "md"
	case ".go":
		return "go"
	}
	return ""
}

func scopeOf(doc knowledge.DocID) string {
	if i := strings.IndexByte(string(doc), '/'); i >= 0 {
		return string(doc[:i])
	}
	return ""
}

// firstMatch returns the first line containing any query word.
func firstMatch(text string, words []string) knowledge.Section {
	n := 0
	for line := range strings.Lines(text) {
		n++
		lower := strings.ToLower(line)
		var hit []string
		for _, w := range words {
			if strings.Contains(lower, w) {
				hit = append(hit, w)
			}
		}
		if len(hit) > 0 {
			return knowledge.Section{
				Lines:   [2]int{n, n},
				Snippet: truncateRunes(strings.TrimSpace(line), grepSnippetLen),
				Why:     []string{"grep: " + strings.Join(hit, ", ")},
			}
		}
	}
	// Unreachable for files that matched: a word never spans a line break
	// because query words contain no whitespace.
	return knowledge.Section{}
}

func truncateRunes(s string, n int) string {
	if utf8.RuneCountInString(s) <= n {
		return s
	}
	i := 0
	for range n {
		_, size := utf8.DecodeRuneInString(s[i:])
		i += size
	}
	return s[:i]
}

// queryWords lower-cases q and splits it into unique words of letters,
// digits and inner -_. connectors, dropping short words and stopwords.
func queryWords(q string) []string {
	var out []string
	fields := strings.FieldsFunc(strings.ToLower(q), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '-' && r != '_' && r != '.'
	})
	for _, w := range fields {
		w = strings.Trim(w, "-_.")
		if utf8.RuneCountInString(w) < grepMinWordLen || grepStopwords[w] || slices.Contains(out, w) {
			continue
		}
		out = append(out, w)
	}
	return out
}

var _ knowledge.Retriever = (*Grep)(nil)

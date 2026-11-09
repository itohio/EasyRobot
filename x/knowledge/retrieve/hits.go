package retrieve

import (
	"cmp"
	"fmt"
	"slices"
	"strings"
	"unicode"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/token"
)

// section is a chunk picked for a hit with the query terms it matches.
type section struct {
	chunk   knowledge.Chunk
	matched []string
}

// hits renders the selected candidates.
func (x *run) hits(sel []*cand) ([]knowledge.Hit, error) {
	out := make([]knowledge.Hit, 0, len(sel))
	x.selected = sel
	for _, c := range sel {
		secs, err := x.sections(c)
		if err != nil {
			return nil, err
		}
		c.sections = secs
		h := knowledge.Hit{
			Doc:   c.id,
			Path:  c.doc.Path,
			Kind:  c.doc.Kind,
			Title: c.doc.Title,
			Score: c.score,
		}
		for _, s := range secs {
			sc := knowledge.Section{
				Heading: strings.Join(s.chunk.Heading, " › "),
				Lines:   s.chunk.Lines,
				Snippet: snippet(s.chunk.Text, s.matched),
			}
			if sc.Heading == "" {
				sc.Heading = s.chunk.Symbol
			}
			if len(s.matched) > 0 {
				sc.Why = []string{"terms: " + strings.Join(s.matched, ", ")}
			}
			h.Sections = append(h.Sections, sc)
		}
		h.Why = append(h.Why, c.exact...)
		h.Why = append(h.Why, c.graphWhy...)
		if c.weight != 1 {
			h.Why = append(h.Why, fmt.Sprintf("weight ×%.3g", c.weight))
		}
		if c.recency >= 1.005 {
			h.Why = append(h.Why, fmt.Sprintf("recency ×%.3g", c.recency))
		}
		switch {
		case len(c.supersededBy) > 0:
			by := make([]string, len(c.supersededBy))
			for i, id := range c.supersededBy {
				by[i] = string(id)
			}
			h.Why = append(h.Why, fmt.Sprintf("superseded by %s ×%.3g", strings.Join(by, ", "), c.status))
		case c.status != 1:
			h.Why = append(h.Why, fmt.Sprintf("status %s ×%.3g", strings.ToLower(strings.TrimSpace(c.doc.Props["status"])), c.status))
		}
		if x.q.Explain {
			exact := 0.0
			if len(c.exact) > 0 {
				exact = 1
			}
			h.Signals = map[string]float64{
				"bm25_rank": float64(c.bm25Rank),
				"exact":     exact,
				"fused":     c.fused,
				"graph":     c.boost,
				"weight":    c.weight,
				"recency":   c.recency,
				"status":    c.status,
			}
		}
		out = append(out, h)
	}
	return out, nil
}

// sections returns up to maxSections best fused chunks of c, or its first
// chunk when it has none (graph-only hits), and records term coverage.
func (x *run) sections(c *cand) ([]section, error) {
	c.covered = map[string]bool{}
	for _, fc := range c.chunks {
		for _, t := range fc.matched {
			c.covered[t] = true
		}
	}
	var out []section
	for _, fc := range c.chunks {
		if len(out) == maxSections {
			break
		}
		ch, ok, err := x.chunk(fc)
		if err != nil {
			return nil, err
		}
		if !ok {
			continue
		}
		out = append(out, x.section(c, ch, fc.matched))
	}
	if len(out) == 0 {
		cs, err := x.docChunks(c.id)
		if err != nil {
			return nil, err
		}
		if len(cs) > 0 {
			out = append(out, x.section(c, cs[0], nil))
		}
	}
	return out, nil
}

func (x *run) section(c *cand, ch knowledge.Chunk, matched []string) section {
	var set map[string]bool
	if matched != nil {
		set = make(map[string]bool, len(matched))
		for _, t := range matched {
			set[t] = true
		}
	} else {
		set = chunkTerms(ch)
	}
	var ordered []string
	for _, t := range x.terms {
		if set[t] {
			ordered = append(ordered, t)
			c.covered[t] = true
		}
	}
	return section{chunk: ch, matched: ordered}
}

func (x *run) chunk(fc *fusedChunk) (knowledge.Chunk, bool, error) {
	for _, ch := range x.docChs[fc.doc] {
		if ch.ID == fc.id {
			return ch, true, nil
		}
	}
	ch, ok, err := x.r.store.GetChunk(x.ctx, fc.id)
	if err != nil {
		return knowledge.Chunk{}, false, fmt.Errorf("retrieve: get chunk %s: %w", fc.id, err)
	}
	return ch, ok, nil
}

// chunkTerms is the term set of a chunk's indexed fields except the path.
func chunkTerms(ch knowledge.Chunk) map[string]bool {
	set := map[string]bool{}
	for _, s := range [...]string{strings.Join(ch.Heading, " "), ch.Symbol, ch.Text} {
		for _, t := range token.Tokens(s) {
			set[t] = true
		}
	}
	return set
}

// confidence grades the answer and suggests terms when it is low.
func (x *run) confidence(hits []knowledge.Hit) (string, []string) {
	if len(hits) == 0 {
		return Low, nil
	}
	top := x.selected[0]
	if len(top.exact) > 0 {
		return High, nil
	}
	if len(x.terms) == 0 {
		return Medium, nil
	}
	coverage := float64(len(top.covered)) / float64(len(x.terms))
	if coverage >= highCoverage && (len(hits) == 1 || hits[0].Score >= highMargin*hits[1].Score) {
		return High, nil
	}
	if coverage < lowCoverage {
		return Low, x.tryTerms()
	}
	return Medium, nil
}

// tryTerms returns the most frequent terms of the top hits' sections that
// are not in the query.
func (x *run) tryTerms() []string {
	counts := map[string]int{}
	for _, c := range x.selected[:min(len(x.selected), tryTermHits)] {
		for _, s := range c.sections {
			for _, t := range token.Tokens(s.chunk.Text) {
				if len(t) >= 3 && !slices.Contains(x.terms, t) {
					counts[t]++
				}
			}
		}
	}
	terms := make([]string, 0, len(counts))
	for t := range counts {
		terms = append(terms, t)
	}
	slices.SortFunc(terms, func(a, b string) int {
		return cmp.Or(cmp.Compare(counts[b], counts[a]), cmp.Compare(a, b))
	})
	return terms[:min(len(terms), maxTryTerms)]
}

// snippet returns up to snippetRunes runes of text with whitespace
// collapsed, centred near the first word carrying a matched term.
func snippet(text string, matched []string) string {
	rs := []rune(strings.Join(strings.Fields(text), " "))
	if len(rs) <= snippetRunes {
		return string(rs)
	}
	pos := firstMatch(rs, matched)
	start := min(max(pos-snippetRunes/4, 0), len(rs)-snippetRunes)
	end := start + snippetRunes
	if start > 0 {
		start++ // room for the leading ellipsis
		// Prefer starting on a word boundary.
		for i := start; i < pos && i < start+20; i++ {
			if rs[i] == ' ' {
				start = i + 1
				break
			}
		}
	}
	if end < len(rs) {
		end-- // room for the trailing ellipsis
	}
	var b strings.Builder
	if start > 0 {
		b.WriteString("…")
	}
	b.WriteString(strings.TrimSpace(string(rs[start:end])))
	if end < len(rs) {
		b.WriteString("…")
	}
	return b.String()
}

// firstMatch returns the rune offset of the first word of rs whose tokens
// include a matched term, or 0.
func firstMatch(rs []rune, matched []string) int {
	if len(matched) == 0 {
		return 0
	}
	for i := 0; i < len(rs); {
		if !isWordRune(rs[i]) {
			i++
			continue
		}
		j := i
		for j < len(rs) && (isWordRune(rs[j]) || isConnector(rs[j])) {
			j++
		}
		for _, t := range token.Tokens(string(rs[i:j])) {
			if slices.Contains(matched, t) {
				return i
			}
		}
		i = j
	}
	return 0
}

func isWordRune(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) }

func isConnector(r rune) bool { return r == '_' || r == '-' || r == '.' }

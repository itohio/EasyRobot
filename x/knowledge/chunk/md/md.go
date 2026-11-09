// Package md is the Markdown chunker: it splits a document into heading
// sections with breadcrumbs and exact line spans, parses flat YAML-ish
// frontmatter, and extracts ids, tags, wikilinks and relative Markdown links.
package md

import (
	"crypto/sha256"
	"net/url"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"unicode"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/token"
)

const (
	defaultMaxTokens = 800
	defaultMinTokens = 12
)

// Option configures a Chunker.
type Option func(*Chunker)

// WithMaxTokens sets the token count above which a section is split at
// blank lines (default 800). Values < 1 are ignored.
func WithMaxTokens(n int) Option {
	return func(c *Chunker) {
		if n > 0 {
			c.maxTokens = n
		}
	}
}

// WithMinTokens sets the body token count below which a section is merged
// into the following sibling or child section (default 12). Values < 0 are
// ignored; 0 disables merging.
func WithMinTokens(n int) Option {
	return func(c *Chunker) {
		if n >= 0 {
			c.minTokens = n
		}
	}
}

// Chunker splits Markdown files. It is immutable after New and safe for
// concurrent use.
type Chunker struct {
	maxTokens int
	minTokens int
}

var _ knowledge.Chunker = (*Chunker)(nil)

// New returns a Markdown chunker.
func New(opts ...Option) *Chunker {
	c := &Chunker{maxTokens: defaultMaxTokens, minTokens: defaultMinTokens}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Accepts reports whether p has a .md or .markdown extension.
func (c *Chunker) Accepts(p string) bool {
	switch strings.ToLower(filepath.Ext(p)) {
	case ".md", ".markdown":
		return true
	}
	return false
}

// section is a heading section before merging and splitting; line indexes
// are 0-based inclusive.
type section struct {
	level      int // 0 for the preamble
	crumbs     []string
	start, end int
	body       int // tokens excluding the heading line
}

// Chunk splits src into heading sections and extracts edges and links.
func (c *Chunker) Chunk(d knowledge.Doc, src []byte) (knowledge.ChunkResult, error) {
	lines := splitLines(string(src))
	props, lists, bodyStart := parseFrontmatter(lines)
	code := fencedLines(lines, bodyStart)

	res := knowledge.ChunkResult{Props: props}

	// Headings and title.
	type heading struct {
		line, level int
		text        string
	}
	var heads []heading
	for i := bodyStart; i < len(lines); i++ {
		if code[i] {
			continue
		}
		if lvl, txt, ok := parseATX(lines[i]); ok {
			heads = append(heads, heading{i, lvl, txt})
		}
	}
	res.Title = strings.TrimSpace(props["title"])
	if res.Title == "" {
		for _, h := range heads {
			if h.level == 1 && h.text != "" {
				res.Title = h.text
				break
			}
		}
	}
	if res.Title == "" {
		res.Title = baseName(d)
	}

	// Raw sections.
	var secs []section
	secs = append(secs, section{level: 0, crumbs: []string{res.Title}, start: bodyStart})
	type frame struct {
		level int
		text  string
	}
	var stack []frame
	for _, h := range heads {
		secs[len(secs)-1].end = h.line - 1
		for len(stack) > 0 && stack[len(stack)-1].level >= h.level {
			stack = stack[:len(stack)-1]
		}
		stack = append(stack, frame{h.level, h.text})
		crumbs := make([]string, len(stack))
		for i, f := range stack {
			crumbs[i] = f.text
		}
		secs = append(secs, section{level: h.level, crumbs: crumbs, start: h.line})
	}
	secs[len(secs)-1].end = len(lines) - 1

	kept := secs[:0]
	for i, s := range secs {
		for s.start <= s.end && isBlank(lines[s.start]) {
			s.start++
		}
		for s.end >= s.start && isBlank(lines[s.end]) {
			s.end--
		}
		if s.start > s.end {
			continue
		}
		from := s.start
		if i > 0 { // heading sections: exclude the heading line from the body
			from++
		}
		for j := from; j <= s.end; j++ {
			s.body += token.Count(lines[j])
		}
		kept = append(kept, s)
	}
	secs = kept

	// Merge tiny sections into the following sibling or child.
	var merged []section
	carry, carryBody := -1, 0
	for i, s := range secs {
		if carry >= 0 {
			s.start, s.body = carry, s.body+carryBody
			carry, carryBody = -1, 0
		}
		if s.body < c.minTokens && i+1 < len(secs) && secs[i+1].level >= s.level {
			carry, carryBody = s.start, s.body
			continue
		}
		merged = append(merged, s)
	}

	// Split long sections and emit chunks.
	for _, s := range merged {
		for _, p := range splitSection(lines, code, s.start, s.end, c.maxTokens) {
			text := strings.Join(lines[p[0]:p[1]+1], "\n")
			ord := len(res.Chunks)
			res.Chunks = append(res.Chunks, knowledge.Chunk{
				ID:      knowledge.NewChunkID(d.ID, ord, text),
				Doc:     d.ID,
				Ord:     ord,
				Heading: slices.Clone(s.crumbs),
				Lines:   [2]int{p[0] + 1, p[1] + 1},
				Text:    text,
				Hash:    sha256.Sum256([]byte(text)),
				Tokens:  token.Count(text),
			})
		}
	}

	res.Edges, res.Links = extract(d.ID, lines, code, bodyStart, props, lists)
	return res, nil
}

// extract finds Defines/Tagged/Mentions edges and raw links.
func extract(doc knowledge.DocID, lines []string, code []bool, bodyStart int, props map[string]string, lists map[string][]string) ([]knowledge.Edge, []knowledge.Link) {
	from := knowledge.DocNode(doc)
	var edges []knowledge.Edge
	seen := map[knowledge.NodeID]bool{}
	add := func(kind knowledge.EdgeKind, to knowledge.NodeID) {
		if seen[to] {
			return
		}
		seen[to] = true
		edges = append(edges, knowledge.Edge{From: from, To: to, Kind: kind, Weight: 1, Source: doc})
	}

	ownID := ""
	if id := strings.TrimSpace(props["id"]); token.IsID(id) {
		ownID = strings.ToUpper(id)
		add(knowledge.Defines, knowledge.IDNode(id))
	}

	tags, isList := lists["tags"]
	if !isList {
		tags = strings.FieldsFunc(props["tags"], func(r rune) bool { return r == ',' || unicode.IsSpace(r) })
	}
	for _, t := range tags {
		if t = strings.TrimPrefix(strings.TrimSpace(t), "#"); t != "" {
			add(knowledge.Tagged, knowledge.TagNode(t))
		}
	}

	var links []knowledge.Link
	var mentions []string
	for i := bodyStart; i < len(lines); i++ {
		if code[i] {
			continue
		}
		mentions = append(mentions, token.IDs(lines[i])...)
		prose := stripInlineCode(lines[i])
		for _, t := range inlineTags(prose) {
			add(knowledge.Tagged, knowledge.TagNode(t))
		}
		links = append(links, lineLinks(prose, i+1)...)
	}
	for _, id := range mentions {
		if id != ownID {
			add(knowledge.Mentions, knowledge.IDNode(id))
		}
	}
	return edges, links
}

var (
	wikiRe   = regexp.MustCompile(`!?\[\[([^\[\]\n]+)\]\]`)
	mdLinkRe = regexp.MustCompile(`!?\[[^\]]*\]\(\s*(<[^>]*>|[^)\s]*)(?:\s+(?:"[^"]*"|'[^']*'))?\s*\)`)
	schemeRe = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9+.\-]*:`)
)

// lineLinks returns the links of one prose line in column order.
func lineLinks(s string, line int) []knowledge.Link {
	type at struct {
		pos  int
		link knowledge.Link
	}
	var found []at
	for _, m := range wikiRe.FindAllStringSubmatchIndex(s, -1) {
		t := s[m[2]:m[3]]
		t, _, _ = strings.Cut(t, "|")
		t, _, _ = strings.Cut(t, "#")
		if t = strings.TrimSpace(t); t != "" {
			found = append(found, at{m[0], knowledge.Link{Target: t, Kind: knowledge.WikiLink, Line: line}})
		}
	}
	for _, m := range mdLinkRe.FindAllStringSubmatchIndex(s, -1) {
		t := strings.TrimSpace(s[m[2]:m[3]])
		t = strings.TrimSuffix(strings.TrimPrefix(t, "<"), ">")
		if t == "" || strings.HasPrefix(t, "#") || schemeRe.MatchString(t) {
			continue
		}
		t, _, _ = strings.Cut(t, "#")
		if u, err := url.PathUnescape(t); err == nil {
			t = u
		}
		if t = strings.TrimSpace(t); t != "" {
			found = append(found, at{m[0], knowledge.Link{Target: t, Kind: knowledge.MarkdownLink, Line: line}})
		}
	}
	sort.SliceStable(found, func(i, j int) bool { return found[i].pos < found[j].pos })
	out := make([]knowledge.Link, len(found))
	for i, f := range found {
		out[i] = f.link
	}
	return out
}

// inlineTags returns #tags in s: '#' at line start or after whitespace,
// followed by letters, digits, '_', '-' or '/', not all digits.
func inlineTags(s string) []string {
	var out []string
	rs := []rune(s)
	for i := 0; i < len(rs); i++ {
		if rs[i] != '#' || (i > 0 && !unicode.IsSpace(rs[i-1])) {
			continue
		}
		j := i + 1
		for j < len(rs) && (unicode.IsLetter(rs[j]) || unicode.IsDigit(rs[j]) || rs[j] == '_' || rs[j] == '-' || rs[j] == '/') {
			j++
		}
		tag := strings.TrimRight(string(rs[i+1:j]), "-/")
		i = j - 1
		if tag == "" || tag[0] == '-' || tag[0] == '/' {
			continue
		}
		if strings.IndexFunc(tag, func(r rune) bool { return !unicode.IsDigit(r) }) < 0 {
			continue
		}
		out = append(out, tag)
	}
	return out
}

// stripInlineCode blanks inline code spans (one or more backticks matched
// by an equal run) on a single line.
func stripInlineCode(s string) string {
	if !strings.Contains(s, "`") {
		return s
	}
	b := []byte(s)
	for i := 0; i < len(b); {
		if b[i] != '`' {
			i++
			continue
		}
		n := runLen(b, i, '`')
		closeAt := -1
		for j := i + n; j < len(b); {
			if b[j] != '`' {
				j++
				continue
			}
			m := runLen(b, j, '`')
			if m == n {
				closeAt = j
				break
			}
			j += m
		}
		if closeAt < 0 {
			i += n
			continue
		}
		for k := i; k < closeAt+n; k++ {
			b[k] = ' '
		}
		i = closeAt + n
	}
	return string(b)
}

func runLen(b []byte, i int, c byte) int {
	n := 0
	for i+n < len(b) && b[i+n] == c {
		n++
	}
	return n
}

// splitSection returns 0-based inclusive line ranges covering [start, end]
// whose token counts do not exceed max where possible. Splits happen at
// blank lines outside fenced code; a single block over max is split at
// line boundaries.
func splitSection(lines []string, code []bool, start, end, max int) [][2]int {
	total := 0
	for i := start; i <= end; i++ {
		total += token.Count(lines[i])
	}
	if total <= max {
		return [][2]int{{start, end}}
	}
	// Blocks: maximal runs of lines not separated by a blank line outside
	// fences.
	type block struct{ a, b, tok int }
	var blocks []block
	for i := start; i <= end; {
		if isBlank(lines[i]) && !code[i] {
			i++
			continue
		}
		bl := block{a: i, b: i}
		for i <= end && !(isBlank(lines[i]) && !code[i]) {
			bl.b = i
			bl.tok += token.Count(lines[i])
			i++
		}
		blocks = append(blocks, bl)
	}
	var out [][2]int
	cur, curTok := [2]int{-1, -1}, 0
	push := func(a, b, tok int) {
		if cur[0] >= 0 && curTok+tok > max {
			out = append(out, cur)
			cur, curTok = [2]int{-1, -1}, 0
		}
		if cur[0] < 0 {
			cur[0] = a
		}
		cur[1] = b
		curTok += tok
	}
	for _, bl := range blocks {
		if bl.tok <= max {
			push(bl.a, bl.b, bl.tok)
			continue
		}
		for i := bl.a; i <= bl.b; i++ {
			push(i, i, token.Count(lines[i]))
		}
	}
	if cur[0] >= 0 {
		out = append(out, cur)
	}
	return out
}

// fencedLines marks lines that belong to fenced code blocks (``` or ~~~),
// delimiters included. An unclosed fence runs to the end of the file.
func fencedLines(lines []string, from int) []bool {
	code := make([]bool, len(lines))
	var fc byte
	fn := 0
	for i := from; i < len(lines); i++ {
		t, ok := trimIndent(lines[i])
		if fn == 0 {
			if ok && len(t) >= 3 && (t[0] == '`' || t[0] == '~') {
				if n := runLen([]byte(t), 0, t[0]); n >= 3 && (t[0] == '~' || !strings.Contains(t[n:], "`")) {
					fc, fn = t[0], n
					code[i] = true
				}
			}
			continue
		}
		code[i] = true
		if ok && len(t) >= fn && t[0] == fc {
			if n := runLen([]byte(t), 0, fc); n >= fn && strings.TrimSpace(t[n:]) == "" {
				fn = 0
			}
		}
	}
	return code
}

// trimIndent strips up to three leading spaces; ok is false when the line
// is indented by four or more (an indented code line).
func trimIndent(s string) (string, bool) {
	n := 0
	for n < len(s) && s[n] == ' ' {
		n++
	}
	if n > 3 {
		return s, false
	}
	return s[n:], true
}

// parseATX parses an ATX heading line.
func parseATX(s string) (level int, text string, ok bool) {
	t, ok := trimIndent(s)
	if !ok {
		return 0, "", false
	}
	level = runLen([]byte(t), 0, '#')
	if level < 1 || level > 6 {
		return 0, "", false
	}
	rest := t[level:]
	if rest != "" && rest[0] != ' ' && rest[0] != '\t' {
		return 0, "", false
	}
	rest = strings.TrimSpace(rest)
	// Optional closing sequence: spaces then #s.
	if k := strings.TrimRight(rest, "#"); k != rest && (k == "" || strings.HasSuffix(k, " ") || strings.HasSuffix(k, "\t")) {
		rest = strings.TrimSpace(k)
	}
	return level, rest, true
}

// parseFrontmatter parses a leading --- block of flat "key: value" lines,
// inline lists "key: [a, b]" and block lists ("key:" then "- a" lines).
// It returns every key's value (lists joined with ","), the list items by
// key, and the index of the first line after the block.
func parseFrontmatter(lines []string) (map[string]string, map[string][]string, int) {
	props := map[string]string{}
	lists := map[string][]string{}
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return props, lists, 0
	}
	end := -1
	for i := 1; i < len(lines); i++ {
		if t := strings.TrimSpace(lines[i]); t == "---" || t == "..." {
			end = i
			break
		}
	}
	if end < 0 {
		return props, lists, 0
	}
	last := ""
	for _, l := range lines[1:end] {
		t := strings.TrimSpace(l)
		if t == "" || strings.HasPrefix(t, "#") {
			continue
		}
		if t == "-" || strings.HasPrefix(t, "- ") {
			if last != "" {
				if item := unquote(strings.TrimSpace(strings.TrimPrefix(t, "-"))); item != "" {
					lists[last] = append(lists[last], item)
					props[last] = strings.Join(lists[last], ",")
				}
			}
			continue
		}
		if l[0] == ' ' || l[0] == '\t' { // nested structure: unsupported
			continue
		}
		k, v, ok := strings.Cut(t, ":")
		k = strings.TrimSpace(k)
		if !ok || k == "" {
			last = ""
			continue
		}
		v = strings.TrimSpace(v)
		last = ""
		switch {
		case v == "":
			props[k] = ""
			last = k
		case strings.HasPrefix(v, "[") && strings.HasSuffix(v, "]"):
			items := []string{}
			for _, it := range strings.Split(v[1:len(v)-1], ",") {
				if it = unquote(strings.TrimSpace(it)); it != "" {
					items = append(items, it)
				}
			}
			lists[k] = items
			props[k] = strings.Join(items, ",")
		default:
			props[k] = unquote(v)
		}
	}
	return props, lists, end + 1
}

func unquote(s string) string {
	if len(s) >= 2 && (s[0] == '"' || s[0] == '\'') && s[len(s)-1] == s[0] {
		return s[1 : len(s)-1]
	}
	return s
}

// splitLines splits s into lines without terminators ("\n" or "\r\n"); a
// trailing newline does not start an extra line.
func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	lines := strings.Split(strings.TrimSuffix(s, "\n"), "\n")
	for i, l := range lines {
		lines[i] = strings.TrimSuffix(l, "\r")
	}
	return lines
}

func isBlank(s string) bool { return strings.TrimSpace(s) == "" }

// baseName is the document's file name without extension.
func baseName(d knowledge.Doc) string {
	b := path.Base(string(d.ID))
	if d.ID == "" {
		b = filepath.Base(d.Path)
	}
	return strings.TrimSuffix(b, path.Ext(b))
}

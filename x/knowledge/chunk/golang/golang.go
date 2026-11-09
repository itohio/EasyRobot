// Package golang is the Go source chunker: one file-header chunk (package
// doc, clause and imports) plus one chunk per top-level declaration with its
// doc comment, Defines edges for declared symbols and Imports edges for
// import paths. Files that do not parse fall back to overlapping line
// windows.
package golang

import (
	"crypto/sha256"
	"go/ast"
	"go/parser"
	gotoken "go/token"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/itohio/EasyRobot/x/knowledge"
	"github.com/itohio/EasyRobot/x/knowledge/token"
)

const (
	defaultMaxTokens = 800
	defaultWindow    = 60
	defaultOverlap   = 10
)

// Option configures a Chunker.
type Option func(*Chunker)

// WithMaxTokens sets the token count above which a declaration is split at
// blank lines (default 800). Values < 1 are ignored.
func WithMaxTokens(n int) Option {
	return func(c *Chunker) {
		if n > 0 {
			c.maxTokens = n
		}
	}
}

// WithWindow sets the line window and overlap used for files that do not
// parse (default 60 lines, 10 overlap). Ignored unless size > 0 and
// 0 <= overlap < size.
func WithWindow(size, overlap int) Option {
	return func(c *Chunker) {
		if size > 0 && overlap >= 0 && overlap < size {
			c.window, c.overlap = size, overlap
		}
	}
}

// Chunker splits Go source files. It is immutable after New and safe for
// concurrent use.
type Chunker struct {
	maxTokens       int
	window, overlap int
}

var _ knowledge.Chunker = (*Chunker)(nil)

// New returns a Go chunker.
func New(opts ...Option) *Chunker {
	c := &Chunker{maxTokens: defaultMaxTokens, window: defaultWindow, overlap: defaultOverlap}
	for _, o := range opts {
		o(c)
	}
	return c
}

// Accepts reports whether p has a .go extension.
func (c *Chunker) Accepts(p string) bool {
	return strings.EqualFold(filepath.Ext(p), ".go")
}

// decl is one chunkable unit; lines are 1-based inclusive.
type decl struct {
	symbol   string
	from, to int
}

// Chunk parses src and splits it into header and declaration chunks. Parse
// errors are not returned: the file is chunked into line windows instead.
func (c *Chunker) Chunk(d knowledge.Doc, src []byte) (knowledge.ChunkResult, error) {
	lines := splitLines(string(src))
	base := baseName(d)
	props := map[string]string{}
	if strings.HasSuffix(base, "_test.go") {
		props["test"] = "true"
	}

	fset := gotoken.NewFileSet()
	f, err := parser.ParseFile(fset, base, src, parser.ParseComments|parser.SkipObjectResolution)
	if err != nil {
		return c.fallback(d, src, lines, base, props), nil
	}
	pkg := f.Name.Name
	props["package"] = pkg
	if ast.IsGenerated(f) {
		props["generated"] = "true"
	}
	res := knowledge.ChunkResult{Title: pkg, Props: props}
	// Raw positions: //line directives (goyacc, cgo, templ) must not remap spans.
	line := func(p gotoken.Pos) int { return fset.PositionFor(p, false).Line }

	from := knowledge.DocNode(d.ID)
	seen := map[knowledge.NodeID]bool{}
	add := func(kind knowledge.EdgeKind, to knowledge.NodeID) {
		if !seen[to] {
			seen[to] = true
			res.Edges = append(res.Edges, knowledge.Edge{From: from, To: to, Kind: kind, Weight: 1, Source: d.ID})
		}
	}
	define := func(name string) {
		if name != "_" {
			add(knowledge.Defines, knowledge.SymNode(pkg+"."+name))
		}
	}

	// Header: package doc + clause + imports.
	head := decl{from: line(f.Package), to: line(f.Name.End())}
	if f.Doc != nil {
		head.from = line(f.Doc.Pos())
	}
	for _, imp := range f.Imports {
		if p, err := strconv.Unquote(imp.Path.Value); err == nil {
			add(knowledge.Imports, knowledge.PkgNode(p))
		}
	}
	var decls []decl
	for _, dl := range f.Decls {
		switch dl := dl.(type) {
		case *ast.FuncDecl:
			name := dl.Name.Name
			if dl.Recv != nil && len(dl.Recv.List) > 0 {
				if r := recvName(dl.Recv.List[0].Type); r != "" {
					name = r + "." + name
				}
			}
			define(name)
			decls = append(decls, decl{symbol: pkg + "." + name, from: line(docStart(dl.Doc, dl.Pos())), to: line(dl.End())})
		case *ast.GenDecl:
			switch dl.Tok {
			case gotoken.IMPORT:
				head.to = max(head.to, line(dl.End()))
			case gotoken.TYPE:
				for k, sp := range dl.Specs {
					ts, ok := sp.(*ast.TypeSpec)
					if !ok {
						continue
					}
					define(ts.Name.Name)
					s, e := docStart(ts.Doc, ts.Pos()), ts.End()
					if ts.Comment != nil {
						e = max(e, ts.Comment.End())
					}
					if k == 0 || !dl.Lparen.IsValid() {
						s = docStart(dl.Doc, dl.Pos())
					}
					if k == len(dl.Specs)-1 || !dl.Lparen.IsValid() {
						e = dl.End()
					}
					decls = append(decls, decl{symbol: pkg + "." + ts.Name.Name, from: line(s), to: line(e)})
				}
			case gotoken.CONST, gotoken.VAR:
				first := ""
				for _, sp := range dl.Specs {
					vs, ok := sp.(*ast.ValueSpec)
					if !ok {
						continue
					}
					for _, n := range vs.Names {
						define(n.Name)
						if first == "" && n.Name != "_" {
							first = n.Name
						}
					}
				}
				if first == "" {
					first = "_"
				}
				decls = append(decls, decl{symbol: pkg + "." + first, from: line(docStart(dl.Doc, dl.Pos())), to: line(dl.End())})
			}
		}
	}

	c.emit(&res, d.ID, lines, []string{pkg}, head)
	for _, dc := range decls {
		c.emit(&res, d.ID, lines, []string{pkg, dc.symbol}, dc)
	}
	return res, nil
}

// emit appends the chunks of one unit, splitting it at blank lines when it
// exceeds the token limit.
func (c *Chunker) emit(res *knowledge.ChunkResult, doc knowledge.DocID, lines []string, heading []string, dc decl) {
	dc.from = max(dc.from, 1)
	dc.to = min(dc.to, len(lines))
	if dc.from > dc.to {
		return
	}
	for _, p := range splitBlocks(lines, dc.from-1, dc.to-1, c.maxTokens) {
		appendChunk(res, doc, lines, heading, dc.symbol, p[0], p[1])
	}
}

// fallback chunks an unparsable file into overlapping line windows.
func (c *Chunker) fallback(d knowledge.Doc, src []byte, lines []string, base string, props map[string]string) knowledge.ChunkResult {
	res := knowledge.ChunkResult{Title: strings.TrimSuffix(base, path.Ext(base)), Props: props}
	if f, err := parser.ParseFile(gotoken.NewFileSet(), base, src, parser.PackageClauseOnly|parser.ParseComments); err == nil {
		res.Title = f.Name.Name
		props["package"] = f.Name.Name
		if ast.IsGenerated(f) {
			props["generated"] = "true"
		}
	}
	heading := []string{base}
	step := c.window - c.overlap
	for s := 0; s < len(lines); s += step {
		e := min(s+c.window, len(lines)) - 1
		appendChunk(&res, d.ID, lines, heading, "", s, e)
		if e == len(lines)-1 {
			break
		}
	}
	return res
}

// appendChunk adds lines[a..b] (0-based inclusive) as the next chunk.
func appendChunk(res *knowledge.ChunkResult, doc knowledge.DocID, lines []string, heading []string, symbol string, a, b int) {
	text := strings.Join(lines[a:b+1], "\n")
	ord := len(res.Chunks)
	res.Chunks = append(res.Chunks, knowledge.Chunk{
		ID:      knowledge.NewChunkID(doc, ord, text),
		Doc:     doc,
		Ord:     ord,
		Heading: append([]string(nil), heading...),
		Symbol:  symbol,
		Lines:   [2]int{a + 1, b + 1},
		Text:    text,
		Hash:    sha256.Sum256([]byte(text)),
		Tokens:  token.Count(text),
	})
}

// splitBlocks returns 0-based inclusive line ranges covering [start, end],
// packing blank-line-separated blocks greedily up to max tokens; a single
// block over max is split at line boundaries.
func splitBlocks(lines []string, start, end, max int) [][2]int {
	total := 0
	for i := start; i <= end; i++ {
		total += token.Count(lines[i])
	}
	if total <= max {
		return [][2]int{{start, end}}
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
	for i := start; i <= end; {
		if isBlank(lines[i]) {
			i++
			continue
		}
		a, tok := i, 0
		for i <= end && !isBlank(lines[i]) {
			tok += token.Count(lines[i])
			i++
		}
		if tok <= max {
			push(a, i-1, tok)
			continue
		}
		for j := a; j < i; j++ {
			push(j, j, token.Count(lines[j]))
		}
	}
	if cur[0] >= 0 {
		out = append(out, cur)
	}
	return out
}

// recvName returns the receiver's base type name without pointer or type
// parameters.
func recvName(e ast.Expr) string {
	for {
		switch t := e.(type) {
		case *ast.StarExpr:
			e = t.X
		case *ast.ParenExpr:
			e = t.X
		case *ast.IndexExpr:
			e = t.X
		case *ast.IndexListExpr:
			e = t.X
		case *ast.Ident:
			return t.Name
		default:
			return ""
		}
	}
}

func docStart(doc *ast.CommentGroup, pos gotoken.Pos) gotoken.Pos {
	if doc != nil {
		return doc.Pos()
	}
	return pos
}

// splitLines splits s into lines without terminators; a trailing newline
// does not start an extra line.
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

// baseName is the document's file name.
func baseName(d knowledge.Doc) string {
	if d.ID == "" {
		return filepath.Base(d.Path)
	}
	return path.Base(string(d.ID))
}

package main

import (
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// Algorithm status vocabulary, in table order.
var statusOrder = []string{"tested", "implemented", "partial", "stub", "unverified"}

// Note is one Markdown file of the vault.
type Note struct {
	Rel    string // path relative to the docs dir, slash separated
	Name   string // base name without .md
	Folder string // first path segment ("" for root notes)
	Raw    string

	HasFM bool
	FM    map[string]*yaml.Node
	FMErr error

	BodyStart int // byte offset of the body in Raw
	BodyLine  int // 1-based file line of the first body line

	Lines    []string // body lines, original text
	Masked   []string // body lines, code fences and inline code blanked (same byte length)
	Headings []string // normalised heading texts
	BlockIDs []string // ^block ids
}

// Vault is the parsed docs tree.
type Vault struct {
	Dir           string
	Notes         []*Note
	byName        map[string][]*Note
	byPath        map[string]*Note
	files         map[string]bool // every non-hidden file, rel path
	VaultRevision string
}

// Load reads every Markdown note under dir, skipping hidden directories (.obsidian).
func Load(dir string) (*Vault, error) {
	v := &Vault{Dir: dir, byName: map[string][]*Note{}, byPath: map[string]*Note{}, files: map[string]bool{}}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if strings.HasPrefix(d.Name(), ".") && p != dir {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		v.files[rel] = true
		if !strings.HasSuffix(rel, ".md") {
			return nil
		}
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		v.Notes = append(v.Notes, parseNote(rel, string(raw)))
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(v.Notes, func(i, j int) bool { return v.Notes[i].Rel < v.Notes[j].Rel })
	for _, n := range v.Notes {
		v.byName[n.Name] = append(v.byName[n.Name], n)
		v.byPath[n.Rel] = n
	}
	for _, ns := range v.byName {
		sortNotes(ns)
	}
	if c := v.byPath["conventions.md"]; c != nil {
		v.VaultRevision = c.Str("vault_revision")
	}
	return v, nil
}

// sortNotes orders candidates by folder depth, then path, so the root
// index.md comes before tickets/index.md and ties are stable.
func sortNotes(ns []*Note) {
	sort.SliceStable(ns, func(i, j int) bool {
		di, dj := strings.Count(ns[i].Rel, "/"), strings.Count(ns[j].Rel, "/")
		if di != dj {
			return di < dj
		}
		return ns[i].Rel < ns[j].Rel
	})
}

func parseNote(rel, raw string) *Note {
	n := &Note{Rel: rel, Raw: raw, Name: strings.TrimSuffix(path.Base(rel), ".md"), FM: map[string]*yaml.Node{}}
	if i := strings.IndexByte(rel, '/'); i >= 0 {
		n.Folder = rel[:i]
	}
	if strings.HasPrefix(raw, "---\n") {
		for off := 4; off <= len(raw); {
			e := strings.IndexByte(raw[off:], '\n')
			line, next := raw[off:], len(raw)
			if e >= 0 {
				line, next = raw[off:off+e], off+e+1
			}
			if strings.TrimRight(line, " \t\r") == "---" {
				n.HasFM = true
				n.parseFM(raw[4:off])
				n.BodyStart = next
				break
			}
			if e < 0 {
				break
			}
			off = next
		}
	}
	n.BodyLine = strings.Count(raw[:n.BodyStart], "\n") + 1
	n.parseBody()
	return n
}

func (n *Note) parseFM(src string) {
	var doc yaml.Node
	if err := yaml.Unmarshal([]byte(src), &doc); err != nil {
		n.FMErr = err
		return
	}
	if len(doc.Content) == 0 {
		return
	}
	m := doc.Content[0]
	if m.Kind != yaml.MappingNode {
		n.FMErr = fmt.Errorf("frontmatter is not a mapping")
		return
	}
	for i := 0; i+1 < len(m.Content); i += 2 {
		n.FM[m.Content[i].Value] = m.Content[i+1]
	}
}

func (n *Note) parseBody() {
	body := n.Raw[n.BodyStart:]
	n.Lines = strings.Split(body, "\n")
	n.Masked = make([]string, len(n.Lines))
	var fenceCh byte
	fenceLen := 0
	for i, line := range n.Lines {
		t := strings.TrimLeft(line, " \t")
		if fenceLen > 0 {
			if c, k := fenceRun(t); c == fenceCh && k >= fenceLen && strings.TrimSpace(t[k:]) == "" {
				fenceLen = 0
			}
			n.Masked[i] = blank(line)
			continue
		}
		if c, k := fenceRun(t); k >= 3 && !(c == '`' && strings.ContainsRune(t[k:], '`')) {
			fenceCh, fenceLen = c, k
			n.Masked[i] = blank(line)
			continue
		}
		n.Masked[i] = maskInline(line)
		if h := headingRe.FindStringSubmatch(line); h != nil {
			n.Headings = append(n.Headings, normHeading(h[1]))
		}
		for _, m := range blockIDRe.FindAllStringSubmatch(n.Masked[i], -1) {
			n.BlockIDs = append(n.BlockIDs, strings.ToLower(m[1]))
		}
	}
}

var (
	headingRe = regexp.MustCompile(`^#{1,6}\s+(.*?)\s*#*\s*$`)
	blockIDRe = regexp.MustCompile(`(?:^|\s)\^([A-Za-z0-9-]+)\s*$`)
)

func fenceRun(t string) (byte, int) {
	if t == "" || (t[0] != '`' && t[0] != '~') {
		return 0, 0
	}
	k := 0
	for k < len(t) && t[k] == t[0] {
		k++
	}
	return t[0], k
}

func blank(s string) string { return strings.Repeat(" ", len(s)) }

// maskInline replaces inline code spans (including the backticks) with spaces.
func maskInline(s string) string {
	if !strings.Contains(s, "`") {
		return s
	}
	b := []byte(s)
	for i := 0; i < len(b); {
		if b[i] != '`' {
			i++
			continue
		}
		k := i
		for k < len(b) && b[k] == '`' {
			k++
		}
		run := k - i
		closeAt := -1
		for j := k; j < len(b); {
			if b[j] != '`' {
				j++
				continue
			}
			e := j
			for e < len(b) && b[e] == '`' {
				e++
			}
			if e-j == run {
				closeAt = e
				break
			}
			j = e
		}
		if closeAt < 0 {
			i = k
			continue
		}
		for x := i; x < closeAt; x++ {
			b[x] = ' '
		}
		i = closeAt
	}
	return string(b)
}

func normHeading(s string) string {
	s = strings.NewReplacer("`", "", "*", "").Replace(s)
	return strings.ToLower(strings.Join(strings.Fields(s), " "))
}

// MaskedBody joins the masked body lines; offsets match Raw[BodyStart:].
func (n *Note) MaskedBody() string { return strings.Join(n.Masked, "\n") }

// Has reports whether a frontmatter key is present and non-empty.
func (n *Note) Has(key string) bool {
	nd, ok := n.FM[key]
	if !ok || nd == nil {
		return false
	}
	if nd.Kind == yaml.ScalarNode {
		return nd.Tag != "!!null" && strings.TrimSpace(nd.Value) != ""
	}
	return true
}

// Str returns a frontmatter value as a string. An unquoted [[link]] (parsed by
// YAML as a nested flow sequence) is turned back into "[[link]]"; lists are
// joined with ", ".
func (n *Note) Str(key string) string {
	nd := n.FM[key]
	if nd == nil {
		return ""
	}
	return nodeStr(nd)
}

func nodeStr(nd *yaml.Node) string {
	switch nd.Kind {
	case yaml.ScalarNode:
		if nd.Tag == "!!null" {
			return ""
		}
		return strings.TrimSpace(nd.Value)
	case yaml.SequenceNode:
		if len(nd.Content) == 1 && nd.Content[0].Kind == yaml.SequenceNode && len(nd.Content[0].Content) == 1 {
			return "[[" + nodeStr(nd.Content[0].Content[0]) + "]]"
		}
		var parts []string
		for _, c := range nd.Content {
			parts = append(parts, nodeStr(c))
		}
		return strings.Join(parts, ", ")
	}
	return ""
}

// List returns a frontmatter value as a list of strings.
func (n *Note) List(key string) []string {
	nd := n.FM[key]
	if nd == nil {
		return nil
	}
	if nd.Kind == yaml.SequenceNode {
		var out []string
		for _, c := range nd.Content {
			if s := nodeStr(c); s != "" {
				out = append(out, s)
			}
		}
		return out
	}
	if s := nodeStr(nd); s != "" {
		return []string{s}
	}
	return nil
}

// Type is the frontmatter type.
func (n *Note) Type() string { return n.Str("type") }

// Foreign reports notes owned by another writer (owner set and not easyrobot);
// only their links are checked. In ticket notes `owner` is the assignee, so
// tickets are never foreign.
func (n *Note) Foreign() bool {
	if n.Folder == "tickets" || n.Type() == "ticket" {
		return false
	}
	o := strings.ToLower(n.Str("owner"))
	return o != "" && o != "easyrobot"
}

// FileLine converts a body line index to a 1-based file line.
func (n *Note) FileLine(i int) int { return n.BodyLine + i }

// linkTarget extracts the note name from "[[name|label]]" or "[[name#h]]".
func linkTarget(s string) string {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "[[")
	s = strings.TrimSuffix(s, "]]")
	if i := strings.IndexAny(s, "|#"); i >= 0 {
		s = s[:i]
	}
	return strings.TrimSuffix(strings.TrimSpace(s), "\\")
}

// Resolve finds the note a wikilink target names: a path relative to docs when
// it contains "/", else a base name. With several candidates the first in
// sortNotes order wins (so [[index]] is the root); Ambiguous reports them.
func (v *Vault) Resolve(target string) *Note {
	if cands := v.candidates(target); len(cands) > 0 {
		return cands[0]
	}
	return nil
}

// Ambiguous returns the candidate paths of a base-name target that more than
// one note matches; [[index]] is never ambiguous (it is the root index.md).
func (v *Vault) Ambiguous(target string) []string {
	t := strings.TrimSuffix(strings.TrimSpace(target), ".md")
	if strings.Contains(t, "/") || strings.EqualFold(t, "index") {
		return nil
	}
	cands := v.candidates(t)
	if len(cands) < 2 {
		return nil
	}
	paths := make([]string, len(cands))
	for i, n := range cands {
		paths[i] = n.Rel
	}
	return paths
}

func (v *Vault) candidates(target string) []*Note {
	t := strings.TrimSuffix(strings.TrimSpace(target), ".md")
	if t == "" {
		return nil
	}
	var cands []*Note
	if strings.Contains(t, "/") {
		t = strings.TrimPrefix(t, "/")
		if n := v.byPath[t+".md"]; n != nil {
			return []*Note{n}
		}
		for _, n := range v.Notes {
			if strings.HasSuffix(n.Rel, "/"+t+".md") || strings.EqualFold(n.Rel, t+".md") {
				cands = append(cands, n)
			}
		}
	} else if ns := v.byName[t]; len(ns) > 0 {
		return ns
	} else {
		for _, name := range sortedIDs(v.byName) {
			if strings.EqualFold(name, t) {
				cands = append(cands, v.byName[name]...)
			}
		}
	}
	sortNotes(cands)
	return cands
}

// fileExists resolves a non-Markdown embed/link target by path or base name.
func (v *Vault) fileExists(target string) bool {
	t := strings.TrimPrefix(target, "/")
	if v.files[t] {
		return true
	}
	for f := range v.files {
		if path.Base(f) == t || strings.HasSuffix(f, "/"+t) {
			return true
		}
	}
	return false
}

// ModuleOf returns the module name an algorithm note's `module` wikilink names
// (resolved note name when it resolves, else the raw target).
func (v *Vault) ModuleOf(n *Note) (string, *Note) {
	t := linkTarget(n.Str("module"))
	if t == "" {
		return "", nil
	}
	if m := v.Resolve(t); m != nil {
		return m.Name, m
	}
	return path.Base(t), nil
}

// skipped reports notes outside the rules (templates).
func (n *Note) template() bool { return n.Folder == "templates" }
